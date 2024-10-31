# import packages
import os
from ast import parse

from torch.utils.data import DataLoader
from profis.gen.dataset import Smiles2SmilesDataset
from profis.gen.generator import SMILES2SMILES
from profis.utils.split import scaffold_split
from profis.utils.modelinit import get_alphabet_len
import time
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.QED as QED
import torch
import wandb

from profis.gen.loss import TCE
from profis.utils.annealing import Annealer
from profis.utils.vectorizer import (
    SMILESVectorizer,
)
from profis.utils.finger import encode

# Suppress RDKit warnings
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def train(learn_rate=0.0002, encoding_size=32, kld_weight=0.01, batch_size=512, run_name='smiles2smiles'):

    print('learn_rate', learn_rate)
    print('encoding_size', encoding_size)
    print('kld_weight', kld_weight)
    print('batch_size', batch_size)
    print('run_name', run_name)

    train_size = 0.9
    random_seed = 42
    data_path = 'data/RNN_dataset_ECFP.parquet'
    dataloader_workers = 3
    use_cuda = True
    out_encoding = "smiles"
    val_size = 0.1
    train_percent = int(train_size * 100)
    val_percent = int(val_size * 100)

    cuda_available = torch.cuda.is_available() and use_cuda
    device = torch.device("cuda" if cuda_available else "cpu")

    # read dataset

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset file {data_path} not found")
    dataset = pd.read_parquet(data_path)

    # create a directory for this model weights if not there

    if not os.path.isdir(f"models/{run_name}"):
        if not os.path.isdir("models"):
            os.mkdir("models")
        os.mkdir(f"models/{run_name}")

    # if train_dataset not generated, perform scaffold split
    if not os.path.isfile(
        data_path.split(".")[0] + f"_train_{train_percent}.parquet"
    ) or not os.path.isfile(data_path.split(".")[0] + f"_val_{val_percent}.parquet"):
        print("Performing scaffold split...")
        train_df, val_df = scaffold_split(
            dataset, train_size, seed=random_seed, shuffle=True
        )
        train_df.to_parquet(data_path.split(".")[0] + f"_train_{train_percent}.parquet")
        val_df.to_parquet(data_path.split(".")[0] + f"_val_{val_percent}.parquet")
        print("Scaffold split complete")
    else:
        train_df = pd.read_parquet(
            data_path.split(".")[0] + f"_train_{train_percent}.parquet"
        )
        val_df = pd.read_parquet(
            data_path.split(".")[0] + f"_val_{val_percent}.parquet"
        )
    scoring_df = val_df

    # prepare dataloaders
    train_dataset = Smiles2SmilesDataset(train_df)
    val_dataset = Smiles2SmilesDataset(val_df)
    scoring_dataset = Smiles2SmilesDataset(scoring_df)

    print("Dataset size:", len(dataset))
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Scoring size:", len(scoring_dataset))

    val_batch_size = batch_size if batch_size < len(val_dataset) else len(val_dataset)
    scoring_batch_size = (
        batch_size if batch_size < len(scoring_dataset) else len(scoring_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
        num_workers=dataloader_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=val_batch_size,
        drop_last=True,
        num_workers=dataloader_workers,
    )
    scoring_loader = DataLoader(
        scoring_dataset,
        shuffle=False,
        batch_size=scoring_batch_size,
        drop_last=True,
        num_workers=dataloader_workers,
    )

    # Init model

    model = SMILES2SMILES(
        encoding_size=encoding_size,
        hidden_size=512,
        num_layers=2,
        output_size=get_alphabet_len('smiles'),
        dropout=0,
        teacher_ratio=0.2
    ).to(device)


    epochs = 500
    kld_backward = True
    start_epoch = 1
    kld_annealing = True
    annealing_max_epoch = 30
    annealing_shape = 'cosine'

    wandb.init(project="profis", name=run_name)

    annealing_agent = Annealer(annealing_max_epoch, annealing_shape)

    # Define dataframe for logging progress
    epochs_range = range(start_epoch, epochs + start_epoch)
    metrics = pd.DataFrame()

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = TCE(notation=out_encoding)

    print("Training PROFIS")
    print(f"Device: {device}")

    # Start training loop
    for epoch in epochs_range:
        model.train()
        start_time = time.time()
        print(f"Epoch: {epoch}")
        epoch_loss = 0
        kld_loss = 0
        for X in train_loader:
            X = X.to(device)
            optimizer.zero_grad()
            output, kld_loss = model(X, X, teacher_forcing=True)
            loss = criterion(X, output)
            kld_weighted = kld_loss * kld_weight
            if kld_annealing:
                kld_weighted = annealing_agent(kld_weighted)
            if kld_backward:
                (loss + kld_weighted).backward()
            else:
                loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion)

        if epoch % 10 == 0:
            start = time.time()
            mean_qed, mean_validity = get_scores(
                model, scoring_loader, device=device)
            end = time.time()
            print(f"QED + fp evaluated in {(end - start) / 60} minutes")
        else:
            mean_qed = None
            mean_validity = None

        metrics_row = pd.DataFrame(
            {
                "epoch": [epoch],
                "kld_loss": [kld_loss.item()],
                "kld_weighted": [kld_weighted.item()],
                "train_loss": [avg_loss],
                "val_loss": [val_loss],
                "mean_qed": [mean_qed],
                "mean_validity": [mean_validity],
            }
        )

        wandb.log({
            "epoch": epoch,
            "kld_loss": kld_loss.item(),
            "kld_weighted": kld_weighted.item(),
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "mean_qed": mean_qed,
            "mean_validity": mean_validity,
        })
        if kld_annealing:
            annealing_agent.step()

        # Update metrics df
        metrics = pd.concat([metrics, metrics_row], ignore_index=True, axis=0)
        if epoch % 50 == 0:
            # save model
            save_path = f"./models/{run_name}/epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv", index=False)
        end_time = time.time()
        loop_time = round((end_time - start_time) / 60, 2)  # in minutes

        print(f"Epoch {epoch} completed in {loop_time} minutes")

    return None


def evaluate(model, val_loader, criterion):
    """
    Evaluates the model on the validation set
    Args:
        model (nn.Module): EncoderDecoderV3 model
        val_loader (DataLoader): validation set loader
        criterion (nn.Module): loss function
    Returns:
        float: average loss on the validation set

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        for batch_idx, X in enumerate(val_loader):
            X = X.to(device)
            output, _ = model(X, X, teacher_forcing=False)
            loss = criterion(X, output)
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(val_loader)
        return avg_loss


def get_scores(model, scoring_loader, device):

    vectorizer = SMILESVectorizer(pad_to_len=100)

    model.eval()
    example_list = []

    with torch.no_grad():
        mean_qed = 0
        mean_validity = 0
        for batch_idx, X in enumerate(scoring_loader):
            X = X.to(device)
            output, _ = model(X, X, teacher_forcing=False)
            seq_list = [
                vectorizer.devectorize(ohe.detach().cpu().numpy(), remove_special=True)
                for ohe in output
            ]

            example_list = seq_list[:10] if batch_idx == 0 else example_list
            smiles_list = seq_list

            mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
            mol_list_valid = [x for x in mol_list if x is not None]

            if mol_list_valid:
                # Calculate validity
                batch_valid = len(mol_list_valid) / len(mol_list)
                mean_validity += batch_valid

                # Calculate QED
                batch_qed = 0
                if len(mol_list_valid) > 0:
                    for mol in mol_list_valid:
                        batch_qed += try_QED(mol)
                    batch_qed = batch_qed / len(mol_list_valid)
                    mean_qed += batch_qed

            else:
                mean_qed += 0
                mean_validity += 0

        mean_validity = mean_validity / len(scoring_loader)
        mean_qed = mean_qed / len(scoring_loader)

        print("Example decoded sequences:")
        [print(seq) for seq in example_list]

        return mean_qed, mean_validity

def try_QED(mol):
    """
    Tries to calculate the QED score for a molecule
    Args:
        mol: rdkit mol object
    Returns:
        qed: float
    """
    try:
        qed = QED.qed(mol)
    except:
        qed = 0
    return qed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--encoding_size", type=int, default=32, help="Latent space size")
    parser.add_argument("--run_name", type=str, default='SMILES2SMILES', help="Run name")
    parser.add_argument("--kld_weight", type=float, default=0.1, help="KLD weight")
    args = parser.parse_args()
    train(learn_rate=args.lr, batch_size=args.batch_size, encoding_size=args.encoding_size)