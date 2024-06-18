# import packages
import wandb

import pandas as pd
import torch
from torch.utils.data import DataLoader
import time
import argparse

from profis.gen.dataset import SELFIESDataset, SMILESDataset, DeepSMILESDataset
from profis.gen.loss import CCE
from profis.utils.modelinit import get_alphabet_len
from profis.gen.generator import ProfisGRU
from profis.utils.vectorizer import (
    SELFIESVectorizer,
    SMILESVectorizer,
    DeepSMILESVectorizer,
)
from profis.gen.train import (
    evaluate,
    get_scores,
    Annealer,
)


def run_train():
    #-----------------------------------------------#
    batch_size = 256
    data_path = "data/RNN_dataset_KRFP.parquet"
    # data_path = "data/RNN_dataset_ECFP.parquet"
    fp_len = 4860
    out_encoding = "smiles"
    train_size = 0.9
    val_size = round(1 - train_size, 1)
    train_percent = int(train_size * 100)
    val_percent = int(val_size * 100)
    dataloader_workers = 3
    # -----------------------------------------------#

    train_df = pd.read_parquet(
        data_path.split(".")[0] + f"_train_{train_percent}.parquet"
    )
    val_df = pd.read_parquet(data_path.split(".")[0] + f"_val_{val_percent}.parquet")
    scoring_df = val_df.sample(frac=0.5, random_state=42)

    # prepare dataloaders
    if out_encoding == "selfies":
        vectorizer = SELFIESVectorizer(pad_to_len=128)
        train_dataset = SELFIESDataset(train_df, vectorizer, fp_len)
        val_dataset = SELFIESDataset(val_df, vectorizer, fp_len)
        scoring_dataset = SELFIESDataset(scoring_df, vectorizer, fp_len)

    elif out_encoding == "smiles":
        vectorizer = SMILESVectorizer(pad_to_len=128)
        train_dataset = SMILESDataset(train_df, vectorizer, fp_len)
        val_dataset = SMILESDataset(val_df, vectorizer, fp_len)
        scoring_dataset = SMILESDataset(scoring_df, vectorizer, fp_len)

    elif out_encoding == "deepsmiles":
        vectorizer = DeepSMILESVectorizer(pad_to_len=128)
        train_dataset = DeepSMILESDataset(train_df, vectorizer, fp_len)
        val_dataset = DeepSMILESDataset(val_df, vectorizer, fp_len)
        scoring_dataset = DeepSMILESDataset(scoring_df, vectorizer, fp_len)
    else:
        raise ValueError(
            "Invalid output encoding (must be selfies, smiles or deepsmiles)"
        )

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

    wandb.init()
    config = wandb.config

    epochs = 100
    fp_size = 4860
    out_encoding = "smiles"
    kld_backward = True
    start_epoch = 1
    kld_annealing = True
    annealing_max_epoch = 20
    annealing_shape = "cosine"
    device = torch.device("cuda")

    model = ProfisGRU(
        fp_size=fp_size,
        encoding_size=config.encoding_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        output_size=get_alphabet_len(out_encoding),
        teacher_ratio=config.teacher_ratio,
        random_seed=42,
        use_cuda=True,
        fc1_size=config.fc1_size,
        fc2_size=config.fc2_size,
        fc3_size=config.fc3_size,
        encoder_activation="relu",
        fc2_enabled=config.fc2_enabled,
        fc3_enabled=config.fc3_enabled,
    ).to(device)

    annealing_agent = Annealer(annealing_max_epoch, annealing_shape)
    epochs_range = range(start_epoch, epochs + start_epoch)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = CCE(notation=out_encoding)

    print("VAE-GRU Training")
    print(f"Device: {device}")

    # Start training loop
    for epoch in epochs_range:
        model.train()
        start_time = time.time()
        print(f"Epoch: {epoch}")
        epoch_loss = 0
        kld_loss = 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output, kld_loss = model(X, y, teacher_forcing=True)
            loss = criterion(y, output)
            kld_weighted = kld_loss * config.kld_weight
            if kld_annealing:
                kld_weighted = annealing_agent(kld_weighted)
            if kld_backward:
                (loss + kld_weighted).backward()
            else:
                loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, notation=out_encoding)

        if epoch % 5 == 0:
            start = time.time()
            mean_qed, mean_fp_recon, mean_validity = get_scores(
                model,
                scoring_loader,
                fp_type=("KRFP" if fp_size == 4860 else "ECFP"),
                format=out_encoding,
            )
            end = time.time()
            print(f"QED + fp evaluated in {(end - start) / 60} minutes")
        else:
            mean_qed = None
            mean_fp_recon = None
            mean_validity = None

        metrics_row = {
                "epoch": epoch,
                "kld_loss": kld_loss.item(),
                "kld_weighted": kld_weighted.item(),
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "mean_qed": mean_qed,
                "mean_fp_recon": mean_fp_recon,
                "mean_validity": mean_validity,
            }

        if kld_annealing:
            annealing_agent.step()
        wandb.log(metrics_row)

        end_time = time.time()
        loop_time = round((end_time - start_time) / 60, 2)  # in minutes
        print(f"Epoch {epoch} completed in {loop_time} minutes")

    wandb.finish()
    return None


def main(sweep_id=None):

    parameters_dict = {
        "learning_rate": {"value": 0.0002},
        "hidden_size": {"values": [1024, 2048]},  # GRU hidden size
        "encoding_size": {"value": 32},  # embedding size
        "dropout": {"values": [0, 0.1, 0.3]},
        "kld_weight": {"value": [0.1]},
        "teacher_ratio": {"values": [0.2, 0.5, 0.9]},
        "num_layers": {"values": [1, 2]},  # number of GRU layers
        "fc1_size": {"values": [1024, 2048]},
        "fc2_size": {"values": [512, 1024]},
        "fc3_size": {"values": [256, 512]},
        "fc2_enabled": {"value": [True]},
        "fc3_enabled": {"values": [True, False]},
        }

    sweep_config = {
        "method": "bayes",
        "parameters": parameters_dict,
        "metric": {"goal": "minimize", "name": "val_loss"},
        "early_terminate": {"type": "hyperband", "min_iter": 20, "eta": 1.5, "strict": True}
    }

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, project="RNN_sweep")
    wandb.agent(
        sweep_id,
        function=run_train,
        project="RNN_sweep"
    )

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-id', type=str
    )
    args = parser.parse_args()
    main(args.id)
