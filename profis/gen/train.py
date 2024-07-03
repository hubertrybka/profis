import time

import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.QED as QED
import rdkit.Chem.AllChem as AllChem
import selfies as sf
import numpy as np
import torch
import deepsmiles as ds

from profis.gen.loss import CCE
from profis.utils.annealing import Annealer
from profis.utils.vectorizer import (
    SELFIESVectorizer,
    SMILESVectorizer,
    DeepSMILESVectorizer,
)
from profis.utils.finger import encode


def train(config, model, train_loader, val_loader, scoring_loader):
    """
    Training loop for the model consisting of a VAE encoder and GRU decoder
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = int(config["RUN"]["epochs"])
    run_name = str(config["RUN"]["run_name"])
    learn_rate = float(config["RUN"]["learn_rate"])
    kld_backward = config.getboolean("RUN", "kld_backward")
    start_epoch = int(config["RUN"]["start_epoch"])
    kld_weight = float(config["RUN"]["kld_weight"])
    kld_annealing = config.getboolean("RUN", "kld_annealing")
    annealing_max_epoch = int(config["RUN"]["annealing_max_epoch"])
    annealing_shape = str(config["RUN"]["annealing_shape"])
    data_path = str(config["RUN"]["data_path"])
    fp_type = data_path.split("_")[-1].split(".")[0]
    out_encoding = str(config["RUN"]["out_encoding"])
    train_size = float(config["RUN"]["train_size"])
    val_percent = int(round(1 - train_size, 1) * 100)

    annealing_agent = Annealer(annealing_max_epoch, annealing_shape)

    # Define dataframe for logging progress
    epochs_range = range(start_epoch, epochs + start_epoch)
    metrics = pd.DataFrame()

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = CCE(notation=out_encoding)

    print("Starting Training of GRU")
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
        val_loss = evaluate(model, val_loader, notation=out_encoding)

        if epoch % 10 == 0:
            start = time.time()
            mean_qed, mean_fp_recon, mean_validity = get_scores(
                model, scoring_loader, fp_type=fp_type, format=out_encoding
            )
            end = time.time()
            print(f"QED + fp evaluated in {(end - start) / 60} minutes")
        else:
            mean_qed = None
            mean_fp_recon = None
            mean_validity = None

        metrics_row = pd.DataFrame(
            {
                "epoch": [epoch],
                "kld_loss": [kld_loss.item()],
                "kld_weighted": [kld_weighted.item()],
                "train_loss": [avg_loss],
                "val_loss": [val_loss],
                "mean_qed": [mean_qed],
                "mean_fp_recon": [mean_fp_recon],
                "mean_validity": [mean_validity],
            }
        )
        if kld_annealing:
            annealing_agent.step()

        # Update metrics df
        metrics = pd.concat([metrics, metrics_row], ignore_index=True, axis=0)
        if epoch % 50 == 0:
            # calculate latent vectors distribution
            val_df = pd.read_parquet(
                data_path.split(".")[0] + f"_val_{val_percent}.parquet"
            )
            mus, _ = encode(val_df, model, device)
            means = mus.mean(axis=0)
            stds = mus.std(axis=0)
            bounds = pd.DataFrame({"mean": means, "std": stds}, index=range(len(means)))
            bounds.to_csv(f"./models/{run_name}/latent_bounds_{epoch}.csv")

            # save model
            save_path = f"./models/{run_name}/epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv", index=False)
        end_time = time.time()
        loop_time = round((end_time - start_time) / 60, 2)  # in minutes
        print(f"Epoch {epoch} completed in {loop_time} minutes")

    return None


def evaluate(model, val_loader, notation="smiles"):
    """
    Evaluates the model on the validation set
    Args:
        model (nn.Module): EncoderDecoderV3 model
        val_loader (DataLoader): validation set loader
        notation (str): output notation, can be "smiles", "selfies" or "deepsmiles"
    Returns:
        float: average loss on the validation set

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        criterion = CCE(notation=notation)
        epoch_loss = 0
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.to(device)
            y = y.to(device)
            output, _ = model(X, y, teacher_forcing=False)
            loss = criterion(y, output)
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(val_loader)
        return avg_loss


def get_scores(model, scoring_loader, fp_type="KRFP", format="selfies"):
    """
    Calculates the QED and FP reconstruction score for the model
    Args:
        model (nn.Module): EncoderDecoderV3 model
        scoring_loader (DataLoader): scoring set loader
        fp_type (str): fingerprint type, either "ECFP" or "KRFP"
        format (str): input format, must be "selfies", "smiles" or "deepsmiles"
    Returns:
        mean_qed (float): average QED score
        mean_fp_recon (float): average FP reconstruction score

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if format == "selfies":
        vectorizer = SELFIESVectorizer(pad_to_len=128)
    elif format == "deepsmiles":
        vectorizer = DeepSMILESVectorizer(pad_to_len=128)
        converter = ds.Converter(rings=True, branches=True)
    elif format == "smiles":
        vectorizer = SMILESVectorizer(pad_to_len=128)
    else:
        raise ValueError("Invalid format, must be 'selfies', 'smiles' or 'deepsmiles'")

    model.eval()
    example_list = []

    with torch.no_grad():
        mean_qed = 0
        mean_fp_recon = 0
        mean_validity = 0
        for batch_idx, (X, y) in enumerate(scoring_loader):
            X = X.to(device)
            y = y.to(device)
            output, _ = model(X, y, teacher_forcing=False)
            seq_list = [
                vectorizer.devectorize(ohe.detach().cpu().numpy(), remove_special=True)
                for ohe in output
            ]

            example_list = seq_list[:10] if batch_idx == 0 else example_list

            if format == "selfies":
                smiles_list = [sf.decoder(x) for x in seq_list]
            elif format == "deepsmiles":
                smiles_list = []
                for x in seq_list:
                    try:
                        smiles_list.append(converter.decode(x))
                    except ds.DecodeError:
                        invalid_string = "invalid"
                        smiles_list.append(invalid_string)
            else:
                smiles_list = seq_list

            mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
            none_idcs = [i for i, x in enumerate(mol_list) if x is None]
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

                # Calculate FP recon score
                X = X.detach().cpu()
                X = np.delete(X, none_idcs, axis=0)
                batch_fp_recon = 0
                for mol, fp in zip(mol_list_valid, X):
                    if fp_type == "ECFP":
                        batch_fp_recon += ECFP_score(mol, fp)
                    elif fp_type == "KRFP":
                        batch_fp_recon += KRFP_score(mol, fp)
                    else:
                        raise ValueError("Invalid fp_type, must be 'ECFP' or 'KRFP'")
                mean_fp_recon += batch_fp_recon / len(mol_list_valid)
            else:
                mean_qed += 0
                mean_fp_recon += 0
                mean_validity += 0

        mean_validity = mean_validity / len(scoring_loader)
        mean_fp_recon = mean_fp_recon / len(scoring_loader)
        mean_qed = mean_qed / len(scoring_loader)

        print("Example decoded sequences:")
        [print(seq) for seq in example_list]

        return mean_qed, mean_fp_recon, mean_validity


def KRFP_score(mol, fp: torch.Tensor):
    """
    Calculates the KRFP fingerprint reconstruction score for a molecule
    Args:
        mol: rdkit mol object
        fp: torch tensor of size (fp_len)
    Returns:
        score: float (0-1)
    """
    score = 0
    key = pd.read_csv("data/KlekFP_keys.txt", header=None)
    fp_len = fp.shape[0]
    for i in range(fp_len):
        if fp[i] == 1:
            frag = Chem.MolFromSmarts(key.iloc[i].values[0])
            score += mol.HasSubstructMatch(frag)
    return score / torch.sum(fp).item()


def ECFP_score(mol, fp: torch.Tensor):
    """
    Calculates the ECFP fingerprint reconstruction score for a molecule
    Args:
        mol: rdkit mol object
        fp: torch tensor of size (fp_len)
    Returns:
        score: float (0-1)
    """
    score = 0
    fp_len = fp.shape[0]
    ECFP_reconstructed = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_len)
    for i in range(fp_len):
        if ECFP_reconstructed[i] and fp[i]:
            score += 1
    return score / torch.sum(fp).item()


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
