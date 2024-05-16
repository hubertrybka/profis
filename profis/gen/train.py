import time

import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.QED as QED
import rdkit.Chem.AllChem as AllChem
import selfies as sf
import numpy as np
import torch

from profis.gen.loss import CCE
from profis.utils.annealing import Annealer
from profis.utils.vectorizer import SELFIESVectorizer, SMILESVectorizer


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
    use_selfies = config.getboolean("RUN", "use_selfies")

    config_dict = {s: dict(config.items(s)) for s in config.sections()}

    annealing_agent = Annealer(annealing_max_epoch, annealing_shape)

    # Define dataframe for logging progress
    epochs_range = range(start_epoch, epochs + start_epoch)
    metrics = pd.DataFrame(
        columns=[
            "epoch",
            "kld_loss",
            "kld_weighted",
            "train_loss",
            "val_loss",
            "mean_qed",
            "mean_fp_recon",
            "mean_validity",
        ]
    )

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = CCE(notation="SELFIES" if use_selfies else "SMILES")

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
        val_loss = evaluate(model, val_loader)

        if epoch % 10 == 0:
            start = time.time()
            mean_qed, mean_fp_recon, mean_validity = get_scores(
                model,
                scoring_loader,
                fp_type=fp_type,
                format="selfies" if use_selfies else "smiles",
            )
            end = time.time()
            print(f"QED + fp evaluated in {(end - start) / 60} minutes")
        else:
            mean_qed = None
            mean_fp_recon = None
            mean_validity = None

        metrics_dict = {
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

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict
        if epoch % 10 == 0 or epoch == 5:
            save_path = f"./models/{run_name}/epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv", index=False)
        end_time = time.time()
        loop_time = (end_time - start_time) / 60  # in minutes
        print(f"Epoch {epoch} completed in {loop_time} minutes")

    return None


def evaluate(model, val_loader):
    """
    Evaluates the model on the validation set
    Args:
        model (nn.Module): EncoderDecoderV3 model
        val_loader (DataLoader): validation set loader
    Returns:
        float: average loss on the validation set

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        criterion = CCE()
        epoch_loss = 0
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.to(device)
            y = y.to(device)
            output, _ = model(X, y, teacher_forcing=False)
            loss = criterion(y, output)
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(val_loader)
        return avg_loss


def get_scores(model, scoring_loader, fp_type="ECFP", format="selfies"):
    """
    Calculates the QED and FP reconstruction score for the model
    Args:
        model (nn.Module): EncoderDecoderV3 model
        scoring_loader (DataLoader): scoring set loader

    Returns:
        mean_qed (float): average QED score
        mean_fp_recon (float): average FP reconstruction score

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if format == "selfies":
        vectorizer = SELFIESVectorizer(pad_to_len=128)
    else:
        vectorizer = SMILESVectorizer(pad_to_len=128)
    model.eval()
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
            if format == "selfies":
                smiles_list = [sf.decoder(x) for x in seq_list]
            else:
                smiles_list = seq_list

            mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
            none_idcs = [i for i, x in enumerate(mol_list) if x is None]
            mol_list_valid = [x for x in mol_list if x is not None]

            # Calculate validity
            batch_valid = 1 - (len(none_idcs) / len(mol_list))
            mean_validity += batch_valid

            # Calculate QED
            batch_qed = 0
            if len(mol_list_valid) > 0:
                for mol in mol_list_valid:
                    batch_qed += QED.qed(mol)
                batch_qed = batch_qed / len(mol_list_valid)
                mean_qed += batch_qed

            X = X.detach().cpu()
            X = np.delete(X, none_idcs, axis=0)

            # Calculate FP recon score
            batch_fp_recon = 0
            for mol, fp in zip(mol_list_valid, X):
                if fp_type == "ECFP":
                    batch_fp_recon += ECFP_score(mol, fp)
                elif fp_type == "KRFP":
                    batch_fp_recon += KRFP_score(mol, fp)
            if len(mol_list_valid) > 0:
                batch_fp_recon = batch_fp_recon / len(mol_list_valid)
            mean_fp_recon += batch_fp_recon

        mean_validity = mean_validity / len(scoring_loader)
        mean_fp_recon = mean_fp_recon / len(scoring_loader)
        mean_qed = mean_qed / len(scoring_loader)
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
