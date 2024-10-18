# import packages
import argparse
import configparser
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from profis.gen.dataset import SELFIESDataset, SMILESDataset, DeepSMILESDataset
from profis.gen.train import train
from profis.utils.split import scaffold_split
from profis.utils.modelinit import initialize_model
from profis.utils.vectorizer import (
    SELFIESVectorizer,
    SMILESVectorizer,
    DeepSMILESVectorizer,
)

# Suppress RDKit warnings
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def main(config_path):
    """
    Training script for model with fully-connected encoder and GRU decoder
    Args:
        config_path: path to config file
    """

    # read config file

    config = configparser.ConfigParser()
    config.read(config_path)
    train_size = float(config["RUN"]["train_size"])
    random_seed = int(config["RUN"]["random_seed"])
    run_name = str(config["RUN"]["run_name"])
    batch_size = int(config["RUN"]["batch_size"])
    data_path = str(config["RUN"]["data_path"])
    dataloader_workers = int(config["RUN"]["num_workers"])
    fp_len = int(config["MODEL"]["fp_len"])
    use_cuda = config.getboolean("RUN", "use_cuda")
    out_encoding = str(config["RUN"]["out_encoding"])
    val_size = round(1 - train_size, 1)
    train_percent = int(train_size * 100)
    val_percent = int(val_size * 100)

    cuda_available = torch.cuda.is_available() and use_cuda
    device = torch.device("cuda" if cuda_available else "cpu")

    print("Using device:", device)

    # read dataset

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset file {data_path} not found")
    dataset = pd.read_parquet(data_path)

    # create a directory for this model weights if not there

    if not os.path.isdir(f"models/{run_name}"):
        if not os.path.isdir("models"):
            os.mkdir("models")
        os.mkdir(f"models/{run_name}")

    with open(f"models/{run_name}/hyperparameters.ini", "w") as configfile:
        config.write(configfile)

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
    scoring_df = val_df.sample(frac=0.1, random_state=random_seed)

    # prepare dataloaders
    if out_encoding == "selfies":
        vectorizer = SELFIESVectorizer(pad_to_len=100)
        train_dataset = SELFIESDataset(train_df, vectorizer, fp_len)
        val_dataset = SELFIESDataset(val_df, vectorizer, fp_len)
        scoring_dataset = SELFIESDataset(scoring_df, vectorizer, fp_len)

    elif out_encoding == "smiles":
        vectorizer = SMILESVectorizer(pad_to_len=100)
        train_dataset = SMILESDataset(train_df, vectorizer, fp_len)
        val_dataset = SMILESDataset(val_df, vectorizer, fp_len)
        scoring_dataset = SMILESDataset(scoring_df, vectorizer, fp_len)

    elif out_encoding == "deepsmiles":
        vectorizer = DeepSMILESVectorizer(pad_to_len=100)
        train_dataset = DeepSMILESDataset(train_df, vectorizer, fp_len)
        val_dataset = DeepSMILESDataset(val_df, vectorizer, fp_len)
        scoring_dataset = DeepSMILESDataset(scoring_df, vectorizer, fp_len)
    else:
        raise ValueError(
            "Invalid output encoding (must be selfies, smiles or deepsmiles)"
        )

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

    model = initialize_model(
        config_path, device=device, use_dropout=True, teacher_forcing=True
    )

    _ = train(config, model, train_loader, val_loader, scoring_loader)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config_files/RNN_config.ini",
        help="Path to config file",
    )
    config_path = parser.parse_args().config
    main(config_path)
