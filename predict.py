import argparse
import configparser
import os
import time

import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
import torch

from profis.pred.pred import predict, filter_dataframe
from profis.utils.modelinit import initialize_model


def main(config_path):
    """
    Predicting molecules using the trained model.

    Args:
        config_path: Path to the config file.
    Returns: None
    """

    start_time = time.time()

    # get config
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_path)
    file_path = config["RUN"]["data_path"]
    model_path = config["RUN"]["model_path"]
    use_cuda = config["RUN"].getboolean("use_cuda")
    verbosity = int(config["RUN"]["verbosity"])

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    # get file name
    dirname = os.path.dirname(file_path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print(f"Using {device} device") if verbosity > 0 else None

    model_epoch = model_path.split("/")[-1]
    model_config_path = model_path.replace(model_epoch, "hyperparameters.ini")
    parser = configparser.ConfigParser(allow_no_value=True)
    parser.read(model_config_path)
    use_selfies = parser["RUN"].getboolean("use_selfies")
    if not os.path.exists(model_config_path):
        raise ValueError(f"Model config file {model_config_path} not found")

    # load model

    model = initialize_model(config_path=model_config_path, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}") if verbosity > 1 else None

    # load data
    if file_path.endswith(".csv"):
        query_df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        query_df = pd.read_parquet(file_path)
    else:
        raise ValueError("Data file format not supported (must be .csv or .parquet)")

    for col in ["smiles", "label", "score", "activity", "norm"]:
        if col in query_df.columns:
            query_df = query_df.drop(columns=[col])
    input_vector = query_df.to_numpy()
    print(f"Loaded data from {file_path}") if verbosity > 1 else None

    # get predictions
    print(f"Getting predictions for file {file_path}...") if verbosity > 1 else None
    df = predict(
        model, input_vector, device=device, use_selfies=use_selfies, batch_size=512
    )

    # filter dataframe
    df = filter_dataframe(df, config)

    # save data as csv
    os.mkdir(f"{dirname}/preds_{timestamp}")
    with open(f"{dirname}/preds_{timestamp}/config.ini", "w") as configfile:
        config.write(configfile)
    df.to_csv(f"{dirname}/preds_{timestamp}/predictions.csv", index=False)

    (
        print(f"Saved data to {dirname}/preds_{timestamp} directory")
        if verbosity > 0
        else None
    )

    # dump config
    with open(f"{dirname}/preds_{timestamp}/config.ini", "w") as configfile:
        config.write(configfile)

    # save images
    os.mkdir(f"{dirname}/preds_{timestamp}/imgs")
    for n, (idx, smiles) in enumerate(zip(df["idx"], df["smiles"])):
        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(
            mol, f"{dirname}/preds_{timestamp}/imgs/{idx}_{n}.png", size=(300, 300)
        )

    time_elapsed = time.time() - start_time
    print(f"{file_path} processed in {(time_elapsed / 60):.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config_files/pred_config.ini",
        help="Path to config file",
    )

    args = parser.parse_args()
    config_path = args.config

    main(config_path=config_path)
