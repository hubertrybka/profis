import argparse
import configparser
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data as D
from sklearn.svm import SVC

from profis.gen.dataset import LatentEncoderDataset
from profis.gen.generator import EncoderDecoderV3
from profis.utils.modelinit import initialize_model
from profis.clf.model_selection import cross_evaluate, grid_search


def main(config_path, verbose=True):
    """
    Trains an SVM classifier on the latent embeddings of the known ligands.
    Args:
        config_path (str): path to the config file
        verbose (bool): whether to print the progress
    """

    # read config file

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_path)
    data_path = str(config["SVC"]["data_path"])
    model_path = str(config["SVC"]["model_path"])
    out_path = str(config["SVC"]["output_dir"])
    c_param = float(config["SVC"]["c_param"])
    kernel = str(config["SVC"]["kernel"])
    gamma = str(config["SVC"]["gamma"])
    use_cuda = config.getboolean("SVC", "use_cuda")
    name = str(config["SVC"]["name"])
    optimize = config.getboolean("SVC", "optimize_hyperparameters")

    start_time = time.time()

    cuda_available = torch.cuda.is_available() and use_cuda
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}") if verbose else None

    # read dataset

    data = pd.read_parquet(data_path, columns=["smiles", "activity", "fps"])
    data.reset_index(drop=True, inplace=True)
    print(f"Loaded data from {data_path}") if verbose else None
    activity = data["activity"]

    # load model

    split = model_path.split("/")
    config_path = "/".join(split[:-1]) + "/hyperparameters.ini"

    if not os.path.exists(config_path):
        raise ValueError(f"Model config file {config_path} not found")
    print(f"Reading model hyperparameters from {config_path}") if verbose else None
    model = initialize_model(config_path, device=device)

    if split[-1] != "dummy.pt":
        print(f"Loading weights from {model_path}") if verbose else None
        model.load_state_dict(torch.load(model_path, map_location=device))

    # encode data into latent space vectors

    print("Encoding data") if verbose else None
    mus, _ = encode(data, model, device)
    data = pd.DataFrame(mus)
    data["activity"] = activity
    data.reset_index(drop=True, inplace=True)
    X = data.drop("activity", axis=1).values
    y = data["activity"].values

    if out_path is None or not os.path.exists(f"{out_path}"):
        out_path = "models"

    if not os.path.exists(f"{out_path}/{name}"):
        os.mkdir(f"{out_path}/{name}")

    # initialize model
    SV_params = {
        "C": c_param,
        "kernel": kernel,
        "gamma": gamma,
        "shrinking": True,
        "probability": True,
        "max_iter": -1,
    }

    svc = SVC(**SV_params)

    # optimize hyperparameters
    if optimize:
        print("Optimizing hyperparameters...") if verbose else None
        param_grid = [
            {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 'scale'], 'kernel': ['rbf']},
        ]
        best_params, cv_results = grid_search(svc, X, y, param_grid, n_splits=10, n_jobs=-1, scoring="roc_auc",
                                              verbose=verbose)
        svc = SVC(**best_params)
        cv_results_df = pd.DataFrame(cv_results)
        cv_results_df.to_csv(f"{out_path}/{name}/cv_results.csv", index=False)
        print(f"CV grid search results saved to {out_path}/{name}/cv_results.csv") if verbose else None

    # train model

    print("Training SVC...") if verbose else None
    svc.fit(X, y)

    # save model

    with open(f"./{out_path}/{name}/clf.pkl", "wb") as file:
        pickle.dump(svc, file)

    # evaluate

    print("Evaluating...") if verbose else None
    metrics = cross_evaluate(svc, X, y)

    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(f"{out_path}/{name}/metrics.csv", index=False)

    # dump config
    with open(f"{out_path}/{name}/config.ini", "w") as configfile:
        config.write(configfile)

    time_elapsed = round((time.time() - start_time), 2)
    if time_elapsed < 60:
        print(f"SVC training finished in {time_elapsed} seconds")
    else:
        print(f"SVC training finished in {round(time_elapsed / 60, 2)} minutes")
    return


def encode(df, model, device):
    """
    Encodes the fingerprints of the molecules in the dataframe using VAE encoder.
    Args:
        df (pd.DataFrame): dataframe containing 'fps' column with Klekota&Roth fingerprints
            in the form of a list of integers (dense representation)
        model (EncoderDecoderV3): model to be used for encoding
        device (torch.device): device to be used for encoding
    Returns:
        mus (np.ndarray): array of means of the latent space
        logvars (np.ndarray): array of logvars of the latent space
    """
    dataset = LatentEncoderDataset(df, fp_len=model.fp_size)
    dataloader = D.DataLoader(dataset, batch_size=1024, shuffle=False)
    mus = []
    logvars = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            X = batch.to(device)
            mu, logvar = model.encoder(X)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())

        mus = np.concatenate(mus, axis=0)
        logvars = np.concatenate(logvars, axis=0)
    return mus, logvars


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config_files/SVC_config.ini",
        help="Path to SVC config file",
    )
    args = parser.parse_args()
    config_path = args.config
    main(config_path)
