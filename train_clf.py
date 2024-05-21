import argparse
import configparser
import os
import pickle
import time

import pandas as pd
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from profis.utils.finger import encode
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
    data_path = str(config["RUN"]["data_path"])
    model_path = str(config["RUN"]["model_path"])
    out_path = str(config["RUN"]["output_dir"])
    use_cuda = config.getboolean("RUN", "use_cuda")
    name = str(config["RUN"]["name"])
    model_type = str(config["INFO"]["model"])
    optimize = config.getboolean("RUN", "optimize_hyperparameters")

    start_time = time.time()

    cuda_available = torch.cuda.is_available() and use_cuda
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}") if verbose else None

    if out_path is None or not os.path.exists(f"{out_path}"):
        out_path = "models"

    if not os.path.exists(f"{out_path}/{name}"):
        os.mkdir(f"{out_path}/{name}")

    # read dataset

    data = pd.read_parquet(data_path, columns=["smiles", "activity", "fps"])
    data.reset_index(drop=True, inplace=True)
    print(f"Loaded data from {data_path}") if verbose else None
    activity = data["activity"]

    # load the big model (VAE)

    split = model_path.split("/")
    config_path = "/".join(split[:-1]) + "/hyperparameters.ini"

    if not os.path.exists(config_path):
        raise ValueError(f"Model config file {config_path} not found")
    print(f"Reading model hyperparameters from {config_path}") if verbose else None
    big_model = initialize_model(config_path, device=device)

    if split[-1] != "dummy.pt":
        print(f"Loading weights from {model_path}") if verbose else None
        big_model.load_state_dict(torch.load(model_path, map_location=device))

    # encode data into latent space vectors

    mus, _ = encode(data, big_model, device)
    data = pd.DataFrame(mus)
    data["activity"] = activity
    data.reset_index(drop=True, inplace=True)
    X = data.drop("activity", axis=1).values
    y = data["activity"].values

    # initialize the classifier

    if model_type == "SVC":
        params = {
            "C": float(config["SVC"]["c_param"]),
            "kernel": str(config["SVC"]["kernel"]),
            "gamma": str(config["SVC"]["gamma"]),
            "shrinking": True,
            "probability": True,
            "max_iter": -1,
        }
        param_grid = [
            {"C": [1, 10, 100, 500], "kernel": ["linear"]},
            {
                "C": [1, 10, 100, 500],
                "gamma": [0.001, 0.0001, "scale"],
                "kernel": ["rbf"],
            },
        ]
        clf = SVC(**params)

    elif model_type == "RF":
        max_depth = (
            None
            if config["RF"]["max_depth"] == "None"
            else int(config["RF"]["max_depth"])
        )
        max_features = (
            None
            if config["RF"]["max_features"] == "None"
            else str(config["RF"]["max_features"])
        )
        max_leaf_nodes = (
            None
            if config["RF"]["max_leaf_nodes"] == "None"
            else int(config["RF"]["max_leaf_nodes"])
        )
        params = {
            "n_estimators": int(config["RF"]["n_estimators"]),
            "max_depth": max_depth,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "random_state": 42,
        }
        param_grid = {
            "n_estimators": [50, 100, 250, 500],
            "max_features": ["sqrt", "log2", None],
            "max_depth": [3, 6, 9, None],
            "max_leaf_nodes": [3, 6, 9, None],
        }
        clf = RandomForestClassifier(**params)

    elif model_type == "XGB":
        params = {
            "learning_rate": float(config["XGB"]["learning_rate"]),
            "n_estimators": int(config["XGB"]["n_estimators"]),
            "max_depth": int(config["XGB"]["max_depth"]),
            "min_child_weight": int(config["XGB"]["min_child_weight"]),
            "gamma": float(config["XGB"]["gamma"]),
            "subsample": float(config["XGB"]["subsample"]),
            "nthread": int(config["XGB"]["nthread"]),
            "seed": 42,
        }
        param_grid = {
            "n_estimators": [50, 100, 250, 500],
            "max_depth": [3, 6, 9, 12],
            "min_child_weight": [1, 3, 6],
            "gamma": [0, 0.1, 0.2, 0.3],
            "subsample": [0.6, 0.8, 1.0],
        }
        clf = XGBClassifier(**params)
    else:
        raise ValueError(
            f"Model type {model_type} not recognized. The config file may be corrupted."
        )

    # optimize hyperparameters
    if optimize:
        print("Optimizing hyperparameters...") if verbose else None

        best_params, cv_results = grid_search(
            clf,
            X,
            y,
            param_grid,
            n_splits=5,
            n_jobs=-1,
            scoring="roc_auc",
            verbose=verbose,
        )
        clf.set_params(**best_params)
        cv_results_df = pd.DataFrame(cv_results)
        cv_results_df.to_csv(f"{out_path}/{name}/cv_results.csv", index=False)
        print(
            f"CV grid search results saved to {out_path}/{name}/cv_results.csv"
        ) if verbose else None

    # train model

    print(f"Training {model_type}...") if verbose else None
    clf.fit(X, y)

    # save model

    with open(f"./{out_path}/{name}/clf.pkl", "wb") as file:
        pickle.dump(clf, file)

    # evaluate

    print("Evaluating...") if verbose else None
    metrics = cross_evaluate(clf, X, y)

    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(f"{out_path}/{name}/metrics.csv", index=False)

    # dump config
    with open(f"{out_path}/{name}/config.ini", "w") as configfile:
        config.write(configfile)

    time_elapsed = round((time.time() - start_time), 2)
    if time_elapsed < 60:
        print(f"Finished in {time_elapsed} seconds")
    else:
        print(f"Finished in {round(time_elapsed / 60, 2)} minutes")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config_files/SVC_config.ini",
        help="Path to config file",
    )
    args = parser.parse_args()
    config_path = args.config
    main(config_path)
