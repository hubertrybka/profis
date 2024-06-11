import numpy as np
import pandas as pd
from torch import device
from profis.utils.finger import encode
from profis.utils.modelinit import initialize_model
import configparser


class SCAvgMeasure:
    """
    Class to calculate the average cosine similarity of the n closest neighbors of a query molecule among the latent
    representations of the training set.
    Args:
        clf_path: path to the trained classifier model
    """

    def __init__(self, clf_path="models/D2_SMILES_KRFP_MLP/clf.pkl"):
        self.clf_path = clf_path
        self.config = self.read_config()
        self.train_path = self.config["RUN"]["data_path"]
        self.encoder_path = self.config["RUN"]["model_path"]
        self.device = device("cpu")
        self.encoder = initialize_model(
            self.encoder_path.replace("model.pt", "hyperparameters.ini")
        )
        self.train_encoded = self.encode_fingerprints()

    def __call__(self, query, n=3):
        """
        Calculate the average cosine similarity of the n closest neighbors of a query molecule among the latent
        representations of the training set.
        Args:
            query: latent representation of the query molecule
            n: number of closest neighbors to consider
        Returns:
            avg: average cosine similarity of the n closest neighbors
        """
        distances = self.calc_cosine_distance(query)

        # find n closest neighbors
        idx = np.argsort(distances)[::-1][:n]

        # calculate average similarity of n closest neighbors
        avg = np.mean(distances[idx])
        return avg

    def read_config(self):
        config = configparser.ConfigParser()
        config.read(self.clf_path.replace("clf.pkl", "config.ini"))
        return config

    def encode_fingerprints(self):
        data = pd.read_parquet(self.train_path)[["fps"]]
        encoded = encode(data, self.encoder, device=self.device)[0]
        return encoded

    def calc_cosine_distance(self, query):
        return np.dot(self.train_encoded, query)
