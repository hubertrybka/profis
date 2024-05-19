import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import selfies as sf
import deepsmiles as ds
import torch
from torch.utils.data import Dataset


class ProfisDataset(Dataset):
    """
    Dataset class for handling RNN training data.
    Parameters:
        df (pd.DataFrame): dataframe containing SMILES and fingerprints,
                           SMILES must be contained in ['smiles'] column as strings,
                           fingerprints must be contained in ['fps'] column as lists
                           of integers (dense vectors).
        vectorizer: vectorizer object instantiated from vectorizer.py
        fp_len (int): length of fingerprint
    """

    def __init__(self, df, vectorizer, fp_len=4860, smiles_enum=False):
        self.smiles = df["smiles"]
        self.fps = df["fps"]
        self.fps = self.prepare_X(self.fps)
        self.smiles = self.prepare_y(self.smiles)
        self.vectorizer = vectorizer
        self.fp_len = fp_len
        self.smiles_enum = smiles_enum

    def __getitem__(self, idx):
        """
        Get item from dataset.
        Args:
            idx (int): index of item to get
        Returns:
            X (torch.Tensor): reconstructed fingerprint
            y (torch.Tensor): vectorized SELFIES
        """
        raw_smile = self.smiles[idx]
        different_encoding = self.prepare_sequential_encoding(raw_smile)
        vectorized_seq = self.vectorizer.vectorize(different_encoding)
        if len(vectorized_seq) > 128:
            vectorized_seq = vectorized_seq[:128]
        raw_X = self.fps[idx]
        X = np.array(raw_X, dtype=int)
        X_reconstructed = self.reconstruct_fp(X)
        return (
            torch.from_numpy(X_reconstructed).float(),
            torch.from_numpy(vectorized_seq).float(),
        )

    def __len__(self):
        return len(self.fps)

    @staticmethod
    def prepare_sequential_encoding(smiles):
        return smiles

    def reconstruct_fp(self, fp):
        fp_rec = np.zeros(self.fp_len)
        fp_rec[fp] = 1
        return fp_rec

    def prepare_X(self, fps):
        fps = fps.apply(lambda x: np.array(x, dtype=int))
        return fps.values

    @staticmethod
    def prepare_y(seq):
        return seq.values


class SELFIESDataset(ProfisDataset):
    def __init__(self, df, vectorizer, fp_len=4860):
        super().__init__(df, vectorizer, fp_len)

    def prepare_sequential_encoding(self, smiles):
        return sf.encoder(smiles, strict=False)

    def __len__(self):
        return len(self.fps)


class SMILESDataset(ProfisDataset):
    def __init__(self, df, vectorizer, fp_len=4860):
        super().__init__(df, vectorizer, fp_len)

    def __len__(self):
        return len(self.fps)

    def prepare_sequential_encoding(self, smiles):
        return smiles


class DeepSMILESDataset(ProfisDataset):
    """
    Dataset class for handling RNN training data.
    Parameters:
       df (pd.DataFrame): dataframe containing SMILES and fingerprints,
                          SMILES must be contained in ['smiles'] column as strings,
                          fingerprints must be contained in ['fps'] column as lists
                          of integers (dense vectors).
       vectorizer: DeepSMILES vectorizer instantiated from vectorizer.py
    """

    def __init__(self, df, vectorizer, fp_len=4860):
        super().__init__(df, vectorizer, fp_len)
        self.converter = ds.Converter(rings=True, branches=True)

    def __len__(self):
        return len(self.fps)

    def prepare_sequential_encoding(self, smiles):
        return self.converter.encode(smiles)


class LatentEncoderDataset(Dataset):
    """
    Dataset for encoding fingerprints into latent space.
    Parameters:
        df (pd.DataFrame): pandas DataFrame object containing 'fps' column, which contains fingerprints
        in the form of lists of integers (dense representation)
        fp_len (int): length of fingerprints
    """

    def __init__(self, df, fp_len):
        self.fps = pd.DataFrame(df["fps"])
        self.fp_len = fp_len

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        raw_X = self.fps.iloc[idx]
        X_prepared = self.prepare_X(raw_X).values[0]
        X = np.array(X_prepared, dtype=int)
        X_reconstructed = self.reconstruct_fp(X)
        return torch.from_numpy(X_reconstructed).float()

    def reconstruct_fp(self, fp):
        fp_rec = np.zeros(self.fp_len)
        fp_rec[fp] = 1
        return fp_rec

    @staticmethod
    def prepare_X(fps):
        fps = fps.apply(lambda x: np.array(x, dtype=int))
        return fps
