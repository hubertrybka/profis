# module for fingerprint manipulation
import numpy as np
import torch
import torch.utils.data as Data
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

from profis.gen.dataset import LatentEncoderDataset


def smiles2sparse_KRFP(smiles):
    """
    Convert SMILES string to sparse Klekota&Roth fingerprint
    Args:
        smiles (str): SMILES string
    Returns:
        np.array: sparse fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    keys = "data/KlekFP_keys.txt"
    klek_keys = [line.strip() for line in open(keys)]
    klek_keys_mols = list(map(Chem.MolFromSmarts, klek_keys))
    fp_list = []
    for i, key in enumerate(klek_keys_mols):
        if mol.HasSubstructMatch(key):
            fp_list.append(1)
        else:
            fp_list.append(0)
    return np.array(fp_list)


def smiles2dense_KRFP(smiles):
    """
    Convert SMILES string to dense Klekota&Roth fingerprint
    Args:
        smiles (str): SMILES string
    Returns:
        np.array: dense fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    keys = "data/KlekFP_keys.txt"
    klek_keys = [line.strip() for line in open(keys)]
    klek_keys_mols = list(map(Chem.MolFromSmarts, klek_keys))
    fp_list = []
    for i, key in enumerate(klek_keys_mols):
        if mol.HasSubstructMatch(key):
            fp_list.append(i)
    return np.array(fp_list)


def sparse2dense(sparse, return_numpy=True):
    """
    Convert sparse fingerprint to dense fingerprint
    Args:
        sparse (np.array): sparse fingerprint
        return_numpy (bool): whether to return numpy array or list
    Returns:
        dense (np.array): dense fingerprint
    """
    dense = []
    for idx, value in enumerate(sparse):
        if value == 1:
            dense.append(idx)
    if return_numpy:
        return np.array(dense)
    else:
        return dense


def dense2sparse(dense, fp_len=4860):
    """
    Convert dense fingerprint to sparse fingerprint
    Args:
        dense (np.array): dense fingerprint
        fp_len (int): length of the fingerprint
    Returns:
        sparse (np.array): sparse fingerprint
    """
    sparse = np.zeros(fp_len, dtype=np.int8)
    for value in dense:
        sparse[value] = 1
    return np.array(sparse)


def encode(df, model, device, batch=1024):
    """
    Encodes the fingerprints of the molecules in the dataframe using VAE encoder.
    Args:
        df (pd.DataFrame): dataframe containing 'fps' column with Klekota&Roth fingerprints
            in the form of a list of integers (dense representation)
        model (EncoderDecoderV3): model to be used for encoding
        device (torch.device): device to be used for encoding
        batch (int): batch size for encoding
    Returns:
        mus (np.ndarray): array of means of the latent space
        logvars (np.ndarray): array of logvars of the latent space
    """
    dataset = LatentEncoderDataset(df, fp_len=model.fp_size)
    dataloader = Data.DataLoader(dataset, batch_size=batch, shuffle=False)
    mus = []
    logvars = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in dataloader:
            X = batch.to(device)
            mu, logvar = model.encoder(X)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())

        mus = np.concatenate(mus, axis=0)
        logvars = np.concatenate(logvars, axis=0)
    return mus, logvars


def smiles2sparse_ECFP(smiles, n_bits=2048):
    """
    Convert SMILES string to sparse ECFP fingerprint
    Args:
        smiles (str): SMILES string
        n_bits (int): number of bits in the fingerprint
    Returns:
        np.array: sparse fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = np.array(GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    return np.array(fp)


def smiles2dense_ECFP(smiles, n_bits=2048):
    """
    Convert SMILES string to dense ECFP fingerprint
    Args:
        smiles (str): SMILES string
        n_bits (int): number of bits in the fingerprint
    Returns:
        np.array: dense fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = np.array(GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    return sparse2dense(fp)
