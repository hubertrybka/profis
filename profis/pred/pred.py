import numpy as np
import pandas as pd
import rdkit.Chem.Crippen as Crippen
import selfies as sf
import torch
import deepsmiles as ds
from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors
from torch.utils.data import DataLoader
from profis.pred.tanimoto import TanimotoSearch
from scipy.special import softmax

from profis.utils.vectorizer import (
    SELFIESVectorizer,
    SMILESVectorizer,
    DeepSMILESVectorizer,
)

# Suppress RDKit warnings
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def predict(
    model,
    latent_vectors: np.array,
    device: torch.device = torch.device("cpu"),
    format: str = "smiles",
    batch_size: int = 512,
    n_trials: int = 1000,
):
    """
    Generate molecules from latent vectors
    Args:
        model (torch.nn.Module): ProfisGRU model.
        latent_vectors (np.array): numpy array of latent vectors. Shape = (n_samples, latent_size).
        device: device to use for prediction. Can be 'cpu' or 'cuda'.
        format: format of the output. Can be 'smiles', 'selfies' or 'deepsmiles'.
        batch_size: batch size for prediction.
        n_trials: number of trials for stochastic decoding.

    Returns:
        pd.DataFrame: Dataframe containing smiles and scores.
    """
    if format == "selfies":
        vectorizer = SELFIESVectorizer(pad_to_len=100)
    elif format == "smiles":
        vectorizer = SMILESVectorizer(pad_to_len=100)
    elif format == "deepsmiles":
        vectorizer = DeepSMILESVectorizer(pad_to_len=100)
    else:
        raise ValueError(
            f"Invalid format. Must be 'smiles', 'selfies' or 'deepsmiles'."
        )

    device = torch.device(device)

    loader = DataLoader(latent_vectors, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        df = pd.DataFrame(columns=["idx", "smiles"])
        preds_list = []
        for X in loader:
            latent_tensor = torch.Tensor(X).type(torch.FloatTensor).to(device)
            preds, _ = model(latent_tensor, None, omit_encoder=True)
            preds = preds.detach().cpu().numpy()
            preds_list.append(preds)
        preds_concat = np.concatenate(preds_list)

        if format == "smiles":
            if n_trials:
                df["smiles"] = [
                    stochastic_decoder(vector, vectorizer, n_trials=n_trials)
                    for vector in preds_concat
                ]
            else:
                df["smiles"] = [
                    simple_decoder(vector, vectorizer) for vector in preds_concat
                ]

        if format == "selfies":
            df["selfies"] = [
                vectorizer.devectorize(pred, remove_special=True, reduction="max")
                for pred in preds_concat
            ]
            df["smiles"] = df["selfies"].apply(sf.decoder)

        elif format == "deepsmiles":
            if n_trials:
                df["smiles"] = [
                    stochastic_decoder(vector, vectorizer, n_trials=n_trials)
                    for vector in preds_concat
                ]
            else:
                df["smiles"] = [
                    simple_decoder(vector, vectorizer) for vector in preds_concat
                ]
        df["idx"] = range(len(df))
    return df


def get_largest_ring(mol):
    """
    Returns the size of the largest ring in a molecule.
    Args:
        mol (rdkit.Chem.Mol): Molecule object.
    """
    ri = mol.GetRingInfo()
    rings = []
    for b in mol.GetBonds():
        ring_len = [len(ring) for ring in ri.BondRings() if b.GetIdx() in ring]
        rings += ring_len
    return max(rings) if rings else 0


def try_sanitize(mol):
    """
    Tries to sanitize a molecule object. If sanitization fails, returns the original molecule.
    Args:
        mol (rdkit.Chem.Mol): Molecule object.
    """
    try:
        output = mol
        Chem.SanitizeMol(output)
        return output
    except:
        return mol


def try_smiles2mol(smiles):
    """
    Tries to convert a SMILES string to a molecule object. If conversion fails, returns None.
    """
    try:
        output = Chem.MolFromSmiles(smiles)
        return output
    except:
        return None


def filter_dataframe(df, config):
    """
    Filters a dataframe of molecules based on the given configuration.
    Args:
        df (pd.DataFrame): Dataframe containing molecules.
        config (dict): Dictionary containing filtering parameters.
    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    df_copy = df.copy()
    df_copy["mols"] = df_copy["smiles"].apply(Chem.MolFromSmiles)
    df_copy.dropna(axis=0, inplace=True, subset=["mols"])

    # filter by largest ring
    df_copy["largest_ring"] = df_copy["mols"].apply(get_largest_ring)
    if config["RING_SIZE"]["min"]:
        df_copy = df_copy[df_copy["largest_ring"] >= int(config["RING_SIZE"]["min"])]
    if config["RING_SIZE"]["max"]:
        df_copy = df_copy[df_copy["largest_ring"] <= int(config["RING_SIZE"]["max"])]
    print(f"Number of molecules after filtering by ring size: {len(df_copy)}")

    # filter by num_rings
    df_copy["num_rings"] = df_copy["mols"].apply(Chem.rdMolDescriptors.CalcNumRings)
    if config["NUM_RINGS"]["min"]:
        df_copy = df_copy[df_copy["num_rings"] >= int(config["NUM_RINGS"]["min"])]
    if config["NUM_RINGS"]["max"]:
        df_copy = df_copy[df_copy["num_rings"] <= int(config["NUM_RINGS"]["max"])]
    print(f"Number of molecules after filtering by num_rings: {len(df_copy)}")

    # filter by QED
    df_copy["qed"] = df_copy["mols"].apply(QED.default)
    if config["QED"]["min"]:
        df_copy = df_copy[df_copy["qed"] >= float(config["QED"]["min"])]
    if config["QED"]["max"]:
        df_copy = df_copy[df_copy["qed"] <= float(config["QED"]["max"])]
    print(f"Number of molecules after filtering by QED: {len(df_copy)}")

    # filter by mol_wt
    df_copy["mol_wt"] = df_copy["mols"].apply(Chem.rdMolDescriptors.CalcExactMolWt)
    if config["MOL_WEIGHT"]["min"]:
        df_copy = df_copy[df_copy["mol_wt"] >= float(config["MOL_WEIGHT"]["min"])]
    if config["MOL_WEIGHT"]["max"]:
        df_copy = df_copy[df_copy["mol_wt"] <= float(config["MOL_WEIGHT"]["max"])]
    print(f"Number of molecules after filtering by mol_wt: {len(df_copy)}")

    # filter by num_HBA
    df_copy["num_HBA"] = df_copy["mols"].apply(rdMolDescriptors.CalcNumHBA)
    if config["NUM_HBA"]["min"]:
        df_copy = df_copy[df_copy["num_HBA"] >= int(config["NUM_HBA"]["min"])]
    if config["NUM_HBA"]["max"]:
        df_copy = df_copy[df_copy["num_HBA"] <= int(config["NUM_HBA"]["max"])]
    print(f"Number of molecules after filtering by num_HBA: {len(df_copy)}")

    # filter by num_HBD
    df_copy["num_HBD"] = df_copy["mols"].apply(rdMolDescriptors.CalcNumHBD)
    if config["NUM_HBD"]["min"]:
        df_copy = df_copy[df_copy["num_HBD"] >= int(config["NUM_HBD"]["min"])]
    if config["NUM_HBD"]["max"]:
        df_copy = df_copy[df_copy["num_HBD"] <= int(config["NUM_HBD"]["max"])]
    print(f"Number of molecules after filtering by num_HBD: {len(df_copy)}")

    # filter by logP
    df_copy["logP"] = df_copy["mols"].apply(Crippen.MolLogP)
    if config["LOGP"]["min"]:
        df_copy = df_copy[df_copy["logP"] >= float(config["LOGP"]["min"])]
    if config["LOGP"]["max"]:
        df_copy = df_copy[df_copy["logP"] <= float(config["LOGP"]["max"])]
    print(f"Number of molecules after filtering by logP: {len(df_copy)}")

    # filter by num_rotatable_bonds
    df_copy["num_rotatable_bonds"] = df_copy["mols"].apply(
        rdMolDescriptors.CalcNumRotatableBonds
    )
    if config["NUM_ROT_BONDS"]["min"]:
        df_copy = df_copy[
            df_copy["num_rotatable_bonds"] >= int(config["NUM_ROTATABLE_BONDS"]["min"])
        ]
    if config["NUM_ROT_BONDS"]["max"]:
        df_copy = df_copy[
            df_copy["num_rotatable_bonds"] <= int(config["NUM_ROTATABLE_BONDS"]["max"])
        ]
    print(f"Number of molecules after filtering by num_rotatable_bonds: {len(df_copy)}")

    # filter by TPSA
    df_copy["tpsa"] = df_copy["mols"].apply(rdMolDescriptors.CalcTPSA)
    if config["TPSA"]["min"]:
        df_copy = df_copy[df_copy["tpsa"] >= float(config["TPSA"]["min"])]
    if config["TPSA"]["max"]:
        df_copy = df_copy[df_copy["tpsa"] <= float(config["TPSA"]["max"])]
    print(f"Number of molecules after filtering by TPSA: {len(df_copy)}")

    # filter by bridgehead atoms
    df_copy["bridgehead_atoms"] = df_copy["mols"].apply(
        rdMolDescriptors.CalcNumBridgeheadAtoms
    )
    if config["NUM_BRIDGEHEAD_ATOMS"]["min"]:
        df_copy = df_copy[
            df_copy["bridgehead_atoms"] >= int(config["NUM_BRIDGEHEAD_ATOMS"]["min"])
        ]
    if config["NUM_BRIDGEHEAD_ATOMS"]["max"]:
        df_copy = df_copy[
            df_copy["bridgehead_atoms"] <= int(config["NUM_BRIDGEHEAD_ATOMS"]["max"])
        ]
    print(f"Number of molecules after filtering by bridgehead atoms: {len(df_copy)}")

    # filter by spiro atoms
    df_copy["spiro_atoms"] = df_copy["mols"].apply(rdMolDescriptors.CalcNumSpiroAtoms)
    if config["NUM_SPIRO_ATOMS"]["min"]:
        df_copy = df_copy[
            df_copy["spiro_atoms"] >= int(config["NUM_SPIRO_ATOMS"]["min"])
        ]
    if config["NUM_SPIRO_ATOMS"]["max"]:
        df_copy = df_copy[
            df_copy["spiro_atoms"] <= int(config["NUM_SPIRO_ATOMS"]["max"])
        ]
    print(f"Number of molecules after filtering by spiro atoms: {len(df_copy)}")

    # filter by novelty score
    if config["RUN"]["clf_data_path"] is not None:
        ts = TanimotoSearch(config["RUN"]["clf_data_path"])

        df_copy["novelty_score"] = df_copy["smiles"].apply(
            lambda x: ts(x, return_similar=False)
        )
        if config["NOVELTY_SCORE"]["min"]:
            df_copy = df_copy[
                df_copy["novelty_score"] >= int(config["NOVELTY_SCORE"]["min"])
            ]
        if config["NOVELTY_SCORE"]["max"]:
            df_copy = df_copy[
                df_copy["novelty_score"] <= int(config["NOVELTY_SCORE"]["max"])
            ]
        print(f"Number of molecules after filtering by novelty score: {len(df_copy)}")

    else:
        print(
            "Path to the QSAR model training set is not provided or invalid. Skipping novelty filtering."
        )

    # drop redundant columns
    df_copy.drop(columns=["mols"], inplace=True)

    return df_copy


def simple_decoder(vector, vectorizer, return_invalid=False):
    """
    Decodes model output to sequence strings using a simple decoder.
    Args:
        vector (np.array): Latent vector.
        vectorizer (Vectorizer): vectorizer object.
    Returns:
        str: SMILES string.
    """
    vector = softmax(vector, axis=-1)

    if isinstance(vectorizer, DeepSMILESVectorizer):
        converter = ds.Converter(rings=True, branches=True)
    else:
        converter = None

    decoded = vectorizer.devectorize(vector, remove_special=True, reduction="max")

    if isinstance(vectorizer, DeepSMILESVectorizer):
        try:
            decoded = converter.decode(decoded)
        except ds.exceptions.DecodeError:
            decoded = ""

    if Chem.MolFromSmiles(decoded) is not None or return_invalid:
        return decoded
    else:
        return "invalid"


def stochastic_decoder(vector, vectorizer, n_trials=1000, verbose=False):
    """
    Decodes model output to sequence strings using a stochastic decoder.
    Out of n_trials, selects only valid SMILES strings returns the sequence with the highest likelihood.
    The first sequence is always the one with the highest likelihood.
    Args:
        vector (np.array): Latent vector.
        vectorizer (Vectorizer): vectorizer object.
        n_trials (int): Number of samples to generate.
        verbose (bool): Whether to print progress.
    Returns:
        str: SMILES string.
    """
    vector = softmax(vector, axis=-1)

    if isinstance(vectorizer, DeepSMILESVectorizer):
        converter = ds.Converter(rings=True, branches=True)
    else:
        converter = None

    seq_list = []
    score_list = np.zeros(n_trials + 1)

    decoded, likelihood_product = vectorizer.devectorize_and_score(
        vector, remove_special=True, reduction="max"
    )
    seq_list.append(decoded)
    score_list[0] = likelihood_product

    for i in range(1, n_trials + 1):
        decoded, likelihood_product = vectorizer.devectorize_and_score(
            vector, remove_special=True, reduction="sample"
        )
        seq_list.append(decoded)
        score_list[i] = likelihood_product

    seq_list = np.array(seq_list)
    if isinstance(vectorizer, DeepSMILESVectorizer):
        smiles_list = np.apply_along_axis(converter.decode, 0, seq_list)

    else:
        smiles_list = seq_list

    valid_idcs = [
        idx for idx, smiles in enumerate(smiles_list) if Chem.MolFromSmiles(smiles)
    ]

    if valid_idcs:
        possible_smiles = smiles_list[valid_idcs]
        best_smile = possible_smiles[np.argmax(score_list[valid_idcs])]
        return best_smile
    else:
        return "invalid"
