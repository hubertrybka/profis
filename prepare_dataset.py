import pandas as pd
import argparse
import rdkit.Chem as Chem
import rdkit.Chem.MolStandardize.rdMolStandardize as rdMolStandardize
from profis.utils.finger import sparse2dense


def prepare(data_path, skip_fingerprint=False):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Check the column names
    column_names = df.columns
    for name in ["smiles", "activity", "fps"]:
        if name not in column_names:
            raise ValueError(f"Column {name} not found in the dataset")

    df = df[["smiles", "activity"]]
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded data from {data_path}")

    # Standardize the SMILES strings and check for validity
    df["smiles"] = df["smiles"].apply(standardize_smiles)

    # Convert sparse fingerprints to dense representation
    if not skip_fingerprint:
        if df["fps"].apply(is_sparse).all():
            print("Converting sparse fingerprints to dense representation")
            df["fps"] = df["fps"].apply(sparse2dense)
        else:
            print("Fingerprints are valid and in dense representation")

    # Save the processed dataset
    out_path = data_path.replace(".parquet", "_processed.parquet")
    df.to_parquet(out_path, index=False)

    return


def standardize_smiles(smiles):
    """
    Standardize the SMILES string.
    """
    u = rdMolStandardize.Uncharger()

    mol = Chem.MolFromSmiles(smiles)
    mol = rdMolStandardize.Cleanup(mol)
    uncharged_mol = u.uncharge(mol)
    return Chem.MolToSmiles(uncharged_mol)


def is_sparse(fp):
    """
    Check if the fingerprint is sparse.
    """
    if isinstance(fp, list):
        for x in fp:
            if not isinstance(x, int):
                raise ValueError("Fingerprint must be a list of integers")
            if x not in [0, 1]:
                return False
            else:
                return True
    else:
        raise ValueError("Fingerprint must be a list")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--skip_fingerprint", action="store_true", help="Skip fingerprint processing")
    args = parser.parse_args()
    prepare(args.data_path, args.skip_fingerprint)
