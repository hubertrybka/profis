import pandas as pd
import argparse
import rdkit.Chem as Chem
import rdkit.Chem.MolStandardize.rdMolStandardize as rdMolStandardize
from profis.utils.finger import sparse2dense, smiles2sparse_KRFP, smiles2sparse_ECFP


def prepare(data_path, gen_ecfp=False, gen_krfp=False, to_dense=True):

    # handle input
    if gen_ecfp and gen_krfp:
        raise ValueError("Please choose only one fingerprint type to generate")

    # Load the dataset
    df = pd.read_csv(data_path)

    # Check the column names
    df.columns = df.columns.str.lower()
    if 'fps' in df.columns and (gen_ecfp or gen_krfp):
        raise ValueError("Fingerprint column already exists in the dataset")

    for name in ["smiles", "activity", "fps"]:
        if name not in df.columns:
            raise ValueError(f"Column {name} not found in the dataset")

    print(f"Loaded data from {data_path}")

    # Standardize the SMILES strings and check for validity
    df["smiles"] = df["smiles"].apply(try_standardize_smiles)
    df = df.dropna(subset=["smiles"])

    if gen_ecfp:
        # Generate ECFP fingerprints
        df["fps"] = df["smiles"].apply(smiles2sparse_ECFP)
    elif gen_krfp:
        # Generate KRFP fingerprints
        df["fps"] = df["smiles"].apply(smiles2sparse_KRFP)

    # Convert sparse fingerprints to dense fingerprints
    if to_dense or gen_ecfp or gen_krfp:
        df["fps"] = df["fps"].apply(sparse2dense)

    # Save the processed dataset
    out_path = data_path.replace(".parquet", "_processed.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Processed data saved to {out_path}")

    return


def try_standardize_smiles(smiles):
    """
    Standardize the SMILES string. Returns None if the SMILES string is invalid or cannot be standardized.
    Args:
        smiles (str): SMILES string.
    Returns:
        str: Standardized SMILES string.
    """
    u = rdMolStandardize.Uncharger()
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = rdMolStandardize.Cleanup(mol)
        uncharged_mol = u.uncharge(mol)
    except:
        return None
    return Chem.MolToSmiles(uncharged_mol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--gen_ecfp", action="store_true", help="Generate ECFP fingerprints")
    parser.add_argument("--gen_krfp", action="store_true", help="Generate KRFP fingerprints")
    parser.add_argument("--to_dense", action="store_true", help="Convert sparse fingerprints to dense fingerprints")
    args = parser.parse_args()
    prepare(args.data_path, args.skip_fingerprint)
