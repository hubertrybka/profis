# Test dataset
from profis.utils.vectorizer import (
    SELFIESVectorizer,
    SMILESVectorizer,
    DeepSMILESVectorizer,
)
from profis.gen.dataset import (
    SELFIESDataset,
    SMILESDataset,
    DeepSMILESDataset,
    LatentEncoderDataset,
)
import pandas as pd


def test_smiles_vectorizer():
    # Test SMILES vectorizer
    seq_length = 128
    smiles_vec = SMILESVectorizer(
        pad_to_len=seq_length, alphabet_path="data/smiles_alphabet.txt"
    )

    assert len(smiles_vec.alphabet) > 0
    test_smile = "C1C2=NN=CN2C3=C(C=C(C=C3)Cl)C(=N1)C4=CC=CC=C4"

    ohe = smiles_vec.vectorize(test_smile)
    reconstructed = smiles_vec.devectorize(ohe, remove_special=True)
    assert ohe.shape == (seq_length, len(smiles_vec.alphabet))
    assert sum(sum(ohe)) == seq_length
    assert reconstructed == test_smile
    return


def test_selfies_vectorizer():
    # Test SELFIES vectorizer
    seq_length = 128
    selfies_vec = SELFIESVectorizer(
        pad_to_len=seq_length, alphabet_path="data/selfies_alphabet.txt"
    )

    assert len(selfies_vec.alphabet) > 0
    test_selfie = (
        "[C][S][=Branch1][C][=O][=Branch1][C][=O][N][C][=C][C][=C][Branch2][Ring2][Ring1][C][=N][N][Branch1]"
        "[=Branch2][S][Branch1][C][C][=Branch1][C][=O][=O][C][Branch1][#C][C][=C][C][=C][N][=C][C][=N][C]"
        "[Ring1][=Branch1][=C][Ring1][#Branch2][C][Ring2][Ring1][Ring1][C][=C][Ring2][Ring1][=Branch2]"
    )
    ohe = selfies_vec.vectorize(test_selfie)
    reconstructed = selfies_vec.devectorize(ohe, remove_special=True)
    assert ohe.shape == (seq_length, len(selfies_vec.alphabet))
    assert sum(sum(ohe)) == seq_length
    assert reconstructed == test_selfie
    return


def test_deepsmiles_vectorizer():
    # Test DeepSMILES vectorizer
    seq_length = 128
    deepsmiles_vec = DeepSMILESVectorizer(
        pad_to_len=seq_length, alphabet_path="data/deepsmiles_alphabet.txt"
    )

    assert len(deepsmiles_vec.alphabet) > 0
    test_deepsmile = "CCCC=O)NCcccco5))))))))nncC)ccC)n-ccccC)cc6))))))nc5c9=O"

    ohe = deepsmiles_vec.vectorize(test_deepsmile)
    reconstructed = deepsmiles_vec.devectorize(ohe, remove_special=True)
    assert ohe.shape == (seq_length, len(deepsmiles_vec.alphabet))
    assert sum(sum(ohe)) == seq_length
    assert reconstructed == test_deepsmile
    return


def test_datasets():
    # Test SELFIES and SMILES datasets
    selfies_vec = SELFIESVectorizer(pad_to_len=128)
    selfies_dataset = SELFIESDataset(
        pd.read_parquet("tests/data/test_RNN_dataset_KRFP.parquet"),
        vectorizer=selfies_vec,
        fp_len=4860,
    )

    smiles_vec = SMILESVectorizer(pad_to_len=128)
    smiles_dataset = SMILESDataset(
        pd.read_parquet("tests/data/test_RNN_dataset_KRFP.parquet"),
        vectorizer=smiles_vec,
        fp_len=4860,
    )

    deepsmiles_vec = DeepSMILESVectorizer(pad_to_len=128)
    deepsmiles_dataset = DeepSMILESDataset(
        pd.read_parquet("tests/data/test_RNN_dataset_KRFP.parquet"),
        vectorizer=deepsmiles_vec,
        fp_len=4860,
    )

    run_dataset_check(selfies_dataset, selfies_vec)
    run_dataset_check(smiles_dataset, smiles_vec)
    run_dataset_check(deepsmiles_dataset, deepsmiles_vec)

    # Test latent encoder dataset
    encoder_dataset = LatentEncoderDataset(
        pd.read_parquet("tests/data/test_RNN_dataset_KRFP.parquet"), fp_len=4860
    )
    run_dataset_check(encoder_dataset, None)

    return


def run_dataset_check(dataset, vectorizer):
    assert len(dataset) == 10
    sums = [18, 36, 53, 50, 35, 56, 50, 89, 69, 73]

    if not isinstance(dataset, LatentEncoderDataset):
        assert vectorizer is not None

    for i, x in enumerate(sums):

        # Check that the sum of the output fingerprint is correct
        if isinstance(dataset, LatentEncoderDataset):
            on_bits = int(sum(dataset[i]))
        else:
            on_bits = int(sum(dataset[i][0]))
        assert on_bits == x

        if not isinstance(dataset, LatentEncoderDataset):
            # Check that the output sequence is the correct length and OHE-encoded
            assert dataset[i][1].shape == (128, len(vectorizer.alphabet))
            assert sum(sum(dataset[0][1])) == 128
    return
