# Test dataset
from profis.utils.vectorizer import SELFIESVectorizer, SMILESVectorizer
from profis.gen.dataset import SELFIESDataset, SMILESDataset, LatentEncoderDataset
import pandas as pd


def test_vectorizers():
    # Test SELFIES and SMILES vectorizers
    seq_length = 128
    selfies_vec = SELFIESVectorizer(
        pad_to_len=seq_length, alphabet_path="data/selfies_alphabet.txt"
    )
    smiles_vec = SMILESVectorizer(
        pad_to_len=seq_length, alphabet_path="data/smiles_alphabet.txt"
    )

    run_vectorizer_check(selfies_vec, seq_length)
    run_vectorizer_check(smiles_vec, seq_length)

    return


def run_vectorizer_check(vectorizer, seq_length):
    assert len(vectorizer.alphabet) > 0
    test_smile = "C1C2=NN=CN2C3=C(C=C(C=C3)Cl)C(=N1)C4=CC=CC=C4"
    test_selfie = (
        "[C][C][=N][N][=C][N][Ring1][Branch1][C][=C][Branch1][#Branch2][C][=C][Branch1][Branch1][C][=C]"
        "[Ring1][=Branch1][Cl][C][=Branch1][Ring2][=N][Ring1][#C][C][=C][C][=C][C][=C][Ring1][=Branch1]"
    )

    if isinstance(vectorizer, SMILESVectorizer):  # SMILES vectorizer
        ohe = vectorizer.vectorize(test_smile)
        reconstructed = vectorizer.devectorize(ohe, remove_special=True)
        assert ohe.shape == (seq_length, len(vectorizer.alphabet))
        assert sum(sum(ohe)) == seq_length
        assert reconstructed == test_smile

    else:  # SELFIES vectorizer
        ohe = vectorizer.vectorize(test_selfie)
        reconstructed = vectorizer.devectorize(ohe, remove_special=True)
        assert ohe.shape == (seq_length, len(vectorizer.alphabet))
        assert sum(sum(ohe)) == seq_length
        assert reconstructed == test_selfie
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

    run_dataset_check(selfies_dataset, selfies_vec)
    run_dataset_check(smiles_dataset, smiles_vec)

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
