import re

import numpy as np


class Vectorizer:
    def __init__(self, pad_to_len=None, alphabet_path=None):
        self.pad_to_len = pad_to_len
        self.alphabet = self.read_alphabet(alphabet_path)
        self.char2idx = {s: i for i, s in enumerate(self.alphabet)}
        self.idx2char = {i: s for i, s in enumerate(self.alphabet)}

    def vectorize(self, sequence):
        """
        Vectorize a list of strings to a numpy array of shape (len(sequence), len(charset))
        Args:
            sequence (string):list of strings
        Returns:
            X (numpy.ndarray): vectorized strings
        """
        if self.pad_to_len is None:
            splited = self.split(sequence)
        else:
            splited = (
                ["[start]"]
                + self.split(sequence)
                + ["[end]"]
                + ["[nop]"] * (self.pad_to_len - len(self.split(sequence)) - 2)
            )
        X = np.zeros((len(splited), len(self.alphabet)))
        for i in range(len(splited)):
            if splited[i] not in self.alphabet:
                raise ValueError(
                    f"Invalid token: {splited[i]} allowed tokens: {self.alphabet}"
                )
            X[i, self.char2idx[splited[i]]] = 1
        return X

    def devectorize(self, ohe, remove_special=False, reduction="max"):
        """
        Devectorize a numpy array of shape (len(sequence), len(charset)) to a string
        Args:
            ohe (numpy.ndarray): one-hot encoded sequence as numpy array
            remove_special (bool): remove special tokens
            reduction (string): reduction method, either 'max' or 'sample'
        Returns:
            sequence_str (string): string
        """
        sequence_str = ""
        for j in range(ohe.shape[0]):
            if reduction == "max":
                idx = np.argmax(ohe[j, :])
            elif reduction == "sample":
                idx = np.random.choice(np.arange(len(self.alphabet)), p=ohe[j, :])
            else:
                raise ValueError('Reduction must be either "max" or "sample"')
            if remove_special and (
                self.idx2char[idx] == "[start]"
                or self.idx2char[idx] == "[end]"
                or self.idx2char[idx] == "[nop]"
            ):
                continue
            sequence_str += self.idx2char[idx]
        return sequence_str

    def idxize(self, sequence, no_special=False):
        if no_special:
            splited = self.split(sequence)

    def deidxize(self, idx, no_special=False):
        if no_special:
            selfie = []
            for i in idx:
                char = self.idx2char[i]
                if char not in ["[end]", "[nop]", "[start]"]:
                    selfie.append(char)
            return "".join(selfie)
        else:
            return "".join([self.idx2char[i] for i in idx])

    @staticmethod
    def split(selfie):
        raise NotImplementedError

    @staticmethod
    def read_alphabet(path):
        with open(path, "r") as f:
            alphabet = f.read().splitlines()
        return alphabet


class SELFIESVectorizer(Vectorizer):
    """
    Vectorizer for SELFIES strings.
    """

    def __init__(self, pad_to_len=None, alphabet_path="data/selfies_alphabet.txt"):
        Vectorizer.__init__(self, pad_to_len, alphabet_path)

    @staticmethod
    def split(selfie):
        pattern = r"(\[[^\[\]]*\])"
        return re.findall(pattern, selfie)


class SMILESVectorizer(Vectorizer):
    """
    Vectorizer for SMILES strings.
    """

    def __init__(self, pad_to_len=None, alphabet_path="data/smiles_alphabet.txt"):
        Vectorizer.__init__(self, pad_to_len, alphabet_path)

    @staticmethod
    def split(smile):
        pattern = (
            r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|"
            r"\*|\$|\%[0-9]{2}|[0-9]|[start]|[nop]|[end])"
        )
        return re.findall(pattern, smile)


class DeepSMILESVectorizer(Vectorizer):
    """
    Vectorizer for DeepSMILES strings.
    """

    def __init__(self, pad_to_len=None, alphabet_path="data/deepsmiles_alphabet.txt"):
        Vectorizer.__init__(self, pad_to_len, alphabet_path)

    @staticmethod
    def split(deepsmile):
        pattern = (
            r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|"
            r"\*|\$|\%[0-9]{2}|[0-9]|[start]|[nop]|[end])"
        )
        return re.findall(pattern, deepsmile)
