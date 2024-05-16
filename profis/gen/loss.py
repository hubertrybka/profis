import torch
import torch.nn as nn


class CCE(nn.Module):
    def __init__(self, notation: str = "SMILES"):
        """
        Conscious Cross-Entropy. Calculates cross-entropy loss on two SMILES or SELFIES,
        ignoring indices of padding tokens.
        Parameters:
            notation (str): format of the input strings. Either "SELFIES" or "SMILES"
        """
        super(CCE, self).__init__()
        alphabet_path = (
            "data/selfies_alphabet.txt"
            if notation == "SELFIES"
            else "data/smiles_alphabet.txt"
        )
        self.idx_ignore = self.determine_nop_idx(alphabet_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, target, predictions):
        """
        Forward pass
        Args:
            target (torch.Tensor): target tensor of shape [batch_size, seq_len, token_idx]
            predictions (torch.Tensor): predictions tensor of shape [batch_size, seq_len, token_idx]
        Returns:
            loss (torch.Tensor): loss value
        """
        # target.shape [batch_size, seq_len, token_idx]
        batch_size = predictions.shape[0]
        one_hot = target.argmax(dim=-1)
        mask = one_hot != self.idx_ignore
        weights = (mask.T / mask.sum(axis=1)).T[mask]
        loss = torch.nn.functional.cross_entropy(
            predictions[mask], one_hot[mask], reduction="none"
        )
        return (weights * loss).sum() / batch_size

    @staticmethod
    def determine_nop_idx(alphabet_path):
        """
        Determine the index of the [nop] token in the alphabet
        """
        with open(alphabet_path, "r") as f:
            alphabet = f.read().splitlines()
            return alphabet.index("[nop]")
