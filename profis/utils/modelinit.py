import configparser

import torch

from profis.gen.generator import EncoderDecoderV3


def initialize_model(
    config_path, device=torch.device("cpu"), use_dropout=False, teacher_forcing=False
):
    """
    Initialize model from a given path
    Args:
        config_path (str): path to the config file
        device (torch.device): device to be used for training
        use_dropout (bool): whether to use dropout
        teacher_forcing (bool): whether to use teacher forcing
    Returns:
        model (torch.nn.Module): initialized model
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    torch_device = device
    out_encoding = config["RUN"]["out_encoding"]
    model = EncoderDecoderV3(
        fp_size=int(config["MODEL"]["fp_len"]),
        encoding_size=int(config["MODEL"]["encoding_size"]),
        hidden_size=int(config["MODEL"]["hidden_size"]),
        num_layers=int(config["MODEL"]["num_layers"]),
        dropout=float(config["MODEL"]["dropout"]) if use_dropout else 0.0,
        output_size=get_alphabet_len(out_encoding),
        teacher_ratio=float(
            config["MODEL"]["teacher_ratio"] if teacher_forcing else 0.0
        ),
        random_seed=42,
        use_cuda=config.getboolean("RUN", "use_cuda"),
        fc1_size=int(config["MODEL"]["fc1_size"]),
        fc2_size=int(config["MODEL"]["fc2_size"]),
        fc3_size=int(config["MODEL"]["fc3_size"]),
        encoder_activation=config["MODEL"]["encoder_activation"],
        fc2_enabled=config.getboolean("MODEL", "fc2_enabled"),
        fc3_enabled=config.getboolean("MODEL", "fc3_enabled"),
    ).to(torch_device)
    return model


def get_alphabet_len(format: str):
    """
    Get the alphabet length (number of tokens) for a given output format
    Args:
        format (str): Can be 'smiles', 'selfies' or 'deepsmiles'
    Returns:
        int: alphabet length
    """
    with open(f"data/{format}_alphabet.txt", "r") as f:
        alphabet = f.read().splitlines()

    return len(alphabet)