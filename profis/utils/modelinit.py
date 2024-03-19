import configparser

import torch

from profis.gen.generator import EncoderDecoderV3


def initialize_model(config_path, device=torch.device("cpu"), use_dropout=False):
    """
    Initialize model from a given path
    Args:
        config_path (str): path to the config file
        use_dropout (bool): whether to use dropout
        device (str): device to be used for training ('cpu' or 'cuda')
    Returns:
        model (torch.nn.Module): initialized model
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    torch_device = device
    use_selfies = config.getboolean("RUN", "use_selfies")
    model = EncoderDecoderV3(
        fp_size=int(config["MODEL"]["fp_len"]),
        encoding_size=int(config["MODEL"]["encoding_size"]),
        hidden_size=int(config["MODEL"]["hidden_size"]),
        num_layers=int(config["MODEL"]["num_layers"]),
        dropout=float(config["MODEL"]["dropout"]) if use_dropout else 0.0,
        output_size=31 if use_selfies else 32,
        teacher_ratio=0.0,
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
