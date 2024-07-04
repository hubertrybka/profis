from profis.utils.finger import encode
import pandas as pd
from profis.utils.modelinit import initialize_model
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model_path", type=str, required=True
)
model_path = parser.parse_args().model_path

config_path = model_path.replace(model_path.split("/")[-1], 'hyperparameters.ini')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_path.split("/")[-2].split("_")[-2] == "KRFP":
    val_df = pd.read_parquet("data/RNN_dataset_KRFP_val_10.parquet")
elif model_path.split("/")[-2].split("_")[-2] == "ECFP":
    val_df = pd.read_parquet("data/RNN_dataset_ECFP_val_10.parquet")
else:
    raise ValueError("Failed to parse FP type from model path.")

model = initialize_model(config_path, device=device)
model.load_state_dict(torch.load(model_path, map_location=device))

mus, _ = encode(val_df, model, device)

means = mus.mean(axis=0)
stds = mus.std(axis=0)
bounds = pd.DataFrame({"mean": means, "std": stds}, index=range(len(means)))

epoch = model_path.split("/")[-1].split("_")[1].split(".")[0]
bounds.to_csv(model_path.replace('epoch_300.pt', f'latent_bounds_{epoch}.csv'))