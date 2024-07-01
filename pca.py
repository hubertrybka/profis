from profis.utils.modelinit import initialize_model
import pandas as pd
from profis.utils.finger import encode
import torch
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import configparser


def main(model_path, samples_path):

    config = configparser.ConfigParser()
    config.read(model_path.replace(model_path.split("/")[-1], f"hyperparameters.ini"))
    if config["MODEL"]["fp_len"] == "4860":
        fp_type = "KRFP"
    elif config["MODEL"]["fp_len"] == "2048":
        fp_type = "ECFP"
    else:
        raise ValueError(f'Invalid fp_len: {config["MODEL"]["fp_len"]}')

    d2_path = f"data/d2_{fp_type}_100nM.parquet"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = initialize_model(
        model_path.replace(model_path.split("/")[-1], f"hyperparameters.ini"), device
    )
    vae.load_state_dict(torch.load(model_path, map_location=device))

    df = pd.read_parquet(d2_path)
    activity_column = df["activity"]
    encoded_d2, _ = encode(df, vae, device)

    samples = pd.read_csv(samples_path).drop(["score", "norm"], axis=1)
    distance_to_model = samples["distance_to_model"].to_numpy()
    samples = samples.drop(["distance_to_model"], axis=1)
    pca = PCA(n_components=2)

    pca.fit(encoded_d2)
    d2_pca = pd.DataFrame(pca.transform(encoded_d2))

    samples_pca = pd.DataFrame(pca.transform(samples)).sample(1000)

    activity_mapped = [
        "train +" if activity == 1 else "train -" for activity in activity_column
    ]
    sns.set_style("white")
    sns.set_context("paper")
    sns.scatterplot(
        data=d2_pca,
        x=0,
        y=1,
        hue=activity_mapped,
        alpha=0.6,
        markers=".",
        size=10,
        palette="viridis",
    ).set_title("PCA")
    sns.scatterplot(
        data=samples_pca, x=0, y=1, alpha=0.6, markers=".", size=10, color="red"
    )
    plt.annotate(
        text=f"mean DM: {round(distance_to_model.mean(axis=0), 3)}",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-s", "--samples_path", type=str)
    args = parser.parse_args()
    main(args.model_path, args.samples_path)
