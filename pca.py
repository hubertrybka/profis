from profis.utils.modelinit import initialize_model
import pandas as pd
from profis.utils.finger import encode
import torch
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
from matplotlib.lines import Line2D

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

    df = pd.read_parquet(d2_path).sample(5000)
    activity_column = df["activity"]
    activity_map = ['train +' if x == 1 else 'train -' for x in activity_column]
    encoded_d2, _ = encode(df, vae, device)

    samples = pd.read_csv(samples_path).drop(["score", "norm"], axis=1)
    distance_to_model = samples["distance_to_model"].to_numpy()
    samples = samples.drop(["distance_to_model"], axis=1)
    pca = PCA(n_components=2)

    pca.fit(encoded_d2)
    d2_pca = pd.DataFrame(pca.transform(encoded_d2))

    samples_pca = pd.DataFrame(pca.transform(samples)).sample(1000)

    palette_mako = sns.color_palette("mako", 2)

    sns.set_style("white")
    sns.set_context("paper")
    sns.scatterplot(
        x=d2_pca[0],
        y=d2_pca[1],
        hue=activity_map,
        alpha=1,
        markers=".",
        size=10,
        palette=palette_mako,
        hue_order=['train +', "train -"],
        linewidth=0,
    )
    sns.scatterplot(
        x=samples_pca[0], y=samples_pca[1], alpha=1, markers=".", size=10, color="indianred", linewidth=0,
    )
    plt.annotate(
        text=f"mean DM: {round(distance_to_model.mean(axis=0), 3)}",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
    )
    lgnd = plt.legend(['train +', 'train -', 'profis'])
    lgnd.legendHandles[0].update({'color': palette_mako[0], 'sizes': [30]})
    lgnd.legendHandles[1].update({'color': palette_mako[1], 'sizes': [30]})
    lgnd.legendHandles[2].update({'color': 'indianred', 'sizes': [30]})

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-s", "--samples_path", type=str)
    args = parser.parse_args()
    main(args.model_path, args.samples_path)
