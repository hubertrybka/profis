import argparse
import configparser
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer

from profis.clf.scorer import SKLearnScorer
from profis.clf.applicability import SCAvgMeasure


# suppress scikit-learn warnings
def warn(*args, **kwargs):
    pass


warnings.warn = warn


def bayesian_search(config, scorer, sc_avg):
    """
    Perform Bayesian optimization on the latent space with respect to the classifier's output class probability.
    Args:
        config (configparser.ConfigParser): config file
        scorer (SKLearnScorer): scorer object
        sc_avg (SCAvgMeasure): distance to model calculator
    """

    # read config file

    latent_size = int(config["SEARCH"]["latent_size"])
    n_init = int(config["SEARCH"]["n_init"])
    n_iter = int(config["SEARCH"]["n_iter"])
    bounds = float(config["SEARCH"]["bounds"])
    verbosity = int(config["SEARCH"]["verbosity"])

    # define bounds
    pbounds = {str(p): (-bounds, bounds) for p in range(latent_size)}
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.2)

    # initialize optimizer
    optimizer = BayesianOptimization(
        f=scorer,
        pbounds=pbounds,
        random_state=(time.time_ns() % 10 ** 6),
        verbose=verbosity > 1,
        bounds_transformer=bounds_transformer,
    )

    vector_list = []
    score_list = []
    model_distance_list = []

    # run optimization:
    optimizer.maximize(
        init_points=n_init,
        n_iter=n_iter,
    )
    vector = np.array(list(optimizer.max["params"].values()))
    print(f"Best score: {optimizer.max['target']}") if verbosity > 1 else None

    score_list.append(float(optimizer.max["target"]))
    model_distance_list.append(sc_avg(vector))
    vector_list.append(vector)

    # append results to return list

    samples = pd.DataFrame(np.array(vector_list))
    samples.columns = [str(n) for n in range(latent_size)]
    samples["score"] = score_list
    samples["score"] = samples["score"].astype(float)
    samples["norm"] = np.linalg.norm(samples.iloc[:, :-1], axis=1)
    samples["distance_to_model"] = model_distance_list
    return samples


def random_search(config, scorer, sc_avg):
    """
    Perform random search on the latent space with respect to the classifier's output class probability.
    Args:
        config (configparser.ConfigParser): config file
        scorer (SKLearnScorer): scorer object
        sc_avg (SCAvgMeasure): distance to model calculator
    """

    # read config file
    latent_size = int(config["SEARCH"]["latent_size"])
    n_samples = int(config["SEARCH"]["n_samples"])
    bounds = float(config["SEARCH"]["bounds"])

    vector_list = []
    score_list = []
    model_distance_list = []

    for i in range(n_samples):
        vector = np.random.uniform(-bounds, bounds, latent_size)
        score = scorer(**{str(p): vector[p] for p in range(latent_size)})
        model_distance = sc_avg(vector)
        vector_list.append(vector)
        score_list.append(score)
        model_distance_list.append(model_distance)

    # append results to return list
    samples = pd.DataFrame(np.array(vector_list))
    samples.columns = [str(n) for n in range(latent_size)]
    samples["score"] = score_list
    samples["score"] = samples["score"].astype(float)
    samples["norm"] = np.linalg.norm(samples.iloc[:, :-1], axis=1)
    samples["distance_to_model"] = model_distance_list
    return samples


if __name__ == "__main__":
    random.seed(42)
    """
    Multiprocessing support and queue handling
    """
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="config_files/search_config.ini"
    )
    config_path = parser.parse_args().config

    # read config file
    config = configparser.ConfigParser()
    config.read(config_path)

    n_workers = int(config["SEARCH"]["n_workers"])
    verbosity = int(config["SEARCH"]["verbosity"])
    n_samples = int(config["SEARCH"]["n_samples"])
    n_init = int(config["SEARCH"]["n_init"])
    n_iter = int(config["SEARCH"]["n_iter"])
    bounds = float(config["SEARCH"]["bounds"])
    latent_size = int(config["SEARCH"]["latent_size"])
    model_path = config["SEARCH"]["model_path"]
    add_timestamp = config["SEARCH"].getboolean("add_timestamp")
    output_path = config["SEARCH"]["output_dir"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # initialize model distance calculator
    sc_avg = SCAvgMeasure(clf_path=model_path)

    # initialize scorer
    scorer = SKLearnScorer(model_path)

    # create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dirname = "latent_vectors_" + timestamp if add_timestamp else "latent_vectors"
    os.mkdir(output_path) if not os.path.isdir(output_path) else None
    (
        os.mkdir(f"{output_path}/{dirname}")
        if not os.path.isdir(f"{output_path}/{dirname}")
        else None
    )

    print("Starting search") if verbosity > 0 else None
    # run search
    samples = []
    for i in range(n_samples):
        vector = bayesian_search(config, scorer, sc_avg)
        # vector = random_search(config, scorer, sc_avg)
        samples.append(vector)
        if i % 100 == 0 or i == (n_samples - 1):
            pd.concat(samples).to_csv(
                f"{output_path}/{dirname}/latent_vectors.csv", index=False
            )

    # read the results
    with open(f"{output_path}/{dirname}/latent_vectors.csv", "r") as f:
        samples = pd.read_csv(f)

    end_time = time.time()
    time_elapsed = end_time - start_time  # in seconds
    if time_elapsed < 60:
        (
            print("Time elapsed: ", round(time_elapsed, 2), "s")
            if verbosity > 0
            else None
        )
    else:
        (
            print(
                "Time elapsed: ",
                int(time_elapsed // 60),
                "min",
                round(time_elapsed % 60, 2),
                "s",
            )
            if verbosity > 0
            else None
        )

    # save the arguments
    with open(f"{output_path}/{dirname}/info.txt", "w") as f:
        text = [
            f"model_path: {model_path}",
            f"latent_size: {latent_size}",
            f"n_samples: {n_samples}",
            f"init_points: {n_init}",
            f"n_iter: {n_iter}",
            f"bounds: {bounds}",
            f"verbosity: {verbosity}",
            f"time elapsed per sample: {round(time_elapsed / n_samples, 2)} s",
            f'mean score: {round(samples["score"].mean(), 2)}'
            if len(samples) > 0
            else round(samples["score"].values[0], 2),
            f'sigma score: {round(samples["score"].std(), 2)}'
            if len(samples) > 0
            else "",
            f'mean norm: {round(samples["norm"].mean(), 2)}'
            if len(samples) > 0
            else round(samples["norm"].values[0], 2),
            f'sigma norm: {round(samples["norm"].std(), 2)}'
            if len(samples) > 0
            else "",
        ]
        text = "\n".join(text)
        f.write(text)

    print(f"Results saved to: {output_path}/{dirname}") if verbosity > 0 else None
