import argparse
import configparser
import os
import random
import time
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer

from profis.clf.scorer import SKLearnScorer
from profis.clf.applicability import SCAvgMeasure


# suppress scikit-learn warnings
def warn(*args, **kwargs):
    pass


warnings.warn = warn


def bayesian_search(job_package):
    """
    Perform Bayesian optimization on the latent space with respect to the classifier's output class probability.
    Args:
        job_package: tuple containing the following elements:
            n_samples  (int): number of samples to generate
            config (configparser.ConfigParser): configuration object
    """

    # unpack job package
    n_samples, config, bounds_path = job_package

    # read config file
    latent_size = int(config["SEARCH"]["latent_size"])
    n_init = int(config["SEARCH"]["n_init"])
    n_iter = int(config["SEARCH"]["n_iter"])
    verbosity = int(config["SEARCH"]["verbosity"])
    if bounds_path is not None:
        bounds = pd.read_csv(bounds_path)
        means = bounds["mean"].to_numpy()
        stds = bounds["std"].to_numpy()
        pbounds = {
            str(p): (means[p] - 2 * stds[p], means[p] + 2 * stds[p])
            for p in range(latent_size)
        }
        pbounds_sizes = np.array(
            [pbounds[str(p)][1] - pbounds[str(p)][0] for p in range(latent_size)]
        )
        min_window = pbounds_sizes.min() * 0.1
        print("Min window: ", min_window) if verbosity > 0 else None
    else:
        bounds = config["SEARCH"]["bounds"]
        pbounds = {str(p): (-bounds, bounds) for p in range(latent_size)}
        min_window = 0.1 * bounds

    worker_id = int(mp.current_process().name.split("-")[-1])
    print(f"(mp debug) Worker {worker_id} started and will generate {n_samples} samples ") if verbosity > 0 else None


    # initialize scorer
    scorer = SKLearnScorer(config["SEARCH"]["model_path"])

    # define bounds transformer
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=min_window)

    vector_list = []
    score_list = []

    # run optimization
    for j in range(n_samples):
        if j % 10 == 0:
            print(f"(mp debug) Worker {worker_id} finished {len(vector_list)} samples") if verbosity > 0 else None
        # initialize optimizer
        optimizer = BayesianOptimization(
            f=scorer,
            pbounds=pbounds,
            verbose=verbosity > 1,
            bounds_transformer=bounds_transformer,
        )
        optimizer.maximize(
            init_points=n_init,
            n_iter=n_iter,
        )
        vector = np.array(list(optimizer.max["params"].values()))
        score_list.append(float(optimizer.max["target"]))
        vector_list.append(vector)

    # create dataframe for the results
    samples = pd.DataFrame(np.array(vector_list))
    samples.columns = [str(n) for n in range(latent_size)]
    samples["score"] = score_list
    samples["score"] = samples["score"].astype(float)
    samples["norm"] = np.linalg.norm(samples.iloc[:, :-1], axis=1)
    return samples


def distribute_jobs(n_samples, n_workers):
    """
    Distribute the jobs among the workers.
    Args:
        n_samples (int): number of samples to generate
        n_workers (int): number of workers
    Returns:
        list: list of integers representing the number of samples each worker should generate
    """
    initial_chunk_size = n_samples // n_workers
    remainder = n_samples % n_workers
    chunk_sizes = [initial_chunk_size] * n_workers
    for i in range(remainder):
        chunk_sizes[i] += 1
    return chunk_sizes


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
    bounds = config["SEARCH"]["bounds"]
    latent_size = int(config["SEARCH"]["latent_size"])
    model_path = config["SEARCH"]["model_path"]
    add_timestamp = config["SEARCH"].getboolean("add_timestamp")
    output_path = config["SEARCH"]["output_dir"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if bounds == "auto":
        config_VAE = configparser.ConfigParser()
        config_VAE.read(config["SEARCH"]["model_path"].replace("clf.pkl", "config.ini"))
        epoch = (
            config_VAE["RUN"]["model_path"].split("/")[-1].split("_")[-1].split(".")[0]
        )
        bounds_path = config_VAE["RUN"]["model_path"].replace(
            config_VAE["RUN"]["model_path"].split("/")[-1], f"latent_bounds_{epoch}.csv"
        )
    else:
        bounds_path = None

    # create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dirname = "latent_vectors_" + timestamp if add_timestamp else "latent_vectors"
    os.mkdir(output_path) if not os.path.isdir(output_path) else None
    (
        os.mkdir(f"{output_path}/{dirname}")
        if not os.path.isdir(f"{output_path}/{dirname}")
        else None
    )

    # determine number of workers
    if n_workers == -1 or n_workers > mp.cpu_count():
        n_workers = mp.cpu_count()
    if n_workers > n_samples:
        n_workers = n_samples

    # determine chunk sizes
    chunks = distribute_jobs(n_samples, n_workers)
    jobs = [(chunk, config, bounds_path) for chunk in chunks]

    # run the search
    print(f"Starting search with {n_workers} workers") if verbosity > 0 else None
    with mp.Pool(n_workers) as pool:
        results = pool.map(bayesian_search, jobs)
        pool.close()
        pool.join()

    # initialize model distance calculator
    sc_avg = SCAvgMeasure(clf_path=model_path)

    results = pd.concat(results)
    results.reset_index(drop=True, inplace=True)
    results["distance_to_model"] = [sc_avg(row[:-2]) for row in results.values]

    # save the results
    with open(f"{output_path}/{dirname}/latent_vectors.csv", "w") as f:
        results.to_csv(f, index=False)
    print(f"Results saved to: {output_path}/{dirname}") if verbosity > 0 else None
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
            f'mean score: {round(results["score"].mean(), 2)}'
            if len(results) > 0
            else round(results["score"].values[0], 2),
            f'sigma score: {round(results["score"].std(), 2)}'
            if len(results) > 0
            else "",
            f'mean norm: {round(results["norm"].mean(), 2)}'
            if len(results) > 0
            else round(results["norm"].values[0], 2),
            f'sigma norm: {round(results["norm"].std(), 2)}'
            if len(results) > 0
            else "",
        ]
        text = "\n".join(text)
        f.write(text)
