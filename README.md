# PROFIS: Design of target-focused libraries by probing continuous fingerprint space with recurrent neural networks
Hubert Rybka, Tomasz Danel, Sabina Podlewska

## Table of contents

<p align="center">
   <img src="https://github.com/hubertrybka/profis/blob/main/figures/TOC.png" width="410">
</p>

   * [General info](#general-info)
   * [Setup](#setup)
   * [Basic usage](#basic-usage)
   * [Advanced usage](#advanced-usage)
   * [Dependencies](#dependencies)


## General info

PROFIS is a generative model that allows for the design of target-focused compound libraries by **pro**bing continuous 
**fi**ngerprint **s**pace with RNNs in a reverse QSAR-type task. PROFIS is a VAE that encodes molecular FPs and decodes molecule structures 
in a sequential notation so that the resulting compounds match the initial FP description. In the process of generating 
potential novel ligands, PROFIS employs a Bayesian search algorithm to explore the space of embedded molecular fingerprints 
and identify subspaces that correspond to the known binders. Because many FPs do not determine the full chemical structure, 
our method can generate a diverse set of molecules that match the FP description.

<p align="center">
   <img src="https://github.com/hubertrybka/profis/blob/main/figures/architecture.png" width="800">
</p>

## Setup

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) following the instructions for your operating
   system.
2. Clone the repository: `git clone https://github.com/hubertrybka/profis.git`
3. Navigate to the profis directory: `cd profis`
4. Install environment from the YAML file: `conda env create -n env -f environment.yml`

## Basic usage

### Activate the environment:

      conda activate env

### Download the pre-trained models:

The trained models (and the dataset of D2 receptor ligands we used in our paper) are available to be downloaded from
our [dropbox](https://www.dropbox.com/scl/fo/z0oet8c9lfgl9wy8cey5e/APNI3cI604jPteati6Ng3SE?rlkey=tecj7pei2c31vw4xxtv64ovie&st=eouv45oa&dl=0)
or by launching:

      get_data.sh

**For successful execution of the script unzip must be installed**

The script will download the datasets and models to the `data` and `models` directories respectively.
If you want to train the RNN models yourself, please refer to the [Advanced usage](#advanced-usage) section.

### Create a directory for you to work in:

      mkdir my_results

### Prepare the dataset:

In order to retrain the latent classifier, you have to provide an appropriate dataset. Put the data into
pandas.DataFrame object. The dataframe must contain the following columns:

* 'smiles' - SMILES strings of known ligands in canonical format.

* 'fps' (optional) - Molecular fingerprints of the ligands. **This column is optional, as later you will be able to generate those from SMILES**.
  The fingerprints have to be saved as ordinary python lists, in dense format (a list of ints designating the indices
  of active bits in the original fingerprint).

* 'activity' - Activity class (True, False). By default, we define active compounds as those having
  Ki value <= 100nM and inactive as those of Ki > 100nM.

Save dataframe to .parquet file in the previously created directory:

```
import pandas as pd
df = pd.DataFrame(columns=['smiles', 'fps', 'activity'])

# ... load data into the dataframe

df.to_parquet('my_results/my_dataset.parquet', index=False)
```

You then have to preprocess the data using:

      python prepare_dataset.py

This script takes the following arguments:
```
--data_path PATH      Path to the dataset in .parquet format
--gen_ecfp            Flag to generate ECFP fingerprints
--gen_krfp            Flag to Generate KRFP fingerprints
--to_dense            Flag to only convert sparse fingerprints in 'fps' column to dense format
```

### Train the latent space QSAR model:

In config_files/MLP_config.ini, provide the path to the preprocessed dataset file (data_path) and the path to the pre-trained RNN model weights.
The model weights are available in the `models` directory if previously downloaded with `./get_data`. 
The path to the RNN weights should look like this:

      models/SMILES_KRFP/model.pt

Please provide the name of the previously created directory and the name of the model in the appropriate rubrics. 
Other parameters can be set according to needs. Config files for SVR, RF, and XGB models are also prepared for the user.

Now, you can train the latent space QSAR classifier by running:
    
      python train_clf.py

For more info on the classifier, please refer
to [scikit-learn documentation](https://scikit-learn.org/stable/api/index.html).

Now a file with the trained model should be saved in 'my_results' directory. Locate the directory,
and save the path to a model.pkl file created by the training script inside.

It should look like this:

      my_results/my_MLP/model.pkl

### Perform bayesian search on the latent space

The trained QSAR classifier can be used to perform bayesian search on the latent space in order to identify the
representations of potential novel ligands. Be sure to provide the path to the trained model pickle (.pkl) and the 
desired number of samples (structures) to be generated in `config_files/search_config.ini`. 
Other parameters can be set according to needs.

To perform bayesian search on the latent space, use the following command:

      python bayesian_search.py

For more info about the bayesian optimization process and the choice of non-default parameters refer to
[bayesian-optimization README](https://github.com/bayesian-optimization/BayesianOptimization).

In the directory of trained QSAR model, a new 'latent_vectors' directory will be created, containing the following 
files:

* latent_vectors.csv - latent vectors identified by the search
* config.ini - a copy of the configuration file used for the search
* info.txt - additional information about the search process

### Generate compound libraries from the latent vectors

The generated compounds are filtered according to criteria that can be modified in `config_files/pred_config.ini`.

To generate a library of ligands, run

      python predict.py

Other parameters and filter criteria can be set according to needs.

As a result, in my_results/my_MLP, a new directory latent_vectors{timestamp} will be created. It contains the
following files:

* predictions.csv, a file containing SMILES of the generated compounds, as well as some calculated molecular properties
  (QED, MW, logP, ring info, RO5 info etc.)
* imgs directory, in which .png files depicting the structures of the generated compounds are located
* config.ini, a copy of the configuration file used for prediction (incl. filter criteria)

<p align="center">
   <img src="https://github.com/hubertrybka/profis/blob/main/figures/scaffold_hopping.png" width="800">
</p>

## Advanced usage

### Train the RNN decoder

This step can normally be omitted, as it is possible to use our pre-trained models. Model weights, as well as datasets
used for training, are available
on [dropbox](https://www.dropbox.com/scl/fo/z0oet8c9lfgl9wy8cey5e/APNI3cI604jPteati6Ng3SE?rlkey=tecj7pei2c31vw4xxtv64ovie&st=eouv45oa&dl=0) and can be
batch downloaded using `get_data.sh`.

If you intend to train the RNN, use the following command:

      python train_RNN.py

Be sure to edit the config file in advance (config_files/RNN_config.ini) to set the desired parameters.
In particular, you should provide a path to the RNN dataset file. This will be `data/RNN_dataset_KRFP.parquet`
or `data/RNN_dataset_ECFP.parquet` provided you used the `get_data.sh` script. Please adjust fp_len parameter 
according to the length of the input fingerprint.

Model weights and training progress will be saved to models/model_name catalog.

## Dependencies

* python 3.8
* bayesian_optimization 1.4.3
* deepsmiles 1.0.1
* numpy 1.24.3
* pandas 2.2.2
* rdkit 2023.9.6
* scikit_learn 1.3.1
* scipy 1.13.0
* selfies 2.1.1
* torch 2.0.1
* xgboost 2.0.3
* unzip 6.0 (automates the downloading of datasets and weights)

