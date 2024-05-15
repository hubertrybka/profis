# PROFIS: Design of target-focused libraries by probing continuous fingerprint space with recurrent neural networks
Hubert Rybka, Mateusz Iwan, Anton Siomchen, Tomasz Danel, Sabina Podlewska

## Table of contents

<p align="center">
   <img src="https://github.com/hubertrybka/profis/blob/main/figures/TOC.png" width="410">
</p>

   * [General info](#general-info)
   * [Setup](#setup)
   * [Usage](#usage)


## General info

PROFIS is a generative model which allows for the design of target-focused compound libraries by probing continuous 
fingerprint space with RNNs. PROFIS is a VAE that encodes molecular FPs and decodes molecule structures 
in a sequential notation so that the resulting compounds match the initial FP description. In the task of generating 
potential novel ligands, PROFIS employs a Bayesian search algorithm to explore the space of embedded molecular fingerprints 
and identify subspaces that correspond to the known binders. Because many FPs do not determine the full chemical structure, 
our method can generate a diverse set of molecules that match the FP description.

## Setup

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) following the instructions for your operating
   system.
2. Clone the repository
3. Install environment from the YAML file: `conda env create -n env -f environment.yml`

## Basic usage

### Activate the environment:

      conda activate env

### Download the pre-trained models:

The trained models (and the dataset of D2 receptor ligands we used in our paper) are available to be downloaded from
our [dropbox](https://www.dropbox.com/scl/fi/n5v2v8e8z63ca3i6byshk/datasets.zip?rlkey=csa1epu0mcuz2fvnevtw8jscl&dl=0) 
or by launching:

      get_data.sh

**For successful execution of the script unzip must be installed**

The script will download the datasets and models to the `data` and `models` directories respectively. The trained
models  are essential for the proper functioning of the program.
If you want to train the models yourself, please refer to the [Advanced usage](#advanced-usage) section.

### Create a directory for you to work in:

      mkdir my_results

### Prepare the dataset:

In order to retrain the latent classifier, you have to provide an appropriate dataset. Put the data into
pandas.DataFrame object. The dataframe must contain the following columns:

* 'smiles' - SMILES strings of known ligands in canonical format.

* 'fps' - Klekota&Roth or Morgan (radius=2, nBits=2048) fingerprints of the ligands.  
  The fingerprints have to be saved as ordinary python lists, in **dense format** (a list of ints designating the indices
  of **active bits** in the original fingerprint).
  For a Python function to convert sparse molecular fingerprints into dense format, see src.utils.finger.sparse2dense.

* 'activity' - Activity class (True, False). By default, we define active compounds as those having
  Ki value <= 100nM and inactive as those of Ki > 100nM.

Save dataframe to .parquet file in the previously created directory:

#

```
import pandas as pd
df = pd.DataFrame(columns=['smiles', 'fps', 'activity'])

# ... load data into the dataframe

df.to_parquet('my_results/my_dataset.parquet', index=False)
```

### Train the SVC activity predictor:

In config_files/SVC_config.ini, provide the path to the dataset file (data_path) and the path to the trained RNN model.
The model weights are available in the `models` directory, if previously downloaded with `./get_data`. 
The path to the RNN weights should look like this:

    models/SMILES_KRFP/epoch_150.pt

Please provide the name of the previously created directory and name for the model in appropriate rubrics. 
Other parameters can be set according to needs.

Now, you can train the SVC activity predictor by running:
    
        python train_clf.py

For more info on the SVC classifier, please refer
to [scikit-learn SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

Now a file with the trained model should be saved in 'my_results' directory. Locate the directory,
and save the path to a model.pkl file created by the training script inside.

It should look like this:

    my_results/SVC/model.pkl

### Perform bayesian search on the latent space

The trained SVC activity predictor can be used to perform bayesian search on the latent space in order to identify latent
representations of potential novel ligands. Be sure to provide the path to the saved SVC model (.pkl) file and the 
desired number of samples (structures) to be generated in `config_files/search_config.ini`. 
Other parameters can be set according to needs.

To perform bayesian search on the latent space, use the following command:

    python bayesian_search.py

For more info about the bayesian optimization process and the choice of non-default parameters refer to
[bayesian-optimization README](https://github.com/bayesian-optimization/BayesianOptimization).

In the directory of the saved SVC model, a new 'latent_vectors' directory will be created, containing the following 
files:

* latent_vectors.csv - latent vectors identified by the search
* info.txt - information about the search process

### Generate compound libraries from the latent vectors

The generated compounds are filtered according to criteria that can be modified in `config_files/pred_config.ini`.

In order to generate a library, run

      python predict.py

Other parameters and filter criteria can be set according to needs.

As a result, in my_dir/mySVC, a new directory latent_vectors{timestamp} will be created. It contains the
following files:

* predictions.csv, a file containing SMILES of the generated compounds, as well as some calculated molecular properties
  (QED, MW, logP, ring info, RO5 info etc.)
* imgs directory, in which .png files depicting the structures of the generated compounds are located
* config.ini, a copy of the configuration file used for prediction (incl. filter criteria)

## Advanced usage

### Train the RNN decoder

This step can normally be omitted, as it is possible to use our pre-trained models. Model weights, as well as datasets
used for training, are available
on [dropbox](https://www.dropbox.com/scl/fi/e4xfi71gr2ih612ud8wai/models.zip?rlkey=8jrut4dexkmqj8egjcsphhdmu&dl=0) and can be
batch downloaded using `get_data.sh`.

If you intend to train the RNN, use the following command:

    python train_RNN.py

Be sure to edit the config file in advance (config_files/RNN_config.ini) to set the desired parameters.
In particular, you should provide a path to the RNN dataset file. This will be `data/RNN_dataset_KRFP.parquet`
or `data/RNN_dataset_ECFP.parquet` provided you used the `get_data.sh` script. Please adjust fp_len parameter 
according to the length of the input fingerprint.

Model weights and training progress will be saved to models/model_name catalog.



