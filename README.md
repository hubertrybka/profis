# PROFIS: Design of target-focused libraries by probing continuous fingerprint space with recurrent neural networks
Hubert Rybka, Mateusz Iwan, Anton Siomchen, Tomasz Danel, Sabina Podlewska

## Table of contents

<center>
   <img src="https://github.com/hubertrybka/profis/blob/main/figures/TOC.png" width="410">
   
   * [General info](#general-info)
   * [Setup](#setup)
   * [Usage](#usage)
</center>

## General info

PROFIS is a generative model which allows for the design of target-focused compound libraries by probing continuous 
fingerprint space with RNNs. PROFIS is a VAE that encodes molecular FPs and decodes molecule structures 
in a sequential notation so that the resulting compounds match the initial FP description. In the task of generating 
potential novel ligands, PROFIS employs a Bayesian search algorithm to explore the space of embedded molecular fingerprints 
and identify subspaces that correspond to the known binders. Because many FPs do not determine the full chemical structure, 
our method can generate a diverse set of molecules that match the FP description.

<img src="https://github.com/hubertrybka/profis/blob/main/figures/architecture.png" width="820">

## Setup

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) following the instructions for your operating
   system.
2. Clone the repository: `git clone'
3. Install environment from the YAML file: `conda env create -n env -f environment.yml`

## Usage

### Activate the environment:

      conda activate env

### Prepare the dataset:

Please note that the dataset of D2 receptor ligands we used in our paper is available to be downloaded from
our [dropbox](https://www.dropbox.com/scl/fi/n5v2v8e8z63ca3i6byshk/datasets.zip?rlkey=csa1epu0mcuz2fvnevtw8jscl&dl=0) or by
launching:

      get_datasets.sh

**For successful execution of the script unzip must be installed**

In order to retrain the latent classifier, you have to provide an appropriate dataset. Put the data into
pandas.DataFrame object. The dataframe must contain the following columns:

* 'smiles' - SMILES strings of known ligands in canonical format.

* 'fps' - Klekota&Roth or Morgan (radius=2, nBits=2048) fingerprints of the ligands.  
  The fingerprints have to be saved as ordinary python lists, in **dense format** (a list of ints designating the indices
  of **active bits** in the original fingerprint).
  For Python script to convert sparse molecular fingerprints into dense format, see src.utils.finger.sparse2dense.

* 'activity' - Activity class (True, False). By default, we define active compounds as those having
  Ki value <= 100nM and inactive as those of Ki > 100nM.

Save dataframe to .parquet file:

```
import pandas as pd
df = pd.DataFrame(columns=['smiles', 'fps', 'activity'])

# ... load data into the dataframe
name = '5ht7_ECFP' # example name for the dataset

df.to_parquet(f'data/activity_data/{name}.parquet', index=False)
```

### Train the RNN decoder

(Advanced) This step can be omitted as it is advised to use our pre-trained models. Model weights, as well as datasets
used for training, are available
on [dropbox](https://www.dropbox.com/scl/fi/e4xfi71gr2ih612ud8wai/models.zip?rlkey=8jrut4dexkmqj8egjcsphhdmu&dl=0) and can be
batch downloaded using `get_datasets.sh` No more steps are needed to use the pre-trained model.

If you intend to train the RNN, use the following command:

    python train_RNN.py

**IMPORTANT**  
Be sure to edit the config file in advance (config_files/RNN_config.ini) to set the desired parameters.
In particular, you should provide a path to the dataset file. This will be `data/RNN_dataset_KRFP.parquet`
or `data/RNN_dataset_ECFP.parquet`
provided you used the `get_datasets.sh` script. Please adjust fp_len parameter according to the length of the input.
fingerprint (KRFP: 4860, ECFP4: 2048).

Model weights and training progress will be saved to models/model_name catalog.

### Train the SVC activity predictor

Use the following command:

    python train_clf.py

**IMPORTANT**  
Be sure to provide the path to the dataset file (data_path) in the config file located
here: `config_files/SVC_config.ini`.  
Provide the path to the weights of RNN decoder (model_path). Here you can use our pre-trained KRFP and ECFP-based models.
This should be `models/SMILES_KRFP/epoch_150.pt` or `models/SMILES_ECFP/epoch_150.pt` provided you used `get_datasets.sh`
script.

Other parameters can be set according to needs.

For more info on the SVC classifier, please refer
to [scikit-learn SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

Now a file with the trained model should be saved in the 'models' directory. Locate the directory,
and save the path to a model.pkl file created by the training script inside.

It should look like this:

    models/model_name/model.pkl

### Perform bayesian search on the latent space

The trained activity predictor can be used to perform bayesian search on the latent space in order to identify latent
representations of potential novel ligands.
To perform bayesian search on the latent space, use the following command:

    python bayesian_search.py

**IMPORTANT**  
Be sure to provide the path to the saved SVC model (.pkl) file, and the desired number of samples to be generated
in `config_files/search_config.ini`  
Other parameters can be set according to needs:

For more info about the bayesian optimization process and the choice of non-default parameters refer to
[bayesian-optimization README](https://github.com/bayesian-optimization/BayesianOptimization).

Results of the search will be saved in 'outputs' directory.

Directory 'SVC_{timestamp}' will be created in outputs directory, containing the following files:

* latent_vectors.csv - latent vectors identified by the search
* info.txt - information about the search process

### Generate compound libraries from the latent vectors

The generated compounds are filtered according to criteria that can be modified in `config_files/pred_config.ini`.

In order to generate a library, run

      python predict.py

**IMPORTANT**  
Be sure to provide the path to `latent_vectors.csv` (latent encodings identified by bayesian search algorithm), as well as
the RNN model weights file in `config_files/pred_config.ini`

Other parameters and filter criteria can be set according to needs.

As a result, in results/SVC_{timestamp} dir, a new directory preds_{new_timestamp} will be created. This contains the
following files:

* predictions.csv, a file containing SMILES of the generated compounds, as well as some calculated molecular properties
  (QED, MW, logP, ring info, RO5 info etc.)
* imgs directory, in which .png files depicting the structures of the generated compounds are located
* config.ini, a copy of the configuration file used for prediction (incl. filter criteria)

