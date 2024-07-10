#!/bin/bash
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python train_clf.py -c config_files/SVC1_config.ini
python train_clf.py -c config_files/SVC1_deepsmiles_config.ini
python calc_bounds.py -m models/dropout_lessKLD_KRFP_SMILES/epoch_200.pt
python calc_bounds.py -m models/dropout_lessKLD_KRFP_DeepSMILES/epoch_200.pt
python bayesian_search.py -c config_files/search_config1.ini
python bayesian_search.py -c config_files/search_config2.ini
python bayesian_search.py -c config_files/search_config3.ini
python bayesian_search.py -c config_files/search_config4.ini