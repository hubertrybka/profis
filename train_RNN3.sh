#!/bin/bash
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python train_RNN.py -c config_files/RNN_config3.ini
python train_clf.py -c config_files/SVC_ECFP_DeepSMILES_config.ini
python train_clf.py -c config_files/MLP_ECFP_DeepSMILES_config.ini
python train_clf.py -c config_files/RF_ECFP_DeepSMILES_config.ini
python train_clf.py -c config_files/XGB_ECFP_DeepSMILES_config.ini