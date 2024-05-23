#!/bin/bash
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
python train_RNN.py -c config_files/train_RNN0.ini