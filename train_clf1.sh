#!/bin/bash
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
python train_clf.py -c config_files/train_clf1.ini