#!/bin/bash
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python train_SMILES2SMILES.py --batch_size 512 --lr 0.0002 --encoding_size 64 --run_name "smiles_to_smiles64"
