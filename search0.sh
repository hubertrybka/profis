#!/bin/bash
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
python bayesian_search.py -c config_files/search_config0.ini