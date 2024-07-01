#!/bin/bash
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python bayesian_search.py -c config_files/search_config1.ini
python bayesian_search.py -c config_files/search_config2.ini
python bayesian_search.py -c config_files/search_config3.ini
python bayesian_search.py -c config_files/search_config4.ini
python bayesian_search.py -c config_files/search_config5.ini
python bayesian_search.py -c config_files/search_config6.ini
python bayesian_search.py -c config_files/search_config7.ini
python bayesian_search.py -c config_files/search_config8.ini