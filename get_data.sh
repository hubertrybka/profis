#!/bin/bash
mkdir -p data
cd data || exit
wget -O datasets.zip 'https://www.dropbox.com/scl/fi/n5v2v8e8z63ca3i6byshk/datasets.zip?rlkey=csa1epu0mcuz2fvnevtw8jscl&dl=1'
unzip datasets.zip
cd ../
mkdir -p models
cd models || exit
wget -O models.zip 'https://www.dropbox.com/scl/fi/5cm7rf49qah1kaee2melp/models.zip?rlkey=mblcvykn0dmn8v77qxxjgn78z&st=l6nis5su&dl=1'
unzip models.zip
