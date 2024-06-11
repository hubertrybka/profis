#!/bin/bash
mkdir -p data
cd data || exit
wget -O datasets.zip 'https://www.dropbox.com/scl/fi/n5v2v8e8z63ca3i6byshk/datasets.zip?rlkey=csa1epu0mcuz2fvnevtw8jscl&dl=1'
unzip datasets.zip
cd ../
mkdir -p models
cd models || exit
wget -O models.zip 'https://www.dropbox.com/scl/fi/ajxbnlzk4lcedj8c78fd5/models.zip?rlkey=p3vn4ao8gjup56ulbm46x3nt7&st=2bywko15&dl=1'
unzip models.zip
