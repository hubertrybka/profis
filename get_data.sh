#!/bin/bash
mkdir -p data
cd data || exit
wget -O datasets.zip 'https://www.dropbox.com/scl/fi/n5v2v8e8z63ca3i6byshk/datasets.zip?rlkey=csa1epu0mcuz2fvnevtw8jscl&dl=1'
unzip datasets.zip
cd ../
mkdir -p models
cd models || exit
wget -O models.zip 'https://www.dropbox.com/scl/fi/vlen613o3phf6oivfwmco/models.zip?rlkey=17rrbssqy28xi0rjc1sgcbx69&st=vjq574o6&dl=1'
unzip models.zip
