#!/bin/bash
mkdir -p data
cd data || exit
wget -O datasets.zip 'https://www.dropbox.com/scl/fi/n5v2v8e8z63ca3i6byshk/datasets.zip?rlkey=csa1epu0mcuz2fvnevtw8jscl&dl=1'
unzip datasets.zip
cd ../
mkdir -p models
cd models || exit
wget -O models.zip 'https://www.dropbox.com/scl/fi/e4xfi71gr2ih612ud8wai/models.zip?rlkey=8jrut4dexkmqj8egjcsphhdmu&dl=1'
unzip models.zip
