#!/bin/bash
mkdir -p data
cd data || exit
wget -O datasets.zip 'https://www.dropbox.com/scl/fi/n5v2v8e8z63ca3i6byshk/datasets.zip?rlkey=csa1epu0mcuz2fvnevtw8jscl&st=k5kzdecs&dl=1'
unzip datasets.zip
cd ../
mkdir -p models
cd models || exit
wget -O models.zip 'https://www.dropbox.com/scl/fi/zw6bu6461o9cq5c6su1q5/models.zip?rlkey=cq308bio1957jlzi437rdhuev&st=izm1gpj1&dl=1'
unzip models.zip
