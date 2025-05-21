#!/bin/bash
mkdir -p data
cd data || exit
wget -O datasets.zip 'https://www.dropbox.com/scl/fi/gm3z8v5wxum16cmb84mhq/datasets.zip?rlkey=yf0cd6w378dgdotgndpudt9ep&st=it4xy9eh&dl=1'
unzip datasets.zip
wget -O models.zip 'https://www.dropbox.com/scl/fi/qc7tov8dsa7hrkkj1d76e/models.zip?rlkey=fyod5iivftcrhy0xe6ukb8jvn&st=eni8lcg4&dl=1'
unzip models.zip