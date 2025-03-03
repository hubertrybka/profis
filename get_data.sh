#!/bin/bash
mkdir -p data
cd data || exit
wget -O datasets.zip 'https://www.dropbox.com/scl/fi/qc7tov8dsa7hrkkj1d76e/models.zip?rlkey=fyod5iivftcrhy0xe6ukb8jvn&st=ha4qt8ul&dl=1'
unzip datasets.zip
