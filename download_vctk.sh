#!/bin/bash
set -e

VCTK_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
TARGET_DIR=${1:-"./data"}
ZIP_NAME="VCTK-Corpus-0.92.zip"

mkdir -p $TARGET_DIR
cd $TARGET_DIR

if [ ! -f $ZIP_NAME ]; then
  echo "Downloading VCTK corpus..."
  wget $VCTK_URL -O $ZIP_NAME
else
  echo "VCTK zip already exists. Skipping download."
fi

if [ ! -d "VCTK-Corpus-0.92" ]; then
  echo "Extracting VCTK corpus..."
  unzip -q $ZIP_NAME
else
  echo "VCTK corpus already extracted."
fi

echo "VCTK dataset ready at: $TARGET_DIR/VCTK-Corpus-0.92" 