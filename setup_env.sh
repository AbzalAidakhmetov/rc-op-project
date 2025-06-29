#!/bin/bash
set -e

ENV_NAME=rcop
PYTHON_VERSION=3.10

# Install Miniconda if not present
if ! command -v conda &> /dev/null; then
  echo "Conda not found. Installing Miniconda..."
  wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  source $HOME/miniconda/etc/profile.d/conda.sh
fi

# Ensure conda is available
source $(conda info --base)/etc/profile.d/conda.sh

# Create environment if it doesn't exist
if ! conda info --envs | grep -q "^$ENV_NAME"; then
  conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

conda activate $ENV_NAME

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment '$ENV_NAME' ready. Activate with: conda activate $ENV_NAME" 