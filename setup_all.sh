#!/bin/bash
set -e

# 1. Setup environment
bash setup_env.sh

# 2. Ensure conda env is activated in THIS shell and dependencies are present
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rcop
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download models
python download_models.py

# 4. Download dataset
echo "Downloading VCTK dataset..."
bash download_vctk.sh ./data

echo "All setup complete!" 