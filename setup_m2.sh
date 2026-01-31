#!/bin/bash
set -e

# 1. Name the environment
ENV_NAME="drug_discovery_gcn"

echo "========================================"
echo "Creating Environment: $ENV_NAME on Apple Silicon"
echo "========================================"

# 2. Create the environment with Python 3.10
# We DO NOT install packages yet to keep it clean.
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists. Skipping create."
else
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# 3. Activate the environment
# This trick allows conda activate to work inside a script
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# 4. Install RDKit via Conda (CRITICAL STEP FOR M2)
# We use the conda-forge channel which is optimized for ARM64
echo "Installing RDKit from conda-forge..."
conda install -c conda-forge rdkit -y

# 5. Install PyTorch and Tools via Pip
# This pulls the standard MacOS version with MPS support automatically
echo "Installing PyTorch, GNN tools, and Data Science libraries..."
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install torch-geometric pandas networkx scikit-learn matplotlib

echo "========================================"
echo "Setup Complete!"
echo "Type: 'conda activate $ENV_NAME' to start."
echo "========================================"
