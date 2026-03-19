#!/bin/bash
set -e
ENV_NAME="drug_discovery_gcn"

# Create Environment
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME exists."
else
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# Activate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Install RDKit (Bioinformatics)
conda install -c conda-forge rdkit -y

# Install PyTorch (AI) - Native Mac Version
pip install torch torchvision torchaudio
pip install torch-geometric pandas networkx scikit-learn matplotlib pubchempy

echo "Setup Complete! Run: conda activate $ENV_NAME"
