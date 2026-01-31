#!/bin/bash
set -e

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in your PATH."
    echo "Please install Miniforge to proceed."
    exit 1
fi

ENV_NAME="drug_discovery_gcn"

echo "Creating Conda environment '$ENV_NAME' with Python 3.10..."
# Create environment, -y to confirm.
# We don't install packages here to ensure we use pip for the requirements.txt as implied by the request structure,
# although mixing conda/pip is common.
conda create -n "$ENV_NAME" python=3.10 -y

# Activate environment
# To activate in a script, we need to source the conda.sh script
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "Warning: Could not find conda.sh to source. Activation might fail."
fi

echo "Activating environment..."
conda activate "$ENV_NAME"

echo "Installing dependencies from requirements.txt..."
# Ensure pip is up to date
pip install --upgrade pip
pip install -r requirements.txt

echo "========================================"
echo "Setup complete!"
echo "To use the environment, run:"
echo "conda activate $ENV_NAME"
echo "========================================"
