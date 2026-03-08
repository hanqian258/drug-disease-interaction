# Drug Discovery GNN: Targeting Alzheimer's Disease

This project utilizes a **Heterogeneous Graph Neural Network (GNN)** to predict novel drug-disease interactions, with a specific focus on the **Tau and Amyloid-beta interactome** for Alzheimer's Disease (AD).

Developed for scientific reproducibility (ISEF standard), the pipeline integrates chemical informatics, protein-protein interaction (PPI) data, and deep learning on graphs.

## 🚀 Recent Updates
- **New Dataset Integration**: Transitioned to a high-density protein interaction network focused on Amyloid-beta and Tau protein pathways.
- **Improved Drug Featurization**: Integrated advanced numerical vector representations for drugs, enhancing the model's ability to learn chemical properties.
- **Link Prediction Architecture**: Refactored the GNN into a link prediction model, allowing it to predict the probability of *any* drug treating Alzheimer's, rather than simple classification.
- **Demo Mode**: Added an automated inference script and a Google Colab demo for easy model testing.

## 📊 Network Visualization
The model operates on a complex biological network connecting drugs, proteins, and diseases. Below is a visual representation of the core network:

![Network Visualization](network_visualization.png)

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/) (Recommended)

### Setup
Run the provided setup script to create the environment and install dependencies:
```bash
chmod +x setup_m2.sh
./setup_m2.sh
conda activate drug_discovery_gcn
```

Alternatively, install via pip:
```bash
pip install torch torch-geometric pandas rdkit scikit-learn matplotlib networkx
```

## 📂 Project Structure
- `00_Raw_Data/`: Original drug and protein datasets.
- `01_Cleaned_Data/`: Processed graph objects and model checkpoints.
- `02_Code/`:
  - `03_build_hetero_graph.py`: Constructs the heterogeneous graph.
  - `04_expand_graph.py`: Adds disease nodes and associations.
  - `05_train_gcn.py`: Trains the HeteroGNN for link prediction.
  - `06_inference.py`: **Main Tool** for predicting drug interactions.
  - `07_visualize_graph.py`: Generates the network visualization.
- `99_ISEF_Docs/`: Technical reports and result logs.

## 🔍 Accessing the Network & Predictions

### 1. Run Inference (Predict a Drug)
You can predict the therapeutic potential of any drug in our library for Alzheimer's:
```bash
python3 02_Code/06_inference.py "Donepezil"
```
*Example Output:*
> Probability of interaction: 0.9984
> Result: High Potential for therapeutic effect.

### 2. Interactive Demo
Try the model in your browser using our **Google Colab Notebook**:
[Link to Colab Demo](Drug_Discovery_GNN_Demo.ipynb) *(Note: Open this file in Google Colab)*

### 3. Training the Model
To re-train the model on new data:
```bash
python3 02_Code/03_build_hetero_graph.py
python3 02_Code/04_expand_graph.py
python3 02_Code/05_train_gcn.py
```

## 🧪 Scientific Approach
Our GNN model uses **HeteroConv** layers with **SAGEConv** operators to perform message passing across different edge types:
- `(drug, binds, protein)`
- `(protein, interacts_with, protein)`
- `(protein, associated_with, disease)`
- `(drug, treats, disease)`

By learning from known "Approved" drugs, the model identifies patterns in how drugs interact with the Tau/Amyloid-beta subnetwork to predict candidate treatments.
