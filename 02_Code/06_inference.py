import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import os
import pandas as pd
from featurizer import DrugFeaturizer
import ast

# Reuse the model architectures from training
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, node_types, edge_types, in_channels_dict):
        super().__init__()
        self.lins = nn.ModuleDict()
        for node_type in node_types:
            self.lins[node_type] = nn.Linear(in_channels_dict[node_type], hidden_channels)

        self.dropout = nn.Dropout(0.3)
        self.convs = nn.ModuleList()
        from torch_geometric.nn import SAGEConv, HeteroConv
        for _ in range(3):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in edge_types
            }, aggr='sum')
            self.convs.append(conv)
        self.final_lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.dropout(F.relu(self.lins[node_type](x)))
            for node_type, x in x_dict.items()
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return {key: self.final_lin(x) for key, x in x_dict.items()}

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, x_drug, x_disease, edge_label_index):
        nodes_s = x_drug[edge_label_index[0]]
        nodes_t = x_disease[edge_label_index[1]]
        x = torch.cat([nodes_s, nodes_t], dim=-1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return torch.sigmoid(x).view(-1)

def run_inference(drug_input):
    """
    drug_input: can be a drug name from the library or a SMILES string
    """
    # 1. Load data and models
    if not os.path.exists('01_Cleaned_Data/expanded_graph.pt'):
        print("Error: Models not trained. Run training pipeline first.")
        return

    data = torch.load('01_Cleaned_Data/expanded_graph.pt', weights_only=False)
    maps = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)
    d_map = maps['d_map']
    drug_names = maps['drug_names']

    in_channels_dict = {node_type: data[node_type].x.size(-1) for node_type in data.node_types}
    hidden_channels = 128

    model = HeteroGNN(hidden_channels, hidden_channels, data.node_types, data.edge_types, in_channels_dict)
    model.load_state_dict(torch.load('01_Cleaned_Data/gnn_model.pt', weights_only=True))
    model.eval()

    predictor = LinkPredictor(hidden_channels, hidden_channels)
    predictor.load_state_dict(torch.load('01_Cleaned_Data/predictor.pt', weights_only=True))
    predictor.eval()

    # 2. Get drug embedding
    drug_idx = None
    if drug_input in d_map:
        drug_idx = d_map[drug_input]
        print(f"Using drug from library: {drug_input} (Index: {drug_idx})")
    else:
        # Try to featurize SMILES
        print(f"Input '{drug_input}' not in library. Attempting to featurize as SMILES...")
        feat = DrugFeaturizer()
        g = feat.smiles_to_graph(drug_input)
        if g:
            # For now, if it's a new drug, we'd ideally re-run the GNN or use an inductive approach.
            # Simplified for demo: if it's new, we use the mean of atom features
            # but our model was trained on the specific 'drug' nodes in the graph.
            # To predict on a NEW drug, we'd need to add it to the graph.
            print("New drug prediction requires re-running graph embedding (Inductive mode).")
            # For the demo, let's just use an existing drug if name matches loosely
            matches = [name for name in drug_names if drug_input.lower() in name.lower()]
            if matches:
                drug_idx = d_map[matches[0]]
                print(f"Loosely matched to library drug: {matches[0]}")
            else:
                print("Could not find or featurize drug. Try a name from the list:")
                print(drug_names[:10])
                return

    if drug_idx is None:
        return

    # 3. Predict
    with torch.no_grad():
        x_dict = model(data.x_dict, data.edge_index_dict)
        # We only have one disease node (Alzheimer's) at index 0
        disease_idx = 0
        edge_label_index = torch.tensor([[drug_idx], [disease_idx]], dtype=torch.long)

        prob = predictor(x_dict['drug'], x_dict['disease'], edge_label_index).item()

    print(f"\nPrediction for {drug_input} and Alzheimer's Disease:")
    print(f"Probability of interaction: {prob:.4f}")
    if prob > 0.5:
        print("Result: High Potential for therapeutic effect.")
    else:
        print("Result: Low Potential.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_inference(sys.argv[1])
    else:
        print("Usage: python3 02_Code/06_inference.py <drug_name_or_smiles>")
        print("Example: python3 02_Code/06_inference.py Tacrine")
