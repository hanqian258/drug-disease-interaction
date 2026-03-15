"""
06_inference.py — Drug → Alzheimer's Disease Correlation Predictor
Outputs a probability score (0.0 – 1.0) for a given drug name.

Usage:
    python3 02_Code/06_inference.py "Donepezil"
    python3 02_Code/06_inference.py "Memantine"

Output format (stdout):
    Drug: Donepezil
    Probability of interaction: 0.9984
    Result: High Potential for therapeutic effect.

The script:
    1. Loads the trained HeteroGNN + LinkPredictor from 01_Cleaned_Data/
    2. Loads the full heterogeneous graph (expanded_graph.pt)
    3. Looks up the drug node by name
    4. Runs a single forward pass to get the drug embedding
    5. Computes link probability against the Alzheimer's disease node
    6. Applies sigmoid → outputs a probability in [0, 1]

NOTE ON PROBABILITY DISTRIBUTION:
    The raw model output is a single logit per (drug, disease) pair.
    We apply torch.sigmoid() to convert it to a probability in [0, 1]:
      - Score > 0.70  → High Potential
      - Score 0.40–0.70 → Moderate Potential
      - Score < 0.40  → Low / No Predicted Correlation
"""

import sys
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, GraphConv
import torch.nn as nn

# ---------------------------------------------------------------------------
# Replicate model architecture (must match 05_train_gcn.py exactly)
# ---------------------------------------------------------------------------
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, node_types, edge_types, in_channels_dict):
        super().__init__()
        self.lins = nn.ModuleDict()
        for node_type in node_types:
            self.lins[node_type] = nn.Linear(in_channels_dict[node_type], hidden_channels)
        self.dropout = nn.Dropout(0.3)
        self.convs = nn.ModuleList()
        for _ in range(3):
            conv = HeteroConv({
                edge_type: GraphConv((-1, -1), hidden_channels)
                for edge_type in edge_types
            }, aggr='sum')
            self.convs.append(conv)
        self.final_lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        x_dict = {
            node_type: self.dropout(F.relu(self.lins[node_type](x)))
            for node_type, x in x_dict.items()
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
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
        return x.view(-1)


# ---------------------------------------------------------------------------
# Inference logic
# ---------------------------------------------------------------------------
SCORE_HIGH     = 0.70
SCORE_MODERATE = 0.40

MODEL_DIR  = "01_Cleaned_Data"
GRAPH_PATH = os.path.join(MODEL_DIR, "expanded_graph.pt")
GNN_PATH   = os.path.join(MODEL_DIR, "gnn_model.pt")
PRED_PATH  = os.path.join(MODEL_DIR, "predictor.pt")


def load_models(data):
    """Load trained GNN and LinkPredictor, return (model, predictor)."""
    hidden_channels = 128
    in_channels_dict = {nt: data[nt].x.size(-1) for nt in data.node_types}

    model = HeteroGNN(
        hidden_channels, hidden_channels,
        data.node_types, data.edge_types, in_channels_dict
    )
    predictor = LinkPredictor(hidden_channels, hidden_channels)

    model.load_state_dict(torch.load(GNN_PATH, weights_only=True, map_location="cpu"))
    predictor.load_state_dict(torch.load(PRED_PATH, weights_only=True, map_location="cpu"))
    model.eval()
    predictor.eval()
    return model, predictor


def find_drug_index(data, drug_name: str):
    """Look up a drug node index by name. Returns int or None."""
    drug_names = data["drug"].node_names  # list of strings set during graph construction
    drug_lower = [n.lower() for n in drug_names]
    target = drug_name.lower().strip()
    if target in drug_lower:
        return drug_lower.index(target)
    # Fuzzy fallback: check if query is a substring of any node name
    for i, n in enumerate(drug_lower):
        if target in n or n in target:
            return i
    return None


def find_disease_index(data, disease_name: str = "alzheimer"):
    """Look up the Alzheimer's disease node index."""
    if not hasattr(data["disease"], "node_names"):
        # If only one disease node exists, return index 0
        return 0
    disease_lower = [n.lower() for n in data["disease"].node_names]
    for i, n in enumerate(disease_lower):
        if disease_name in n:
            return i
    return 0  # default to first disease node


def predict(drug_name: str) -> float | None:
    """
    Returns probability in [0, 1] that `drug_name` treats Alzheimer's disease.
    Returns None if the drug is not found in the graph.
    """
    # Load graph
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"Graph not found at {GRAPH_PATH}. Run 04_expand_graph.py first.")
    data = torch.load(GRAPH_PATH, weights_only=False)

    # Find node indices
    drug_idx = find_drug_index(data, drug_name)
    if drug_idx is None:
        return None
    disease_idx = find_disease_index(data)

    # Load models
    model, predictor = load_models(data)

    with torch.no_grad():
        # Full graph forward pass to get embeddings
        x_dict = model(data.x_dict, data.edge_index_dict,
                       getattr(data, "edge_attr_dict", None))

        # Construct a single edge: (drug_idx) → (disease_idx)
        edge_label_index = torch.tensor([[drug_idx], [disease_idx]], dtype=torch.long)

        # Raw logit → sigmoid → probability
        logit = predictor(x_dict["drug"], x_dict["disease"], edge_label_index)
        prob  = torch.sigmoid(logit).item()

    return prob


def interpret(prob: float) -> str:
    if prob >= SCORE_HIGH:
        return "High Potential for therapeutic effect."
    elif prob >= SCORE_MODERATE:
        return "Moderate Potential — warrants further investigation."
    else:
        return "Low / No Predicted Correlation with Alzheimer's disease."


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 02_Code/06_inference.py \"DrugName\"")
        sys.exit(1)

    drug_name = sys.argv[1]

    # Check model files exist
    for path in [GNN_PATH, PRED_PATH]:
        if not os.path.exists(path):
            print(f"[ERROR] Model file not found: {path}")
            print("Run 02_Code/05_train_gcn.py first to generate the model checkpoints.")
            sys.exit(1)

    try:
        prob = predict(drug_name)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if prob is None:
        print(f"Drug: {drug_name}")
        print(f"Result: Drug not found in the model's library.")
        print(f"Probability of interaction: N/A")
        sys.exit(0)

    print(f"Drug: {drug_name}")
    print(f"Probability of interaction: {prob:.4f}")
    print(f"Result: {interpret(prob)}")

    # Also print a simple probability distribution across threshold buckets
    # (useful for understanding score context)
    print()
    print("Score interpretation:")
    print(f"  >= {SCORE_HIGH}  : High Potential     {'<-- YOUR SCORE' if prob >= SCORE_HIGH else ''}")
    print(f"  {SCORE_MODERATE}–{SCORE_HIGH} : Moderate Potential {'<-- YOUR SCORE' if SCORE_MODERATE <= prob < SCORE_HIGH else ''}")
    print(f"  <  {SCORE_MODERATE}  : Low Correlation    {'<-- YOUR SCORE' if prob < SCORE_MODERATE else ''}")


if __name__ == "__main__":
    main()
