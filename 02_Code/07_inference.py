import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, HeteroConv
import os
import pandas as pd
from featurizer import DrugFeaturizer
import ast
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# Reuse the model architectures from training
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, node_types, edge_types, in_channels_dict):
        super().__init__()
        self.lins = nn.ModuleDict()
        for node_type in node_types:
            self.lins[node_type] = nn.Linear(in_channels_dict[node_type], hidden_channels)

        self.dropout = nn.Dropout(0.3)
        self.convs = nn.ModuleList()
        from torch_geometric.nn import GraphConv, HeteroConv
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
        return torch.sigmoid(x).view(-1)

def calculate_drug_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    properties = {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP (Lipophilicity)": round(Descriptors.MolLogP(mol), 2),
        "QED (Drug-likeness)": round(Descriptors.qed(mol), 2),
        "H-Bond Donors": Descriptors.NumHDonors(mol),
        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
        "BBB Permeable (Likely)": "Yes" if Descriptors.MolLogP(mol) < 5 and Descriptors.MolWt(mol) < 450 else "Limited"
    }
    return properties

def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return torch.tensor(list(fp), dtype=torch.float).view(1, -1)

def run_inference(drug_input, target_protein=None):
    """
    drug_input: can be a drug name from the library or a SMILES string
    target_protein: optional specific protein target name
    """
    # 1. Load data and models
    if not os.path.exists('01_Cleaned_Data/expanded_graph.pt'):
        print("Error: Models not trained. Run training pipeline first.")
        return

    data = torch.load('01_Cleaned_Data/expanded_graph.pt', weights_only=False)
    maps = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)
    d_map = maps['d_map']
    p_map = maps['p_map']
    drug_names = maps['drug_names']
    all_proteins = maps['all_proteins']

    in_channels_dict = {node_type: data[node_type].x.size(-1) for node_type in data.node_types}
    hidden_channels = 128

    model = HeteroGNN(hidden_channels, hidden_channels, data.node_types, data.edge_types, in_channels_dict)
    model.load_state_dict(torch.load('01_Cleaned_Data/gnn_model.pt', weights_only=True))
    model.eval()

    predictor = LinkPredictor(hidden_channels, hidden_channels)
    predictor.load_state_dict(torch.load('01_Cleaned_Data/predictor.pt', weights_only=True))
    predictor.eval()

    # 2. Handle Drug Input (Existing or Virtual)
    drug_x = None
    is_new_drug = False
    active_smiles = None

    if drug_input in d_map:
        drug_idx = d_map[drug_input]
        active_smiles = pd.read_csv('00_Raw_Data/drugs_raw.csv').set_index('Drug Name/Treatment').loc[drug_input, 'Drug Structure']
        print(f"Using drug from library: {drug_input}")
    else:
        # Check if it's a known drug but with different casing
        matches = [name for name in drug_names if drug_input.lower() == name.lower()]
        if matches:
            drug_idx = d_map[matches[0]]
            active_smiles = pd.read_csv('00_Raw_Data/drugs_raw.csv').set_index('Drug Name/Treatment').loc[matches[0], 'Drug Structure']
            print(f"Matched '{drug_input}' to library drug: {matches[0]}")
        else:
            # Try to fetch SMILES from PubChem if not a valid SMILES already
            mol = Chem.MolFromSmiles(drug_input)
            active_smiles = drug_input
            if not mol:
                print(f"'{drug_input}' not a valid SMILES. Searching PubChem...")
                import pubchempy as pcp
                try:
                    pcp_matches = pcp.get_compounds(drug_input, 'name')
                    if pcp_matches:
                        active_smiles = pcp_matches[0].canonical_smiles
                        print(f"Found SMILES for {drug_input} on PubChem: {active_smiles}")
                except:
                    pass

            # Inductive Mode: Create Virtual Node
            drug_x = get_morgan_fingerprint(active_smiles)
            if drug_x is not None:
                is_new_drug = True
                print(f"Inductive Mode: Created virtual node for '{drug_input}'.")
            else:
                # Fallback to loose match in library
                matches = [name for name in drug_names if drug_input.lower() in name.lower()]
                if matches:
                    drug_idx = d_map[matches[0]]
                    active_smiles = pd.read_csv('00_Raw_Data/drugs_raw.csv').set_index('Drug Name/Treatment').loc[matches[0], 'Drug Structure']
                    print(f"Loosely matched '{drug_input}' to library drug: {matches[0]}")
                else:
                    print(f"Error: Could not find or featurize '{drug_input}'.")
                    return

    # 3. Create prediction graph state
    with torch.no_grad():
        if is_new_drug:
            # Inject new node into data temporarily
            orig_drug_x = data['drug'].x
            data['drug'].x = torch.cat([orig_drug_x, drug_x], dim=0)
            drug_idx = data['drug'].x.size(0) - 1

            # If target protein provided, add edge
            if target_protein and target_protein in p_map:
                p_idx = p_map[target_protein]
                new_edge = torch.tensor([[drug_idx], [p_idx]], dtype=torch.long)
                data['drug', 'binds', 'protein'].edge_index = torch.cat([data['drug', 'binds', 'protein'].edge_index, new_edge], dim=1)
                new_attr = torch.tensor([[1.0]], dtype=torch.float)
                data['drug', 'binds', 'protein'].edge_attr = torch.cat([data['drug', 'binds', 'protein'].edge_attr, new_attr], dim=0)
                print(f"Connected virtual drug to target protein: {target_protein}")

        # 4. Predict Top Proteins (Binding Affinity Inference)
        # We look at the drug's embedding similarity to all protein embeddings
        x_dict = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        drug_emb = x_dict['drug'][drug_idx].view(1, -1)
        prot_embs = x_dict['protein']

        # Calculate cosine similarity as proxy for binding potential
        sims = F.cosine_similarity(drug_emb, prot_embs)
        top_prot_indices = torch.topk(sims, k=min(3, len(all_proteins))).indices
        top_prots = [all_proteins[i] for i in top_prot_indices]

        # 5. Predict Alzheimer's Interaction
        # Alzheimer's is disease node 0
        disease_idx = 0
        edge_label_index = torch.tensor([[drug_idx], [disease_idx]], dtype=torch.long)
        prob = predictor(x_dict['drug'], x_dict['disease'], edge_label_index).item()

        # 6. Biological Metrics
        props = calculate_drug_properties(active_smiles)

    # Output Results
    print("-" * 30)
    print(f"RESULTS FOR: {drug_input}")
    print("-" * 30)
    print(f"Alzheimer's Interaction Probability: {prob:.4f}")
    print(f"Result: {'HIGH POTENTIAL' if prob > 0.5 else 'LOW POTENTIAL'}")
    print(f"\nTop 3 Predicted Protein Targets:")
    for i, p in enumerate(top_prots):
        print(f" {i+1}. {p}")

    if props:
        print(f"\nBiological Properties:")
        for k, v in props.items():
            print(f" - {k}: {v}")

    # Restore data state if modified
    if is_new_drug:
        data['drug'].x = orig_drug_x
        # (Simplified: we don't bother trimming edges as they are in-memory local to this run)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_inference(sys.argv[1])
    else:
        print("Usage: python3 02_Code/06_inference.py <drug_name_or_smiles>")
        print("Example: python3 02_Code/06_inference.py Tacrine")
