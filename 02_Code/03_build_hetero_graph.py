import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from featurizer import DrugFeaturizer

def build_hetero_graph():
    print("--- Building Heterogeneous Graph ---")

    # 1. Load Cleaned Data
    pos = pd.read_csv('01_Cleaned_Data/positive_drugs.csv')
    neg = pd.read_csv('01_Cleaned_Data/negative_controls.csv')
    links = pd.read_csv('01_Cleaned_Data/drug_links.csv')
    ppi = pd.read_csv('01_Cleaned_Data/ppi_interactions.csv')

    all_drugs = pd.concat([pos, neg], ignore_index=True)

    # 2. Process Drug Nodes (Features from SMILES)
    feat = DrugFeaturizer()
    drug_embeds = []
    valid_drug_idxs = []

    for i, row in all_drugs.iterrows():
        g = feat.smiles_to_graph(row['smiles'])
        if g:
            # Aggregate atom features for the drug (e.g., mean)
            drug_embeds.append(torch.mean(g.x, dim=0, keepdim=True))
            valid_drug_idxs.append(i)
        else:
            print(f"Warning: Failed to featurize {row['name']}")

    data = HeteroData()
    data['drug'].x = torch.cat(drug_embeds, dim=0)
    data['drug'].y = torch.tensor(all_drugs.loc[valid_drug_idxs, 'label'].values, dtype=torch.float)

    # 3. Process Protein Nodes
    # All proteins from drug-target links and PPI interactions
    all_proteins = sorted(list(set(links['protein_target'].unique()) |
                               set(ppi['preferredName_A'].unique()) |
                               set(ppi['preferredName_B'].unique())))

    p_map = {p: i for i, p in enumerate(all_proteins)}
    data['protein'].x = torch.eye(len(all_proteins)) # Simple one-hot features

    # 4. Process Drug-Binds-Protein Edges
    src, dst = [], []
    for _, row in links.iterrows():
        d_name, p_name = row['drug_name'], row['protein_target']
        # Find index in valid drugs
        match = all_drugs[(all_drugs['name'] == d_name) & (all_drugs.index.isin(valid_drug_idxs))]
        if not match.empty:
            d_idx = valid_drug_idxs.index(match.index[0])
            if p_name in p_map:
                src.append(d_idx)
                dst.append(p_map[p_name])

    data['drug', 'binds', 'protein'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 5. Process Protein-Interacts-Protein Edges (PPI)
    p_src, p_dst, p_weights = [], [], []
    for _, row in ppi.iterrows():
        p1, p2, score = row['preferredName_A'], row['preferredName_B'], row['score']
        if p1 in p_map and p2 in p_map:
            p_src.extend([p_map[p1], p_map[p2]])
            p_dst.extend([p_map[p2], p_map[p1]])
            p_weights.extend([score, score])

    data['protein', 'interacts_with', 'protein'].edge_index = torch.tensor([p_src, p_dst], dtype=torch.long)
    data['protein', 'interacts_with', 'protein'].edge_attr = torch.tensor(p_weights, dtype=torch.float).view(-1, 1)

    print("\nGraph Construction Complete:")
    print(data)

    os.makedirs('01_Cleaned_Data', exist_ok=True)
    torch.save(data, '01_Cleaned_Data/master_graph.pt')
    print("\nSaved master_graph.pt to 01_Cleaned_Data/")

if __name__ == "__main__":
    build_hetero_graph()
