import torch
import pandas as pd
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import os

def expand_graph():
    print("--- Expanding Heterogeneous Graph (Targeting Alzheimer's) ---")

    # Load existing graph and mappings
    if not os.path.exists('01_Cleaned_Data/master_graph.pt') or not os.path.exists('01_Cleaned_Data/mappings.pt'):
        print("Error: Required files not found. Run 02_Code/03_build_hetero_graph.py first.")
        return

    data = torch.load('01_Cleaned_Data/master_graph.pt', weights_only=False)
    maps = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)

    d_map = maps['d_map']
    p_map = maps['p_map']
    drug_names = maps['drug_names']
    all_proteins = maps['all_proteins']

    # 1. Add Disease Nodes
    # We'll focus on Alzheimer's as requested
    diseases = ['Alzheimer\'s Disease']
    data['disease'].x = torch.eye(len(diseases))
    dis_map = {name: i for i, name in enumerate(diseases)}

    # 2. Protein-Disease Associations
    # Associate ALL proteins in our Amyloid/Tau focused network with AD
    prot_dis_edges = []
    for p in all_proteins:
        prot_dis_edges.append((p_map[p], dis_map['Alzheimer\'s Disease']))

    src, dst = zip(*prot_dis_edges)
    data['protein', 'associated_with', 'disease'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 3. Drug-Treats-Disease Edges (Ground Truth)
    # We need to identify which drugs in our list are "Approved" or "Successful" for AD
    drugs_df = pd.read_csv('00_Raw_Data/drugs_raw.csv')

    drug_dis_edges = []
    for i, row in drugs_df.iterrows():
        d_name = row['Drug Name/Treatment']
        status = row['Current Status']

        if d_name in d_map:
            # If status is "Approved", we consider it a ground truth link for AD
            if status == 'Approved':
                drug_dis_edges.append((d_map[d_name], dis_map['Alzheimer\'s Disease']))

    if drug_dis_edges:
        d_src, d_dst = zip(*drug_dis_edges)
        data['drug', 'treats', 'disease'].edge_index = torch.tensor([d_src, d_dst], dtype=torch.long)
    else:
        print("Warning: No 'Approved' drugs found for ground truth.")

    # 4. Add Reverse Edges
    data = T.ToUndirected()(data)

    print("\nExpanded Graph Summary:")
    print(data)

    torch.save(data, '01_Cleaned_Data/expanded_graph.pt')
    print("\nSaved expanded_graph.pt to 01_Cleaned_Data/")

if __name__ == "__main__":
    expand_graph()
