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
    diseases = ['Alzheimer\'s Disease', 'Healthy Control']
    data['disease'].x = torch.eye(len(diseases))
    dis_map = {name: i for i, name in enumerate(diseases)}

    # 2. Protein-Disease Associations
    # Associate ALL proteins in our Amyloid/Tau focused network with AD
    prot_dis_edges = []
    for p in all_proteins:
        prot_dis_edges.append((p_map[p], dis_map['Alzheimer\'s Disease']))

    src, dst = zip(*prot_dis_edges)
    data['protein', 'associated_with', 'disease'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 3. Drug-Treats-Disease Edges with Confidence Weights
    # Logic: Approved = 1.0, Experimental/Trial = 0.5, Failed/Other = 0.1
    drugs_df = pd.read_csv('00_Raw_Data/drugs_raw.csv')

    drug_dis_edges, d_dis_weights = [], []
    existing_edges = set()

    def add_drug_disease_edge(drug_name, weight):
        drug_key = drug_name.strip()
        if drug_key not in d_map:
            return False
        edge = (d_map[drug_key], dis_map['Alzheimer\'s Disease'])
        if edge in existing_edges:
            return False
        existing_edges.add(edge)
        drug_dis_edges.append(edge)
        d_dis_weights.append(weight)
        return True

    for i, row in drugs_df.iterrows():
        d_name = str(row.get('Drug Name/Treatment', '')).strip()
        status = str(row.get('Current Status', '')).strip()

        if d_name and d_name in d_map:
            if status == 'Approved':
                weight = 1.0
            elif status in ['Experimental', 'Clinical Trial']:
                weight = 0.5
            else:
                weight = 0.1  # Helps model learn "low potential"

            add_drug_disease_edge(d_name, weight)

    # 3a. Add CTD Alzheimer-associated chemicals from the curated file
    #    Use adjusted confidence based on CTD inference score.
    ctd_path = '01_Cleaned_Data/CTD_D000544_chemicals_20260315024131.csv'
    if os.path.exists(ctd_path):
        ctd_df = pd.read_csv(ctd_path)
        for _, row in ctd_df.iterrows():
            chem = str(row.get('Chemical Name', '')).strip()
            score = row.get('Inference Score', None)
            if pd.isna(chem) or not chem:
                continue
            try:
                score = float(score) if score is not None and not pd.isna(score) else 0.0
            except Exception:
                score = 0.0

            # Rescale inference score 0-100 to 0.2-0.8
            weight = 0.2 + 0.6 * min(max(score / 100.0, 0.0), 1.0)
            if add_drug_disease_edge(chem, weight):
                print(f"Added CTD drug-disease edge {chem} (score={score:.2f}, weight={weight:.3f})")
            elif chem in d_map:
                # already present via drugs_raw route, possibly upweight if higher
                pass
            else:
                print(f"CTD chemical not in drug map (skip): {chem}")
    else:
        print(f"CTD file not found: {ctd_path}")

    if drug_dis_edges:
        d_src, d_dst = zip(*drug_dis_edges)
        data['drug', 'treats', 'disease'].edge_index = torch.tensor([d_src, d_dst], dtype=torch.long)
        data['drug', 'treats', 'disease'].edge_attr = torch.tensor(d_dis_weights, dtype=torch.float).view(-1, 1)
    else:
        print("Warning: No drugs found for AD therapeutic links.")

    # 4. Add Reverse Edges
    data = T.ToUndirected()(data)

    print("\nExpanded Graph Summary:")
    print(data)

    torch.save(data, '01_Cleaned_Data/expanded_graph.pt')
    print("\nSaved expanded_graph.pt to 01_Cleaned_Data/")

if __name__ == "__main__":
    expand_graph()
