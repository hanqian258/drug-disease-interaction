import torch
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_graph(graph_path):
    if not os.path.exists(graph_path):
        logging.error(f"Graph file not found: {graph_path}")
        return

    data = torch.load(graph_path, weights_only=False)

    num_drugs = data['drug'].num_nodes
    num_proteins = data['protein'].num_nodes
    num_binds = data['drug', 'binds', 'protein'].num_edges
    num_ppi = data['protein', 'interacts_with', 'protein'].num_edges // 2 # Bidirectional edges

    print("--- Graph Summary Report ---")
    print(f"Total Drugs: {num_drugs}")
    print(f"Total Proteins: {num_proteins}")
    print(f"Drug-Protein Binding Edges: {num_binds}")
    print(f"Protein-Protein Interaction Edges: {num_ppi}")

    # For link prediction, check ('drug', 'treats', 'disease')
    if 'disease' in data.node_types:
        num_treats = data['drug', 'treats', 'disease'].num_edges
        print(f"Drug-Treats-Disease Edges (Ground Truth): {num_treats}")

if __name__ == "__main__":
    if os.path.exists('01_Cleaned_Data/expanded_graph.pt'):
        validate_graph('01_Cleaned_Data/expanded_graph.pt')
    else:
        validate_graph('01_Cleaned_Data/master_graph.pt')

##Checks##
import torch
data = torch.load('01_Cleaned_Data/expanded_graph.pt', weights_only=False)
maps = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)
ei = data['drug', 'treats', 'disease'].edge_index
dis_map = maps['dis_map']
idx_to_dis = {v: k for k, v in dis_map.items()}
from collections import Counter
counts = Counter(ei[1].tolist())
for dis_idx, count in sorted(counts.items()):
    print(f"  {idx_to_dis[dis_idx]}: {count} edges")

import torch, pandas as pd

maps = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)
d_map = maps['d_map']

for fname, disease in [
    ('01_Cleaned_Data/positive_drugs_parkinsons.csv', "Parkinson's"),
    ('01_Cleaned_Data/positive_drugs_adhd.csv',       'ADHD'),
]:
    df = pd.read_csv(fname)
    missing = [r['name'] for _, r in df.iterrows()
               if str(r['name']).strip() not in d_map]
    print(f"\n{disease} — {len(missing)} drugs not in graph:")
    for m in missing:
        print(f"  {m}")