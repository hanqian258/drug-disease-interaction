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
