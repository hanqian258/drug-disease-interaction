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
    print(f"Protein-Protein Interaction Edges (High Confidence): {num_ppi}")

    # Check for labels
    pos_labels = (data['drug'].y == 1).sum().item()
    neg_labels = (data['drug'].y == 0).sum().item()
    print(f"Positive Samples (Alzheimer's Drugs): {pos_labels}")
    print(f"Negative Samples (Controls): {neg_labels}")

    # Check for internal controls in PPI
    ppi_df = pd.read_csv('01_Cleaned_Data/ppi_interactions.csv')
    controls = [('APP', 'BACE1'), ('MAPT', 'GSK3B')]
    for p1, p2 in controls:
        match = ppi_df[((ppi_df['preferredName_A'] == p1) & (ppi_df['preferredName_B'] == p2)) |
                       ((ppi_df['preferredName_A'] == p2) & (ppi_df['preferredName_B'] == p1))]
        if not match.empty:
            print(f"Internal PPI Control: {p1}-{p2} - PASSED (Score: {match['score'].values[0]})")
        else:
            print(f"Internal PPI Control: {p1}-{p2} - FAILED")

if __name__ == "__main__":
    validate_graph('01_Cleaned_Data/master_graph.pt')
