import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
import os

def visualize_graph():
    print("--- Visualizing Heterogeneous Graph ---")

    if not os.path.exists('01_Cleaned_Data/expanded_graph.pt'):
        print("Error: expanded_graph.pt not found.")
        return

    data = torch.load('01_Cleaned_Data/expanded_graph.pt', weights_only=False)
    maps = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)

    drug_names = maps['drug_names']
    all_proteins = maps['all_proteins']

    G = nx.Graph()

    # Add nodes with types
    # To keep the plot readable, we'll sample a subset if it's too large
    # Let's focus on Approved drugs and their targets, plus some PPI

    import pandas as pd
    drugs_df = pd.read_csv('00_Raw_Data/drugs_raw.csv')
    approved_drugs = drugs_df[drugs_df['Current Status'] == 'Approved']['Drug Name/Treatment'].tolist()
    approved_indices = [maps['d_map'][name] for name in approved_drugs if name in maps['d_map']]

    # Add Drugs
    for i, name in enumerate(drug_names):
        if i in approved_indices:
            G.add_node(f"Drug: {name}", type='drug', color='lightblue')

    # Add Proteins
    for i, name in enumerate(all_proteins):
        G.add_node(f"Protein: {name}", type='protein', color='lightgreen')

    # Add Disease
    G.add_node("Alzheimer's Disease", type='disease', color='salmon')

    # Add Edges
    # Drug -> Protein
    edge_index = data['drug', 'binds', 'protein'].edge_index
    for i in range(edge_index.size(1)):
        d_idx = edge_index[0, i].item()
        p_idx = edge_index[1, i].item()
        if d_idx in approved_indices:
            G.add_edge(f"Drug: {drug_names[d_idx]}", f"Protein: {all_proteins[p_idx]}", label='binds')

    # Protein -> Protein
    edge_index = data['protein', 'interacts_with', 'protein'].edge_index
    # Only add PPI for proteins connected to approved drugs or key AD proteins to keep it clean
    for i in range(edge_index.size(1)):
        p1_idx = edge_index[0, i].item()
        p2_idx = edge_index[1, i].item()
        # Sampling PPI to avoid hairball
        if i % 20 == 0:
            G.add_edge(f"Protein: {all_proteins[p1_idx]}", f"Protein: {all_proteins[p2_idx]}", label='interacts')

    # Protein -> Disease
    edge_index = data['protein', 'associated_with', 'disease'].edge_index
    for i in range(edge_index.size(1)):
        p_idx = edge_index[0, i].item()
        # Dis index is 0
        G.add_edge(f"Protein: {all_proteins[p_idx]}", "Alzheimer's Disease", label='associated_with')

    # Drug -> Disease (Approved links)
    edge_index = data['drug', 'treats', 'disease'].edge_index
    for i in range(edge_index.size(1)):
        d_idx = edge_index[0, i].item()
        G.add_edge(f"Drug: {drug_names[d_idx]}", "Alzheimer's Disease", label='treats')

    # Plot
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.15, iterations=20)

    node_colors = [G.nodes[n]['color'] for n in G.nodes]

    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, font_size=8, edge_color='gray', alpha=0.7)

    plt.title("Heterogeneous Interaction Network: Drugs, Proteins, and Alzheimer's Disease")
    plt.savefig('network_visualization.png')
    print("Graph visualization saved to network_visualization.png")

if __name__ == "__main__":
    visualize_graph()
