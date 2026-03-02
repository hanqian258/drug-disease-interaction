import torch
import pandas as pd
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from featurizer import DrugFeaturizer
import os

def expand_graph():
    print("--- Expanding Heterogeneous Graph ---")

    # Load existing graph
    if not os.path.exists('01_Cleaned_Data/master_graph.pt'):
        print("Error: master_graph.pt not found. Run 02_Code/03_build_hetero_graph.py first.")
        return

    data = torch.load('01_Cleaned_Data/master_graph.pt', weights_only=False)

    # 1. Add Levodopa
    levodopa_smiles = "C1=CC(=C(C=C1CC(C(=O)O)N)O)O"
    feat = DrugFeaturizer()
    levo_g = feat.smiles_to_graph(levodopa_smiles)
    levo_embed = torch.mean(levo_g.x, dim=0, keepdim=True)

    # Append to drug nodes
    data['drug'].x = torch.cat([data['drug'].x, levo_embed], dim=0)
    # Label for Levodopa (it's not an AD drug, so 0 for AD classification,
    # though we are moving to link prediction)
    data['drug'].y = torch.cat([data['drug'].y, torch.tensor([0.0])])

    levo_idx = data['drug'].num_nodes - 1

    # 2. Add Disease Nodes (AD, PD, FTD)
    diseases = ['Alzheimer\'s Disease', 'Parkinson\'s Disease', 'Frontotemporal Dementia']
    data['disease'].x = torch.eye(len(diseases))
    dis_map = {name: i for i, name in enumerate(diseases)}

    # 3. Protein-Disease Associations
    # AD: All 11 proteins
    # FTD: MAPT, GSK3B, APOE, TREM2
    # PD: MAPT, APOE, GSK3B

    # Need to get p_map from original build script or derive it
    # Since we don't have the original p_map, let's look at ppi_interactions.csv to rebuild it
    ppi = pd.read_csv('01_Cleaned_Data/ppi_interactions.csv')
    links = pd.read_csv('01_Cleaned_Data/drug_links.csv')
    all_proteins = sorted(list(set(links['protein_target'].unique()) |
                               set(ppi['preferredName_A'].unique()) |
                               set(ppi['preferredName_B'].unique())))
    p_map = {p: i for i, p in enumerate(all_proteins)}

    prot_dis_edges = []
    # AD
    for p in all_proteins:
        prot_dis_edges.append((p_map[p], dis_map['Alzheimer\'s Disease']))

    # FTD
    for p in ['MAPT', 'GSK3B', 'APOE', 'TREM2']:
        if p in p_map:
            prot_dis_edges.append((p_map[p], dis_map['Frontotemporal Dementia']))

    # PD
    for p in ['MAPT', 'APOE', 'GSK3B']:
        if p in p_map:
            prot_dis_edges.append((p_map[p], dis_map['Parkinson\'s Disease']))

    src, dst = zip(*prot_dis_edges)
    data['protein', 'associated_with', 'disease'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 4. Drug-Treats-Disease Edges (Ground Truth)
    # AD drugs (first 5 in original) to AD
    # Levodopa (last one) to PD

    drug_dis_edges = []
    # Original AD drugs: Tacrine, Donepezil, Memantine, Rivastigmine, Galantamine
    # They should be indices 0 to 4
    for i in range(5):
        drug_dis_edges.append((i, dis_map['Alzheimer\'s Disease']))

    # Levodopa to PD
    drug_dis_edges.append((levo_idx, dis_map['Parkinson\'s Disease']))

    d_src, d_dst = zip(*drug_dis_edges)
    data['drug', 'treats', 'disease'].edge_index = torch.tensor([d_src, d_dst], dtype=torch.long)

    # 5. Add Reverse Edges
    # This is important for message passing
    data = T.ToUndirected()(data)

    print("\nExpanded Graph Summary:")
    print(data)

    torch.save(data, '01_Cleaned_Data/expanded_graph.pt')
    print("\nSaved expanded_graph.pt to 01_Cleaned_Data/")

if __name__ == "__main__":
    expand_graph()
