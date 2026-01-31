import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from featurizer import DrugFeaturizer

def build_hetero_graph():
    print("--- Processing ---")
    pos = pd.read_csv('data/positive_drugs.csv')
    neg = pd.read_csv('data/negative_controls.csv')
    links = pd.read_csv('data/drug_links.csv')
    all_drugs = pd.concat([pos, neg], ignore_index=True)
    
    feat = DrugFeaturizer()
    drug_embeds = []
    valid_idxs = []
    
    for i, row in all_drugs.iterrows():
        g = feat.smiles_to_graph(row['smiles'])
        if g:
            drug_embeds.append(torch.mean(g.x, dim=0, keepdim=True))
            valid_idxs.append(i)
            
    data = HeteroData()
    data['drug'].x = torch.cat(drug_embeds, dim=0)
    data['drug'].y = torch.tensor(all_drugs.loc[valid_idxs, 'label'].values, dtype=torch.float)
    
    targets = links['protein_target'].unique().tolist()
    t_map = {t: i for i, t in enumerate(targets)}
    data['protein'].x = torch.eye(len(targets))
    
    src, dst = [], []
    for _, row in links.iterrows():
        d_name, p_name = row['drug_name'], row['protein_target']
        match = all_drugs[(all_drugs['name'] == d_name) & (all_drugs.index.isin(valid_idxs))]
        if not match.empty:
            tensor_idx = valid_idxs.index(match.index[0])
            src.append(tensor_idx)
            dst.append(t_map[p_name])
            
    data['drug', 'binds', 'protein'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    print(data)
    torch.save(data, 'data/master_graph.pt')

if __name__ == "__main__":
    build_hetero_graph()
