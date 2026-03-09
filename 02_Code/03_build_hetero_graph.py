import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from featurizer import DrugFeaturizer
import ast

def build_hetero_graph():
    print("--- Building Heterogeneous Graph (Updated Data) ---")

    # 1. Load Data
    # New Drug Data
    drugs_df = pd.read_csv('00_Raw_Data/drugs_raw.csv')
    # New PPI Data
    ppi_df = pd.read_csv('00_Raw_Data/ppi_raw.tsv', sep='\t')

    # 2. Process Drug Nodes
    # We'll use the provided vectors if possible, or featurize from SMILES if they are better
    # The user provided vectors in "Numerical_Vector" column

    drug_names = drugs_df['Drug Name/Treatment'].tolist()
    smiles_list = drugs_df['Drug Structure'].tolist()

    feat = DrugFeaturizer()
    drug_embeds = []
    valid_drug_indices = []

    for i, row in drugs_df.iterrows():
        # Use DrugFeaturizer to get consistent embedding size if possible
        # or use the provided vector. The provided vector is very long (1024 or so?)
        # Let's see the length of the vector
        vec_str = row['Numerical_Vector']
        try:
            vec = ast.literal_eval(vec_str)
            drug_embeds.append(torch.tensor(vec, dtype=torch.float).view(1, -1))
            valid_drug_indices.append(i)
        except:
            print(f"Warning: Failed to parse vector for {row['Drug Name/Treatment']}")

    data = HeteroData()
    data['drug'].x = torch.cat(drug_embeds, dim=0)

    # Map drug names to indices for edge creation
    d_map = {name: i for i, name in enumerate(drug_names) if i in valid_drug_indices}

    # 3. Process Protein Nodes
    all_proteins = sorted(list(set(ppi_df['#node1'].unique()) | set(ppi_df['node2'].unique())))
    p_map = {p: i for i, p in enumerate(all_proteins)}
    data['protein'].x = torch.eye(len(all_proteins))

    # 4. Drug-Binds-Protein Edges with Confidence Weights
    # Logic: Specific Gene Target = 1.0, Pathway Target (e.g. beta-amyloid) = 0.5, Unknown/Neither = 0.1
    src, dst, d_p_weights = [], [], []
    # Manual mapping for common non-gene targets
    target_mapping = {
        'beta-amyloid': 'APP',
        'tau protein': 'MAPT'
    }

    for i, row in drugs_df.iterrows():
        d_name = row['Drug Name/Treatment']
        raw_target_str = str(row['Targeted protein']).strip()

        # Handle multiple targets (e.g. "beta-amyloid, Tau")
        raw_targets = [t.strip() for t in raw_target_str.replace('and', ',').split(',') if t.strip()]

        for rt in raw_targets:
            # Determine weight based on specificity
            if rt.lower() in ['neither', 'nan', 'unknown', 'none', '']:
                weight = 0.1
                # For 'Neither', we don't have a specific protein node to bind to.
                # The 'treats' edge in expand_graph will handle the drug-disease connection.
                continue
            elif rt.lower() in ['beta-amyloid', 'tau protein', 'tau']:
                weight = 0.5
            else:
                weight = 1.0 # Specific gene target

            # Apply mapping to proteins present in our PPI graph
            # Since APP is missing from the TSV, we map amyloid-targeting drugs to BACE1 (the enzyme that creates it)
            p_target = rt
            if rt.lower() == 'beta-amyloid': p_target = 'BACE1'
            if rt.lower() in ['tau', 'tau protein']: p_target = 'MAPT'

            if d_name in d_map and p_target in p_map:
                src.append(d_map[d_name])
                dst.append(p_map[p_target])
                d_p_weights.append(weight)
            else:
                if d_name in d_map and rt.lower() not in ['neither', 'nan', 'none']:
                    print(f"Skipping edge for {d_name}: Target '{p_target}' (from '{rt}') not in protein map.")

    data['drug', 'binds', 'protein'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    data['drug', 'binds', 'protein'].edge_attr = torch.tensor(d_p_weights, dtype=torch.float).view(-1, 1)

    # 5. Protein-Interacts-Protein Edges (PPI)
    p_src, p_dst, p_weights = [], [], []
    for _, row in ppi_df.iterrows():
        p1, p2, score = row['#node1'], row['node2'], row['combined_score']
        if p1 in p_map and p2 in p_map:
            p_src.extend([p_map[p1], p_map[p2]])
            p_dst.extend([p_map[p2], p_map[p1]])
            # Combined score is often out of 1000 in STRING, normalize to 0-1
            norm_score = score / 1000.0
            p_weights.extend([norm_score, norm_score])

    data['protein', 'interacts_with', 'protein'].edge_index = torch.tensor([p_src, p_dst], dtype=torch.long)
    data['protein', 'interacts_with', 'protein'].edge_attr = torch.tensor(p_weights, dtype=torch.float).view(-1, 1)

    print("\nGraph Construction Complete:")
    print(data)

    os.makedirs('01_Cleaned_Data', exist_ok=True)
    torch.save(data, '01_Cleaned_Data/master_graph.pt')
    # Save mappings for later use
    torch.save({'d_map': d_map, 'p_map': p_map, 'drug_names': drug_names, 'all_proteins': all_proteins}, '01_Cleaned_Data/mappings.pt')
    print("\nSaved master_graph.pt and mappings.pt to 01_Cleaned_Data/")

if __name__ == "__main__":
    build_hetero_graph()
