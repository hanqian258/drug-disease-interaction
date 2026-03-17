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
    drugs_df = pd.read_csv('00_Raw_Data/drugs_raw_augmented.csv')
    ppi_df   = pd.read_csv('01_Cleaned_Data/ppi_interactions.csv')

    # 2. Process Drug Nodes
    drug_names = drugs_df['Drug Name/Treatment'].tolist()

    drug_embeds        = []
    valid_drug_indices = []

    for i, row in drugs_df.iterrows():
        vec_str = row.get('Numerical_Vector', None)
        try:
            vec = ast.literal_eval(str(vec_str))
            drug_embeds.append(torch.tensor(vec, dtype=torch.float).view(1, -1))
            valid_drug_indices.append(i)
        except Exception:
            # CTD-derived drugs have no Numerical_Vector — featurize from SMILES
            smiles = str(row.get('Drug Structure', '')).strip()
            if smiles and smiles != 'nan':
                from rdkit import Chem
                from rdkit.Chem import AllChem
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    drug_embeds.append(
                        torch.tensor(list(fp), dtype=torch.float).view(1, -1)
                    )
                    valid_drug_indices.append(i)
                else:
                    print(f"Warning: invalid SMILES for {row['Drug Name/Treatment']}")
            else:
                print(f"Warning: no vector or SMILES for {row['Drug Name/Treatment']}")

    data = HeteroData()
    data['drug'].x = torch.cat(drug_embeds, dim=0)

    # Map drug names to indices
    # Build d_map using sequential position in drug_embeds, not original dataframe index
        d_map = {}
        embed_position = 0
        for i, row in drugs_df.iterrows():
            if i in valid_drug_indices:
                d_map[row['Drug Name/Treatment']] = embed_position
                embed_position += 1

    # 3. Process Protein Nodes
    all_proteins = sorted(
        set(ppi_df['preferredName_A'].unique()) |
        set(ppi_df['preferredName_B'].unique())
    )
    p_map = {p: i for i, p in enumerate(all_proteins)}
    data['protein'].x = torch.eye(len(all_proteins))
    print(f"  Protein nodes: {len(all_proteins)}")

    # 4. Drug-Binds-Protein Edges — weights from CTD inference scores
    links_df = pd.read_csv('00_Raw_Data/drug_links.csv')
    max_score = links_df['inference_score'].max()
    links_df['weight'] = links_df['inference_score'] / max_score

    src, dst, d_p_weights = [], [], []
    skipped_drugs    = set()
    skipped_proteins = set()

    for _, row in links_df.iterrows():
        d_name   = str(row['drug_name']).strip()
        p_target = str(row['protein_target']).strip()
        weight   = float(row['weight'])

        if d_name in d_map and p_target in p_map:
            src.append(d_map[d_name])
            dst.append(p_map[p_target])
            d_p_weights.append(weight)
        else:
            if d_name not in d_map:
                skipped_drugs.add(d_name)
            if p_target not in p_map:
                skipped_proteins.add(p_target)

    if skipped_drugs:
        print(f"  Drugs not in graph ({len(skipped_drugs)}): {sorted(skipped_drugs)[:5]} ...")
    if skipped_proteins:
        print(f"  Proteins not in graph ({len(skipped_proteins)}): {sorted(skipped_proteins)[:5]} ...")

    if src:
        data['drug', 'binds', 'protein'].edge_index = torch.tensor(
            [src, dst], dtype=torch.long)
        data['drug', 'binds', 'protein'].edge_attr  = torch.tensor(
            d_p_weights, dtype=torch.float).view(-1, 1)
        print(f"  Drug-protein binding edges: {len(src)}")
    else:
        print("  WARNING: zero drug-protein edges created — check name matching")

    # 5. Protein-Interacts-Protein Edges
    p_src, p_dst, p_weights = [], [], []
    for _, row in ppi_df.iterrows():
        p1    = str(row['preferredName_A']).strip()
        p2    = str(row['preferredName_B']).strip()
        score = float(row['score'])
        if p1 in p_map and p2 in p_map:
            p_src.extend([p_map[p1], p_map[p2]])
            p_dst.extend([p_map[p2], p_map[p1]])
            p_weights.extend([score, score])

    data['protein', 'interacts_with', 'protein'].edge_index = torch.tensor(
        [p_src, p_dst], dtype=torch.long)
    data['protein', 'interacts_with', 'protein'].edge_attr  = torch.tensor(
        p_weights, dtype=torch.float).view(-1, 1)
    print(f"  PPI edges: {len(p_src) // 2}")

    print("\nGraph Construction Complete:")
    print(data)

    os.makedirs('01_Cleaned_Data', exist_ok=True)
    torch.save(data, '01_Cleaned_Data/master_graph.pt')
    torch.save(
        {'d_map': d_map, 'p_map': p_map,
         'drug_names': drug_names, 'all_proteins': all_proteins},
        '01_Cleaned_Data/mappings.pt'
    )
    print("\nSaved master_graph.pt and mappings.pt to 01_Cleaned_Data/")

if __name__ == "__main__":
    build_hetero_graph()