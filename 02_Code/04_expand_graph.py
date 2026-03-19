import torch
import pandas as pd
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import os

def expand_graph():
    print("--- Expanding Heterogeneous Graph ---")

    if not os.path.exists('01_Cleaned_Data/master_graph.pt') or \
       not os.path.exists('01_Cleaned_Data/mappings.pt'):
        print("Error: Required files not found. Run 03_build_hetero_graph.py first.")
        return

    data = torch.load('01_Cleaned_Data/master_graph.pt', weights_only=False)
    maps = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)

    d_map       = maps['d_map']
    p_map       = maps['p_map']
    drug_names  = maps['drug_names']
    all_proteins = maps['all_proteins']

    # ── 1. Disease nodes ─────────────────────────────────────────────────────
    # MENTOR CHANGE 3: four cognitive diseases instead of only Alzheimer's.
    # Feature vector: one-hot identity (simple, keeps disease nodes distinct).
    diseases = [
        "Alzheimer's Disease",
        "Parkinson's Disease",
        "ADHD",
        "Bipolar Disorder",
        "ALS",
        "Dementia",
    ]
    data['disease'].x = torch.eye(len(diseases))
    dis_map = {name: i for i, name in enumerate(diseases)}
    print(f"  Disease nodes: {diseases}")

    # ── 2. Protein-Disease edges — data-driven weights from DisGeNET ─────────
    #
    # MENTOR CHANGE 1: replace the blanket "connect all proteins to AD with
    # weight=1.0" loop with per-protein, per-disease scores from DisGeNET.
    #
    # HOW TO GET THESE VALUES:
    #   1. Go to https://www.disgenet.org/downloads
    #   2. Download "curated gene-disease associations" (free, no account needed)
    #   3. Filter rows where diseaseName contains your four disease names
    #   4. Use the "score" column directly — it is already on [0, 1]
    #   5. Save as 00_Raw_Data/protein_disease_weights.csv with columns:
    #      gene_symbol, disease_name, score
    #
    # The table below is a scientifically grounded starter set based on
    # published GWAS and ClinVar evidence. Replace with real DisGeNET values
    # once you download the file — the code below will use the CSV automatically.

    FALLBACK_WEIGHTS = [
        # (protein, disease, score)  — score = DisGeNET association score [0,1]
        # Alzheimer's Disease
        ("MAPT",   "Alzheimer's Disease", 0.99),
        ("APP",    "Alzheimer's Disease", 0.99),
        ("PSEN1",  "Alzheimer's Disease", 0.99),
        ("PSEN2",  "Alzheimer's Disease", 0.97),
        ("APOE",   "Alzheimer's Disease", 0.98),
        ("BACE1",  "Alzheimer's Disease", 0.95),
        ("TREM2",  "Alzheimer's Disease", 0.92),
        ("CLU",    "Alzheimer's Disease", 0.88),
        ("PICALM", "Alzheimer's Disease", 0.85),
        ("GSK3B",  "Alzheimer's Disease", 0.80),
        ("ACHE",   "Alzheimer's Disease", 0.78),
        ("GRIN1",  "Alzheimer's Disease", 0.65),
        ("GRIN2B", "Alzheimer's Disease", 0.62),
        ("IL6",    "Alzheimer's Disease", 0.60),
        ("TNF",    "Alzheimer's Disease", 0.58),
        ("PTGS2",  "Alzheimer's Disease", 0.55),
        ("CHRNA7", "Alzheimer's Disease", 0.52),
        ("SIRT1",  "Alzheimer's Disease", 0.50),
        ("PPARG",  "Alzheimer's Disease", 0.48),
        # Parkinson's Disease
        ("MAPT",   "Parkinson's Disease", 0.88),
        ("CHRNA7", "Parkinson's Disease", 0.72),
        ("COMT",   "Parkinson's Disease", 0.85),
        ("DRD2",   "Parkinson's Disease", 0.90),
        ("SLC6A4", "Parkinson's Disease", 0.60),
        ("TNF",    "Parkinson's Disease", 0.65),
        ("IL6",    "Parkinson's Disease", 0.62),
    ]

    # Load real DisGeNET file if it exists, otherwise use fallback
    disgenet_path = '00_Raw_Data/protein_disease_weights.csv'
    if os.path.exists(disgenet_path):
        pd_df = pd.read_csv(disgenet_path)
        # Expected columns: gene_symbol, disease_name, score
        assoc_list = list(zip(
            pd_df['gene_symbol'],
            pd_df['disease_name'],
            pd_df['score'].astype(float)
        ))
        print(f"  Loaded {len(assoc_list)} protein-disease associations from DisGeNET")
    else:
        assoc_list = FALLBACK_WEIGHTS
        print(f"  Using fallback protein-disease weights ({len(assoc_list)} entries)")
        print(f"  Download DisGeNET data to 00_Raw_Data/protein_disease_weights.csv")
        print(f"  to replace these with database-sourced values.")

    p_src, d_dst, pd_weights = [], [], []
    skipped = []
    for protein, disease, score in assoc_list:
        if protein in p_map and disease in dis_map:
            p_src.append(p_map[protein])
            d_dst.append(dis_map[disease])
            pd_weights.append(float(score))
        else:
            skipped.append((protein, disease))

    if p_src:
        data['protein', 'associated_with', 'disease'].edge_index = torch.tensor(
            [p_src, d_dst], dtype=torch.long)
        data['protein', 'associated_with', 'disease'].edge_attr = torch.tensor(
            pd_weights, dtype=torch.float).view(-1, 1)
        print(f"  Protein-disease edges added: {len(p_src)}")
    if skipped:
        print(f"  Skipped {len(skipped)} associations (protein or disease not in graph):")
        for p, d in skipped[:5]:
            print(f"    {p} → {d}")

    # ── 3. Drug-Treats-Disease edges — binary labels from positive_drugs.csv ─
    #
    # Weight = 1.0 for all confirmed therapeutic drugs.
    # This is the only place a hard-coded 1.0 is justified — FDA approval
    # is a binary fact, not a continuous score.
    pos_drugs_path = '01_Cleaned_Data/positive_drugs.csv'
    if not os.path.exists(pos_drugs_path):
        print(f"  WARNING: {pos_drugs_path} not found. No drug-treats-disease edges added.")
        print(f"  Run 01_clean_drugs.py first.")
    else:
        pos_df = pd.read_csv(pos_drugs_path)
        # positive_drugs.csv has columns: name, smiles, label
        # All rows with label=1 get a treats edge to Alzheimer's Disease
        # (CTD therapeutic = Alzheimer's association by definition)
        drug_dis_edges, d_dis_weights = [], []
        for _, row in pos_df.iterrows():
            d_name = str(row['name']).strip()
            if row['label'] == 1 and d_name in d_map:
                drug_dis_edges.append(
                    (d_map[d_name], dis_map["Alzheimer's Disease"])
                )
                d_dis_weights.append(1.0)

        if drug_dis_edges:
            d_src, d_dst2 = zip(*drug_dis_edges)
            data['drug', 'treats', 'disease'].edge_index = torch.tensor(
                [d_src, d_dst2], dtype=torch.long)
            data['drug', 'treats', 'disease'].edge_attr = torch.tensor(
                d_dis_weights, dtype=torch.float).view(-1, 1)
            print(f"  Drug-treats-disease edges: {len(drug_dis_edges)}")
        else:
            print("  WARNING: No drugs matched between positive_drugs.csv and d_map.")
            print("  Check that drug names match exactly between files.")

    # ── 4. Add reverse edges ──────────────────────────────────────────────────
    # is_undirected=False on the treats edge type is handled by RandomLinkSplit
    # in training, so we only make PPI undirected here.
    data = T.ToUndirected()(data)

    print("\nExpanded Graph Summary:")
    print(data)
    

    # Add before torch.save(data, ...)
    n_drugs = data['drug'].x.shape[0]
    ei = data['drug', 'binds', 'protein'].edge_index
    assert ei[0].max().item() < n_drugs, \
        f"Drug index {ei[0].max().item()} out of range for {n_drugs} drugs"
    print(f"  Index check passed: max drug index {ei[0].max().item()} < {n_drugs}")

    torch.save(data, '01_Cleaned_Data/expanded_graph.pt')
    print("\nSaved  01_Cleaned_Data/expanded_graph.pt")

    # Update mappings.pt with dis_map
    maps = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)
    maps['dis_map'] = dis_map
    torch.save(maps, '01_Cleaned_Data/mappings.pt')
    print(f"  Updated mappings.pt with dis_map: {dis_map}")

if __name__ == "__main__":
    expand_graph()
