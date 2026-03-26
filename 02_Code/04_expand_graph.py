"""
04_expand_graph.py — Expand master graph with disease nodes, protein-disease
edges, drug-treats-disease edges, and hard negative edges.

CHANGES FROM PREVIOUS VERSION
──────────────────────────────
1. DISEASE_DRUG_FILES now includes the three new CTD-derived files:
      positive_drugs_ad.csv        (full AD CTD chemical query, 58 drugs)
      positive_drugs_adhd.csv      (ADHD CTD chemical query, 16 drugs)
      positive_drugs_parkinsons.csv (Parkinson's CTD query, 49 drugs)
   These activate automatically once 01_clean_drugs.py has been run.

2. FALLBACK_WEIGHTS completely replaced with CTD-data-derived scores.
   Scores are frequency-normalised across the new drug_links files —
   proteins shared by more drugs get higher association scores.
   This covers all 85 AD proteins, 14 ADHD proteins, and 78 Parkinson's
   proteins from the real CTD data, replacing the hand-picked fallback.

3. drug_links.csv merger in 01_clean_drugs.py now includes:
      drug_links_ad.csv, drug_links_adhd.csv, drug_links_parkinsons.csv
   so 03_build_hetero_graph.py automatically picks up the new protein
   targets without any changes to that file.

PREVIOUS CHANGES (still in effect)
────────────────────────────────────
- Multi-disease positive edges (ALS, Bipolar, Dementia)
- Hard negative edges from failed AD trials
- Graded evidence weights (FDA=1.0, CTD=0.9, Phase2=0.8, Preclinical=0.7)
"""

import torch
import pandas as pd
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import os
from collections import Counter


# ── Disease name normalisation ────────────────────────────────────────────────

DISEASE_ALIASES = {
    "alzhimers":            "Alzheimer's Disease",
    "alzheimer disease":    "Alzheimer's Disease",
    "alzheimers disease":   "Alzheimer's Disease",
    "alzheimer's disease":  "Alzheimer's Disease",
    "parkinson disease":    "Parkinson's Disease",
    "parkinsons disease":   "Parkinson's Disease",
    "parkinson's disease":  "Parkinson's Disease",
    "adhd":                 "ADHD",
    "attention deficit hyperactivity disorder": "ADHD",
    "bipolar disorder":     "Bipolar Disorder",
    "als":                  "ALS",
    "amyotrophic lateral sclerosis": "ALS",
    "dementia":             "Dementia",
}


def normalize_disease_name(name: str) -> str:
    key = str(name).strip().lower()
    return DISEASE_ALIASES.get(key, str(name).strip())


# ── Evidence tier weights ─────────────────────────────────────────────────────

EVIDENCE_WEIGHTS = {
    'Approved':    1.00,
    'CTD':         0.90,
    'Phase2':      0.80,
    'Preclinical': 0.70,
}

# ── Hard negatives: confirmed failed AD clinical trials ───────────────────────

FAILED_AD_DRUGS = [
    "Semagacestat",        # gamma-secretase inhibitor — Phase 3 cognitive worsening
    "Tarenflurbil",        # gamma-secretase modulator — Phase 3 no effect
    "Verubecestat",        # BACE1 inhibitor — Phase 3 stopped early, no benefit
    "Hydromethylthionine", # tau aggregation inhibitor — Phase 3 failed
    "Methylthioninium",    # tau aggregation inhibitor — Phase 3 failed
]

# ── Per-disease positive drug file mapping ────────────────────────────────────

DISEASE_DRUG_FILES = {
    "Alzheimer's Disease": [
        ('01_Cleaned_Data/positive_drugs.csv',          'CTD'),
        ('01_Cleaned_Data/positive_drugs_ad.csv',       'CTD'),
    ],
    "Dementia": [
        ('01_Cleaned_Data/positive_drugs_dementia.csv', 'CTD'),
    ],
    "ALS": [
        ('01_Cleaned_Data/positive_drugs_als.csv', 'CTD'),
    ],
    "Bipolar Disorder": [
        ('01_Cleaned_Data/positive_drugs_bipolar.csv', 'CTD'),
    ],
    "Parkinson's Disease": [
        ('01_Cleaned_Data/positive_drugs_parkinsons.csv', 'CTD'),
    ],
    "ADHD": [
        ('01_Cleaned_Data/positive_drugs_adhd.csv', 'CTD'),
    ],
}

# ── Protein-Disease association weights ───────────────────────────────────────
# Derived from CTD drug_links files — score = frequency-normalised across drugs.
# Only used if 00_Raw_Data/protein_disease_weights.csv does NOT exist.

FALLBACK_WEIGHTS = [

    # ── Alzheimer's Disease (from drug_links_ad.csv) ──────────────────────────
    ("CASP3",    "Alzheimer's Disease", 0.99),
    ("TNF",      "Alzheimer's Disease", 0.98),
    ("IL1B",     "Alzheimer's Disease", 0.94),
    ("BCL2",     "Alzheimer's Disease", 0.91),
    ("APP",      "Alzheimer's Disease", 0.90),
    ("BAX",      "Alzheimer's Disease", 0.89),
    ("HMOX1",    "Alzheimer's Disease", 0.84),
    ("ACHE",     "Alzheimer's Disease", 0.83),
    ("SOD2",     "Alzheimer's Disease", 0.78),
    ("VEGFA",    "Alzheimer's Disease", 0.78),
    ("PPARG",    "Alzheimer's Disease", 0.75),
    ("BDNF",     "Alzheimer's Disease", 0.73),
    ("GSK3B",    "Alzheimer's Disease", 0.73),
    ("ESR1",     "Alzheimer's Disease", 0.68),
    ("NOS3",     "Alzheimer's Disease", 0.68),
    ("LEP",      "Alzheimer's Disease", 0.66),
    ("BCHE",     "Alzheimer's Disease", 0.65),
    ("IGF1",     "Alzheimer's Disease", 0.65),
    ("MPO",      "Alzheimer's Disease", 0.65),
    ("BACE1",    "Alzheimer's Disease", 0.64),
    ("INS",      "Alzheimer's Disease", 0.64),
    ("PLAU",     "Alzheimer's Disease", 0.63),
    ("SLC2A4",   "Alzheimer's Disease", 0.63),
    ("ENO1",     "Alzheimer's Disease", 0.63),
    ("CYP2D6",   "Alzheimer's Disease", 0.63),
    ("ACE",      "Alzheimer's Disease", 0.61),
    ("MAPT",     "Alzheimer's Disease", 0.61),
    ("APOE",     "Alzheimer's Disease", 0.61),
    ("CRH",      "Alzheimer's Disease", 0.60),
    ("TF",       "Alzheimer's Disease", 0.59),
    ("ATP5F1A",  "Alzheimer's Disease", 0.59),
    ("ARC",      "Alzheimer's Disease", 0.59),
    ("IGF1R",    "Alzheimer's Disease", 0.59),
    ("PSEN1",    "Alzheimer's Disease", 0.59),
    ("TFAM",     "Alzheimer's Disease", 0.59),
    ("INSR",     "Alzheimer's Disease", 0.59),
    ("CLU",      "Alzheimer's Disease", 0.58),
    ("CHRNB2",   "Alzheimer's Disease", 0.58),
    ("DHCR24",   "Alzheimer's Disease", 0.58),
    ("NPY",      "Alzheimer's Disease", 0.58),
    ("MAOB",     "Alzheimer's Disease", 0.57),
    ("TPI1",     "Alzheimer's Disease", 0.57),
    ("F2",       "Alzheimer's Disease", 0.57),
    ("EIF2S1",   "Alzheimer's Disease", 0.57),
    ("IDE",      "Alzheimer's Disease", 0.56),
    ("APOC1",    "Alzheimer's Disease", 0.56),
    ("IGF2",     "Alzheimer's Disease", 0.56),
    ("CHRNA7",   "Alzheimer's Disease", 0.56),
    ("CST3",     "Alzheimer's Disease", 0.56),
    ("RELN",     "Alzheimer's Disease", 0.56),
    ("ADAMTS1",  "Alzheimer's Disease", 0.55),
    ("CALM1",    "Alzheimer's Disease", 0.53),
    ("A2M",      "Alzheimer's Disease", 0.53),
    ("IGF2R",    "Alzheimer's Disease", 0.53),
    ("IQCK",     "Alzheimer's Disease", 0.53),
    ("PYY",      "Alzheimer's Disease", 0.53),
    ("NCSTN",    "Alzheimer's Disease", 0.53),
    ("AMFR",     "Alzheimer's Disease", 0.53),
    ("TREM2",    "Alzheimer's Disease", 0.53),
    ("VSNL1",    "Alzheimer's Disease", 0.53),
    ("ADAM10",   "Alzheimer's Disease", 0.52),
    ("WWOX",     "Alzheimer's Disease", 0.52),
    ("TOMM40",   "Alzheimer's Disease", 0.52),
    ("PTK2B",    "Alzheimer's Disease", 0.52),
    ("PSEN2",    "Alzheimer's Disease", 0.52),
    ("PICALM",   "Alzheimer's Disease", 0.52),
    ("GAPDHS",   "Alzheimer's Disease", 0.52),
    ("INPP5D",   "Alzheimer's Disease", 0.52),
    ("EPHA1",    "Alzheimer's Disease", 0.52),
    ("PGRMC1",   "Alzheimer's Disease", 0.52),
    ("MTHFR",    "Alzheimer's Disease", 0.52),
    ("PRNP",     "Alzheimer's Disease", 0.52),
    ("ABI3",     "Alzheimer's Disease", 0.51),
    ("CD2AP",    "Alzheimer's Disease", 0.51),
    ("ABCA7",    "Alzheimer's Disease", 0.51),
    ("CD33",     "Alzheimer's Disease", 0.51),
    ("DPYSL2",   "Alzheimer's Disease", 0.51),
    ("HLA-DRB5", "Alzheimer's Disease", 0.51),
    ("HFE",      "Alzheimer's Disease", 0.51),
    ("CYP46A1",  "Alzheimer's Disease", 0.51),
    ("NECTIN2",  "Alzheimer's Disease", 0.51),
    ("PILRA",    "Alzheimer's Disease", 0.51),
    ("PLCG2",    "Alzheimer's Disease", 0.51),
    ("SLC30A6",  "Alzheimer's Disease", 0.51),
    ("TPP1",     "Alzheimer's Disease", 0.51),

    # ── ADHD (from drug_links_adhd.csv) ───────────────────────────────────────
    ("DRD2",   "ADHD", 0.99),
    ("SLC6A3", "ADHD", 0.87),
    ("COMT",   "ADHD", 0.81),
    ("CNR1",   "ADHD", 0.74),
    ("TPH2",   "ADHD", 0.68),
    ("CHRNB2", "ADHD", 0.62),
    ("CHRNA4", "ADHD", 0.62),
    ("AS3MT",  "ADHD", 0.56),
    ("CIC",    "ADHD", 0.56),
    ("DRD4",   "ADHD", 0.56),
    ("GIT1",   "ADHD", 0.56),
    ("FGD1",   "ADHD", 0.56),
    ("GRM1",   "ADHD", 0.56),
    ("STS",    "ADHD", 0.56),

    # ── Parkinson's Disease (from drug_links_parkinsons.csv) ──────────────────
    ("TNF",      "Parkinson's Disease", 0.99),
    ("HMOX1",    "Parkinson's Disease", 0.99),
    ("DRD2",     "Parkinson's Disease", 0.88),
    ("IL6",      "Parkinson's Disease", 0.88),
    ("BDNF",     "Parkinson's Disease", 0.88),
    ("SOD1",     "Parkinson's Disease", 0.88),
    ("SOD2",     "Parkinson's Disease", 0.81),
    ("NQO1",     "Parkinson's Disease", 0.79),
    ("DRD1",     "Parkinson's Disease", 0.79),
    ("ABCB1",    "Parkinson's Disease", 0.79),
    ("CYP2E1",   "Parkinson's Disease", 0.79),
    ("TH",       "Parkinson's Disease", 0.79),
    ("SNCA",     "Parkinson's Disease", 0.79),
    ("PPARGC1A", "Parkinson's Disease", 0.77),
    ("CYP2D6",   "Parkinson's Disease", 0.72),
    ("NGF",      "Parkinson's Disease", 0.70),
    ("INS",      "Parkinson's Disease", 0.70),
    ("GSTP1",    "Parkinson's Disease", 0.68),
    ("MAOB",     "Parkinson's Disease", 0.66),
    ("MAOA",     "Parkinson's Disease", 0.66),
    ("GFAP",     "Parkinson's Disease", 0.66),
    ("GDNF",     "Parkinson's Disease", 0.66),
    ("PRKN",     "Parkinson's Disease", 0.66),
    ("SLC6A3",   "Parkinson's Disease", 0.66),
    ("GSTM1",    "Parkinson's Disease", 0.63),
    ("PARK7",    "Parkinson's Disease", 0.63),
    ("DNM1L",    "Parkinson's Disease", 0.61),
    ("DDC",      "Parkinson's Disease", 0.61),
    ("SLC38A2",  "Parkinson's Disease", 0.61),
    ("PINK1",    "Parkinson's Disease", 0.61),
    ("SLC18A2",  "Parkinson's Disease", 0.59),
    ("MAP3K5",   "Parkinson's Disease", 0.59),
    ("NECTIN2",  "Parkinson's Disease", 0.59),
    ("IGF2",     "Parkinson's Disease", 0.59),
    ("HSPA9",    "Parkinson's Disease", 0.59),
    ("HSPA1A",   "Parkinson's Disease", 0.59),
    ("HGF",      "Parkinson's Disease", 0.59),
    ("EDN1",     "Parkinson's Disease", 0.59),
    ("CP",       "Parkinson's Disease", 0.59),
    ("INSR",     "Parkinson's Disease", 0.59),
    ("MAPT",     "Parkinson's Disease", 0.59),
    ("TRPM2",    "Parkinson's Disease", 0.57),
    ("IGF2R",    "Parkinson's Disease", 0.57),
    ("IGF1R",    "Parkinson's Disease", 0.57),
    ("MTA1",     "Parkinson's Disease", 0.57),
    ("AIF1",     "Parkinson's Disease", 0.57),
    ("FGB",      "Parkinson's Disease", 0.57),
    ("ENO2",     "Parkinson's Disease", 0.57),
    ("LRRK2",    "Parkinson's Disease", 0.57),
    ("MAP2",     "Parkinson's Disease", 0.57),
    ("RPL6",     "Parkinson's Disease", 0.54),
    ("MTHFR",    "Parkinson's Disease", 0.54),
    ("DDIT4",    "Parkinson's Disease", 0.54),
    ("VPS35",    "Parkinson's Disease", 0.54),
    ("RPS8",     "Parkinson's Disease", 0.54),
    ("SLC2A14",  "Parkinson's Disease", 0.54),
    ("GSTA4",    "Parkinson's Disease", 0.54),
    ("FBP1",     "Parkinson's Disease", 0.54),
    ("ALDH2",    "Parkinson's Disease", 0.52),
    ("ADARB2",   "Parkinson's Disease", 0.52),
    ("BAG5",     "Parkinson's Disease", 0.52),
    ("CEACAM6",  "Parkinson's Disease", 0.52),
    ("BST1",     "Parkinson's Disease", 0.52),
    ("MAG",      "Parkinson's Disease", 0.52),
    ("HLA-DRA",  "Parkinson's Disease", 0.52),
    ("FCER2",    "Parkinson's Disease", 0.52),
    ("HBG1",     "Parkinson's Disease", 0.52),
    ("HFE",      "Parkinson's Disease", 0.52),
    ("RPL14",    "Parkinson's Disease", 0.52),
    ("RAB32",    "Parkinson's Disease", 0.52),
    ("NCAPG2",   "Parkinson's Disease", 0.52),
    ("NOS1",     "Parkinson's Disease", 0.52),
    ("SLC30A10", "Parkinson's Disease", 0.52),
    ("TALDO1",   "Parkinson's Disease", 0.52),
    ("TCL1B",    "Parkinson's Disease", 0.52),
    ("TMEM230",  "Parkinson's Disease", 0.52),

    # ── ALS ───────────────────────────────────────────────────────────────────
    ("SOD1",     "ALS", 0.99),
    ("TARDBP",   "ALS", 0.97),
    ("FUS",      "ALS", 0.95),
    ("C9ORF72",  "ALS", 0.93),
    ("OPTN",     "ALS", 0.88),
    ("TBK1",     "ALS", 0.85),
    ("NEK1",     "ALS", 0.82),
    ("NFE2L2",   "ALS", 0.75),
    ("SOD2",     "ALS", 0.72),
    ("PPARGC1A", "ALS", 0.70),
    ("TNF",      "ALS", 0.65),
    ("GFAP",     "ALS", 0.63),
    ("GSR",      "ALS", 0.60),
    ("GSTP1",    "ALS", 0.58),
    ("MFN1",     "ALS", 0.56),
    ("MFN2",     "ALS", 0.56),
    ("NRF1",     "ALS", 0.54),
    ("PTGS2",    "ALS", 0.52),
    ("SQSTM1",   "ALS", 0.52),
    ("UBB",      "ALS", 0.51),

    # ── Bipolar Disorder ──────────────────────────────────────────────────────
    ("ANK3",    "Bipolar Disorder", 0.92),
    ("CACNA1C", "Bipolar Disorder", 0.90),
    ("BDNF",    "Bipolar Disorder", 0.88),
    ("GSK3B",   "Bipolar Disorder", 0.85),
    ("SLC6A4",  "Bipolar Disorder", 0.82),
    ("DRD2",    "Bipolar Disorder", 0.80),
    ("HTR2A",   "Bipolar Disorder", 0.78),
    ("COMT",    "Bipolar Disorder", 0.75),
    ("TPH2",    "Bipolar Disorder", 0.72),
    ("NRG1",    "Bipolar Disorder", 0.70),
    ("DRD1",    "Bipolar Disorder", 0.68),
    ("GAD1",    "Bipolar Disorder", 0.65),
    ("GAD2",    "Bipolar Disorder", 0.63),
    ("GRIK2",   "Bipolar Disorder", 0.61),
    ("GRIN2A",  "Bipolar Disorder", 0.59),
    ("INS",     "Bipolar Disorder", 0.57),
    ("NCAN",    "Bipolar Disorder", 0.55),
    ("PDE4B",   "Bipolar Disorder", 0.53),
    ("PVALB",   "Bipolar Disorder", 0.51),
    ("RELN",    "Bipolar Disorder", 0.51),
    ("SNAP25",  "Bipolar Disorder", 0.51),
    ("TAC1",    "Bipolar Disorder", 0.51),

    # ── Dementia ──────────────────────────────────────────────────────────────
    ("APOE",   "Dementia", 0.95),
    ("MAPT",   "Dementia", 0.92),
    ("GRN",    "Dementia", 0.88),
    ("APP",    "Dementia", 0.85),
    ("PSEN1",  "Dementia", 0.83),
    ("PSEN2",  "Dementia", 0.80),
    ("TNF",    "Dementia", 0.70),
    ("IL6",    "Dementia", 0.68),
    ("BDNF",   "Dementia", 0.65),
    ("CASP3",  "Dementia", 0.62),
    ("HMOX1",  "Dementia", 0.60),
    ("GSK3B",  "Dementia", 0.58),
    ("BCL2",   "Dementia", 0.55),
    ("BAX",    "Dementia", 0.53),
    ("SOD2",   "Dementia", 0.51),
    ("ACHE",   "Dementia", 0.50),
    ("BCHE",   "Dementia", 0.50),
    ("IGF1",   "Dementia", 0.49),
    ("VEGFA",  "Dementia", 0.48),
    ("NOS3",   "Dementia", 0.47),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def expand_graph():
    print("--- Expanding Heterogeneous Graph ---")

    for req in ['01_Cleaned_Data/master_graph.pt', '01_Cleaned_Data/mappings.pt']:
        if not os.path.exists(req):
            print(f"Error: {req} not found. Run 03_build_hetero_graph.py first.")
            return

    data = torch.load('01_Cleaned_Data/master_graph.pt', weights_only=False)
    maps = torch.load('01_Cleaned_Data/mappings.pt',     weights_only=False)

    d_map        = maps['d_map']
    p_map        = maps['p_map']
    drug_names   = maps['drug_names']
    all_proteins = maps['all_proteins']

    # ── 1. Disease nodes ──────────────────────────────────────────────────────
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
    print(f"  Disease nodes ({len(diseases)}): {diseases}")

    # ── 2. Protein-Disease edges ──────────────────────────────────────────────
    disgenet_path = '00_Raw_Data/protein_disease_weights.csv'
    if os.path.exists(disgenet_path):
        pd_df = pd.read_csv(disgenet_path)
        pd_df['disease_name'] = pd_df['disease_name'].map(normalize_disease_name)
        assoc_list = list(zip(
            pd_df['gene_symbol'],
            pd_df['disease_name'],
            pd_df['score'].astype(float)
        ))
        print(f"  Loaded {len(assoc_list)} protein-disease associations from DisGeNET")
    else:
        assoc_list = FALLBACK_WEIGHTS
        print(f"  Using CTD-derived protein-disease weights ({len(assoc_list)} entries)")

    p_src, d_dst, pd_weights = [], [], []
    skipped_pd = []
    for protein, disease, score in assoc_list:
        if protein in p_map and disease in dis_map:
            p_src.append(p_map[protein])
            d_dst.append(dis_map[disease])
            pd_weights.append(float(score))
        else:
            skipped_pd.append((protein, disease))

    if p_src:
        data['protein', 'associated_with', 'disease'].edge_index = torch.tensor(
            [p_src, d_dst], dtype=torch.long)
        data['protein', 'associated_with', 'disease'].edge_attr = torch.tensor(
            pd_weights, dtype=torch.float).view(-1, 1)
        print(f"  Protein-disease edges: {len(p_src)}")
    if skipped_pd:
        print(f"  Skipped {len(skipped_pd)} protein-disease pairs "
              f"(re-run 02_fetch_string_interactions.py to add them): "
              f"{skipped_pd[:3]} ...")

    # ── 3. Drug-Treats-Disease edges (ALL DISEASES) ───────────────────────────
    print("\n  Building drug-treats-disease edges across all diseases...")

    edge_weight_map: dict = {}

    for disease_name, file_list in DISEASE_DRUG_FILES.items():
        if disease_name not in dis_map:
            print(f"  SKIP: '{disease_name}' not in dis_map")
            continue

        dis_idx  = dis_map[disease_name]
        n_added  = 0

        for csv_path, evidence_tier in file_list:
            if not os.path.exists(csv_path):
                print(f"  WARNING: {csv_path} not found — skipping "
                      f"{disease_name} ({evidence_tier})")
                continue

            df = pd.read_csv(csv_path)

            if 'name' in df.columns:
                name_col = 'name'
            elif 'Drug Name/Treatment' in df.columns:
                name_col = 'Drug Name/Treatment'
            else:
                name_col = df.columns[0]

            weight    = EVIDENCE_WEIGHTS.get(evidence_tier,
                                             EVIDENCE_WEIGHTS['Preclinical'])
            has_label = 'label' in df.columns

            for _, row in df.iterrows():
                if has_label and int(row['label']) != 1:
                    continue
                d_name = str(row[name_col]).strip()
                if d_name not in d_map:
                    continue
                key = (d_map[d_name], dis_idx)
                if key not in edge_weight_map or edge_weight_map[key] < weight:
                    edge_weight_map[key] = weight
                    n_added += 1

        print(f"  {disease_name:<25} → {n_added} edges "
              f"(weight={EVIDENCE_WEIGHTS.get(evidence_tier, 0.70):.2f})")

    if edge_weight_map:
        d_src_list = [k[0] for k in edge_weight_map]
        d_dst_list = [k[1] for k in edge_weight_map]
        d_w_list   = [edge_weight_map[k] for k in edge_weight_map]

        data['drug', 'treats', 'disease'].edge_index = torch.tensor(
            [d_src_list, d_dst_list], dtype=torch.long)
        data['drug', 'treats', 'disease'].edge_attr = torch.tensor(
            d_w_list, dtype=torch.float).view(-1, 1)

        total_edges    = len(edge_weight_map)
        idx_to_disease = {v: k for k, v in dis_map.items()}
        counts         = Counter(k[1] for k in edge_weight_map)

        print(f"\n  Total drug-treats-disease edges: {total_edges}")
        for dis_idx_c, count in sorted(counts.items()):
            print(f"    {idx_to_disease[dis_idx_c]:<25} {count} edges")
    else:
        print("  WARNING: Zero drug-treats-disease edges created.")

    # ── 4. Hard Negative Edges ────────────────────────────────────────────────
    print("\n  Building hard negative edges from failed AD trials...")
    hard_neg_src, hard_neg_dst = [], []
    ad_idx         = dis_map["Alzheimer's Disease"]
    missing_failed = []

    for drug_name in FAILED_AD_DRUGS:
        if drug_name in d_map:
            hard_neg_src.append(d_map[drug_name])
            hard_neg_dst.append(ad_idx)
            print(f"    ✓ {drug_name}")
        else:
            missing_failed.append(drug_name)
            print(f"    ✗ {drug_name} — not in graph")

    if hard_neg_src:
        data['drug', 'failed', 'disease'].edge_index = torch.tensor(
            [hard_neg_src, hard_neg_dst], dtype=torch.long)
        data['drug', 'failed', 'disease'].edge_attr = torch.zeros(
            len(hard_neg_src), 1)
        print(f"  Hard negative edges: {len(hard_neg_src)}")
    else:
        print("  WARNING: No hard negative edges created.")

    if missing_failed:
        print(f"\n  Missing hard negatives (add SMILES to drugs_raw_augmented.csv):")
        for name in missing_failed:
            print(f"    - {name}")

    # ── 5. Add reverse edges ──────────────────────────────────────────────────
    data = T.ToUndirected()(data)
    rev_key = ('disease', 'rev_failed', 'drug')
    if rev_key in data.edge_types:
        del data[rev_key]
        print("  Removed spurious rev_failed reverse edge type.")

    # ── 6. Validation checks ──────────────────────────────────────────────────
    print("\n  Running index validation checks...")
    n_drugs    = data['drug'].x.shape[0]
    n_proteins = data['protein'].x.shape[0]
    n_diseases = data['disease'].x.shape[0]

    ei_dp = data['drug', 'binds', 'protein'].edge_index
    assert ei_dp[0].max().item() < n_drugs
    assert ei_dp[1].max().item() < n_proteins

    if ('drug', 'treats', 'disease') in data.edge_types:
        ei_dd = data['drug', 'treats', 'disease'].edge_index
        assert ei_dd[0].max().item() < n_drugs
        assert ei_dd[1].max().item() < n_diseases
        print(f"  ✓ treats edge index valid")

    if ('drug', 'failed', 'disease') in data.edge_types:
        ei_fn = data['drug', 'failed', 'disease'].edge_index
        assert ei_fn[0].max().item() < n_drugs
        assert ei_fn[1].max().item() < n_diseases
        print(f"  ✓ failed edge index valid")

    print(f"  ✓ index check passed: {n_drugs} drugs, "
          f"{n_proteins} proteins, {n_diseases} diseases")

    # ── 7. Summary & Save ─────────────────────────────────────────────────────
    print("\nExpanded Graph Summary:")
    print(data)

    torch.save(data, '01_Cleaned_Data/expanded_graph.pt')
    print("\nSaved → 01_Cleaned_Data/expanded_graph.pt")

    maps = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)
    maps['dis_map'] = dis_map
    torch.save(maps, '01_Cleaned_Data/mappings.pt')
    print(f"  Updated mappings.pt with dis_map: {dis_map}")

    n_pos = len(edge_weight_map) if edge_weight_map else 0
    n_neg = len(hard_neg_src)
    print(f"\n  Training signal summary:")
    print(f"    Positive edges (drug treats disease) : {n_pos}")
    print(f"    Hard negative edges (failed trials)  : {n_neg}")
    print(f"    Random negatives (sampled at train)  : ~{n_pos * 2}")
    print(f"    Total training signal                : ~{n_pos + n_neg + n_pos * 2}")


if __name__ == "__main__":
    expand_graph()

