# Drug Discovery & Repurposing GNN: Targeting Alzheimer's Disease

**Alzheimer's disease (AD)** is a complex neurodegenerative disorder characterized by complicated **protein-protein interactions (PPI)**. While it takes years to develop new treatments, computational bioinformatics offers a localized path to repurpose existing drugs by predicting their effects on specific pathological pathways. In this study, we developed a 3-layer **Heterogeneous Graph Convolutional Network (GCN)** to model the AD interactome. The structure includes three distinct layers: drug-protein interactions, protein-protein interaction (PPI) networks, and protein-disease correlations. By looking at all these layers together, the model can identify the optimal drug candidates capable of interacting with specific protein pathways, utimately treating the disease. The results confirm the model's ability to learn complex biological graphs with high precision, providing a framework for future drug discovery for other neurodegenerative diseases.  

## Network Visualization
The model operates on a complex biological network connecting drugs, proteins, and diseases. Below is a visual representation of the core network (Created via Cytoscape):

<iframe src="file:///Users/kellychang/Downloads/web_session/index.html#/" width="600" height="400"></iframe>

## Interactive Demo
Try the model live without installations -- 

Download file: Drug_Discovery_GCN_Demo.ipynb 
Open Google Colab or Jupyter Notebook: Import file

## Getting Started

### Prerequisites
- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/) (Recommended)

### Setup
Run the provided setup script to create the environment and install dependencies:
```bash
chmod +x setup_m2.sh
./setup_m2.sh
conda activate drug_discovery_gcn
```

Alternatively, install via pip:
```bash
pip install torch torch-geometric pandas rdkit scikit-learn matplotlib networkx pubchempy
```

# Drug-Disease Interaction GNN — Project Structure

**ISEF Science Fair Project**
Predicting therapeutic drug-disease interactions across 6 cognitive diseases using a Heterogeneous Graph Neural Network.

---

## Repository Layout

```
drug-disease-interaction/
│
├── 00_Raw_Data/                         # Original source data
│   ├── drugs_raw.csv                    # 67 original drugs with Numerical_Vector features
│   ├── drugs_raw_augmented.csv          # 160 drugs after CTD merge + SMILES filled
│   ├── drug_links.csv                   # Original AD drug-protein links (CTD inference scores)
│   ├── drug_links_als.csv               # ALS drug-protein links
│   ├── drug_links_bipolar.csv           # Bipolar drug-protein links
│   ├── drug_links_dementia.csv          # Dementia drug-protein links
│   ├── positive_drugs_ctd.csv           # 58 AD therapeutic drugs (cleaned)
│   ├── positive_drugs_als.csv           # 20 ALS therapeutic drugs (cleaned)
│   ├── positive_drugs_bipolar.csv       # 27 Bipolar therapeutic drugs (cleaned)
│   ├── positive_drugs_dementia.csv      # 69 Dementia therapeutic drugs (cleaned)
│   ├── negative_controls.csv            # Non-CNS reference drugs
│   └── protein_disease_weights.csv      # DisGeNET DPI scores (180 entries, 6 diseases)
│
├── 01_Cleaned_Data/                     # Processed data & trained model files
│   ├── ppi_interactions.csv             # STRING PPI fetch output (96 proteins, score ≥ 400)
│   ├── drug_links.csv                   # Merged drug-protein links across all 4 disease files
│   ├── positive_drugs.csv               # Merged positive drugs across all 4 disease files
│   ├── negative_controls.csv            # Cleaned negative controls with SMILES
│   ├── master_graph.pt                  # Drug + protein nodes + binding/PPI edges
│   ├── expanded_graph.pt                # Full graph with 6 disease nodes
│   ├── mappings.pt                      # d_map, p_map, dis_map, drug_names, all_proteins
│   ├── gnn_model_best.pt                # Best validation checkpoint (use for inference)
│   ├── predictor_best.pt                # Best validation checkpoint (use for inference)
│   ├── gnn_model.pt                     # Final epoch weights
│   └── predictor.pt                     # Final epoch weights
│
├── 02_Code/                             # All pipeline scripts
│   ├── 01_clean_drugs.py                # Fetch SMILES from PubChem; merge all disease files
│   ├── 02_fetch_string_interactions.py  # Fetch PPI from STRING API (96 proteins, score ≥ 400)
│   ├── 03_build_hetero_graph.py         # Build master_graph.pt from drugs + proteins
│   ├── 04_expand_graph.py               # Add 6 disease nodes, DisGeNET weights, treats edges
│   ├── 04a_inject_ctd_drug_names.py     # Merge all 4 CTD disease files → drugs_raw_augmented.csv
│   ├── 05_validate_graph.py             # Print graph statistics and edge counts
│   ├── 05b_kfold_eval.py                # 5-fold cross-validation → kfold_results.txt
│   ├── 06_train_gcn.py                  # Train HeteroGNN with SAGEConv architecture
│   ├── 07_inference.py                  # predict(drug_name) → sigmoid probability per disease
│   ├── 08_visualize_graph.py            # Community-layout network PNG + Cytoscape GraphML
│   ├── 09_results_validation.py         # Full results report: metric/dummy/kfold/discovery tests
│   └── featurizer.py                    # DrugFeaturizer helper (Morgan fingerprints)
│
├── 99_ISEF_Docs/                        # Output reports for paper and poster
│   ├── kfold_results.txt                # 5-fold AUC per fold + mean ± std
│   ├── results_validation.txt           # Full 4-section results report
│   └── discovery_candidates.csv        # Ranked repurposing candidates with scores
│          
├── network_visualization.graphml        # Cytoscape import file
├── Drug_Discovery_GNN_Demo.ipynb        # Google Colab live demo notebook
└── README.md
```

---

## Pipeline Run Order

Run these scripts in sequence from the project root directory.

### First-time setup (data collection)

```bash
# Step 1 — Merge CTD drug names from all 4 disease files into augmented drug list
python3 02_Code/04a_inject_ctd_drug_names.py

# Step 2 — Fetch SMILES from PubChem + merge all drug_links and positive_drugs files
python3 02_Code/01_clean_drugs.py

# Step 3 — Fetch protein-protein interactions from STRING (score ≥ 400)
python3 02_Code/02_fetch_string_interactions.py
```

### Graph construction

```bash
# Step 4 — Build heterogeneous graph (drug + protein nodes, binding + PPI edges)
python3 02_Code/03_build_hetero_graph.py

# Step 5 — Add 6 disease nodes with DisGeNET weights and drug-treats-disease edges
python3 02_Code/04_expand_graph.py

# Step 6 — Verify graph statistics
python3 02_Code/05_validate_graph.py
```

### Training & evaluation

```bash
# Step 7 — Train the GNN (saves gnn_model_best.pt + predictor_best.pt)
python3 02_Code/06_train_gcn.py

# Step 8 — 5-fold cross-validation
python3 02_Code/05b_kfold_eval.py
```

### Results & visualization

```bash
# Step 9 — Full results validation report (metric / dummy / kfold / discovery)
python3 02_Code/09_results_validation.py

# Step 10 — Network visualization PNG + Cytoscape GraphML
python3 02_Code/08_visualize_graph.py

# Single drug inference
python3 02_Code/07_inference.py "Metformin"
python3 02_Code/07_inference.py "Donepezil"
```

---

## Graph Statistics (Current Model)

| Component | Value |
|-----------|-------|
| Drug nodes | 160 |
| Protein nodes | 96 |
| Disease nodes | 6 |
| Drug→protein binding edges | 770 |
| Protein→protein interaction edges | 1058 (bidirectional) |
| Protein→disease association edges | 79 |
| Drug→treats→disease edges | 105 |

### Diseases modeled
| Disease | Index | Source |
|---------|-------|--------|
| Alzheimer's Disease | 0 | CTD + DisGeNET |
| Parkinson's Disease | 1 | DisGeNET |
| ADHD | 2 | DisGeNET |
| Bipolar Disorder | 3 | CTD + DisGeNET |
| ALS | 4 | CTD + DisGeNET |
| Dementia | 5 | CTD + DisGeNET |

---

## Model Architecture

```
Input features
  Drug    : 2048-dim Morgan fingerprint (radius=2, RDKit)
  Protein : 96-dim identity matrix
  Disease : 6-dim identity matrix

HeteroGNN (3 message-passing layers)
  Conv type  : SAGEConv (mean aggregation)
  Hidden dim : 64
  Norm       : BatchNorm1d + residual connections
  Dropout    : 0.3

LinkPredictor MLP
  Input  : concat(drug_emb, disease_emb) → 128-dim
  Layers : Linear(128→64) → BatchNorm → ReLU → Dropout
           Linear(64→32)  → BatchNorm → ReLU
           Linear(32→1)   → sigmoid

Training
  Split      : RandomLinkSplit 70/20/10
  Loss       : BCEWithLogitsLoss, pos_weight=2.0
  Optimizer  : Adam, lr=3e-4
  Epochs     : 200, best checkpoint saved
```

---

## Validated Results

| Metric | Value |
|--------|-------|
| 5-fold cross-validation AUC | 0.9966 ± 0.0012 |
| Approved AD drugs mean score | ~0.44 |
| Non-CNS reference drugs mean score | ~0.39 |
| Mann-Whitney U p-value | p = 0.004 |
| Perfect separation | YES (all approved drugs rank above all negatives) |
| Dummy test (10/10 off-pathway drugs) | All scored below approved-drug mean |

> **Note on AUC:** 0.9966 is near-perfect and likely reflects some overfitting given the small dataset (105 positive edges). The model's discrimination ability is genuine — evidenced by perfect separation on the metric test — but predictions should be validated against independent clinical trial data before biological conclusions are drawn.

---

## Score Interpretation

Scores are compressed below 0.5 due to `pos_weight=2.0` training. The meaningful measure is relative ranking, not absolute value.

| Score | Interpretation |
|-------|---------------|
| ≥ 0.43 | 🟢 High Potential |
| 0.40 – 0.43 | 🟡 Moderate Potential |
| < 0.40 | 🔴 Low / No Predicted Correlation |

---

## Key Data Sources

| Source | Usage |
|--------|-------|
| CTD (Comparative Toxicogenomics Database) | Drug-disease therapeutic associations + drug-protein inference networks |
| STRING (v11) | Protein-protein interaction network (score ≥ 400) |
| DisGeNET | Protein-disease association scores (DPI weights) |
| PubChem | SMILES strings for drug featurization |
| RDKit | Morgan fingerprint generation (2048-bit, radius=2) |

---

## Data Cleaning Decisions

The following were removed from all CTD files (not featurizable as single molecules):

- Plant Preparations, Plant Extracts, Biological Products
- Cholinesterase Inhibitors, Antipsychotic Agents, Drugs Chinese Herbal
- Androgens, Lecithins, Heparin Low-Molecular-Weight
- Ginkgo biloba extract, Kai-Xin-San, Anti-Inflammatory Agents Non-Steroidal
- Nanotubes Carbon, DP 155

Name corrections applied (CTD name → PubChem canonical):

| Original | Corrected |
|----------|-----------|
| Thioctic Acid | Lipoic acid |
| Vitamin E | Tocopherol |
| Vitamin D | Cholecalciferol |
| Raloxifene Hydrochloride | Raloxifene |
| Quetiapine Fumarate | Quetiapine |
| Lithium Chloride / Lithium carbonate | Lithium |
| Acetylcysteine | N-Acetylcysteine |
| kenpaullone | Kenpaullone |
| 2-(4-morpholino)ethyl-1-phenylcyclohexane-1-carboxylate | PRE-084 |
| BMS 708163 | Avagacestat |
| Long IUPAC string | Semagacestat |
| 3-methyl-5-(1-methyl-2-pyrrolidinyl)isoxazole | ABT-418 |
| ginsenoside Rg1 | Ginsenoside Rg1 |
