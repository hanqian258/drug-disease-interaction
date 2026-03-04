# Technical Status Report: The Great Data Harvest

## Progress Summary

### Environment & Repo
- **Status:** Completed
- **Details:** Initialized the ISEF-standard directory structure: `00_Raw_Data`, `01_Cleaned_Data`, `02_Code`, `03_Literature`, and `99_ISEF_Docs`. Verified repository organization for scientific reproducibility.

### Chemical Informatics (The SMILES Script)
- **Status:** Completed
- **Details:** Implemented `02_Code/01_clean_drugs.py` using `pubchempy` to fetch missing Canonical SMILES from the PubChem API. Integrated this into the pipeline to handle new drug lists automatically.
- **Data:** Successfully processed 5 positive Alzheimer's drugs and 5 negative controls.

### Graph Node Definition (The Protein List)
- **Status:** Completed
- **Details:**
  - Defined a high-confidence subnetwork with 11 key proteins: `MAPT`, `APP`, `APOE`, `BACE1`, `PSEN1`, `PSEN2`, `TREM2`, `CLU`, `PICALM`, `CR1`, and `GSK3B`.
  - Implemented `02_Code/02_fetch_string_interactions.py` to retrieve interaction data from the STRING DB API with a `required_score` of 700.
  - Successfully fetched 33 high-confidence PPI edges.

### Biological Validation
- **Status:** Completed
- **Details:**
  - Verified presence of "Positive Controls" (Known Alzheimer's drugs) in the dataset.
  - Confirmed that critical interactions `APP-BACE1` and `MAPT-GSK3B` are present in the ground truth with a confidence score of 0.999.

### Technical Stack
- **Status:** Completed
- **Details:** Configured environment with `PyTorch 2.10`, `PyTorch Geometric`, `RDKit`, and `Pandas`.

---

## Code Snippet: Data Cleaning Function

```python
import pubchempy as pcp

def get_smiles(drug_name):
    try:
        results = pcp.get_compounds(drug_name, 'name')
        if results:
            return results[0].canonical_smiles
    except Exception as e:
        print(f"Error fetching SMILES for {drug_name}: {e}")
    return None
```

---

## Technical Progress: Graph Expansion & GNN Pipeline

### Graph Expansion (The Disease Node)
- **Status:** Completed
- **Details:**
  - Implemented `02_Code/04_expand_graph.py` to transition from node classification to link prediction.
  - Added 3 disease nodes: `Alzheimer's Disease`, `Parkinson's Disease`, and `Frontotemporal Dementia`.
  - Added `Levodopa` as a new drug node to serve as a biological control for PD.
  - Manually mapped 11 proteins and 6 drugs to diseases based on biological consensus to ensure specific pathway biases (e.g., MAPT/Tau bias).
  - Automatically generated reverse edges for all 4 primary edge types to support deep message passing.

### GNN Model Architecture (HeteroGNN v2)
- **Status:** Refactored
- **Details:**
  - Implemented an enhanced `HeteroGNN` in `02_Code/05_train_gcn.py` with:
    - **Initial Projections**: Linear layers project all heterogeneous node features (drug, protein, disease) to a common 64-dimensional latent space.
    - **Non-Linearities & Dropout**: Integrated ReLU activation and 0.2 Dropout after the projection to improve robustness and prevent overfitting.
    - **Lazy Initialization**: Leveraged PyG's dynamic input dimension inference (`-1`) for all layers.
    - **Separate Edge weights**: 3-layer `HeteroConv` now uses distinct `SAGEConv` kernels for each relationship type (forward and reverse).
- **Results:** Successfully executed the first training run. The model achieves rapid loss reduction (Loss: 0.0000 at Epoch 100), demonstrating high capacity for learning the training graph.

---

## Scientific Audit & Next Steps (Tau Focus)

### Audit Findings
- **Status:** Evaluated for ISEF Scientific Rigor.
- **Details:**
    - Pipeline is functionally sound but requires a dedicated `test` split (currently `num_test=0.0`) to meet fair standards.
    - Protein featurization is currently identity-based (one-hot), which limits generalization to the broader Tau interactome.
    - Inference logic is not yet automated; predicting novel drug-disease links requires manual script execution.

### Tau-Specific Expansion Plan
1. **Network Densitification:** Transition from the broad AD network to a dense Tau-focused interactome (MAPT, GSK3B, CDK5, etc.) using BioGRID/STRING.
2. **Inference & Drug Repurposing:** Implement a dedicated `06_inference.py` script to score the library against Tau-associated diseases.
3. **Biological Featurization:** Replace one-hot protein features with Gene Ontology (GO) or UniProt embeddings to allow the model to learn biological properties rather than just node identities.
4. **Validation Rigor:** Implement a 3-way split (Train/Val/Test) and k-fold cross-validation to ensure results are statistically significant and free of data leakage.

**Immediate Priority:** Integrate the BioGRID API to fetch a high-density Tau-protein network and refactor `02_Code/04_expand_graph.py` for automated disease mapping.
