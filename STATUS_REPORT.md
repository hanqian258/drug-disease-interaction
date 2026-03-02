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

### GNN Model Architecture (HeteroGNN)
- **Status:** Completed
- **Details:**
  - Implemented a 3-layer `HeteroConv` architecture in `02_Code/05_train_gcn.py` using `SAGEConv` layers.
  - Designed a custom `LinkPredictor` MLP head (Linear -> ReLU -> Linear) for drug-disease association scoring.
  - Configured `T.RandomLinkSplit` to handle heterogeneous bipartite edge splitting with negative sampling.
- **Results:** Successfully executed the training pipeline. The model achieves rapid convergence on the training set, establishing a functional baseline for drug repurposing.

---

## Next Steps
1. **Inference & Drug Repurposing:** Use the trained `LinkPredictor` to score all drugs against `Frontotemporal Dementia` to identify potential repurposing candidates.
2. **Feature Engineering:** Replace one-hot protein features with UniProt/Gene Ontology embeddings to improve generalization.
3. **Cross-Validation:** Implement k-fold cross-validation to robustly evaluate performance on the small-scale biological network.

**Immediate Priority (Next 48 Hours):** Perform the first inference run on FTD and validate the top 3 drug hits against existing literature in `03_Literature`.
