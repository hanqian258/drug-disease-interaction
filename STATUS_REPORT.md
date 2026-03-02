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

## Next Steps
1. **Model Architecture Design:** Define the 3-layer GCN architecture in `02_Code/05_gcn_model.py`.
2. **Feature Engineering:** Expand protein node features beyond one-hot encoding (e.g., using UniProt annotations).
3. **Training Pipeline:** Set up the training loop with cross-validation to assess model performance on the small dataset.

**Immediate Priority (Next 48 Hours):** Finalize the multimodal feature integration to ensure the GCN has rich node embeddings before the training phase begins.
