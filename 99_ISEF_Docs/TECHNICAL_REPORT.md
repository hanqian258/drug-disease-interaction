# Technical Audit Report: Alzheimer's GNN Project

## 1. Pipeline Functionality & Structure

### Is it a fully functional pipeline?
Yes, the repository contains a **fully functional end-to-end pipeline** that progresses from raw drug/protein lists to a trained Link Prediction model.

### Repository Map
*   **00_Raw_Data/**: Initial CSV files defining drug names (`positive_drugs.csv`, `negative_controls.csv`) and their known protein targets (`drug_links.csv`).
*   **01_Cleaned_Data/**: Intermediary artifacts. Contains processed CSVs with SMILES strings, the retrieved STRING DB interactions, and serialized PyTorch Geometric graph objects (`master_graph.pt`, `expanded_graph.pt`).
*   **02_Code/**:
    *   `01_clean_drugs.py`: Automates SMILES retrieval via PubChem API.
    *   `02_fetch_string_interactions.py`: Retrieves high-confidence PPI data from STRING DB API.
    *   `03_build_hetero_graph.py`: Constructs the initial `HeteroData` object with 'drug' and 'protein' nodes.
    *   `04_expand_graph.py`: Adds 'disease' nodes and maps associations to transition from classification to link prediction.
    *   `05_train_gcn.py`: The core GNN implementation using PyTorch Geometric.
    *   `featurizer.py`: Utility for converting SMILES into molecular graph embeddings for drug nodes.
*   **Standard Library**: The code utilizes **PyTorch Geometric (PyG)** extensively for graph operations and **RDKit** for chemical featurization.

### Modularity Assessment
The pipeline is **moderately modular**. While the logic for building and training is decoupled from the data, the specific protein lists and disease mappings are currently hardcoded in scripts `02` and `04`. Swapping in your own Tau-specific data will require updating these lists, but the underlying `HeteroData` construction logic is reusable.

---

## 2. Data Status

### Is data included?
Yes, the repository contains both raw inputs and pre-processed outputs.

*   **Directories**: `/00_Raw_Data` and `/01_Cleaned_Data`.
*   **Data Types**:
    *   **Nodes**: Drugs (featurized from SMILES via RDKit), Proteins (currently one-hot encoded), Diseases (one-hot encoded).
    *   **Edges**:
        *   `('drug', 'binds', 'protein')`: Known target interactions.
        *   `('protein', 'interacts_with', 'protein')`: High-confidence STRING PPI (weighted by confidence score).
        *   `('protein', 'associated_with', 'disease')`: Biological associations.
        *   `('drug', 'treats', 'disease')`: Ground truth for supervised learning/validation.
*   **External Reliance**: The code fetches data dynamically if missing using `pubchempy` (PubChem) and `requests` (STRING DB).

---

## 3. Execution Flow & Algorithmic Approach

### Main Execution Path
1.  **Data Ingestion**: `01_clean_drugs.py` ensures all drugs have SMILES.
2.  **Network Retrieval**: `02_fetch_string_interactions.py` builds the protein interactome layer.
3.  **Graph Assembly**: `03_build_hetero_graph.py` & `04_expand_graph.py` synthesize nodes and edges into a `HeteroData` object.
4.  **Learning**: `05_train_gcn.py` executes the training loop.

### Mathematical/Algorithmic Approach
*   **Heterogeneous Graph Convolution (HeteroConv)**: The model uses 3 layers of `HeteroConv` wrapped around `SAGEConv`. This allows the model to learn different message-passing weights for each relationship type (e.g., drug-binds-protein vs protein-interacts-protein).
*   **Node Embeddings**:
    *   **Drugs**: Mean-aggregated atom features from their molecular graphs.
    *   **Proteins/Diseases**: Identity-based one-hot features.
*   **Link Prediction Head**: A custom `LinkPredictor` MLP. It concatenates the final embeddings of a drug and a disease ($h_{drug} || h_{disease}$) and passes them through a 2-layer MLP to output a probability score of the 'treats' relationship.

### Step-by-Step Training Epoch
1.  **GNN Forward Pass**: The `HeteroGNN` processes the entire graph (`edge_index_dict`) to compute updated embeddings for all nodes.
2.  **Negative Sampling**: `RandomLinkSplit` (inside PyG) automatically generates "negative edges" (pairs of drugs and diseases that do not have a known 'treats' relationship).
3.  **Link Scoring**: The `LinkPredictor` takes the embeddings of the source (drug) and target (disease) for both positive and negative edges in the current batch.
4.  **Loss Calculation**: `BCEWithLogitsLoss` is used to compare predicted scores against ground truth (1 for known treatment, 0 for sampled negative).
5.  **Backpropagation**: The optimizer updates both the GNN weights and the MLP head weights simultaneously.

---

## 4. Assessment for ISEF (Scientific Rigor)

### Salvageability
*   **Salvage (80%)**: The `HeteroGNN` architecture, the multi-stage build process, and the `DrugFeaturizer` are excellent foundations. The use of `RandomLinkSplit` is a standard PyG practice.
*   **Rewrite (20%)**: To meet ISEF's high standards for scientific rigor, you should focus on:
    *   **Validation**: Add a strict `test` split (currently `num_test=0.0`).
    *   **Protein Featurization**: Replace one-hot encoding with biologically meaningful features (e.g., UniProt embeddings or Gene Ontology vectors) to avoid "memorizing" the small graph.
    *   **Inference Script**: The repo lacks a dedicated script to run the model on *unlabeled* drug-disease pairs to predict novel Tau treatments.
    *   **Parameterization**: Move hardcoded protein lists to a configuration file (`config.yaml`) to improve reproducibility.
