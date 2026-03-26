"""
07_inference.py — Drug → Disease Correlation Predictor
Outputs a probability score (0.0 – 1.0) for a given drug name.

Usage:
    python3 02_Code/07_inference.py "Donepezil"
    python3 02_Code/07_inference.py "Memantine"

Output format (stdout):
    Drug: Donepezil
    Probability of interaction: 0.4421
    Result: Moderate Potential — warrants further investigation.

NOTE ON PROBABILITY DISTRIBUTION:
    The raw model output is a single logit per (drug, disease) pair.
    We apply torch.sigmoid() to convert it to a probability in [0, 1].
    Due to pos_weight training bias, scores are compressed below 0.5.
    Interpretation thresholds are calibrated to the training score distribution:
      - Score >= 0.43 : High Potential
      - Score 0.40–0.43: Moderate Potential
      - Score <  0.40 : Low / No Predicted Correlation
"""

import sys
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

try:
    from rdkit.Chem import rdFingerprintGenerator
except Exception:
    rdFingerprintGenerator = None

# ── Model architecture — must match 06_train_gcn.py exactly ──────────────────

class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, node_types, edge_types, in_channels_dict):
        super().__init__()
        self.input_lins  = nn.ModuleDict()
        self.input_norms = nn.ModuleDict()
        for ntype in node_types:
            self.input_lins[ntype]  = nn.Linear(in_channels_dict[ntype], hidden_channels)
            self.input_norms[ntype] = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.3)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(3):
            conv = HeteroConv(
                {etype: SAGEConv((-1, -1), hidden_channels) for etype in edge_types},
                aggr='mean'
            )
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict(
                {ntype: nn.BatchNorm1d(hidden_channels) for ntype in node_types}
            ))
        self.final_lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            ntype: self.dropout(F.relu(self.input_norms[ntype](self.input_lins[ntype](x))))
            for ntype, x in x_dict.items()
        }
        for conv, norms in zip(self.convs, self.norms):
            residual = x_dict
            x_dict   = conv(x_dict, edge_index_dict)
            x_dict   = {
                ntype: F.relu(norms[ntype](x)) + residual.get(ntype, 0)
                for ntype, x in x_dict.items()
            }
        return {ntype: self.final_lin(x) for ntype, x in x_dict.items()}


class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x_drug, x_disease, edge_label_index):
        src = x_drug[edge_label_index[0]]
        dst = x_disease[edge_label_index[1]]
        return self.net(torch.cat([src, dst], dim=-1)).view(-1)


# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_DIR  = "01_Cleaned_Data"
GRAPH_PATH = os.path.join(MODEL_DIR, "expanded_graph.pt")
GNN_PATH   = os.path.join(MODEL_DIR, "gnn_model_best.pt")
PRED_PATH  = os.path.join(MODEL_DIR, "predictor_best.pt")
MAP_PATH   = os.path.join(MODEL_DIR, "mappings.pt")

# ── Score calibration ─────────────────────────────────────────────────────────
# The multi-disease model (316 training edges, 6 diseases) produces compressed
# raw scores because its embeddings encode general disease relevance rather than
# AD-specific signal. Raw approved drugs cluster at 0.458–0.489, non-CNS at
# 0.354–0.397 — perfect separation, but a narrow absolute range.
#
# We apply a linear calibration (analogous to temperature scaling) that maps:
#   mean(non-CNS)  → 0.39   (reference low anchor)
#   mean(approved) → 0.80   (reference high anchor)
#
# This preserves the relative ordering and separation completely — it only
# stretches the range to match intuitive [0,1] expectations.
# Calibration parameters derived from 09_results_validation.py metric test.

_CALIB_LOW    = 0.3717   # mean raw score of non-CNS reference drugs
_CALIB_HIGH   = 0.4709   # mean raw score of FDA-approved drugs
_TARGET_LOW   = 0.39     # calibrated target for non-CNS mean
_TARGET_HIGH  = 0.80     # calibrated target for approved drug mean
_SLOPE        = (_TARGET_HIGH - _TARGET_LOW) / (_CALIB_HIGH - _CALIB_LOW)
_INTERCEPT    = _TARGET_LOW - _SLOPE * _CALIB_LOW


def calibrate(raw_prob: float) -> float:
    """Linearly rescale raw sigmoid output to calibrated probability."""
    import numpy as np
    return float(np.clip(_SLOPE * raw_prob + _INTERCEPT, 0.0, 1.0))


# Thresholds applied to CALIBRATED scores
#   HIGH     >= 0.70  → within calibrated approved drug range (0.75–0.87)
#   MODERATE  0.50–0.70 → above calibrated non-CNS baseline, below approved
#   LOW       < 0.50  → at or below non-CNS reference level
SCORE_HIGH     = 0.70
SCORE_MODERATE = 0.50


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if rdFingerprintGenerator is not None:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = generator.GetFingerprint(mol)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return torch.tensor(list(fp), dtype=torch.float).view(1, -1)


def calculate_drug_properties(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {}
    return {
        "Molecular Weight":       round(Descriptors.MolWt(mol), 2),
        "LogP (Lipophilicity)":   round(Descriptors.MolLogP(mol), 2),
        "QED (Drug-likeness)":    round(Descriptors.qed(mol), 2),
        "H-Bond Donors":          Descriptors.NumHDonors(mol),
        "H-Bond Acceptors":       Descriptors.NumHAcceptors(mol),
        "BBB Permeable (Likely)": "Yes" if Descriptors.MolLogP(mol) < 5
                                           and Descriptors.MolWt(mol) < 450
                                        else "Limited",
    }


def load_everything():
    """Load graph, mappings, and trained models. Returns (data, maps, model, predictor)."""
    for path in [GRAPH_PATH, GNN_PATH, PRED_PATH, MAP_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run the full pipeline (03 → 04 → 06_train_gcn.py) first."
            )

    data = torch.load(GRAPH_PATH, weights_only=False)
    maps = torch.load(MAP_PATH,   weights_only=False)

    hidden = 64   # must match 06_train_gcn.py hidden_channels
    in_ch  = {nt: data[nt].x.size(-1) for nt in data.node_types}

    model     = HeteroGNN(hidden, hidden, data.node_types, data.edge_types, in_ch)
    predictor = LinkPredictor(hidden, hidden)

    model.load_state_dict(
        torch.load(GNN_PATH, weights_only=True, map_location="cpu"))
    predictor.load_state_dict(
        torch.load(PRED_PATH, weights_only=True, map_location="cpu"))
    model.eval()
    predictor.eval()

    return data, maps, model, predictor


# ── Core prediction ───────────────────────────────────────────────────────────

def predict(drug_name: str):
    """
    Returns (prob, smiles, top_proteins) where:
      prob          — float in [0,1], sigmoid of model logit
      smiles        — SMILES string used for featurization (or None)
      top_proteins  — list of top-3 protein names by embedding similarity
    Returns (None, None, None) if drug not found and cannot be featurized.
    """
    data, maps, model, predictor = load_everything()
    d_map       = maps['d_map']
    all_proteins = maps['all_proteins']
    drug_names  = maps['drug_names']

    is_new_drug  = False
    active_smiles = None
    drug_idx      = None

    # ── Look up in library ────────────────────────────────────────────────────
   # Exact match
    if drug_name in d_map:
        drug_idx = d_map[drug_name]
    else:
    # Case-insensitive exact match
        match = next((k for k in d_map if k.lower() == drug_name.lower()), None)
        if match:
            drug_idx = d_map[match]
        else:
        # Substring match (drug_name contained in key)
            match = next((k for k in d_map if drug_name.lower() in k.lower()), None)
        if match:
            drug_idx = d_map[match]
            print(f"  Matched '{drug_name}' to '{match}'")
    # ── Inductive mode for new drugs ──────────────────────────────────────────
    if drug_idx is None:
        mol = Chem.MolFromSmiles(drug_name)
        if mol:
            active_smiles = drug_name
        else:
            # Try PubChem
            try:
                import pubchempy as pcp
                hits = pcp.get_compounds(drug_name, 'name')
                if hits:
                    active_smiles = hits[0].connectivity_smiles or hits[0].canonical_smiles
                    print(f"  Found SMILES on PubChem for '{drug_name}'")
            except Exception:
                pass

        if active_smiles:
            fp = get_morgan_fp(active_smiles)
            if fp is not None:
                is_new_drug = True
                orig_x = data['drug'].x
                data['drug'].x = torch.cat([orig_x, fp], dim=0)
                drug_idx = data['drug'].x.shape[0] - 1
                print(f"  Inductive mode: created virtual node for '{drug_name}'")
            else:
                return None, None, None
        else:
            return None, None, None

    # ── Retrieve SMILES for property display ──────────────────────────────────
    if active_smiles is None and drug_idx is not None:
        try:
            import pandas as pd
            df = pd.read_csv('00_Raw_Data/drugs_raw_augmented.csv')
            row = df[df['Drug Name/Treatment'].str.lower() == drug_name.lower()]
            if not row.empty:
                active_smiles = str(row.iloc[0].get('Drug Structure', '')).strip()
                if active_smiles in ('', 'nan'):
                    active_smiles = None
        except Exception:
            pass

    # ── Disease index — Alzheimer's is always index 0 ─────────────────────────
    disease_idx = 0

    # ── Forward pass ──────────────────────────────────────────────────────────
    with torch.no_grad():
        x_dict = model(data.x_dict, data.edge_index_dict)

        # Top-3 protein targets by cosine similarity
        drug_emb  = x_dict['drug'][drug_idx].view(1, -1)
        prot_embs = x_dict['protein']
        sims      = F.cosine_similarity(drug_emb, prot_embs)
        top_idx   = torch.topk(sims, k=min(3, len(all_proteins))).indices
        top_prots = [all_proteins[i] for i in top_idx]

        ei       = torch.tensor([[drug_idx], [disease_idx]], dtype=torch.long)
        # Link probability — apply calibration before returning
        raw_prob = torch.sigmoid(predictor(x_dict['drug'], x_dict['disease'], ei)).item()
        prob     = calibrate(raw_prob)

    # Restore graph if modified
    if is_new_drug:
        data['drug'].x = data['drug'].x[:-1]

    return prob, active_smiles, top_prots


def interpret(prob: float) -> str:
    if prob >= SCORE_HIGH:
        return "High Potential — score within calibrated approved drug range."
    elif prob >= SCORE_MODERATE:
        return "Moderate Potential — above non-CNS baseline, warrants investigation."
    else:
        return "Low / No Predicted Correlation."


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 02_Code/07_inference.py \"DrugName\"")
        sys.exit(1)

    drug_name = sys.argv[1].strip()

    try:
        prob, smiles, top_prots = predict(drug_name)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("-" * 40)
    print(f"RESULTS FOR: {drug_name}")
    print("-" * 40)

    if prob is None:
        print(f"Drug not found in library and could not be featurized.")
        print(f"Probability of interaction: N/A")
        sys.exit(0)

    print(f"Probability of interaction: {prob:.4f}")
    print(f"Result: {interpret(prob)}")

    if top_prots:
        print(f"\nTop 3 Predicted Protein Targets:")
        for i, p in enumerate(top_prots, 1):
            print(f"  {i}. {p}")

    if smiles:
        props = calculate_drug_properties(smiles)
        if props:
            print(f"\nBiological Properties:")
            for k, v in props.items():
                print(f"  - {k}: {v}")

    print()
    print("Score interpretation (linearly calibrated — approved drugs → 0.75–0.87):")
    print(f"  >= {SCORE_HIGH}  : High Potential     "
          f"{'<-- YOUR SCORE' if prob >= SCORE_HIGH else ''}")
    print(f"  {SCORE_MODERATE}–{SCORE_HIGH} : Moderate Potential "
          f"{'<-- YOUR SCORE' if SCORE_MODERATE <= prob < SCORE_HIGH else ''}")
    print(f"  <  {SCORE_MODERATE}  : Low / No Correlation "
          f"{'<-- YOUR SCORE' if prob < SCORE_MODERATE else ''}")


if __name__ == "__main__":
    main()