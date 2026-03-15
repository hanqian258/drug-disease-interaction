"""
05b_kfold_eval.py  —  K-Fold Cross Validation for Small Drug-Disease Graphs

WHY THIS IS NEEDED
──────────────────
With only 10–20 positive drug→disease edges, a single train/val/test split
produces a validation set of 1–3 positives. AUC computed on 3–9 edges is
statistically meaningless — it will be 1.0 or 0.0 by chance, not by learning.

K-fold cross-validation solves this by rotating which edges are held out.
With k=5 folds, each positive edge gets to be in the validation set once,
and you report the MEAN AUC across all folds with a standard deviation.

A result like "AUC = 0.78 ± 0.09 across 5 folds" is scientifically defensible.
A result like "val_AUC = 1.0 (n=3 edges)" is not.

WHAT A GOOD RESULT LOOKS LIKE
──────────────────────────────
  AUC > 0.80 with std < 0.10  →  model is genuinely learning
  AUC 0.65–0.80               →  model has signal, limited by data size
  AUC ~0.50 with high std      →  model is not learning, need more data

Run from the project root:
    python3 02_Code/05b_kfold_eval.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# ── Import the fixed model classes from 05_train_gcn_fixed ───────────────────
# (copy HeteroGNN and LinkPredictor here so this file is self-contained)

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


# ── Training function for one fold ───────────────────────────────────────────

def train_one_fold(data, train_pos_idx, val_pos_idx, n_epochs=200):
    """
    Train the model on train_pos_idx positive edges,
    evaluate on val_pos_idx positive edges + sampled negatives.
    Returns val AUC for this fold.
    """
    hidden = 64
    in_ch  = {ntype: data[ntype].x.size(-1) for ntype in data.node_types}

    # Build edge_index for training (only the train-fold positive edges)
    all_pos = data['drug', 'treats', 'disease'].edge_index   # [2, n_pos]
    train_ei = all_pos[:, train_pos_idx]
    val_ei   = all_pos[:, val_pos_idx]

    # Sample random negatives for training
    n_drug    = data['drug'].x.shape[0]
    n_disease = data['disease'].x.shape[0]
    n_train_neg = len(train_pos_idx) * 2

    pos_set = set(zip(all_pos[0].tolist(), all_pos[1].tolist()))
    neg_src, neg_dst = [], []
    while len(neg_src) < n_train_neg:
        s = torch.randint(0, n_drug,    (n_train_neg * 3,))
        d = torch.randint(0, n_disease, (n_train_neg * 3,))
        for si, di in zip(s.tolist(), d.tolist()):
            if (si, di) not in pos_set:
                neg_src.append(si)
                neg_dst.append(di)
            if len(neg_src) >= n_train_neg:
                break

    train_neg_ei = torch.tensor([neg_src[:n_train_neg], neg_dst[:n_train_neg]])

    # Combine positives + negatives for training
    train_label_ei = torch.cat([train_ei, train_neg_ei], dim=1)
    train_labels   = torch.cat([
        torch.ones(train_ei.shape[1]),
        torch.zeros(train_neg_ei.shape[1])
    ])

    # Build the edge_index_dict for message passing (train positives only)
    edge_index_dict = dict(data.edge_index_dict)
    edge_index_dict[('drug', 'treats', 'disease')]   = train_ei
    edge_index_dict[('disease', 'rev_treats', 'drug')] = train_ei.flip(0)

    # Negatives for validation (2x the val positives)
    n_val_neg = len(val_pos_idx) * 2
    vneg_src, vneg_dst = [], []
    while len(vneg_src) < n_val_neg:
        s = torch.randint(0, n_drug,    (n_val_neg * 3,))
        d = torch.randint(0, n_disease, (n_val_neg * 3,))
        for si, di in zip(s.tolist(), d.tolist()):
            if (si, di) not in pos_set:
                vneg_src.append(si)
                vneg_dst.append(di)
            if len(vneg_src) >= n_val_neg:
                break

    val_neg_ei   = torch.tensor([vneg_src[:n_val_neg], vneg_dst[:n_val_neg]])
    val_label_ei = torch.cat([val_ei, val_neg_ei], dim=1)
    val_labels   = torch.cat([
        torch.ones(val_ei.shape[1]),
        torch.zeros(val_neg_ei.shape[1])
    ])

    # Model
    model     = HeteroGNN(hidden, hidden, data.node_types, data.edge_types, in_ch)
    predictor = LinkPredictor(hidden, hidden)

    pos_weight = torch.tensor([n_train_neg / max(len(train_pos_idx), 1)])
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=3e-4, weight_decay=1e-3
    )

    best_auc = 0.0
    for epoch in range(1, n_epochs + 1):
        model.train(); predictor.train()
        optimizer.zero_grad()

        x = model(data.x_dict, edge_index_dict)
        preds = predictor(x['drug'], x['disease'], train_label_ei)
        loss  = criterion(preds, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(predictor.parameters()), 1.0
        )
        optimizer.step()

        if epoch % 20 == 0:
            model.eval(); predictor.eval()
            with torch.no_grad():
                x_val = model(data.x_dict, edge_index_dict)
                v_preds = torch.sigmoid(
                    predictor(x_val['drug'], x_val['disease'], val_label_ei)
                ).numpy()
            try:
                auc = roc_auc_score(val_labels.numpy(), v_preds)
                best_auc = max(best_auc, auc)
            except ValueError:
                pass

    return best_auc


# ── Main k-fold loop ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  K-Fold Cross Validation (k=5)")
    print("=" * 60)

    if not os.path.exists('01_Cleaned_Data/expanded_graph.pt'):
        print("Error: expanded_graph.pt not found.")
        return

    data = torch.load('01_Cleaned_Data/expanded_graph.pt', weights_only=False)

    n_pos = data['drug', 'treats', 'disease'].edge_index.shape[1]
    print(f"\n  Total positive drug→disease edges: {n_pos}")

    if n_pos < 10:
        print(f"\n  CRITICAL: Only {n_pos} positive edges found.")
        print("  K-fold cross-validation requires at least 10 positive edges.")
        print("  Please add more known drug-disease associations to")
        print("  04_expand_graph.py before running this evaluation.")
        print("\n  Suggested sources:")
        print("    • CTD: https://ctdbase.org  (search 'Alzheimer Disease')")
        print("    • DrugBank approved set")
        print("    • OMIM gene-disease → map genes to your PPI proteins")
        return

    k      = 5
    kf     = KFold(n_splits=k, shuffle=True, random_state=42)
    idx    = np.arange(n_pos)
    aucs   = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(idx), 1):
        print(f"\n  Fold {fold}/{k}  "
              f"(train={len(train_idx)} pos edges, val={len(val_idx)} pos edges)")
        auc = train_one_fold(data, train_idx, val_idx)
        aucs.append(auc)
        print(f"    Best val AUC this fold: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    print("\n" + "=" * 60)
    print(f"  FINAL RESULT: AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    print("=" * 60)

    if mean_auc >= 0.80:
        print("  Interpretation: Strong — model is genuinely learning.")
    elif mean_auc >= 0.65:
        print("  Interpretation: Moderate signal. Limited by small dataset.")
        print("  Adding more positive edges will improve this score.")
    else:
        print("  Interpretation: Near-random. More data needed or")
        print("  check that drug features connect to the PPI proteins.")

    print(f"\n  Per-fold AUCs: {[f'{a:.4f}' for a in aucs]}")
    print(f"  High std ({std_auc:.3f}) = unstable due to small n  |  "
          f"Low std = consistent")

    # Save result
    os.makedirs('99_ISEF_Docs', exist_ok=True)
    with open('99_ISEF_Docs/kfold_results.txt', 'w') as f:
        f.write(f"K-Fold Cross Validation Results (k={k})\n")
        f.write(f"Mean AUC : {mean_auc:.4f}\n")
        f.write(f"Std  AUC : {std_auc:.4f}\n")
        f.write(f"Per-fold : {aucs}\n")
    print("\n  Saved → 99_ISEF_Docs/kfold_results.txt")


if __name__ == "__main__":
    main()