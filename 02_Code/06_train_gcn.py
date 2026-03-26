"""
06_train_gcn.py  —  HeteroGNN Training Script

CHANGES FROM ORIGINAL
─────────────────────
1. HARD NEGATIVE LOSS
   Original only trained on RandomLinkSplit positives + random negatives.
   Now also incorporates ('drug', 'failed', 'disease') edges — confirmed
   failed AD trial drugs — as guaranteed label=0 examples. These are
   weighted separately (hard_neg_weight=0.5) so they contribute to the
   loss without dominating it.

2. GRADED EDGE WEIGHTS IN LOSS
   Original treated all positive edges equally (weight=1.0).
   Now reads edge_attr from the treats edge type and uses those graded
   weights (FDA=1.0, CTD=0.9, Phase2=0.8, Preclinical=0.7) to scale
   the per-sample loss. Higher-evidence edges contribute more to learning.

3. MULTI-DISEASE TRAINING
   Graph now has edges for AD, ALS, Bipolar, Dementia, Parkinson's, ADHD.
   No code change needed here —
   RandomLinkSplit automatically handles all drug→treats→disease edges
   regardless of which disease they point to.

ROOT CAUSE FIXES (preserved from previous version)
───────────────────────────────────────────────────
- SAGEConv + mean aggregation (not sum) → prevents embedding explosion
- BatchNorm + residual connections → stable training on small graphs
- is_undirected=False for drug→disease → no label leakage
- pos_weight balancing → handles class imbalance
- Gradient clipping → prevents exploding gradients
- LR scheduler → adapts learning rate when val AUC plateaus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import os
import numpy as np
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (unchanged from previous version — architecture is correct)
# ─────────────────────────────────────────────────────────────────────────────

class HeteroGNN(nn.Module):
    """
    Three-layer heterogeneous GNN.
      • SAGEConv (mean aggregation)   → inductive, handles new nodes at inference
      • BatchNorm after each layer    → stabilises training on small graphs
      • Residual (skip) connections   → prevents over-smoothing
      • Dropout on input projection   → regularisation without disrupting message passing
    """
    def __init__(self, hidden_channels, out_channels,
                 node_types, edge_types, in_channels_dict):
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
            ntype: self.dropout(
                F.relu(self.input_norms[ntype](self.input_lins[ntype](x)))
            )
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
    """
    MLP that scores a (drug, disease) pair given their embeddings.
    Input: concatenation of drug embedding + disease embedding (2 * hidden).
    Output: single logit (apply sigmoid for probability).
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTED LOSS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def weighted_bce_loss(logits: torch.Tensor,
                      labels: torch.Tensor,
                      sample_weights: torch.Tensor,
                      pos_weight: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss with:
      - pos_weight: upweights positive class globally (handles imbalance)
      - sample_weights: per-sample multiplier (handles evidence tier)

    Formula per sample:
        loss_i = -w_pos * y_i * log(σ(x_i)) - (1-y_i) * log(1-σ(x_i))
        weighted_loss = mean(sample_weights * loss_i)
    """
    # Unweighted BCE per sample
    base_loss = F.binary_cross_entropy_with_logits(
        logits, labels,
        pos_weight=pos_weight,
        reduction='none'    # keep per-sample so we can multiply by sample_weights
    )
    return (sample_weights * base_loss).mean()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  HeteroGNN Training — Multi-Disease Drug Discovery")
    print("=" * 60)

    graph_path = '01_Cleaned_Data/expanded_graph.pt'
    if not os.path.exists(graph_path):
        print("Error: expanded_graph.pt not found. Run 04_expand_graph.py first.")
        return

    data = torch.load(graph_path, weights_only=False)

    # ── Graph statistics ──────────────────────────────────────────────────────
    print("\n[Graph statistics]")
    for ntype in data.node_types:
        print(f"  Node type '{ntype}': {data[ntype].x.shape[0]} nodes, "
              f"{data[ntype].x.shape[1]}-dim features")
    for etype in data.edge_types:
        print(f"  Edge type {etype}: "
              f"{data[etype].edge_index.shape[1]} edges")

    n_pos = data['drug', 'treats', 'disease'].edge_index.shape[1]
    print(f"\n  Positive drug→disease edges: {n_pos}")

    # Check for hard negatives
    has_hard_negs = ('drug', 'failed', 'disease') in data.edge_types
    if has_hard_negs:
        n_hard_neg = data['drug', 'failed', 'disease'].edge_index.shape[1]
        print(f"  Hard negative edges (failed trials): {n_hard_neg}")
    else:
        n_hard_neg = 0
        print("  Hard negative edges: 0 (run 04_expand_graph.py to add them)")

    if n_pos < 10:
        print("  WARNING: very few positive edges — consider adding more "
              "drug-disease data before training.")

    # ── RandomLinkSplit on positive edges only ────────────────────────────────
    # is_undirected=False: drug→disease is directed — no label leakage
    # The 'failed' edge type is excluded from splitting; it's handled separately
    transform = T.RandomLinkSplit(
        num_val=0.15,
        num_test=0.10,
        is_undirected=False,
        edge_types=[('drug', 'treats', 'disease')],
        rev_edge_types=[('disease', 'rev_treats', 'drug')],
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,   # reduced from 2.0 — fewer negatives = less compression
    )
    train_data, val_data, test_data = transform(data)

    n_train_pos = int(train_data['drug', 'treats', 'disease'].edge_label.sum())
    n_train_neg = int((1 - train_data['drug', 'treats', 'disease'].edge_label).sum())
    print(f"\n  Train split: {n_train_pos} pos / {n_train_neg} neg edges")
    print(f"  Val   split: positive edges in val set")
    print(f"  Test  split: positive edges in test set")

    # ── pos_weight: penalise missed positives more ────────────────────────────
    # Cap pos_weight at 2.0 — as the dataset grows, higher values push
    # the model toward predicting ~0.5 for everything (score collapse).
    raw_pw = n_train_neg / max(n_train_pos, 1)
    pos_weight = torch.tensor([min(raw_pw, 2.0)], dtype=torch.float)
    print(f"  pos_weight: {pos_weight.item():.2f} (raw={raw_pw:.2f}, capped at 2.0)")

    # ── Hard negative edge index (used every epoch if available) ─────────────
    # We use the original data's failed edges, not the split version,
    # because we always want ALL failed drugs seen during training.
    if has_hard_negs:
        hard_neg_index = data['drug', 'failed', 'disease'].edge_index
        hard_neg_labels = torch.zeros(hard_neg_index.shape[1])
        # Hard negatives have neutral sample weight (no grading needed —
        # they are confirmed failures so weight = 1.0)
        hard_neg_weights = torch.ones(hard_neg_index.shape[1])
        print(f"  Hard negatives will be added to every training batch.")

    # ── Model ─────────────────────────────────────────────────────────────────
    hidden_channels  = 64
    in_channels_dict = {ntype: data[ntype].x.size(-1)
                        for ntype in data.node_types}

    model = HeteroGNN(
        hidden_channels, hidden_channels,
        train_data.node_types, train_data.edge_types,
        in_channels_dict
    )
    predictor = LinkPredictor(hidden_channels, hidden_channels)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=3e-4, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-5
    )

    best_val_auc = 0.0
    best_epoch   = 0
    os.makedirs('01_Cleaned_Data', exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  Training for 300 epochs...")
    for epoch in range(1, 301):
        model.train()
        predictor.train()
        optimizer.zero_grad()

        x_dict = model(train_data.x_dict, train_data.edge_index_dict)

        edge_label_index = train_data['drug', 'treats', 'disease'].edge_label_index
        edge_label       = train_data['drug', 'treats', 'disease'].edge_label.float()
        preds            = predictor(x_dict['drug'], x_dict['disease'],
                                     edge_label_index)

        # ── Graded sample weights from edge_attr ─────────────────────────────
        # edge_label_index contains both positives (from original edge_attr)
        # and random negatives (sampled by RandomLinkSplit, no edge_attr).
        # Strategy: positives get their graded weight; negatives get weight=1.0
        n_labels = edge_label.shape[0]
        sample_weights = torch.ones(n_labels)

        # The first n_pos_train entries in edge_label_index are real positives
        # (RandomLinkSplit appends negatives at the end)
        if ('drug', 'treats', 'disease') in train_data.edge_types:
            ea = train_data['drug', 'treats', 'disease'].edge_attr
            if ea is not None:
                n_real_pos = int(edge_label.sum().item())
                # Only assign graded weights to positive entries
                pos_mask = edge_label.bool()
                # ea may be shorter than pos_mask if RandomLinkSplit reindexed
                n_assign = min(ea.shape[0], pos_mask.sum().item())
                pos_indices = pos_mask.nonzero(as_tuple=True)[0][:n_assign]
                sample_weights[pos_indices] = ea[:n_assign].view(-1)

        # ── Main loss (positives + random negatives) ──────────────────────────
        loss = weighted_bce_loss(preds, edge_label, sample_weights, pos_weight)

        # ── Hard negative loss ────────────────────────────────────────────────
        # Hard negative weight reduced to 0.2 — at 316 training edges the
        # 5 hard negatives are less critical and a weight of 0.5 was
        # contributing to score compression toward 0.5.
        hard_neg_weight = 0.2
        if has_hard_negs:
            hard_preds = predictor(x_dict['drug'], x_dict['disease'],
                                   hard_neg_index)
            hard_loss  = F.binary_cross_entropy_with_logits(
                hard_preds, hard_neg_labels, reduction='mean'
            )
            loss = loss + hard_neg_weight * hard_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(predictor.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        # ── Validation every 10 epochs ────────────────────────────────────────
        if epoch % 10 == 0:
            model.eval()
            predictor.eval()
            with torch.no_grad():
                x_val   = model(val_data.x_dict, val_data.edge_index_dict)
                v_idx   = val_data['drug', 'treats', 'disease'].edge_label_index
                v_label = val_data['drug', 'treats', 'disease'].edge_label.float()
                v_preds = predictor(x_val['drug'], x_val['disease'], v_idx)
                v_loss  = F.binary_cross_entropy_with_logits(
                    v_preds, v_label, pos_weight=pos_weight
                )

                try:
                    val_auc = roc_auc_score(
                        v_label.cpu().numpy(),
                        torch.sigmoid(v_preds).cpu().numpy()
                    )
                except ValueError:
                    val_auc = 0.0

            scheduler.step(val_auc)

            flag = " ← best" if val_auc > best_val_auc else ""
            print(f"  Epoch {epoch:03d} | loss {loss:.4f} | "
                  f"val_loss {v_loss:.4f} | val_AUC {val_auc:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}{flag}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch   = epoch
                torch.save(model.state_dict(),
                           '01_Cleaned_Data/gnn_model_best.pt')
                torch.save(predictor.state_dict(),
                           '01_Cleaned_Data/predictor_best.pt')

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n  Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    model.load_state_dict(
        torch.load('01_Cleaned_Data/gnn_model_best.pt', weights_only=True))
    predictor.load_state_dict(
        torch.load('01_Cleaned_Data/predictor_best.pt', weights_only=True))
    model.eval()
    predictor.eval()

    with torch.no_grad():
        x_test  = model(test_data.x_dict, test_data.edge_index_dict)
        t_idx   = test_data['drug', 'treats', 'disease'].edge_label_index
        t_label = test_data['drug', 'treats', 'disease'].edge_label.float()
        t_preds = predictor(x_test['drug'], x_test['disease'], t_idx)
        try:
            test_auc = roc_auc_score(
                t_label.cpu().numpy(),
                torch.sigmoid(t_preds).cpu().numpy()
            )
            print(f"  Final test AUC: {test_auc:.4f}")
        except ValueError:
            print("  Test AUC could not be computed (too few samples).")

    # Save final weights
    torch.save(model.state_dict(),     '01_Cleaned_Data/gnn_model.pt')
    torch.save(predictor.state_dict(), '01_Cleaned_Data/predictor.pt')
    print("\n  Saved: gnn_model_best.pt / predictor_best.pt  ← use for inference")
    print("  Saved: gnn_model.pt / predictor.pt            ← final epoch weights")

    # ── Score distribution sanity check ──────────────────────────────────────
    print("\n[Sanity check — score distribution across all drugs vs AD]")
    print("  Healthy: scores spread across [0,1], std > 0.05")
    print("  Collapsed: all scores ≈ 1.0, std < 0.05")
    with torch.no_grad():
        x_all            = model(data.x_dict, data.edge_index_dict)
        n_drugs          = data['drug'].x.shape[0]
        all_drug_idx     = torch.arange(n_drugs)
        disease_idx_zero = torch.zeros(n_drugs, dtype=torch.long)  # AD = index 0
        dummy_idx        = torch.stack([all_drug_idx, disease_idx_zero])
        all_scores       = torch.sigmoid(
            predictor(x_all['drug'], x_all['disease'], dummy_idx)
        ).cpu().numpy()

    print(f"  min={all_scores.min():.4f}  max={all_scores.max():.4f}  "
          f"mean={all_scores.mean():.4f}  std={all_scores.std():.4f}")

    if all_scores.std() < 0.05:
        print("  WARNING: scores are collapsed — model may be memorising.")
        print("  Check positive/negative edge balance in 04_expand_graph.py.")
    else:
        print("  Score distribution is healthy — model is discriminating.")

    # ── Per-disease score check ───────────────────────────────────────────────
    maps = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)
    if 'dis_map' in maps:
        print("\n[Per-disease mean scores across all drugs]")
        dis_map = maps['dis_map']
        with torch.no_grad():
            x_all = model(data.x_dict, data.edge_index_dict)
            for dis_name, dis_idx in dis_map.items():
                dis_tensor = torch.full((n_drugs,), dis_idx, dtype=torch.long)
                ei = torch.stack([all_drug_idx, dis_tensor])
                scores = torch.sigmoid(
                    predictor(x_all['drug'], x_all['disease'], ei)
                ).cpu().numpy()
                print(f"  {dis_name:<25} mean={scores.mean():.4f}  "
                      f"std={scores.std():.4f}  "
                      f"max={scores.max():.4f}")


if __name__ == "__main__":
    train()

  