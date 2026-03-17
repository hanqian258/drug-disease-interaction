"""
05_train_gcn_fixed.py  —  Fixed HeteroGNN training script

ROOT CAUSE ANALYSIS: why all scores were 1.0
─────────────────────────────────────────────
The original model had four compounding problems that together caused the
network to collapse: it stopped learning and simply predicted 1.0 for
everything it had seen during training.

PROBLEM 1 — Dataset too small / no negative edges at inference time
  The graph only contains KNOWN drug→disease edges (all positives).
  RandomLinkSplit creates negative samples for training, but the saved
  graph.pt still has only positives. So at inference time, any drug that
  exists in the graph gets the same positive embedding and scores 1.0.
  FIX: explicitly add hard negative drug→disease pairs during graph
       construction AND use those negatives consistently at inference.

PROBLEM 2 — Label leakage through is_undirected=True
  Setting is_undirected=True on a directed (drug→disease) edge type
  causes the transform to duplicate every edge as its reverse, so the
  model sees the disease→drug direction as a second positive signal.
  This inflates scores for anything in-distribution.
  FIX: set is_undirected=False for drug→disease edges.

PROBLEM 3 — No gradient clipping / loss collapse
  With only ~5–15 positive drug→disease edges (typical for a focused AD
  graph), BCEWithLogitsLoss converges almost immediately to predict 1.0
  for all positives and never moves. The small dataset lets the model
  memorize rather than generalize.
  FIX: gradient clipping + lower learning rate + more epochs + use
       pos_weight to balance loss when positive examples are rare.

PROBLEM 4 — Validation AUC of 0.5 is the symptom, not the disease
  AUC = 0.5 means the model is no better than random on held-out data,
  which confirms it memorized training positives and cannot discriminate.
  FIX: all of the above, plus add BatchNorm and residual connections to
       give the GNN a better inductive bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import os
from sklearn.metrics import roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class HeteroGNN(nn.Module):
    """
    Three-layer heterogeneous GNN with:
      • BatchNorm after each conv layer   → stabilizes training on small graphs
      • Residual (skip) connections       → prevents over-smoothing of embeddings
      • SAGEConv instead of GraphConv     → better inductive generalization
      • Dropout only on input projection  → keeps message passing stable
    """
    def __init__(self, hidden_channels, out_channels, node_types, edge_types, in_channels_dict):
        super().__init__()

        # Input projection: map each node type to the same hidden dimension
        self.input_lins = nn.ModuleDict()
        self.input_norms = nn.ModuleDict()
        for ntype in node_types:
            self.input_lins[ntype] = nn.Linear(in_channels_dict[ntype], hidden_channels)
            self.input_norms[ntype] = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(0.3)

        # FIX: use SAGEConv (mean aggregation) instead of GraphConv (sum).
        # Sum aggregation on a small graph causes embedding values to grow
        # unboundedly, pushing sigmoid outputs toward 1.0 for all nodes.
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(3):
            conv = HeteroConv(
                {etype: SAGEConv((-1, -1), hidden_channels) for etype in edge_types},
                aggr='mean'   # FIX: was 'sum' — mean prevents value explosion
            )
            self.convs.append(conv)
            # Per-layer norm dict (one BN per node type per layer)
            layer_norms = nn.ModuleDict({
                ntype: nn.BatchNorm1d(hidden_channels) for ntype in node_types
            })
            self.norms.append(layer_norms)

        self.final_lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        # --- Input projection ---
        x_dict = {
            ntype: self.dropout(
                F.relu(self.input_norms[ntype](self.input_lins[ntype](x)))
            )
            for ntype, x in x_dict.items()
        }

        # --- Message passing with residual connections ---
        for conv, norms in zip(self.convs, self.norms):
            residual = x_dict                         # save for skip connection
            x_dict = conv(x_dict, edge_index_dict)    # no edge_weight_dict for SAGEConv
            x_dict = {
                ntype: F.relu(norms[ntype](x)) + residual.get(ntype, 0)
                for ntype, x in x_dict.items()
            }

        return {ntype: self.final_lin(x) for ntype, x in x_dict.items()}


class LinkPredictor(nn.Module):
    """
    MLP that takes the concatenation of a drug embedding and a disease
    embedding and outputs a single logit (pre-sigmoid score).

    Added BatchNorm and a third hidden layer so it can learn non-trivial
    decision boundaries even when the GNN embeddings are similar.
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
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  Fixed HeteroGNN Training — Alzheimer's Drug Discovery")
    print("=" * 60)

    if not os.path.exists('01_Cleaned_Data/expanded_graph.pt'):
        print("Error: expanded_graph.pt not found. Run 04_expand_graph.py first.")
        return

    data = torch.load('01_Cleaned_Data/expanded_graph.pt', weights_only=False)

    # ── Diagnostic: print graph statistics ───────────────────────────────────
    print("\n[Graph statistics]")
    for ntype in data.node_types:
        print(f"  Node type '{ntype}': {data[ntype].x.shape[0]} nodes, "
              f"{data[ntype].x.shape[1]}-dim features")
    for etype in data.edge_types:
        n_edges = data[etype].edge_index.shape[1]
        print(f"  Edge type {etype}: {n_edges} edges")

    n_pos = data['drug', 'treats', 'disease'].edge_index.shape[1]
    print(f"\n  Positive drug→disease edges: {n_pos}")
    if n_pos < 10:
        print("  WARNING: very few positive edges. Consider adding more known")
        print("           drug-disease pairs to 04_expand_graph.py.")

    # ── FIX 2: is_undirected=False for directed drug→disease edges ───────────
    transform = T.RandomLinkSplit(
        num_val=0.2,
        num_test=0.1,          # keep a held-out test split
        is_undirected=False,   # FIX: was True — caused label leakage
        edge_types=[('drug', 'treats', 'disease')],
        rev_edge_types=[('disease', 'rev_treats', 'drug')],
        add_negative_train_samples=True,
        neg_sampling_ratio=2.0  # 2 negatives per positive → harder training signal
    )

    train_data, val_data, test_data = transform(data)

    n_train_pos = int(train_data['drug', 'treats', 'disease'].edge_label.sum())
    n_train_neg = int((1 - train_data['drug', 'treats', 'disease'].edge_label).sum())
    print(f"\n  Train: {n_train_pos} pos / {n_train_neg} neg edges")

    # ── FIX 3: pos_weight to handle class imbalance ──────────────────────────
    # When negatives outnumber positives, the model learns to predict 0 for
    # everything. pos_weight tells the loss to penalise missed positives more.
    pos_weight = torch.tensor([n_train_neg / max(n_train_pos, 1)], dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  pos_weight set to {pos_weight.item():.2f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    hidden_channels = 64   # reduced from 128: avoids overfitting on small graphs
    in_channels_dict = {ntype: data[ntype].x.size(-1) for ntype in data.node_types}

    model = HeteroGNN(
        hidden_channels, hidden_channels,
        train_data.node_types, train_data.edge_types,
        in_channels_dict
    )
    predictor = LinkPredictor(hidden_channels, hidden_channels)

    # ── FIX 3 cont.: lower LR + LR scheduler ─────────────────────────────────
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=3e-4,          # lower than original 1e-3: slows convergence, better generalization
        weight_decay=1e-3
    )
    # Reduce LR when validation AUC stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-5
    )

    best_val_auc = 0.0
    best_epoch   = 0

    # ── Training loop (200 epochs — more than 100 needed for small graphs) ────
    for epoch in range(1, 201):
        model.train()
        predictor.train()
        optimizer.zero_grad()

        x_dict = model(train_data.x_dict, train_data.edge_index_dict)

        edge_label_index = train_data['drug', 'treats', 'disease'].edge_label_index
        edge_label       = train_data['drug', 'treats', 'disease'].edge_label.float()

        preds = predictor(x_dict['drug'], x_dict['disease'], edge_label_index)
        loss  = criterion(preds, edge_label)

        loss.backward()

        # FIX 3: gradient clipping prevents exploding gradients on small graphs
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(predictor.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % 10 == 0:
            model.eval()
            predictor.eval()
            with torch.no_grad():
                x_val = model(val_data.x_dict, val_data.edge_index_dict)
                v_idx   = val_data['drug', 'treats', 'disease'].edge_label_index
                v_label = val_data['drug', 'treats', 'disease'].edge_label.float()
                v_preds = predictor(x_val['drug'], x_val['disease'], v_idx)
                v_loss  = criterion(v_preds, v_label)

                try:
                    val_auc = roc_auc_score(
                        v_label.cpu().numpy(),
                        torch.sigmoid(v_preds).cpu().numpy()  # FIX: apply sigmoid before AUC
                    )
                except ValueError:
                    val_auc = 0.0

            scheduler.step(val_auc)

            flag = " ← best" if val_auc > best_val_auc else ""
            print(f"  Epoch {epoch:03d} | loss {loss:.4f} | val_loss {v_loss:.4f} "
                  f"| val_AUC {val_auc:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}{flag}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch   = epoch
                # Save best checkpoint (not just the final one)
                os.makedirs('01_Cleaned_Data', exist_ok=True)
                torch.save(model.state_dict(),     '01_Cleaned_Data/gnn_model_best.pt')
                torch.save(predictor.state_dict(), '01_Cleaned_Data/predictor_best.pt')

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n  Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    # Load best weights for test evaluation
    model.load_state_dict(torch.load('01_Cleaned_Data/gnn_model_best.pt'))
    predictor.load_state_dict(torch.load('01_Cleaned_Data/predictor_best.pt'))
    model.eval()
    predictor.eval()

    with torch.no_grad():
        x_test = model(test_data.x_dict, test_data.edge_index_dict)
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

    # Also save final weights
    torch.save(model.state_dict(),     '01_Cleaned_Data/gnn_model.pt')
    torch.save(predictor.state_dict(), '01_Cleaned_Data/predictor.pt')
    print("\n  Saved: gnn_model_best.pt / predictor_best.pt  (use these for inference)")
    print("  Saved: gnn_model.pt / predictor.pt            (final epoch weights)")

    # ── Sanity check: score distribution on training drugs ───────────────────
    print("\n[Sanity check — score distribution on all training drugs]")
    print("  If the model learned properly, scores should NOT all be 1.0")
    with torch.no_grad():
        x_all = model(data.x_dict, data.edge_index_dict)
        all_drug_idx     = torch.arange(data['drug'].x.shape[0])
        disease_idx_zero = torch.zeros(data['drug'].x.shape[0], dtype=torch.long)
        dummy_idx = torch.stack([all_drug_idx, disease_idx_zero])
        all_scores = torch.sigmoid(
            predictor(x_all['drug'], x_all['disease'], dummy_idx)
        ).cpu().numpy()

    import numpy as np
    print(f"  min={all_scores.min():.4f}  max={all_scores.max():.4f}  "
          f"mean={all_scores.mean():.4f}  std={all_scores.std():.4f}")
    if all_scores.std() < 0.05:
        print("  WARNING: scores are still collapsed. Check that your graph")
        print("  has enough positive AND negative drug-disease edges.")
    else:
        print("  Score distribution looks healthy — model is discriminating.")


if __name__ == "__main__":
    train()
