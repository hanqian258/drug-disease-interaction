# 📓 ISEF Lab Notebook: GNN Training Logs

## 📅 March 4, 2026: Optimization Cycle
**Goal:** Stabilize the 3-Layer HeteroGNN and reduce validation loss.

### 🧪 Run #1 (Initial Optimization)
- **Settings:** Hidden Channels: 128, Dropout: 0.3, LR: 0.01
- **Outcome:** Extreme Overfitting. Training Loss: 0.00 | Val Loss: 297.97

### 🧪 Run #2 (Final Stabilization)
- **Settings:** Hidden Channels: 128, Dropout: 0.5, LR: 0.001, Weight Decay: 1e-4
- **Outcome:** Success. Val Loss reduced to ~34. High AUC maintained.
