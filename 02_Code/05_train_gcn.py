import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import os

class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, node_types, edge_types):
        super().__init__()

        # Initial projection layers for each node type to match hidden_channels
        self.lins = nn.ModuleDict()
        for node_type in node_types:
            self.lins[node_type] = Linear(-1, hidden_channels)

        self.dropout = nn.Dropout(0.2)

        self.convs = nn.ModuleList()
        for _ in range(3):
            # Each layer uses separate learned weights for each edge type
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in edge_types
            }, aggr='sum')
            self.convs.append(conv)

        # Final projection to output channels
        self.final_lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 1. Project all node features to the same dimension (hidden_channels)
        x_dict = {
            node_type: self.dropout(F.relu(self.lins[node_type](x)))
            for node_type, x in x_dict.items()
        }

        # 2. Pass through 3 HeteroConv layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # 3. Final linear layer to get the final embeddings
        return {key: self.final_lin(x) for key, x in x_dict.items()}

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, x_drug, x_disease, edge_label_index):
        # edge_label_index is [2, num_edges]
        # node_s are drugs, node_t are diseases
        nodes_s = x_drug[edge_label_index[0]]
        nodes_t = x_disease[edge_label_index[1]]

        x = torch.cat([nodes_s, nodes_t], dim=-1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x.view(-1)

def train():
    print("--- Training Heterogeneous GNN for Link Prediction ---")

    # Load expanded graph
    if not os.path.exists('01_Cleaned_Data/expanded_graph.pt'):
        print("Error: expanded_graph.pt not found. Run 02_Code/04_expand_graph.py first.")
        return

    data = torch.load('01_Cleaned_Data/expanded_graph.pt', weights_only=False)

    # Random Link Split for ('drug', 'treats', 'disease')
    # Note: expanded_graph.pt already has reverse edges from T.ToUndirected()
    transform = T.RandomLinkSplit(
        num_val=0.2,
        num_test=0.0,
        is_undirected=True,
        edge_types=[('drug', 'treats', 'disease')],
        rev_edge_types=[('disease', 'rev_treats', 'drug')],
        add_negative_train_samples=True
    )

    train_data, val_data, _ = transform(data)

    # Model Setup
    hidden_channels = 64
    # Model now accepts node_types for initial projection
    model = HeteroGNN(hidden_channels, hidden_channels, train_data.node_types, train_data.edge_types)
    predictor = LinkPredictor(hidden_channels, hidden_channels)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, 101):
        model.train()
        predictor.train()
        optimizer.zero_grad()

        # 1. Forward pass GNN to get embeddings
        x_dict = model(train_data.x_dict, train_data.edge_index_dict)

        # 2. Predict on training edges
        edge_label_index = train_data['drug', 'treats', 'disease'].edge_label_index
        edge_label = train_data['drug', 'treats', 'disease'].edge_label

        preds = predictor(x_dict['drug'], x_dict['disease'], edge_label_index)
        loss = criterion(preds, edge_label)

        loss.backward()
        optimizer.step()

        # Validation
        if epoch % 10 == 0:
            model.eval()
            predictor.eval()
            with torch.no_grad():
                x_dict_val = model(val_data.x_dict, val_data.edge_index_dict)
                val_edge_label_index = val_data['drug', 'treats', 'disease'].edge_label_index
                val_edge_label = val_data['drug', 'treats', 'disease'].edge_label
                val_preds = predictor(x_dict_val['drug'], x_dict_val['disease'], val_edge_label_index)
                val_loss = criterion(val_preds, val_edge_label)

                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

    print("\nTraining Complete.")
    os.makedirs('01_Cleaned_Data', exist_ok=True)
    torch.save(model.state_dict(), '01_Cleaned_Data/gnn_model.pt')
    torch.save(predictor.state_dict(), '01_Cleaned_Data/predictor.pt')
    print("Models saved to 01_Cleaned_Data/")

if __name__ == "__main__":
    train()
