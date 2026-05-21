import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def build_graph(edges_df, node_df, allowed_edge_types):
    node_ids = node_df["node_id"].tolist()
    node_map = {nid: i for i, nid in enumerate(node_ids)}

    data = HeteroData()
    data["node"].num_nodes = len(node_ids)

    for etype in allowed_edge_types:
        df = edges_df[edges_df.edge_type == etype]
        src = torch.tensor([node_map[i] for i in df.src], dtype=torch.long)
        dst = torch.tensor([node_map[i] for i in df.dst], dtype=torch.long)
        data["node", etype, "node"].edge_index = torch.stack([src, dst])

    return data, node_map

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train(model, data, train_idx, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data["node"].x, data["node", "similarity", "node"].edge_index)
    loss = criterion(out[train_idx], data["node"].y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, idx):
    model.eval()
    with torch.no_grad():
        out = model(data["node"].x, data["node", "similarity", "node"].edge_index)
        pred = out.argmax(dim=-1)
        true = data["node"].y[idx]
        f1 = f1_score(true.cpu(), pred.cpu(), average='binary')
        acc = accuracy_score(true.cpu(), pred.cpu())
        prec = precision_score(true.cpu(), pred.cpu(), average='binary')
        rec = recall_score(true.cpu(), pred.cpu(), average='binary')
    return f1, acc, prec, rec

def main():
    data_dir = "data/public"
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    edges_df = pd.read_csv(os.path.join(data_dir, "graph_edges.csv"))
    node_df = pd.read_csv(os.path.join(data_dir, "node_types.csv"))

    # Identify target column (train uses 'disease_labels' or 'target')
    target_col_train = "disease_labels" if "disease_labels" in train_df.columns else "target"

    # Build graphs
    train_graph, node_map = build_graph(edges_df, node_df, ["similarity"])
    test_graph, _ = build_graph(edges_df, node_df, ["similarity"])

    # Node features: only shared columns between train and test
    shared_cols = set(train_df.columns).intersection(set(test_df.columns))
    feat_cols = sorted([c for c in shared_cols if c not in ["node_id", target_col_train, "sample_id"]])

    x = torch.zeros((len(node_map), len(feat_cols)), dtype=torch.float32)
    train_idx = torch.tensor([node_map[i] for i in train_df.node_id], dtype=torch.long)
    test_idx = torch.tensor([node_map[i] for i in test_df.node_id], dtype=torch.long)
    x[train_idx] = torch.tensor(train_df[feat_cols].values, dtype=torch.float32)
    x[test_idx] = torch.tensor(test_df[feat_cols].values, dtype=torch.float32)

    train_graph["node"].x = x
    test_graph["node"].x = x

    # Labels (train only; others are -1)
    y = -1 * np.ones(len(node_map), dtype=int)
    y[train_idx] = train_df[target_col_train].values.astype(int)
    y = torch.tensor(y, dtype=torch.long)
    train_graph["node"].y = y
    test_graph["node"].y = y

    # Split training data into train and validation sets
    train_indices, val_indices = train_test_split(train_idx.numpy(), test_size=0.2, random_state=42)
    train_idx = torch.tensor(train_indices, dtype=torch.long)
    val_idx = torch.tensor(val_indices, dtype=torch.long)

    # Define the model
    in_channels = train_graph["node"].x.size(1)
    hidden_channels = 128
    out_channels = 2
    model = GNN(in_channels, hidden_channels, out_channels)

    # Move model to CPU
    device = torch.device('cpu')
    model.to(device)
    train_graph.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    best_f1 = 0
    for epoch in range(1, 201):
        loss = train(model, train_graph, train_idx, optimizer, criterion)
        f1, acc, prec, rec = evaluate(model, train_graph, val_idx)
        if f1 > best_f1:
            best_f1 = f1
            best_state_dict = model.state_dict()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {f1:.4f}, Val Acc: {acc:.4f}, Val Prec: {prec:.4f}, Val Rec: {rec:.4f}')

    # Load best model
    model.load_state_dict(best_state_dict)

    # Generate predictions for the test set
    model.eval()
    with torch.no_grad():
        test_out = model(test_graph["node"].x, test_graph["node", "similarity", "node"].edge_index)
        test_pred = test_out.argmax(dim=-1)

    # Prepare submission DataFrame
    test_node_ids = test_df["node_id"].tolist()
    submission_df = pd.DataFrame({
        "node_id": test_node_ids,
        "target": test_pred[test_idx].cpu().numpy()  # Ensure we only take predictions for test nodes
    })

    # Save submission file
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")

if __name__ == "__main__":
    main()