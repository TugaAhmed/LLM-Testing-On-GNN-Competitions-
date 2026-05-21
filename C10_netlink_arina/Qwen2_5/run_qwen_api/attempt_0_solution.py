import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load data
node_features = pd.read_csv('data/public/node_features.csv')
train_edges = pd.read_csv('data/public/train_edges.csv')
val_edges = pd.read_csv('data/public/val_edges.csv')
test_edges = pd.read_csv('data/public/test_nodes.csv')

# Preprocess data
x = torch.tensor(node_features.iloc[:, 1:].values, dtype=torch.float32)
edge_index = torch.tensor([train_edges['source'].values, train_edges['target'].values], dtype=torch.long)

# Create train and validation sets
train_pos_edge_index = edge_index
train_neg_edge_index = torch.tensor(
    train_test_split(
        np.array([[i, j] for i in range(3327) for j in range(i+1, 3327) if j not in train_edges[train_edges['source'] == i]['target'].values]),
        train_size=2730, shuffle=True, random_state=42
    )[0].T, dtype=torch.long
)

val_pos_edge_index = torch.tensor([val_edges['source'].values, val_edges['target'].values], dtype=torch.long)
val_neg_edge_index = torch.tensor(
    train_test_split(
        np.array([[i, j] for i in range(3327) for j in range(i+1, 3327) if j not in val_edges[val_edges['source'] == i]['target'].values]),
        train_size=911, shuffle=True, random_state=42
    )[0].T, dtype=torch.long
)

# Define the GraphSAGE model
class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

# Initialize the model, optimizer, and loss function
model = GraphSAGEModel(in_channels=x.size(1), hidden_channels=128, out_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    
    z = model(x, train_pos_edge_index)
    logits = model.decode(z, train_pos_edge_index, train_neg_edge_index)
    y_true = torch.cat([torch.ones(train_pos_edge_index.size(1)), torch.zeros(train_neg_edge_index.size(1))], dim=0)
    
    loss = criterion(logits, y_true)
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Evaluation loop
@torch.no_grad()
def evaluate(edge_index, neg_edge_index):
    model.eval()
    
    z = model(x, train_pos_edge_index)
    logits = model.decode(z, edge_index, neg_edge_index)
    y_true = torch.cat([torch.ones(edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0)
    
    auc_roc = roc_auc_score(y_true.numpy(), logits.numpy())
    return auc_roc

# Train the model
best_val_auc_roc = 0.0
for epoch in range(1, 101):
    loss = train()
    val_auc_roc = evaluate(val_pos_edge_index, val_neg_edge_index)
    
    if val_auc_roc > best_val_auc_roc:
        best_val_auc_roc = val_auc_roc
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC-ROC: {val_auc_roc:.4f}')

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Generate predictions for the test set
test_edge_index = torch.tensor([test_edges['source'].values, test_edges['target'].values], dtype=torch.long)
z = model(x, train_pos_edge_index)
logits = model.decode(z, test_edge_index, torch.empty((2, 0), dtype=torch.long))
pred = torch.sigmoid(logits)

# Write the submission file
submission_df = pd.DataFrame({
    'id': test_edges['id'],
    'y_pred': pred.numpy()
})
submission_df.to_csv('submission.csv', index=False)

print("Submission file generated successfully.")