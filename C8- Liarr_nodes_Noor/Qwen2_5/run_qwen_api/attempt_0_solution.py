import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.loader import NeighborSampler, NeighborLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load data
train = pd.read_csv('data/train_compressed.csv')
labels = pd.read_csv('data/labels.csv')
edges = pd.read_csv('data/edges.csv')
test = pd.read_csv('data/test.csv')

# Prepare node features
y_train = labels['label'].values
classes = np.unique(y_train)
num_nodes = max(train["cell_type"].max(), test["cell_type"].max()) + 1
num_features = train.shape[1] - 2  # Exclude 'cell_type' and 'is_perturbed'

x = torch.zeros((num_nodes, num_features), dtype=torch.float32)
x[train["cell_type"].values] = torch.tensor(train.iloc[:, 1:-1].values, dtype=torch.float32)
x[test["cell_type"].values] = torch.tensor(test.iloc[:, 1:].values, dtype=torch.float32)

# Prepare edge indices and weights
edge_index = torch.tensor(edges[["source", "target"]].values.T, dtype=torch.long)
edge_weight = torch.tensor(edges["weight"].values, dtype=torch.float32)

# Prepare labels
y = torch.full((num_nodes,), -1, dtype=torch.long)
y[labels["cell_type"].values] = torch.tensor(labels["label"].values, dtype=torch.long)

# Create masks
train_nodes = labels["cell_type"].values
train_ids, val_ids = train_test_split(train_nodes, test_size=0.2, random_state=42)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_ids] = True
val_mask[val_ids] = True
test_mask[test["cell_type"].values] = True

# Create Data object
data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)

# Define NeighborLoaders
train_loader = NeighborLoader(data, input_nodes=train_ids, sizes=[10, 10], batch_size=256, shuffle=True)
val_loader = NeighborLoader(data, input_nodes=val_ids, sizes=[10, 10], batch_size=256, shuffle=False)
test_loader = NeighborLoader(data, input_nodes=test["cell_type"].values, sizes=[10, 10], batch_size=256, shuffle=False)

# Define the GNN model with attention mechanism
class AttentionGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

# Initialize model, optimizer, and loss function
model = AttentionGNN(num_features, 8, len(classes))
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.NLLLoss()

# Compute class weights
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Training loop
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, edge_weight=batch.edge_weight)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.batch_size
    return total_loss / len(train_ids)

# Validation loop
@torch.no_grad()
def validate():
    model.eval()
    ys, preds = [], []
    for batch in val_loader:
        ys.append(batch.y)
        out = model(batch.x, batch.edge_index, edge_weight=batch.edge_weight)
        preds.append(out.argmax(dim=-1))
    y, pred = torch.cat(ys), torch.cat(preds)
    acc = accuracy_score(y.cpu().numpy(), pred.cpu().numpy())
    f1 = f1_score(y.cpu().numpy(), pred.cpu().numpy(), average='weighted')
    return acc, f1

# Training and validation
best_val_acc = 0
best_model_state = None
for epoch in range(1, 201):
    loss = train()
    val_acc, val_f1 = validate()
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()

# Load best model state
model.load_state_dict(best_model_state)
model.eval()

# Generate predictions for the test set
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        out = model(batch.x, batch.edge_index, edge_weight=batch.edge_weight)
        preds = out[:batch.batch_size].argmax(dim=1)
        all_test_preds.append(preds)

all_test_preds = torch.cat(all_test_preds).cpu().numpy()

# Write submission file
submission_df = pd.DataFrame({
    'id': test["cell_type"].values,
    'cell_type': all_test_preds
})
submission_df.to_csv('submission.csv', index=False)