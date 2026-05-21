import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load data
train_features = np.load("data/public/node_features_train.npy").astype(np.float32)  # shape (484, 200, 200)
test_features = np.load("data/public/node_features_test.npy").astype(np.float32)  # shape (153, 200, 200)
adj_train = np.load("data/public/adj_train.npy").astype(np.float32)  # shape  (484, 200, 200)
adj_test = np.load("data/public/adj_test.npy").astype(np.float32)  # (153, 200, 200)
train_labels = pd.read_csv("data/public/train_label.csv")  # (484, 2)

# Build graph data objects
def build_graph(adj_matrix, node_features, label=None):
    num_nodes = adj_matrix.shape[0]

    # Get edges (non-zero connections)
    src, dst = np.nonzero(adj_matrix)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = adj_matrix[src, dst]
    edge_weight = np.clip(edge_weight, 0, 1)  # prevent explosion
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    edge_weight = edge_weight.view(-1, 1)

    x = torch.tensor(node_features, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight
    )

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    return data

graphs = []
for i in range(len(adj_train)):
    g = build_graph(adj_train[i], train_features[i], train_labels['label'].to_numpy()[i])
    graphs.append(g)

# Split into train and validation sets
train_graphs, val_graphs = train_test_split(
    graphs, test_size=0.2, random_state=SEED, stratify=train_labels['label'].to_numpy())

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32)

# Define the GNN model
class GNNModel(nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# Initialize model, loss function, and optimizer
device = torch.device('cpu')
model = GNNModel(num_node_features=200, hidden_channels=128, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# Training loop with early stopping
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def validate():
    model.eval()
    pred = []
    true = []
    for data in val_loader:
        data = data.to(device)
        out = model(data)
        pred.extend(out.argmax(dim=1).cpu().numpy())
        true.extend(data.y.cpu().numpy())
    return f1_score(true, pred, average='macro')

best_val_f1 = 0
patience_counter = 0
max_patience = 10

for epoch in range(1, 1001):
    loss = train()
    val_f1 = validate()
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Macro F1: {val_f1:.4f}')

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print('Early stopping triggered.')
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Generate predictions for the test set
test_graphs = []
for i in range(len(adj_test)):
    g = build_graph(adj_test[i], test_features[i])
    test_graphs.append(g)

test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

preds = []
ids = list(range(1, len(adj_test) + 1))

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        preds.extend(out.argmax(dim=1).cpu().numpy())

# Write the submission file
submission_df = pd.DataFrame({'id': ids, 'y_pred': preds})
submission_df.to_csv('submission.csv', index=False)

# Encrypt the submission file
os.system('python extra/encrypt.py submission.csv')