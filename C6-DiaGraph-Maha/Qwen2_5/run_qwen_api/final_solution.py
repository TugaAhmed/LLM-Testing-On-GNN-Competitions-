import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
import torch.nn as nn

# Set seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# Load CSVs
nodes_df = pd.read_csv("data/public/nodes.csv")  # nodes * features
edges_df = pd.read_csv("data/public/edges.csv")  # edges_count * 2 
train_df = pd.read_csv("data/public/train.csv")  # id , label
val_df   = pd.read_csv("data/public/val.csv")    # id , label
test_df  = pd.read_csv("data/public/test_nodes.csv")  # id 

# Handle NaN and Inf values in node features
nodes_df.replace([np.inf, -np.inf], np.nan, inplace=True)
nodes_df.fillna(nodes_df.mean(), inplace=True)

node_ids = nodes_df.iloc[:, 0].values
features = nodes_df.iloc[:, 1:].values

x = torch.tensor(features, dtype=torch.float32)
src = edges_df.iloc[:, 0].values
dst = edges_df.iloc[:, 1].values
edge_index = torch.tensor([src, dst], dtype=torch.long)

# Initialize y with -1 values 
num_nodes = len(node_ids)
y = torch.full((num_nodes,), -1, dtype=torch.long)

# Fill y with labels from train.csv and val.csv 
for _, row in train_df.iterrows():
    y[row['id']] = int(row['diabetes'])

for _, row in val_df.iterrows():
    y[row['id']] = int(row['diabetes'])

# Masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool) 
val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_df.iloc[:,0]] = True
val_mask[val_df.iloc[:,0]] = True
test_mask[test_df.iloc[:,0]] = True

data = Data(
    x=x,                     # all nodes
    edge_index=edge_index,   # all edges
    y=y,                     # labels (-1 for test)
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)

# Define the GNN model
class GNNModel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model, loss function, and optimizer
device = torch.device('cpu')
model = GNNModel(num_features=data.num_features, hidden_channels=64, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

# Training loop with early stopping
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        acc = pred[mask] == data.y[mask]
        return acc.sum() / mask.sum().item(), f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='macro')

best_val_f1 = 0
patience_counter = 0
patience_limit = 10

for epoch in range(1, 201):
    train_loss = train()
    train_acc, train_f1 = evaluate(data.train_mask)
    val_acc, val_f1 = evaluate(data.val_mask)
    
    print(f'Epoch {epoch:03d}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= patience_limit:
        print('Early stopping triggered.')
        break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Generate predictions for the test set
with torch.no_grad():
    test_out = model(data.x, data.edge_index)
    test_pred = test_out.argmax(dim=1)[data.test_mask]

# Prepare the submission DataFrame
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'y_pred': test_pred.cpu().numpy()
})

# Write the predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)

# Encrypt the predictions file
os.system('python encryption/encrypt.py submission.csv encryption/public_key.pem submission.csv.enc')

# Clean up
del model, data, submission_df, test_out, test_pred
torch.cuda.empty_cache() if torch.cuda.is_available() else None