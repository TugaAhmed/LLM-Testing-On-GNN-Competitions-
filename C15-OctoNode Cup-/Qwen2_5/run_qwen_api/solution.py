import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split  # Import train_test_split
import numpy as np
import random

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Load data
node_features = np.load('data/public/node_features.npy').astype(np.float32)
edges = pd.read_csv("data/public/musae_git_edges.csv")
train_labels = pd.read_csv('data/public/train_target.csv')
test_ids = pd.read_csv('data/public/test_target_without_labels.csv')

# Preprocess data
num_nodes = node_features.shape[0]
num_features = node_features.shape[1]
x = torch.tensor(node_features, dtype=torch.float32)

edge_index = torch.tensor(edges[['id_1', 'id_2']].values.T, dtype=torch.long)
y = torch.full((num_nodes,), -1, dtype=torch.long)

for _, row in train_labels.iterrows():
    y[row['id']] = row['ml_target']

# Create masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_ids, val_ids = train_test_split(
    train_labels['id'].values,
    test_size=0.2,
    random_state=SEED,
    stratify=train_labels['ml_target']
)

train_mask[train_ids] = True
val_mask[val_ids] = True

data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask)

# Define the GNN model
class SimpleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model, optimizer, and loss function
model = SimpleGNN(num_features, 64, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Compute class weights for handling class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels['ml_target']), y=train_labels['ml_target'])
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Training loop with early stopping
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(mask):
    model.eval()
    out = model(data.x, data.edge_index)[mask]
    pred = out.argmax(dim=1)
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    f1 = f1_score(data.y[mask].cpu(), pred.cpu(), average='macro')
    return acc, f1

best_val_f1 = 0
patience_counter = 0
max_patience = 10

for epoch in range(1, 201):
    loss = train()
    train_acc, _ = test(data.train_mask)
    val_acc, val_f1 = test(data.val_mask)
    
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= max_patience:
        print('Early stopping')
        break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Generate predictions for the test set
with torch.no_grad():
    test_out = model(data.x, data.edge_index)[~data.train_mask & ~data.val_mask]
    test_pred = test_out.argmax(dim=1)

# Prepare the submission file
submission_df = pd.DataFrame({
    'id': test_ids['id'],
    'name': test_ids['name'],
    'ml_target': test_pred.cpu().numpy()
})

submission_df.to_csv('submission.csv', index=False)

print('Submission file generated successfully.')