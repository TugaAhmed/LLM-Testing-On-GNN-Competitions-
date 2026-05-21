import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import random

# Set seeds for reproducibility
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

# Load data
train_nodes = pd.read_csv('data/public/train_nodes.csv', dtype={'id': int, 'timestep': int})
train_labels = pd.read_csv('data/public/train_labels.csv', dtype={'id': int, 'y': int})
test_nodes = pd.read_csv('data/public/test_nodes.csv', dtype={'id': int, 'timestep': int})
edgelist = pd.read_csv('data/public/edgelist.csv', dtype={'txId1': int, 'txId2': int})

# Preprocess data
train_nodes.set_index('id', inplace=True)
train_labels.set_index('id', inplace=True)
test_nodes.set_index('id', inplace=True)

# Convert features to float32
x_train = torch.tensor(train_nodes.values, dtype=torch.float32)
x_test = torch.tensor(test_nodes.values, dtype=torch.float32)

# Convert labels to tensor
y_train = torch.tensor(train_labels['y'].values, dtype=torch.long)

# Filter edgelist to ensure edge indices are within the bounds of the number of nodes
valid_edges = edgelist[(edgelist['txId1'].isin(train_nodes.index)) & (edgelist['txId2'].isin(train_nodes.index))]
edge_index = torch.tensor([valid_edges['txId1'].values, valid_edges['txId2'].values], dtype=torch.long)

# Create a PyTorch Geometric Data object
data = Data(x=x_train, edge_index=edge_index, y=y_train)

# Split training data into training and validation sets
train_indices, val_indices = train_test_split(range(len(data.y)), test_size=0.2, stratify=data.y.numpy(), random_state=seed_value)
train_mask = torch.zeros(len(data.y), dtype=torch.bool)
val_mask = torch.zeros(len(data.y), dtype=torch.bool)
train_mask[train_indices] = True
val_mask[val_indices] = True
data.train_mask = train_mask
data.val_mask = val_mask

# Define the GNN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model, optimizer, and loss function
device = torch.device('cpu')
model = GCN(in_channels=data.num_features, hidden_channels=128, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Validation function
def validate():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        val_f1 = f1_score(data.y[data.val_mask].numpy(), pred[data.val_mask].numpy(), average='macro')
    return val_f1

# Training loop with early stopping
best_val_f1 = 0
patience = 10
trigger_times = 0
epochs = 200

for epoch in range(1, epochs + 1):
    loss = train()
    val_f1 = validate()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Macro-F1: {val_f1:.4f}')
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!')
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Generate predictions for the test set
with torch.no_grad():
    test_out = model(x_test, data.edge_index)
    test_pred = test_out.argmax(dim=1) + 1  # Convert to 1-based class labels

# Prepare submission DataFrame
submission_df = pd.DataFrame({'id': test_nodes.index, 'y_pred': test_pred.numpy()})
submission_df['id'] = submission_df['id'].astype(int)

# Write the submission file
submission_df.to_csv('submissions/submission.csv', index=False)

print("Submission file generated successfully.")