import os
import numpy as np
import pandas as pd
import torch
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
from utils import perturb_graph

# Set seeds for reproducibility
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

# Load the MUTAG dataset
dataset = TUDataset(root='./data', name='MUTAG')

# Read train and test indices
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

train_indices = train_df['graph_index'].values - 1  # Convert to 0-based indexing
test_indices = test_df['graph_index'].values - 1    # Convert to 0-based indexing

# Filter the dataset to include only the relevant graphs
train_dataset = [dataset[i] for i in train_indices]
test_dataset = [dataset[i] for i in test_indices]

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        return x

# Initialize model, loss function, and optimizer
model = GNN(dataset.num_features, dataset.num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop with early stopping
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_dataset)

def validate():
    model.eval()
    correct = 0
    preds = []
    true_labels = []
    for data in test_loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        preds.extend(pred.tolist())
        true_labels.extend(data.y.tolist())
    val_acc = correct / len(test_dataset)
    val_f1 = f1_score(true_labels, preds, average='macro')
    return val_acc, val_f1

best_val_f1 = 0
patience_counter = 0
max_patience = 10

for epoch in range(1, 201):
    loss = train()
    val_acc, val_f1 = validate()
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    
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

# Generate predictions for ideal test set
ideal_preds = []
for data in test_loader:
    out = model(data)
    pred = out.argmax(dim=1)
    ideal_preds.extend(pred.tolist())

# Generate predictions for perturbed test set
perturbed_preds = []
perturbed_loader = DataLoader([perturb_graph(graph) for graph in test_dataset], batch_size=32, shuffle=False)
for data in perturbed_loader:
    out = model(data)
    pred = out.argmax(dim=1)
    perturbed_preds.extend(pred.tolist())

# Write the submission files
ideal_submission_df = pd.DataFrame({'graph_index': test_df['graph_index'], 'label': ideal_preds})
perturbed_submission_df = pd.DataFrame({'graph_index': test_df['graph_index'], 'label': perturbed_preds})

ideal_submission_df.to_csv('submissions/ideal_submission.csv', index=False)
perturbed_submission_df.to_csv('submissions/perturbed_submission.csv', index=False)

print("Submission files generated successfully.")