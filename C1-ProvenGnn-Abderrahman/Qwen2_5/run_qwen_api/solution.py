import pandas as pd
import numpy as np
import torch
import random
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.nn import Linear, Sequential, ReLU, Dropout
from torch.optim import Adam

# Function to build dataloader (copied from baseline.py)
def build_dataloader(df, batch_size=32, shuffle=False, has_labels=True):
    graph_list = []

    for _, row in df.iterrows():
        x = torch.tensor(np.vstack(row["node_feat"]).astype(np.float32))
        edge_index = torch.tensor(np.vstack(row["edge_index"]).astype(np.int64))
        edge_attr = torch.tensor(np.vstack(row["edge_attr"]).astype(np.float32))

        if has_labels:
            y = torch.tensor(row["label"], dtype=torch.long)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
            )
        else:
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

        graph_list.append(data)

    return DataLoader(graph_list, batch_size=batch_size, shuffle=shuffle)

# Set seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# Load the dataset
train_df = pd.read_parquet('data/public/train_data.parquet')
test_df = pd.read_parquet('data/public/test_data.parquet')

# Split the training data into training and validation sets
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=seed_value, stratify=train_df['label'])

# Build dataloaders
train_loader = build_dataloader(train_df, batch_size=32, shuffle=True, has_labels=True)
val_loader = build_dataloader(val_df, batch_size=32, shuffle=False, has_labels=True)
test_loader = build_dataloader(test_df, batch_size=32, shuffle=False, has_labels=False)

# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, hidden_channels=256, heads=8, dropout=0.6):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(527, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.lin = Linear(hidden_channels * heads, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

# Initialize the model, loss function, and optimizer
device = torch.device('cpu')
model = GNNModel().to(device)
optimizer = Adam(model.parameters(), lr=0.005)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train(loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# Validation function
def validate(loader):
    model.eval()
    correct = 0
    preds = []
    true_labels = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        preds.extend(pred.cpu().numpy())
        true_labels.extend(data.y.cpu().numpy())
    val_f1 = f1_score(true_labels, preds, average='macro')
    return correct / len(loader.dataset), val_f1

# Training loop with early stopping
best_val_f1 = 0
patience = 10
trigger_times = 0
num_epochs = 100

for epoch in range(1, num_epochs + 1):
    loss = train(train_loader)
    train_acc, _ = validate(train_loader)
    val_acc, val_f1 = validate(val_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    
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
test_preds = []
for data in test_loader:
    data = data.to(device)
    with torch.no_grad():
        out = model(data)
    pred = out.argmax(dim=1)
    test_preds.extend(pred.cpu().numpy())

# Write the submission file
submission_df = pd.DataFrame({'id': test_df['id'], 'y_pred': test_preds})
submission_df.to_csv('submissions/submission.csv', index=False)