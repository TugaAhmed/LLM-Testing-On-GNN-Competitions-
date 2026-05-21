import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, random_split
from torch.optim import Adam
from tqdm import tqdm

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# MANDATORY graph loading function
def nx_to_pyg(path):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    # OSMnx node IDs are large integers — must remap to 0..N-1
    nodes = list(G.nodes())
    id_map = {n: i for i, n in enumerate(nodes)}
    # edges: G.edges() returns (u, v, key) for MultiDiGraph — strip key
    edges = [(id_map[u], id_map[v]) for u, v, *_ in G.edges()
             if u in id_map and v in id_map]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    x_list = [[G.nodes[n].get('x', 0.0),
               G.nodes[n].get('y', 0.0),
               float(G.degree(n))] for n in nodes]
    x = torch.tensor(x_list, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, num_nodes=len(nodes))

class CityGraphDataset(Dataset):
    def __init__(self, root_dir, labels_df=None):
        self.root_dir = root_dir
        self.labels_df = labels_df
        self.filenames = os.listdir(root_dir)
        if labels_df is not None:
            self.labels = labels_df.set_index('filename')['target'].to_dict()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.root_dir, filename)
        data = nx_to_pyg(path)
        if self.labels_df is not None:
            data.y = torch.tensor([self.labels[filename]], dtype=torch.long)
        return data

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global pooling
        x_mean = torch_geometric.nn.global_mean_pool(x, batch)
        x_max = torch_geometric.nn.global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Classifier
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=100):
    best_val_f1 = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        model.eval()
        val_preds, val_true = [], []
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            val_preds.extend(pred.cpu().numpy())
            val_true.extend(data.y.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='macro')

        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')

def main():
    device = torch.device('cpu')
    train_root = 'data/train'
    test_root = 'data/test'
    train_labels_path = 'data/train_labels.csv'

    # Load train labels
    train_labels_df = pd.read_csv(train_labels_path)

    # Create datasets
    train_dataset = CityGraphDataset(train_root, train_labels_df)
    test_dataset = CityGraphDataset(test_root)

    # Calculate class weights
    class_counts = train_labels_df['target'].value_counts().sort_index().values
    class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Split train dataset into train and validation sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_idx = next(sss.split(train_dataset, train_dataset.labels_df['target']))
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model, optimizer
    model = GCN(in_channels=3, hidden_channels=64, out_channels=3).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # Train model
    train_model(model, train_loader, val_loader, optimizer, criterion, device)

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Generate predictions for test set
    test_preds = []
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        test_preds.extend(pred.cpu().numpy())

    # Write submission file
    submission_df = pd.DataFrame({
        'filename': test_dataset.filenames,
        'prediction': test_preds
    })
    submission_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()