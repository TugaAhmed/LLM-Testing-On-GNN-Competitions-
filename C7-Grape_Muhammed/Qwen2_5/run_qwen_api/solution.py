import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch.nn as nn

# Set seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

def load_graphs(graph_path, label_path=None):
    df = pd.read_csv(graph_path)
    labels = pd.read_csv(label_path) if label_path else None
    graphs = []
    
    for gid in df['graph_id'].unique():
        g = df[df['graph_id']==gid].reset_index(drop=True)
        
        # Node features: x, y, width + node type encoding
        node_type_map = {'junction': 0, 'endpoint': 1}
        type_vals = g['type'].map(lambda t: node_type_map.get(t, 0)).values if 'type' in g.columns else np.zeros(len(g))
        
        x = np.column_stack([
            g['x'].values / 600,
            g['y'].values / 600,
            g['width'].values / 20,
            type_vals,  # Node type
        ])
        x = torch.tensor(x, dtype=torch.float32)
        
        # Build edge list
        edges = []
        node_map = {row['node_id']: i for i, row in g.iterrows()}
        for i, row in g.iterrows():
            if pd.notna(row['edges']) and row['edges']:
                for tgt in str(row['edges']).split(';'):
                    if tgt.strip().isdigit():
                        tgt_id = int(tgt)
                        if tgt_id in node_map:
                            edges.append([i, node_map[tgt_id]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2,0), dtype=torch.long)
        
        y = torch.tensor([labels[labels['graph_id']==gid]['label'].values[0]], dtype=torch.long) if labels is not None else None
        data = Data(x=x, edge_index=edge_index, y=y)
        data.gid = gid
        graphs.append(data)
    
    return graphs

class SimpleGCN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

def train_model(model, train_loader, val_loader, optimizer, criterion, device, patience=5):
    best_val_f1 = 0
    epochs_without_improvement = 0
    
    for epoch in range(1, 101):  # Limit to 100 epochs for time constraint
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
        
        avg_loss = total_loss / len(train_loader.dataset)
        
        model.eval()
        correct = 0
        preds = []
        true_labels = []
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            preds.extend(pred.cpu().numpy())
            true_labels.extend(data.y.cpu().numpy())
        
        val_acc = correct / len(val_loader.dataset)
        val_f1 = f1_score(true_labels, preds, average='macro')
        
        print(f'Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print('Early stopping triggered.')
            break

def generate_predictions(model, test_loader, device):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    predictions = {}
    
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        for i in range(data.num_graphs):
            predictions[data.gid[i]] = pred[i].item()
    
    return predictions

def main():
    data_dir = 'data/public/'
    train_data_path = os.path.join(data_dir, 'train_data.csv')
    train_labels_path = os.path.join(data_dir, 'train_labels.csv')
    test_data_path = os.path.join(data_dir, 'test_data.csv')
    
    graphs = load_graphs(train_data_path, train_labels_path)
    test_graphs = load_graphs(test_data_path)
    
    # Split into train and validation sets
    train_graphs, val_graphs = train_test_split(graphs, test_size=0.15, stratify=[g.y.item() for g in graphs], random_state=seed_value)
    
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
    
    device = torch.device('cpu')
    model = SimpleGCN(num_node_features=4, hidden_channels=32, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    train_model(model, train_loader, val_loader, optimizer, criterion, device)
    
    predictions = generate_predictions(model, test_loader, device)
    
    # Write predictions to submission.csv
    submission_df = pd.DataFrame(list(predictions.items()), columns=['graph_id', 'label'])
    submission_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()