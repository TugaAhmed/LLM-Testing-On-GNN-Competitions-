import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_max_pool
from scipy.sparse import csr_matrix
import random
from sklearn.metrics import accuracy_score, f1_score

# Set seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

class GraphDataset:
    def __init__(self, split='train'):
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of train/val/test")
        
        self.split = split

        # Load graph structure
        self.edges = np.loadtxt("data/public/A.txt", delimiter=",", dtype=int)
        self.node_graph_id = np.load("data/public/node_graph_id.npy")

        # Load graph IDs and labels for this split
        self.graph_ids = np.load(f"data/public/{self.split}_idx.npy")
        if self.split != 'test':
            self.graph_labels = pd.read_csv(f"data/public/{self.split}_labels.csv")
            self.label_map = dict(zip(self.graph_labels["id"], self.graph_labels["y_true"]))  # fast O(1) lookup

        # Load and combine features
        spacy_features = np.load("data/public/new_spacy_feature.npz")
        bert_features = np.load("data/public/new_bert_feature.npz")
        profile_features = np.load("data/public/new_profile_feature.npz")

        spacy_sparse = csr_matrix((spacy_features["data"], spacy_features["indices"], spacy_features["indptr"]), shape=tuple(spacy_features["shape"]))
        bert_sparse = csr_matrix((bert_features["data"], bert_features["indices"], bert_features["indptr"]), shape=tuple(bert_features["shape"]))
        profile_sparse = csr_matrix((profile_features["data"], profile_features["indices"], profile_features["indptr"]), shape=tuple(profile_features["shape"]))

        self.node_features = torch.tensor(np.hstack((spacy_sparse.toarray(), bert_sparse.toarray(), profile_sparse.toarray())), dtype=torch.float)

    def build_graph(self, g_id):
        nodes = np.where(self.node_graph_id == g_id)[0]
        mask = np.isin(self.edges[:,0], nodes) & np.isin(self.edges[:,1], nodes)
        edge_index = self.edges[mask]

        node_map = {node:i for i,node in enumerate(nodes)}
        edge_index = np.array([[node_map[u], node_map[v]] for u,v in edge_index]).T
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        x = self.node_features[nodes]
        if self.split != 'test':
            y = torch.tensor([self.label_map[g_id]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y)
        else:
            data = Data(x=x, edge_index=edge_index)  # no label for test set

        data.graph_id = torch.tensor([g_id], dtype=torch.long)

        return data

    def get_loader(self, batch_size=128, shuffle=True):
        dataset = [self.build_graph(g) for g in self.graph_ids]
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class GNN(torch.nn.Module):
    def __init__(self, num_features):
        super(GNN, self).__init__()
        self.conv1 = GATConv(num_features, 256, heads=8, dropout=0.6)
        self.conv2 = GATConv(256 * 8, 256, heads=8, concat=False, dropout=0.6)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.6)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = global_max_pool(x, batch)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return torch.sigmoid(x)

epochs_no = 30
device = torch.device("cpu")
train_loader = GraphDataset(split='train').get_loader(batch_size=32) 
val_loader   = GraphDataset(split='val').get_loader(batch_size=32, shuffle=False)
test_loader  = GraphDataset(split='test').get_loader(batch_size=32, shuffle=False)

# Model
sample_data = next(iter(train_loader)) 
model = GNN(sample_data.num_features).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
loss_fnc = torch.nn.BCELoss()

def metrics(preds, gts):
    preds = torch.round(torch.cat(preds)).int()
    gts = torch.cat(gts).int()
    return accuracy_score(preds, gts), f1_score(preds, gts)

def train_epoch():
    model.train()
    total_loss = 0
    for data in train_loader: 
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fnc(out.view(-1), data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def validate():
    model.eval()
    all_preds, all_labels = [], []
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        all_preds.append(out.view(-1).cpu())
        all_labels.append(data.y.cpu())
    return metrics(all_preds, all_labels)

@torch.no_grad()
def test():
    model.eval()
    all_preds, all_ids = [], []
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        all_preds.append(torch.round(out.view(-1)).int().cpu())
        all_ids.append(data.graph_id.cpu())
    return torch.cat(all_preds), torch.cat(all_ids)

# Training loop
best_val_acc = 0.0

for epoch in range(epochs_no):
    train_loss = train_epoch()
    val_acc, val_f1 = validate()
    print(f"Epoch {epoch:02d} | TrainLoss: {train_loss:.4f} | ValAcc: {val_acc:.4f} | ValF1: {val_f1:.4f}")
   
    # Save if this is the best validation accuracy so far
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/saved_model.model")

# Load the best model
model.load_state_dict(torch.load("models/saved_model.model"))
model.to(device)

# Generate predictions for the test set
preds, ids = test()

# Write the submission file
submission_df = pd.DataFrame({'id': ids.numpy(), 'y_pred': preds.numpy()})
submission_df.to_csv('submission.csv', index=False)