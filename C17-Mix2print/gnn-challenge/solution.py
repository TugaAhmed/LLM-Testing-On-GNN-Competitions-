import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import random
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Load the dataset
def load_dataset(data_dir):
    train_ids = sorted([int(f.split('_')[1]) for f in os.listdir(os.path.join(data_dir, 'train_graphs')) if f.endswith('_X.npy')])
    test_ids = sorted([int(f.split('_')[1]) for f in os.listdir(os.path.join(data_dir, 'test_graphs')) if f.endswith('_X.npy')])
    
    train_data_list = []
    for graph_id in train_ids:
        X = np.load(os.path.join(data_dir, 'train_graphs', f'graph_{graph_id}_X.npy'))
        A = np.load(os.path.join(data_dir, 'train_graphs', f'graph_{graph_id}_A.npy'))
        y = np.load(os.path.join(data_dir, 'train_graphs', f'graph_{graph_id}_y.npy'))
        edge_index = torch.tensor(np.array((A > 0).nonzero()), dtype=torch.long)
        data = Data(x=torch.from_numpy(X).float(), edge_index=edge_index, y=torch.from_numpy(y).float().unsqueeze(0))
        train_data_list.append(data)
    
    test_data_list = []
    for graph_id in test_ids:
        X = np.load(os.path.join(data_dir, 'test_graphs', f'graph_{graph_id}_X.npy'))
        A = np.load(os.path.join(data_dir, 'test_graphs', f'graph_{graph_id}_A.npy'))
        edge_index = torch.tensor(np.array((A > 0).nonzero()), dtype=torch.long)
        data = Data(x=torch.from_numpy(X).float(), edge_index=edge_index)
        test_data_list.append(data)
    
    return train_data_list, test_data_list, train_ids, test_ids

# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Training function
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=100, patience=10):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                out = model(data)
                loss = criterion(out, data.y.squeeze(1))
                val_loss += loss.item() * data.num_graphs
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            print("Early stopping")
            break

# Main function
def main():
    data_dir = './data/public'
    train_data_list, test_data_list, train_ids, test_ids = load_dataset(data_dir)
    
    # Split into train and validation sets
    train_indices, val_indices = train_test_split(range(len(train_data_list)), test_size=0.2, random_state=42)
    train_loader = DataLoader([train_data_list[i] for i in train_indices], batch_size=32, shuffle=True)
    val_loader = DataLoader([train_data_list[i] for i in val_indices], batch_size=32, shuffle=False)
    
    # Initialize model, optimizer, and loss function
    input_dim = train_data_list[0].x.shape[1]
    hidden_dim = 64
    output_dim = 3
    model = GNNModel(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Train the model
    train(model, train_loader, val_loader, optimizer, criterion)
    
    # Generate predictions for the test set
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_data_list:
            out = model(data)
            predictions.append(out.numpy().flatten())  # Flatten the output to ensure correct shape
    
    predictions = np.array(predictions)
    
    # Write the predictions to a CSV file
    submission_df = pd.DataFrame({
        'id': test_ids,
        'pressure': predictions[:, 0],
        'temperature': predictions[:, 1],
        'speed': predictions[:, 2]
    })
    submission_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()