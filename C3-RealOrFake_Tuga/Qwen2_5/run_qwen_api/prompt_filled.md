You are solving a GNN coding competition.

You will receive:
  (1) the competition README,
  (2) the repository file tree,
  (3) a data-sample summary,
  (4) the required submission format,
  (5) allowed libraries.
  (6) baseline code
  (7) important notes

Produce a single, self-contained Python script that, when executed
from the repository root on CPU, trains a GNN, generates predictions
for the test set, and writes the submission file.

Respond in EXACTLY this format:

<plan>
A 5-10 bullet plan covering: features, model architecture, training,
validation, decoding, and how the submission file will be written.
</plan>

<code>
A complete, runnable Python script. Constraints:
  - Use only the libraries listed under "Allowed libraries"
    (plus the Python standard library).
  - Runnable as `python solution.py` from the repo root.
  - Set seeds (numpy, torch, random) for reproducibility.
  - Must complete on CPU within 60 minutes.
  - Must write the submission file in the exact required format.
  - Do not download external data; use only files already on disk.
</code>

---





=== SECTION 1: README ===

# 📰 Real Or Fake?! 🕵️‍♂️
## GNN-based Fake News Detection Challenge

Welcome to the **GNN-based Fake News Detection Challenge**! This competition focuses on detecting fake news propagation on Twitter using Graph Neural Networks (GNNs). 


### ⭐📊 **[Live Leaderboard](https://tugaahmed.github.io/Real_Or_Fake/leaderboard.html)** 📊⭐

---



## Repository Structure

```text
Real_Or_Fake/
├── data/
│   ├── public/
│   │   ├── A.txt
│   │   ├── new_bert_feature.npz
│   │   ├── new_spacy_feature.npz
│   │   ├── new_profile_feature.npz
│   │   ├── node_graph_id.npy
│   │   ├── train_idx.npy
│   │   ├── train_labels.csv
│   │   ├── val_idx.npy
│   │   ├── val_labels.csv
│   │   ├── test_idx.csv
│   │   └── test_idx.npy
│   └── test_labels.csv   (kept private, restored via GitHub Secret)
│
├── submissions/
│   ├── sample_submission/
│   │   └── predictions.csv
│   └── inbox/
│       └── team_name/
│           └── run_name/
│               ├── metadata.json
│               └── predictions.csv.gpg
│
├── competition/
│   ├── metrics.py
│   └── scoring_script.py
│
├── docs/
│   ├── leaderboard.css
│   ├── leaderboard.csv
│   ├── leaderboard.html
│   └── leaderboard.js
│
├── key/
│   └── competition_public_key.asc
│
├── models/
│   ├── model.py
│   └── saved_model.model
│
├── dataloader.py
├── evaluate.py
├── test.py
├── train.py
├── update_leaderboard.py
├── validate_submission.py
├── requirements.txt
├── README.md
└── LICENSE
```


---

## 📚 Dataset Source

The dataset used in this repository is from the paper:

> Dou, Y., Shu, K., Xia, C., Yu, P. S., & Sun, L. (2021). *User Preference-aware Fake News Detection*. In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval* (SIGIR '21), pp. 2051–2055. [https://doi.org/10.1145/3404835.3462990](https://doi.org/10.1145/3404835.3462990)




## 📦 Dataset Overview

This competition uses the **GossipCop** dataset, which contains Twitter news propagation graphs. Each graph represents the spread of a single news article

Each graph corresponds to a news article (root node) and all users who engaged with it (child nodes).
  - **Nodes:** represent either the news article or a user who interacted with it.
  - **Edges:** represent interactions or retweets between nodes. Only edges connecting nodes in the same graph are used for that graph
  - The **root node** corresponds to the news article itself.
  - **Child nodes** correspond to users who retweeted or engaged with the news.

Graphs are used as input to Graph Neural Networks (GNNs) to classify news as **Real (0)** or **Fake (1)**.

The dataset is split into **public** and **private** parts:

- **Public:** Available to participants for training, validation, and testing.
- **Private:** Hidden labels used for submission and leaderboard evaluation.

---

### Graph Structure 
<p align="center">
  <img src="images/graph_plot.png" alt="Graph Plot" width="600"/>
</p> 
<p align="center"> <em>Graph visualization generated using ChatGPT</em> </p>

The graph connectivity and graph assignment information are stored in the following files:

- **`A.txt`**  
  Contains all edges in the dataset. Each row is an edge represented by two node IDs (source and target).  
  **Type:** Integer array, shape `(num_edges, 2)`  

- **`node_graph_id.npy`**  
  Maps each node to its corresponding graph. The value at index `i` indicates the graph ID of node `i`.  
  **Type:** Integer array, shape `(num_nodes,)`  
  

### Node Features

Each node in the graph has **text embeddings** and optionally **user profile features**.  

#### 1. Text Embeddings
- **BERT embeddings:** `768-dim` vectors representing the content of the news or user historical tweets.  
  File: `new_bert_feature.npz`  
- **spaCy embeddings:** `300-dim` vectors representing the content of the news or user historical tweets.  
  File: `new_spacy_feature.npz`  

#### 2. User Profile Features (10-dim)
These features are derived from the Twitter user object using the Twitter API:  

1. Verified? (`0` or `1`)  
2. Geo-spatial enabled? (`0` or `1`)  
3. Number of followers  
4. Number of friends  
5. Status/tweet count  
6. Number of favorites  
7. Number of lists the user is part of  
8. Account age (months since Twitter launch)  
9. Number of words in the user’s name  
10. Number of words in the user’s description  

File: `new_profile_feature.npz`  

### Data Splits
- **`train_idx.npy`**  
  Contains the list of **`3826`** graph IDs used for training.  
  **Type:** Integer array, shape `(num_train_graphs,)`  
 

- **`val_idx.npy`**  
  Contains the list of **`546`** graph IDs used for validation.  
  **Type:** Integer array, shape `(num_val_graphs,)`  


- **`test_idx.npy`**  
  Contains the list of **`1092`** graph IDs used for testing. Labels are hidden in the private folder for competition evaluation.  
  **Type:** Integer array, shape `(num_test_graphs,)`  
  
  
### Graph Labels

Each graph in the dataset has a label indicating whether the news is **real** or **fake**:

- `0` → Real news  
- `1` → Fake news  

Graph labels are stored separately for different splits:

- **Training labels:** `train_labels.csv`  
- **Validation labels:** `val_labels.csv`  
- **Test labels (hidden for competition evaluation):** `test_labels.csv`  
- Each CSV file contains two columns:
    1. `id` → Graph ID  
    2. `y_true` → Label (0 or 1)


### Dataset Statistics

Here are some key statistics for the news propagation graphs in the competition datasets:

| Dataset      | #Graphs (Fake) | #Total Nodes | #Total Edges | Avg. Nodes per Graph |
|-------------|----------------|--------------|--------------|--------------------|
| GossipCop (GOS)  | 5,464 (2,732)   | 314,262      | 308,798      | 58                 |


---

## 📝 Problem Statement

**Task:** Classify each news propagation graph as real or fake.

### Baseline Model Description

The baseline model is a **Graph Neural Network (GNN)** for fake news detection implemented in `model.py`. Its main components:

- **Graph Attention Layers (GAT):** 3 layers to learn node embeddings from the propagation graph.  
- **Global Max Pooling:** Aggregates node embeddings to a single graph-level representation.  
- **Root Node Transformation:** Linear layer processes the root node (news article) features.  
- **Concatenation & Output:** Combines graph representation and root node features, then passes through a linear layer with **sigmoid** to predict fake/real news.  

**Features used in baseline:**  
  - spaCy Text embeddings of news and historical user tweets
    
**Output:**
  - Predicted class 

## 🚀 Getting Started

Follow these steps to replicate the baseline results and build your own implementation.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/TugaAhmed/Real_Or_Fake.git
cd Real_Or_Fake
```

### 2️⃣ Set Up Environment

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Download and Prepare the Dataset

Download the public dataset ZIP file from this [link](https://drive.google.com/file/d/1sn_xFAM_9StN49AL50XNJaV3f9_TVKn5/view?usp=sharing) .

Extract the contents inside the data/public folder so that the folder structure looks like this:

```text
├── data/
│   ├── public/
│   │   ├── A.txt
│   │   ├── new_bert_feature.npz
│   │   ├── new_spacy_feature.npz
│   │   ├── new_profile_feature.npz
│   │   ├── node_graph_id.npy
│   │   ├── train_idx.npy
│   │   ├── train_labels.csv
│   │   ├── val_idx.npy
│   │   ├── val_labels.csv
│   │   ├── test_idx.csv
│   │   └── test_idx.npy

```

### 4️⃣ Train the Baseline Model

Run the training script to train the GNN on the dataset:
```bash
python train.py
```

This will train the model and generate `saved_model.model` in the `models/` folder.

The saved model corresponds to the one with the best validation accuracy.

Metrics tracked during training: **Accuracy and F1 score**.

### 5️⃣ Generate Predictions

After training, run the test script to generate predictions:
```bash
python test.py
```
This will create a predictions.csv file inside the submissions/ folder.

The CSV contains two columns: **id** , **y_pred**

### 6️⃣ Evaluate Predictions

You can evaluate your predictions using the evaluation script:

```bash
python evaluate.py
```

⚠️ **Note:**

- The file `test_labels.csv` is **not publicly available**.
- The evaluation script is provided **only to demonstrate how scoring works**.
- The script will run successfully **only when the ground-truth labels are available**.
- Final scoring is performed on the competition server after submission.


**Metrics reported include:**    
   - Accuracy
   - F1 Score
     
## 🎯 Competition Task

Your goal is to **beat the baseline accuracy** on the fake news detection task using the provided Twitter news propagation graphs.

### Baseline Overview
- Uses **only spaCy text embeddings** of news articles and user historical tweets.
- Achieves:
  - **Accuracy:** 0.7216
  - **F1 score:** 0.7071

### Your Task
1. Build a **Graph Neural Network (GNN)** based pipeline.
2. Use **any combination of available features**:
   - SpaCy text embeddings (baseline feature)
   - BERT embeddings (`new_bert_feature.npz`)
   - User profile features (`new_profile_feature.npz`)
3. Train your model on the **public training data** and validate on the validation set.
4. Generate predictions for the **test set** as `predictions.csv`.

### Rules
- Your model **must use a GNN**; other models alone will not be accepted.
- You may combine features in **any way** to improve performance.
- The objective is to **maximize accuracy** on the hidden test set.

---



## 📤 Submission Workflow

Follow these steps to participate in the competition and submit your results.

---

### 1️⃣ Train Your Model Locally

- Use the public dataset in `data/public/`.
- Train your model using your own implementation.
- Generate predictions for the **test set**.
- Create a `predictions.csv` file locally.

---

### 2️⃣ Prepare Submission Files

Each submission must include:

#### ✅ `predictions.csv` (Local File Only – DO NOT Upload)

Must contain exactly two columns:

| Column  | Description |
|----------|------------|
| `id`     | Graph identifier (must exactly match public test IDs) |
| `y_pred` | Predicted Class |

⚠️ IDs must exactly match those in the public test input file.  
Incorrect formatting will cause automatic validation failure.

🚫 **Do NOT upload `predictions.csv` to the repository.**

---

#### 🔐 Encrypt Your Predictions File

Before submission, you must encrypt your `predictions.csv` using the competition public key located in:

```
key/competition_public_key.asc
```

Run the following commands in bash:

```bash
# Import the public key
gpg --import competition_public_key.asc

# Encrypt predictions file
gpg --output predictions.csv.gpg \
    --encrypt \
    --recipient "GNN competition (Real or Fake) " \
    predictions.csv
```

This will generate:

```
predictions.csv.gpg
```

✅ **Only this encrypted `.gpg` file is allowed for submission.**

---

#### ✅ `metadata.json`

```json
{
  "team": "example_team",
  "run_id": "example_run_id",
  "type": "human",
  "model": "GAT",
  "notes": "Additional notes"
}
```

`type` must be one of:

- `"human"`
- `"llm-only"`
- `"human+llm"`

---

### 3️⃣ Submission Directory Structure

Your Pull Request must add files in the following structure:

```
submissions/inbox/<team_name>/<run_id>/
    ├── predictions.csv.gpg
    └── metadata.json
```

Example:

```
submissions/inbox/team_alpha/run_01/
    ├── predictions.csv.gpg
    └── metadata.json
```

🚫 Uploading `predictions.csv` will result in automatic rejection.

---

### 4️⃣ Submit via Pull Request

1. Fork the repository.
2. Add your metadata and encrypted submission files in the correct directory.
3. Open a Pull Request (PR) to the main repository.

---

### 5️⃣ Automatic Validation & Scoring

When the Pull Request is opened:

- Submission format is validated
- The encrypted file is securely decrypted by the competition server
- Predictions are scored using hidden test labels
- Score is posted automatically as a PR comment
- Invalid submissions fail automatically

---

### 6️⃣ Leaderboard Update

- Your score is appended to `docs/leaderboard.csv`
- The leaderboard page is automatically updated



## 🏫 Mentorship Program

This competition is part of the [BASIRA-LAB](https://basira-lab.com/) **(GNNs) for Rising Stars Mentorship Program**  

The lab’s tutorials on **Deep Graph Learning** served as guidance for preparing this challenge: [Tutorials Link](https://www.youtube.com/watch?v=gQRV_jUyaDw&list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T)




=== SECTION 2: REPO TREE ===

```text
Real_Or_Fake/
├── data/
│   ├── public/
│   │   ├── A.txt
│   │   ├── new_bert_feature.npz
│   │   ├── new_spacy_feature.npz
│   │   ├── new_profile_feature.npz
│   │   ├── node_graph_id.npy
│   │   ├── train_idx.npy
│   │   ├── train_labels.csv
│   │   ├── val_idx.npy
│   │   ├── val_labels.csv
│   │   ├── test_idx.csv
│   │   └── test_idx.npy
│   └── test_labels.csv   (kept private, restored via GitHub Secret)
│
├── submissions/
│   ├── sample_submission/
│   │   └── predictions.csv
│   └── inbox/
│       └── team_name/
│           └── run_name/
│               ├── metadata.json
│               └── predictions.csv.gpg
│
├── competition/
│   ├── metrics.py
│   └── scoring_script.py
│
├── docs/
│   ├── leaderboard.css
│   ├── leaderboard.csv
│   ├── leaderboard.html
│   └── leaderboard.js
│
├── key/
│   └── competition_public_key.asc
│
├── models/
│   ├── model.py
│   └── saved_model.model
│
├── dataloader.py
├── evaluate.py
├── test.py
├── train.py
├── update_leaderboard.py
├── validate_submission.py
├── requirements.txt
├── README.md
└── LICENSE
```


=== SECTION 3: DATA SAMPLE ===
A.txt: 
Contains all edges in the dataset. Each row is an edge represented by two node IDs (source and target).  
Type : Integer array
Shape: (num_edges, 2) = (308798, 2)
samples:
0, 1
0, 2
0, 3
57, 59
57, 60
57, 61

node_graph_id.npy: 
Maps each node to its corresponding graph. The value at index `i` indicates the graph ID of node `i`.  
Type: Integer array, 
shape: (num_nodes,) = (314262,)
samples :
array([   0,    0,    0, ..., 5463, 5463, 5463],)

new_spacy_feature.npz: 
Each node in the graph has text embeddings vector (using BERT model) representing the content of the news or user historical tweets.
type:
Sparse matrix stored in .npz format (CSR components: data, indices, indptr, shape), converted to a dense float tensor
shape:
(num_nodes, 300) = (314262, 300)
samples:
[-0.0053,  0.1056, -0.0178, ..., -0.0471, -0.0365,  0.0221]
[-0.0428,  0.0817, -0.0160, ..., -0.0790, -0.0202,  0.0365]
[-0.0401,  0.1270, -0.0552, ..., -0.0514, -0.0139,  0.0367]



new_bert_feature.npz: 
Each node in the graph has text embeddings vector (using spaCy) representing the content of the news or user historical tweets.
type:
Sparse matrix stored in .npz format (CSR components), converted to a dense float tensor
shape:
(num_nodes, 768) = (314262, 768)
samples:
[-0.1684, -0.3371, -0.4174, ..., -0.1544, -0.0898,  0.4973]
[-0.0330, -0.0308, -0.0759, ...,  0.5105, -0.0533,  0.3666]
[ 0.0062, -0.1012, -0.0058, ...,  0.4706, -0.1281,  0.2801]
...


new_profile_feature: 
These features are derived from the Twitter user object using the Twitter API including:  
1. Verified? (`0` or `1`)  
2. Geo-spatial enabled? (`0` or `1`)  
3. Number of followers  
4. Number of friends  
5. Status/tweet count  
6. Number of favorites  
7. Number of lists the user is part of  
8. Account age (months since Twitter launch)  
9. Number of words in the user’s name  
10. Number of words in the user’s description  
type:
Sparse matrix stored in .npz format (CSR components), converted to a dense float tensor
shape:
(num_nodes, 10) = (314262, 10)
samples:
[0.0000, 0.0000, 9.2164e-05, 1.5072e-03, 3.7818e-02,
 1.1493e-03, 7.2193e-04, 6.9244e-01, 1.1004e-01, 1.1538e-01]

[0.0000, 1.0000, 2.5605e-05, 7.1639e-04, 1.0968e-01,
 3.2587e-04, 2.5984e-03, 3.2877e-01, 1.6667e-01, 2.0588e-01]

[0.0000, 1.0000, 1.9822e-05, 1.0794e-03, 4.6407e-02,
 3.9404e-06, 1.9924e-04, 6.1644e-01, 5.5556e-02, 5.8824e-02]


### Data Splits
train_idx.npy:
Contains the list of **`3826`** graph IDs used for training.  
Type: Integer array, 
shape (num_train_graphs,) = (3826,)
samples:
array([5453, 2803, 1401, 4838, 1643, 1806, 1754, 3668, 2043,  825]))


val_idx.npy
Contains the list of **`546`** graph IDs used for validation.  
Type: Integer array, 
shape `(num_val_graphs,) = (546,)
samples :
array ([3570, 4908,  365, 2913, 1374,  818,  865,  721, 4385, 3771])

test_idx.csv
Contains the list of **`1092`** graph IDs used for testing. Labels are hidden in the private folder for competition evaluation.  
Type: Integer array, 
shape `(num_test_graphs,1)`   = (1092, 1))
samples: 
id
1
2
5
8
11


### Graph Labels
Each graph in the dataset has a label indicating whether the news is **real** or **fake**:

- `0` → Real news  
- `1` → Fake news  

Graph labels are stored separately for different splits:

- **Training labels:** `train_labels.csv`  , shape : (3826, 2)

- **Validation labels:** `val_labels.csv`  , shape :  (546, 2)
- **Test labels (hidden for competition evaluation):** `test_labels.csv`  , shape (1092,2)

- Each CSV file contains two columns:
    1. `id` → Graph ID  
    2. `y_true` → Label (0 or 1)



=== SECTION 4: SUBMISSION FORMAT ===
After training the model on train_idx and validate on val_idx , use the trained model to make predictions on test_idx
--
Create a `submission.csv` in the following format and save it in the competition main directory :
Real Or Fake requires: id, y_pred columns, 1092 rows (one per test graph).

```csv
id,y_pred
271,0.0
3681,0.0
4887,1.0
...


=== SECTION 5: REQUIREMENTS ===
networkx>=3.1
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.10
torch>=2.0
torch_geometric  (PyTorch Geometric — available, use for GNN layers)

=== SECTION 6: BASELINE CODE  ===
dataloader.py file code :

import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data, DataLoader

from scipy.sparse import csr_matrix

class GraphDataset:
    def __init__(self, split='train'):

        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of train/val/test")
        
        self.split = split

        # Load graph structure
        self.edges = np.loadtxt("data/public/A.txt", delimiter=",", dtype=int)
        self.node_graph_id = np.load("data/public/node_graph_id.npy")

        # Load graph IDs and labels for this split
        self.graph_ids =  np.load(f"data/public/{self.split}_idx.npy")
        if self.split != 'test' :
            self.graph_labels = pd.read_csv(f"data/public/{self.split}_labels.csv")
            self.label_map = dict(zip(self.graph_labels["id"], self.graph_labels["y_true"]))  # fast O(1) lookup

      
        features = np.load("data/public/new_spacy_feature.npz")
        sparse = csr_matrix(
            (features["data"], features["indices"], features["indptr"]),
            shape=tuple(features["shape"])
        )
        self.node_features = torch.tensor(sparse.toarray(), dtype=torch.float)


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

----------

train.py file code : 
import torch
from dataloader import GraphDataset
from models.model import GNN
from sklearn.metrics import accuracy_score, f1_score

epochs_no = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = GraphDataset(split='train').get_loader() 
val_loader   = GraphDataset(split='val').get_loader(shuffle=False)

# Model
sample_data = next(iter(train_loader)) 
model = GNN(sample_data.num_features).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
loss_fnc = torch.nn.BCELoss()

def metrics(preds, gts):
    preds = torch.round(torch.cat(preds)) 
    gts = torch.cat(gts) 
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



=== SECTION 7: IMPORTANT NOTES  ===

- Data paths: 'Real_Or_Fake/data/public/'.
- The final saved file should be submission.csv 
- predictions should be classes (0 or 1) not probabilities.
- The model MUST be a Graph Neural Network (GNN) using torch_geometric.
- Always use float32 instead of float64
- Delete large objects immediately from memory after they are no longer needed.
- Make use of the baseline code provided for you 
