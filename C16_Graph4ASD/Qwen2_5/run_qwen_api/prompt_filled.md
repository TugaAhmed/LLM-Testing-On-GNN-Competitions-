You are solving a GNN coding competition.

You will receive:
  (1) the competition README,
  (2) the repository file tree,
  (3) a data-sample summary,
  (4) the required submission format,
  (5) allowed libraries.
  (6) baseline code.

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
# 🧠 Graph4ASD Challenge 

**[Live Leaderboard](https://rosepy.github.io/Graph4ASD-Challenge/docs/leaderboard.html)**

## 📌 Overview

This repository hosts the **Graph4ASD Challenge**, a graph machine learning competition focused on **Autism Spectrum Disorder (ASD) classification** using resting-state fMRI functional connectivity data from the ABIDE dataset.

Participants must design and train **Graph Neural Network (GNN)** models to classify subjects as:

- ASD  
- Typical Control (TC)  

based on brain connectivity graphs.

Each subject is represented as a graph where:

- Nodes → brain regions (Craddock 200 atlas)  
- Edges → functional connectivity between brain regions  

---

## 🏆 Leaderboard

Leaderboard scores are automatically updated. 

👉 **[Live Leaderboard](https://rosepy.github.io/Graph4ASD-Challenge/docs/leaderboard.html)**

The evaluation metric is **Macro F1-Score**:

$$
\text{Macro F1} = \frac{F1_{ASD} + F1_{TC}}{2}
$$

Rankings are sorted by **descending score**.

---

## 🧠 Task Description

Each sample corresponds to one subject’s brain graph.

### Graph Definition

\[
G = (A, X)
\]

Where:

- **A** → adjacency matrix (functional connectivity)  
- **X** → node feature matrix (identity features)
- **Nodes** → 200 brain regions (Craddock atlas)  

### Objective

Train a model that correctly predicts whether a subject:

- has Autism Spectrum Disorder (ASD)  
- or is a Typical Control (TC)  

using:

- Graph structure  
- Node features  

This is a **graph-level classification task**.

---

## 📊 Dataset Details

The dataset is derived from:

> **Autism Brain Imaging Data Exchange (ABIDE)**

ABIDE aggregates anonymized rs-fMRI data from 17 international sites.

### Full Dataset (original ABIDE)

| Property    | Value        |
| ----------- | ------------ |
| Subjects    | 1,009        |
| ASD         | 516 (51.14%) |
| Controls    | 493 (48.86%) |
| Brain atlas | Craddock 200 |

⚠️ For the challenge, a processed subset is provided, 484 samples for train and 153 for test. Participants must use data as-is, since original data has been exclusively processed for this challenge. 

---

## 🧩 Data Representation

Each subject is stored as a graph:

- **Adjacency matrix (A)** → connectivity between brain regions  
- **Feature matrix (X)** → node-level features  

All graphs share:

- Same number of nodes → 200  
- Same node ordering → Craddock atlas  

---

## 📂 Dataset Access

The dataset is hosted in folder `data/public`. 



## 📝 How to Submit Your Results

### Step 1: Train
Train your model using:
- Graphs: Adjacency matrix and node feature matrix in `data/public/adj_train.npy` and `data/public/node_features.npy`. Each 
- labels in `data/public/train_label.csv`

### Step 2: Predict
Predict labels for every graph in:
- `data/public/adj_test.npy` and `data/public/node_features_test.npy`

### Step 3: Prepare your submission files

Create a `metadata.json` that contains metadata about your submission:

```json
{
  "team": "example_team",
  "run_id": "example_run_id",
  "type": "human",   // must be "human", "llm-only", or "human+llm"
  "model": "GAT",
  "notes": "Additional notes"
}
```

Create a CSV with columns `id` and `y_pred` (same format as `data/public/sample_submission.csv`):

```csv
id,y_pred
1,0
2,0
3,1
...
```
Encrypt your `predictions.csv` using: ``` python  extra/encrypt.py predictions.csv ```. 

Then put  `predictions.csv.enc` and `metadata.json` in the **`submissions/`** folder.  
**Note:** ❗ Do NOT upload the raw CSV. You need to submit an **encrypted** version of your predictions file to keep privacy.

### Step 4: Create a Pull Request
Commit and push the `predictions.csv.enc` and `metadata.json`  to the repository. Then, open  Pull Request or push to the main branch. The automated pipeline will decrypt and score your submission to update the leaderboard.

⚠️ Only **one submission per participant** is allowed.

## 🏁 Challenge Rules

- You are free to use any Graph Neural Network.
- You are not allowed to use external data.
- You must not try to identify each subject.


## 📚 References

Cameron Craddock, Yassine Benhajali, Carlton Chu, Francois Chouinard, Alan Evans, András Jakab, Budhachandra Singh Khundrakpam, John David Lewis, Qingyang Li, Michael Milham, Chaogan Yan, Pierre Bellec (2013). The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives. In Neuroinformatics 2013, Stockholm, Sweden.


=== SECTION 2: REPO TREE ===
./extra/encrypt.py
./extra/__pycache__/decrypt.cpython-314.pyc
./extra/generate_keys.py
./extra/decrypt.py
./extra/public_key.pem
./extra/.DS_Store
./data/public/sample_submission.csv
./data/public/node_features_test.npy
./data/public/node_features_train.npy
./data/public/adj_train.npy
./data/public/adj_test.npy
./data/public/.DS_Store
./data/public/train_label.csv
./data/.DS_Store
./LICENSE
./.gitattributes
./docs/leaderboard.html
./docs/leaderboard.css
./docs/leaderboard.js
./docs/.DS_Store
./requirements.txt
./submissions/README.md
./submissions/.DS_Store
./competition/update_leaderboard.py
./competition/render_leaderboard.py
./competition/validate_submission.py
./competition/evaluate.py
./competition/config.yaml
./competition/.DS_Store
./README.md
./.github/scripts/process_submission.py
./.github/workflows/score_submission.yml
./.github/workflows/publish_leaderboard.yml
./leaderboard/leaderboard.csv
./leaderboard/leaderboard.md
./leaderboard/.DS_Store
./.DS_Store

=== SECTION 3: DATA SAMPLE ===
`node_features_train.npy`:
A numpy array containing the feature matrices of training graphs, its shape is (484, 200, 200) (which represent the feature matrices 200x200 of the 484 graphs) and its dtype is float64.

`node_features_test.npy`:
A numpy array containing the feature matrices of testing graphs, its shape is (153, 200, 200) (which represent the feature matrices 200x200 of the 153 graphs) and its dtype is float64.

`adj_train.npy`:
A numpy array containing the adjacency matrices of training graphs, its shape is (484, 200, 200) (which represent the adjacency matrices 200x200 of the 484 graphs) and its dtype is float64.

`adj_test.npy`:
A numpy array containing the adjacency matrices of testing graphs, its shape is (153, 200, 200) (which represent the adjacency matrices 200x200 of the 153 graphs) and its dtype is float64.

`train_label.csv`:

Contains the labels of the training graphs, it has this format:

| Column  | Dtype | Description     |
| ------- | ----- | --------------- |
| `id`    | int64 | The graph id    |
| `label` | int64 | The graph label |

**Total_rows**: 484 

=== SECTION 4: SUBMISSION FORMAT ===
After training the model on training data, use the trained model to make predictions on testing data
--
Create a `submission.csv` in the following format and save it in the competition main directory :

```csv
id,y_pred
1,1
2,0
3,1
4,0
...


=== SECTION 5: REQUIREMENTS ===
pandas>=3.0.0
scikit-learn>=1.8.0
numpy==2.4.2
scipy==1.17.0
torch
torch-geometric



=== SECTION 6: BASELINE CODE ===

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score, precision_score

import matplotlib.pyplot as plt

# PyTorch Geometric
from torch_geometric.data import Data, DataLoader

from torch_geometric.loader import NeighborSampler , NeighborLoader

train_features = np.load("node_features_train.npy") # shape (484, 200, 200)
test_features = np.load("node_features_test.npy") # shape (153, 200, 200)
adj_train = np.load("adj_train.npy")  # shape  (484, 200, 200)
adj_test = np.load("adj_test.npy")  # (153, 200, 200)
train_labels = pd.read_csv("train_label.csv") #  (484, 2)

def build_graph(adj_matrix, label=None):
    num_nodes = adj_matrix.shape[0]

    # Get edges (non-zero connections)
    src, dst = np.nonzero(adj_matrix)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # edge_weight = torch.tensor(adj_matrix[src, dst], dtype=torch.float)
    edge_weight = adj_matrix[src, dst]
    edge_weight = np.clip(edge_weight, 0, 1)  # prevent explosion
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    edge_weight = edge_weight.view(-1, 1) 

    # compute node features 

    x = torch.tensor(x, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight
    )

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    return data

graphs = []
for i in range(len(adj_train)):
    g = build_graph(adj_train[i], train_labels['label'].to_numpy()[i])
    graphs.append(g)

train_graphs, val_graphs = train_test_split(
    graphs, test_size=0.2, random_state=SEED, stratify=train_labels['label'].to_numpy())

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32)


Important notes : 
- This is Binary Graph Classification not node classification , so you need to Aggregate node embeddings
- Data paths: 'Graph4ASD-Challenge/data/public'.
- The final saved file should be submission.csv 
- predictions should be classes not probabilities.
- The model MUST be a Graph Neural Network (GNN) using torch_geometric.
- Always use float32 instead of float64
- Delete large objects immediately from memory after they are no longer needed.
- Make use of the baseline code if provided for you