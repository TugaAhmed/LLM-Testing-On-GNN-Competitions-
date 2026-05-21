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

# DiaGraph - Robust Diabetes Prediction on Noisy Graphs

**DiaGraph** is a **prediction-only GitHub mini-competition** for **graph machine learning**.

It is **not an application**.  
It is a **benchmark + evaluation pipeline** where participants train models **locally** and submit **only encrypted prediction files** via Pull Requests.

---

## What this project is (exactly)

This project is:

- **Supervised Learning** (train/val labels are provided)
- **Graph Machine Learning** task
- **Node Classification** problem  
  (each node = one patient, predict diabetes label per node)
- A **robustness / noise-oriented benchmark**  
  (noisy + imbalanced data, hidden test labels)

**One-line description:**

> **A supervised graph node classification benchmark for robust diabetes prediction under noisy conditions.**

---

## Objective

Predict whether a node represents a **diabetic patient** (binary classification) using:

- **Node features** (clinical attributes)
- **Graph edges / adjacency** (similarity relations between patients)

Both **GNN models** and **non-GNN baselines** are allowed.

---
## Leaderboard

Live leaderboard:  
<https://mahatrabelsi1.github.io/GNN-MVP-Node-Classification-Under-Noise/leaderboard.html>

Source files:

- `leaderboard/leaderboard.csv`
- `leaderboard/leaderboard.md`
- `docs/leaderboard.csv`
- `docs/leaderboard.html`

---

## Graph Setting

- Each row in `nodes.csv` represents one **patient node**
- `edges.csv` contains the **graph structure**
- Your model can use edges (GNNs) or ignore them (baseline)

This is a **graph node classification** task.

---

## Repository Data

### Public data (committed)

```text
data/public/
├── nodes.csv           # node features ONLY (no labels)
├── edges.csv           # adjacency list (graph edges)
├── train.csv           # node ids + labels (training)
├── val.csv             # node ids + labels (validation)
├── test_nodes.csv      # node ids ONLY (final test for predictions)
├── sample_submission.csv
```

### Private data (never committed)

```text
data/private/
└── test_labels.csv     # hidden ground truth (used only in CI scoring)
```

Test labels are injected securely during GitHub Actions via Secrets.

---

## Evaluation Metric

**Macro F1-score** with **threshold = 0.5**

- evaluates both classes equally
- strongly penalizes predicting only the majority class

Leaderboard ranking uses competition-style ties (same score => same rank).

---

## Security and Privacy

Private submissions are submitted in encrypted form:

- Participants encrypt predictions locally with public key `encryption/public_key.pem`
- Repo stores only `predictions.csv.enc` (encrypted payload)
- GitHub Actions restores private decryption key from Secrets
- Decryption and scoring happen only inside runner environment
- Leaderboard updates are committed to `main`
- Submission PR is commented and auto-closed (no manual merge needed)

---

## Submission (STRICT)

### Required folder structure

```text
submissions/inbox/<team>/<run_id>/
├── predictions.csv.enc
└── metadata.json
```

### Plain `predictions.csv` format (before encryption)

```csv
id,y_pred
123,0.82
124,0.11
125,0.93
```

### Encryption command

```bash
python encryption/encrypt.py predictions.csv encryption/public_key.pem predictions.csv.enc
```

**Rules**

- `id` must match exactly the ids in `data/public/test_nodes.csv`
- `y_pred` must be a probability in [0, 1]
- Row count must match `test_nodes.csv`
- Submit encrypted file only (`predictions.csv.enc`), not plain `predictions.csv`

---

### `metadata.json` format

Copy from:

```text
submissions/inbox/metadata_template.json
```

Example:

```json
{
  "team": "my_team",
  "run_id": "gcn_v1",
  "author_type": "human",
  "model": "GCN + MLP",
  "notes": "Used edges + class weights"
}
```

Allowed `author_type` values:

```text
human
llm
hybrid
```

---

## How to Participate

1. Clone the repository.
2. Train your model locally using:
```text
data/public/nodes.csv
data/public/edges.csv
data/public/train.csv + val.csv
```
3. Generate predictions for all ids in `data/public/test_nodes.csv`.
4. Encrypt predictions:
```bash
python encryption/encrypt.py predictions.csv encryption/public_key.pem predictions.csv.enc
```
5. Create:
```text
predictions.csv.enc
metadata.json
```
6. Add them to:
```text
submissions/inbox/<team>/<run_id>/
```
7. Open a Pull Request to `main`.

The PR must modify only these two files.

---

## Automatic Scoring (GitHub Actions)

On Pull Request:

- validates folder structure
- validates `metadata.json`
- restores hidden labels and decryption key from Secrets
- decrypts encrypted submission
- scores Macro-F1
- updates leaderboard files directly on `main`
- posts Macro-F1 + metadata as a PR comment
- closes PR automatically

No manual merge is required for submission PRs.

---



## Rules

- No external data
- Do not submit training code; predictions + metadata only
- One submission attempt per team (enforced in CI)
- LLMs must not fully design dataset/task/evaluation logic

Invalid submissions are rejected automatically.



=== SECTION 2: REPO TREE ===

GNN-MVP-Node-Classification-Under-Noise/
├── .github/
│   └── workflows/
│       ├── score_submission.yml
│       └── update_leaderboard.yml
├── .gitignore
├── README.md
├── data/
│   └── public/
│       ├── edges.csv
│       ├── nodes.csv
│       ├── test_nodes.csv
│       ├── train.csv
│       └── val.csv
├── docs/
│   ├── index.html
│   ├── leaderboard.csv
│   └── leaderboard.html
├── encryption/
│   ├── decrypt.py
│   ├── encrypt.py
│   ├── generate_keys.py
│   └── public_key.pem
├── leaderboard/
│   ├── leaderboard.csv
│   └── leaderboard.md
├── scoring/
│   ├── score_submission.py
│   ├── update_leaderboard.py
│   ├── update_leaderboard_all.py
│   ├── update_leaderboard_all_encrypted.py
│   └── validate_metadata.py
└── submissions/
    └── inbox/
        ├── .gitkeep
        ├── Idrees_Bhat/
        │   └── run_01/
        │       ├── metadata.json
        │       └── predictions.csv.enc
        ├── README.md
        ├── metadata_template.json
        └── samuel/
            └── run_01/
                ├── metadata.json
                └── predictions.csv.enc

=== SECTION 3: DATA SAMPLE ===
nodes.csv :
Contains node features ONLY (no labels) where each row in `nodes.csv` represents one patient node
Shape: (nodes * features ) = (96128, 13)
columns : ['id', 'age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
        'hypertension', 'heart_disease', 'gender_Female', 'gender_Male',
        'smoking_history_current', 'smoking_history_non_smoker',
        'smoking_history_past', 'HbA1c_missing'],
samples:
id       age       bmi  HbA1c_level  blood_glucose_level  hypertension   heart_disease  gender_Female  gender_Male  smoking_history_current
 0  1.700709 -0.314941     1.024820             0.089266             0    1              1                0                        0
 1  0.543261 -0.000214          NaN            -1.579039             0    0              1                0                        0  
 2 -0.614187 -0.000214          NaN             0.596105             0    0              0                1                        0 
 3 -0.258049 -0.572042    -0.520527             0.551292             0    0              1                0                        1 
 4  1.522640 -1.061124    -0.660453             0.117552             1    1              0                1                        1  
...........

edges.csv:
contains the graph structure which is the adjacency list (graph edges)
shape :  (edges * [src', 'dst] ) =(1224756, 2)
samples: 
src       dst
0        79043
79043      0
0        64282
64282      0
0        11861
.........

train.csv:
contains node ids and labels used for training
shape :  (training_nodes * [ id ,  diabetes]) = (67289, 2)  --> diabetes is the label for prediction
samples : 
  id    diabetes
88223         0
51099         0
56627         1
35363         0
53951         0

........

val.csv : 
contains node ids and labels used for validation
shape : (validation_nodes * [ id ,  diabetes]) =  (9613, 2)  
  id     diabetes
73459         0
60675         0
53589         0
11236         0
91950         0

........

test_nodes.csv:
contain node ids used for testing (without labels)
shape :  (testing_nodes * 1 ) = (19226, 1)  
  id
  699
  64298
  56235
  32411
  62655



### Data Splits
- All nodes : all nodes(train,val,test) ids and features are in nodes.csv
- All edges : all nodes(train,val,test) edges are in edges.csv 
- Training nodes: included in train.csv
- Validation nodes : included in val.csv
- Test set:  included in test_nodes.csv (labels are hidden for competition fair assessment)
- Labels: binary classification (0 or 1 ) . Labels only available for training nodes (in train.csv) and validation nodes (in val.csv) but hidden for testing nodes 

## Problem Type
- Node Classification problem : each node = one patient, predict diabetes label per node

## Objective
Predict whether a node represents a diabetic patient(binary classification) using:

- Node features (clinical attributes)
- Graph edges / adjacency (similarity relations between patients)



=== SECTION 4: SUBMISSION FORMAT ===
After training the model using train.csv , and val.csv, use the trained model to make predictions on testing set exist in test_nodes.csv 
--
Create a `submission.csv` in the following format and save it in the competition main directory :
This challenge requires: id,y_pred columns, 19226 rows (one per test node in test_nodes.csv).

```csv
id,y_pred
699,0
64298,0
56235,0
...

=== SECTION 5: REQUIREMENTS ===
networkx>=3.1
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.10
torch>=2.0
torch_geometric  (PyTorch Geometric — available, use for GNN layers)
matplotlib

=== SECTION 6: BASELINE CODE  ===

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader , GraphSAINTNodeSampler
from torch_geometric.nn import GCNConv , GATConv , SAGEConv
from sklearn.metrics import f1_score
import torch.nn as nn

# Load CSVs
nodes_df = pd.read_csv("data/public/nodes.csv")  # nodes * features
edges_df = pd.read_csv("data/public/edges.csv")  # edges_count * 2 

train_df = pd.read_csv("data/public/train.csv")  # id , label
val_df   = pd.read_csv("data/public/val.csv")    # id , label
test_df  = pd.read_csv("data/public/test_nodes.csv")  # id 

node_ids = nodes_df.iloc[:, 0].values
features = nodes_df.iloc[:, 1:].values

x = torch.tensor(features, dtype=torch.float)
src = edges_df.iloc[:, 0].values
dst = edges_df.iloc[:, 1].values
edge_index = torch.tensor([src, dst], dtype=torch.long)

# initilize y with -1 values 
num_nodes = len(node_ids)
y = torch.full((num_nodes,), -1, dtype=torch.long)

# fill y with labels from train.csv and val.csv 
for _, row in train_df.iterrows():
    y[row['id']] = int(row['diabetes'])

for _, row in val_df.iterrows():
    y[row['id']] = int(row['diabetes'])

# Masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool) 
val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[[ i for i in train_df.iloc[:,0]]] = True
val_mask[[ i for i in val_df.iloc[:,0]]] = True
test_mask[[ i for i in test_df.iloc[:,0]]] = True


data = Data(
    x=x,                     # all nodes
    edge_index=edge_index,   # all edges
    y=y,                     # labels (-1 for test)
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)




=== SECTION 7: IMPORTANT NOTES  ===

- Data path: 'GNN-MVP-Node-Classification-Under-Noise/data/public'.
- Submission path should be in the competition main directory  'GNN-MVP-Node-Classification-Under-Noise/submission.csv'.
- The final saved file should be submission.csv 
- predictions should be classes (0 or 1) not probabilities.
- The model MUST be a Graph Neural Network (GNN) using torch_geometric.
- Always use float32 instead of float64
- Delete large objects immediately from memory after they are no longer needed.
- Make use of the baseline code if provided for you 
- Make sure that you handle noisy features with NaN of Inf values 
