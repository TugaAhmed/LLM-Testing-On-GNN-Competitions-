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
<div align="center">

# GRAPE

### **G**raph **R**etinal **A**nalysis for **P**rediction and **E**valuation

[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-blue)](https://muhammad0isah.github.io/GRAPE/leaderboard.html) [![Dataset](https://img.shields.io/badge/Dataset-DRIVE_|_STARE_|_HRF-green)](#data-sources) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![CI](https://img.shields.io/badge/CI-Automated_Scoring-orange)](.github/workflows/score_submission.yml) [![Encryption](https://img.shields.io/badge/Submissions-RSA_Encrypted-red)](encryption/)

</div>

## Overview

**GRAPE** is a GNN benchmark competition for diabetic retinopathy classification from retinal vessel graphs.

**Task:** Binary graph classification (healthy vs DR)  
**Metric:** Macro F1 Score (leaderboard_score), AUROC  
**Data:** 70 retinal vessel graphs from DRIVE[<sup>1</sup>](#data-sources) + STARE[<sup>2</sup>](#data-sources) + HRF[<sup>3</sup>](#data-sources)  
**Leaderboard:** [muhammad0isah.github.io/GRAPE/leaderboard.html](https://muhammad0isah.github.io/GRAPE/leaderboard.html)

---

## Background

Diabetic retinopathy (DR) is the leading cause of blindness in working-age adults. Retinal blood vessels form natural graphs where bifurcation points are nodes and vessel segments are edges. Changes in vessel topology (branching patterns, tortuosity, connectivity) indicate disease progression. This competition benchmarks GNN methods on classifying these graphs as healthy or DR-positive.

---

## Data Sources

| Dataset | Graphs | Healthy | DR | Source |
|---------|--------|---------|-----|--------|
| DRIVE | 20 | 17 | 3 | [drive.grand-challenge.org](https://drive.grand-challenge.org/) |
| STARE | 20 | 16 | 4 | [cecas.clemson.edu/~ahoover/stare](https://cecas.clemson.edu/~ahoover/stare/) |
| HRF | 30 | 15 | 15 | [www5.cs.fau.de/research/data/fundus-images](https://www5.cs.fau.de/research/data/fundus-images/) |
| **Total** | **70** | **48** | **22** | |

**Split:** 55 train / 15 test (stratified).

**Graph ID prefixes:** `D_XX` = DRIVE, `S_XX` = STARE, `H_XX` = HRF healthy, `R_XX` = HRF DR.

---

## Dataset Challenges

- **Class imbalance** — 48 healthy vs 22 DR (~69%/31%)
- **Cross-domain shift** — three imaging sources with different resolutions and protocols
- **Variable graph sizes** — ~30 to 500+ nodes per graph
- **Noisy topology** — graph extraction from segmentation introduces structural noise

---

## Repository Structure

```
GRAPE/
├── data/
│   └── public/
│       ├── train_data.csv          # 55 graphs (nodes + edges)
│       ├── train_labels.csv        # training labels
│       ├── test_data.csv           # 15 graphs for prediction
│       └── sample_submission.csv   # expected output format
├── encryption/
│   ├── public_key.pem              # RSA public key (for encrypting submissions)
│   ├── encrypt.py                  # encryption script
│   └── decrypt.py                  # decryption (CI-only)
├── competition/
│   ├── evaluate.py                 # scoring script
│   ├── validate_submission.py      # format validation
│   └── metrics.py                  # macro F1, AUROC
├── baseline.py                     # GAT baseline model
├── submissions/
│   └── inbox/<team>/               # place your .enc file here
└── leaderboard/
    └── leaderboard.csv             # auto-updated scores
```

---

## Graph Specification (A, X)

Each graph is defined by a **node feature matrix X** and an **adjacency matrix A**.

### Node Feature Matrix X

Each node has 4 features:

| Feature | Column | Description |
|---------|--------|-------------|
| $x_1$ | `x` | horizontal coordinate (pixels) |
| $x_2$ | `y` | vertical coordinate (pixels) |
| $x_3$ | `width` | vessel width at the node |
| $x_4$ | `type` | junction or endpoint |

### Adjacency Matrix A

The `edges` column encodes adjacency. Each node lists its neighbors as semicolon-separated IDs. This defines an undirected, unweighted adjacency matrix:

If node 0 has `edges = "3;9;121"`, then $A_{0,3} = A_{0,9} = A_{0,121} = 1$.

### CSV Columns

**train_data.csv / test_data.csv:**

| Column | Description |
|--------|-------------|
| `graph_id` | graph identifier (e.g. `D_21`, `S_44`) |
| `node_id` | node index within graph |
| `x`, `y` | pixel coordinates |
| `width` | vessel width |
| `type` | `junction` or `endpoint` |
| `edges` | adjacent node IDs (semicolon-separated) |

**train_labels.csv:**

| Column | Description |
|--------|-------------|
| `graph_id` | graph identifier |
| `label` | `0` = healthy, `1` = diabetic retinopathy |

---

## How to Participate (Step by Step)

### Step 1: Clone the Repository

```bash
git clone https://github.com/muhammad0isah/GRAPE.git
cd GRAPE
```

### Step 2: Install Dependencies

```bash
pip install pandas scikit-learn cryptography torch torch-geometric
```

### Step 3: Train Your Model and Generate Predictions

Use `data/public/train_data.csv` and `data/public/train_labels.csv` to train a GNN, then predict labels for each graph in `data/public/test_data.csv`.

A baseline GAT model is provided. Running it trains the model and generates `submission.csv` automatically:

```bash
python baseline.py
```

This outputs a file called `submission.csv` in the project root with the required format:

```csv
graph_id,label
D_25,0
R_2,1
S_235,0
H_10,0
...
```

You can build your own model — just make sure the output CSV has exactly these two columns, includes all 15 test graph IDs, and labels are `0` or `1`.

### Step 4: Encrypt Your Predictions

Submissions are encrypted so that other participants cannot see your predictions.

```bash
mkdir -p submissions/inbox/YOUR_TEAM_NAME
python encryption/encrypt.py submission.csv submissions/inbox/YOUR_TEAM_NAME/submission.csv.enc
```

Replace `YOUR_TEAM_NAME` with your team name (no spaces, use underscores).

Add a `meta.yaml` in your team folder to display your model name and notes on the leaderboard:

```yaml
model: VesselGCN                          # Name of your model (optional)
type: human                               # human, llm, or human+llm
notes: 3-layer GCN with skip connections  # Brief description (optional)
```

Example folder structure:

```
submissions/inbox/<team>/
├── submission.csv.enc   # Required (encrypted predictions)
└── meta.yaml            # Describes your submission
```

### Step 5: Fork, Commit, and Open a Pull Request

```bash
# Fork this repo on GitHub first, then:
git checkout -b submission/YOUR_TEAM_NAME
git add submissions/inbox/YOUR_TEAM_NAME/
git commit -m "Submission: YOUR_TEAM_NAME"
git push origin submission/YOUR_TEAM_NAME
```

Then open a Pull Request from your fork to the main repository.

### Step 6: Wait for Automated Scoring

The CI pipeline will automatically:
1. Decrypt your `.enc` file using the organizer's private key
2. Validate the submission format
3. Score against the hidden test labels
4. Report your Macro F1 and AUROC in the PR

Scores are published on the [leaderboard](https://muhammad0isah.github.io/GRAPE/leaderboard.html) after the PR is merged.

---

## Rules

- **One submission per team.** Only your first submission is scored.
- **Predictions must be encrypted.** Raw `.csv` files are gitignored and will not be accepted.
- **Training must complete within 3 hours on CPU.**
- **No access to test labels.** They are stored as a GitHub Secret and never exposed.

---

## Evaluation

Submissions are ranked by **Macro F1 Score** on the hidden test set. AUROC is reported as a secondary metric. Tied scores share the same rank.

---

## Baseline

The provided baseline (`baseline.py`) uses a 3-layer GAT with multi-head attention, multi-pool readout, and graph-level topological features. It achieves a Macro F1 of **0.830**.

Dependencies: `torch`, `torch-geometric`, `pandas`, `numpy`.

---

## License

MIT

---

## Citation

```bibtex
@misc{grape_2025,
  title={GRAPE:Graph Retinal Analysis for Prediction and Evaluation},
  author={Muhammad Ibrahim Isah},
  year={2026},
  url={https://github.com/Muhammad0isah/GRAPE}
}
```

=== SECTION 2: REPO TREE ===

```
GRAPE/
├── data/
│   └── public/
│       ├── train_data.csv          # 55 graphs (nodes + edges)
│       ├── train_labels.csv        # training labels
│       ├── test_data.csv           # 15 graphs for prediction
│       └── sample_submission.csv   # expected output format
├── encryption/
│   ├── public_key.pem              # RSA public key (for encrypting submissions)
│   ├── encrypt.py                  # encryption script
│   └── decrypt.py                  # decryption (CI-only)
├── competition/
│   ├── evaluate.py                 # scoring script
│   ├── validate_submission.py      # format validation
│   └── metrics.py                  # macro F1, AUROC
├── baseline.py                     # GAT baseline model
├── submissions/
│   └── inbox/<team>/               # place your .enc file here
└── leaderboard/
    └── leaderboard.csv             # auto-updated scores
```
=== SECTION 3: DATA SAMPLE ===
Each graph is defined by a **node feature matrix X** and an **adjacency matrix A**.
T
train_data.csv 
Contains 55 graphs (nodes + edges) for training   
Shape: (nodes, [node_id , graph_id , 4features , edges]) = (14156, 7)
columns : ['graph_id', 'node_id', 'x', 'y', 'width', 'type', 'edges']

| Column | Description |
|--------|-------------|
| `graph_id` | graph identifier (e.g. `D_21`, `S_44`) |
| `node_id` | node index within graph |
| `x`, `y` | pixel coordinates |
| `width` | vessel width |
| `type` | `junction` or `endpoint` |
| `edges` | adjacent node IDs to node_id(semicolon-separated) |

The `edges` column encodes adjacency. Each node lists its neighbors as semicolon-separated IDs. This defines an undirected, unweighted adjacency matrix:

samples:
graph_id  node_id      x     y   width  type                   edges
D_21        0      165.8   64.8    2.8     1                   7;105
D_21        1      324.7   77.5    4.0     1    3;10;101;102;107;113
D_21        2      333.3   78.7    2.8     1    3;10;101;102;107;109
D_21        3      270.7   94.7    2.8     1                     1;2
D_21        4      244.0   95.8    5.7     1  6;7;9;10;11;18;100;102 

......
train_labels.csv
contains graph labels for training graphs (testing graphs labels are hidden for competition)
shape : (graphs , [graph_id , label]) = (55, 2)

| Column | Description |
|--------|-------------|
| `graph_id` | graph identifier |
| `label` | `0` = healthy, `1` = diabetic retinopathy |

samples:
graph_id  label
R_13      1
S_1       1
R_10      0
R_12      1
R_14      0 


......

test_data.csv
contains 15 graphs (nodes+edges) for prediction 
shape :  (nodes * [graph_id,node_id,4features,edges])  = (3458, 7)

| Column | Description |
|--------|-------------|
| `graph_id` | graph identifier (e.g. `D_21`, `S_44`) |
| `node_id` | node index within graph |
| `x`, `y` | pixel coordinates |
| `width` | vessel width |
| `type` | `junction` or `endpoint` |
| `edges` | adjacent node IDs (semicolon-separated) |

samples: 
graph_id  node_id      x      y  width  type    edges
D_25        0      382.7   61.3    6.0     1   3;9;121 
D_25        1      261.0   77.0    8.2     1   2;4;8;12;13;117;120;125;127;132 
D_25        2      247.6   86.3    2.0     1   1;4;6;8;12;13;117;120;125;127;132
D_25        3      380.7   96.7    8.2     1   0;9;21;118;121;128 
D_25        4      247.0  103.8    0.0     1   1;2;6;8;12;13;23;117;120;127;132 


.......



### Data Splits
train_data.csv : contains all graphs for training with their nodes and edges 
test_data.csv : contains all graphs for testing with their nodes and edges
train_labels.csv : contains labels for training graphs in train_data.csv 



### Graph Labels
Binary graph classification (healthy vs DR) . Each graph in the dataset has a label 0 or 1



=== SECTION 4: SUBMISSION FORMAT ===
After training the model on graphs in train_data.csv and train_labels.csv use the trained model to make predictions on graphs in test_data.csv 
--
Create a `submission.csv` in the following format and save it in the competition main directory :
Real Or Fake requires: graph_id, label columns, 15 rows (one per test graph).

```csv
graph_id,label
D_25,0
D_28,0
D_36,0
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

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.data import Data , DataLoader


from torch_geometric.loader import NeighborLoader, GraphSAINTNodeSampler
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from torch_geometric.nn import GCNConv , GATConv , SAGEConv
from sklearn.metrics import f1_score
import torch.nn as nn

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
        x = torch.tensor(x, dtype=torch.float)
        
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
        
        edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros(2,0,dtype=torch.long)
        
        # Compute graph-level features
        graph_feats = compute_graph_features(g, node_map, edges)
        
        y = torch.tensor([labels[labels['graph_id']==gid]['label'].values[0]]) if labels is not None else None
        data = Data(x=x, edge_index=edge_index, y=y)
        data.gid = gid
        data.graph_feats = torch.tensor([graph_feats], dtype=torch.float)
        graphs.append(data)
    
    return graphs

=== SECTION 7: IMPORTANT NOTES  ===

- Data paths: 'GRAPE/data/public/'.
- The final saved file should be submission.csv 
- predictions should be classes (0 or 1) not probabilities.
- The model MUST be a Graph Neural Network (GNN) using torch_geometric.
- Always use float32 instead of float64
- Delete large objects immediately from memory after they are no longer needed.
- Make use of the baseline code if provided for you 
- Make sure that graph id in submission.csv is not between []
