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
# 🧬 GLIMPS-GNN
**Graph-based Liquid-biopsy Inductive Modeling for PreeclampSia**

## GNN Challenge: cfRNA → Placenta Inductive Prediction

<div align="center">
    <img src="images/IMG3.png" width='640' /> 
</div>

<br>
<br>


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?logo=github)](LICENSE)
[![License: CC-BY-SA-4.0](https://img.shields.io/badge/License-CC--BY--SA--4.0-green.svg?logo=github)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-blue?logo=googlechrome)](https://mubarraqqq.github.io/gnn-challenge/leaderboard.html)

This repository hosts a prediction-only challenge focused on maternal-fetal health modeling using graph learning.
Participant code is run outside this repository. Submissions are scored in CI against hidden labels.

**[🏆 Click me to join competition](https://github.com/Mubarraqqq/gnn-challenge/blob/main/CONTRIBUTING.md)** 
## Scientific Focus

- Inductive graph learning across cfRNA and placental transcriptomics to detect maternal-fetal health issues.
- Learn transferable representations that generalize to unseen samples and domains rather than treating each dataset independently.

## Alignment with [BASIRA Lab's](https://basira-lab.com) Mission

- Prioritizes robust generalization across heterogeneous datasets.
- Uses compute-efficient, non-data-hungry graph learning methods that can run on standard hardware.

## Inspiration from GNN Literature

- Draws from studies on inductive learning, message passing, and representation transfer.
- Model design follows [DGL Lectures 1.1-4.6](https://www.youtube.com/watch?v=gQRV_jUyaDw&list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T), covering:
  - Graph construction from tabular data
  - Node feature encoding
  - Neighborhood aggregation (GraphSAGE-style inductive updates)
  - Mini-batch training via neighborhood sampling
  - Inductive inference on unseen nodes

## Overview

- Task: Binary classification (`0=Control`, `1=Preeclampsia`)
- Setting: Inductive transfer from cfRNA (train) to placenta (test)
- Primary metric: F1 Score
- Additional metrics: Accuracy, Precision, Recall
- Public leaderboard: Auto-updated after merged submissions

<div align="center">
    <img src="images/IMG4.jpeg" width='750' /> 
</div>

<br>
<br>

## Dataset Source and Description

### Source

- Public datasets from Gene Expression Omnibus (GEO, NIH)
- Maternal plasma cfRNA: `GSE192902`
- Placental RNA-seq: `GSE234729`

### Data Splits

- Training set: cfRNA samples
- Test set: placenta samples (unseen during training)
- Labels: binary disease status

### Purpose and Integration Goal

- Identify and validate cfRNA biomarkers for early prediction of preeclampsia, often before clinical symptoms appear.
- Support research in maternal-fetal health and early detection of preeclampsia.
- Integrate gene expression and clinical metadata to capture subtle risk patterns while handling noisy and imbalanced data for robust and equitable predictions.

### 🧩 Mandatory Graph Specification

This competition explicitly provides both required graph components:

- Adjacency matrix `A`: `data/public/adjacency_matrix.csv`
- Node feature matrix `X`: derived from `data/public/train.csv` and `data/public/test.csv`

Related graph files:

- `data/public/graph_edges.csv`
- `data/public/node_types.csv`
- `data/public/graph_artifacts.pt`

Interpretation:

- `A[i, j] = 1` indicates an edge between nodes `i` and `j`, else `0`
- `X` is node-by-feature and includes harmonized expression features and released covariates
- Node alignment is by `node_id`; use `data/public/test_nodes.csv` (and node files) as the ordering reference so rows in `X` correspond to the same nodes indexed in `A`.

### 🌍 Dataset Difficulty and Realism

The benchmark includes meaningful modeling difficulty:

- 🧪 Noisy and partially missing metadata
- ⚖️ Label imbalance pressure
- 🧬 High-dimensional features relative to sample size (sparsity pressure)
- 🔄 Cross-domain distribution shift (cfRNA -> placenta)
- 🕸️ Inductive generalization to unseen test nodes

### ⏱️ Computational Affordability

- Full training should not exceed **3 hours on CPU** per competition.
- If needed, downsize graph complexity (for example by reducing node count, edge density, or neighborhood sampling size) while preserving task integrity.

## Dataset Construction and Preprocessing

[`build_dataset.ipynb`](./organizer_scripts/build_dataset.ipynb) and [Kaggle](https://www.kaggle.com/code/freeeman/maternal-2014425c3f4)

Objective: Ensure structural compatibility for graph construction and inductive learning by handling expression data, parsing and cleaning metadata, and expression-metadata fusion.

## Advanced GNN Implementation

[`advanced_GNN_model.py`](./starter_code/advanced_GNN_model.py)

Objective: Implement an advanced inductive GNN for cfRNA -> placenta prediction, ensuring generalizable node representations and inductive learning.

Key Components:

- Graph Construction: Build hetero-graphs using similarity and ancestry edges.
- Node Feature Encoding: Integrate gene expression and metadata into node-level features.
- Neighborhood Aggregation: GraphSAGE-style layers with BatchNorm and ReLU for neighbor information propagation.
- Mini-Batch Training: Use neighborhood sampling for efficient training on large graphs.
- Inductive Inference: Generate predictions for unseen placenta nodes without label leakage.

## Starter Assets

- `starter_code/advanced_GNN_model.py`
- `starter_code/baseline.py`
- `starter_code/build_adjacency_matrix.py`
- `starter_code/build_graph_artifacts.py`

## Submission Policy

Submission instructions are in `CONTRIBUTING.md`.

Key policy:

- Only one submission attempt per participant (enforced in CI)
- Submission files are public but participant predictions are encrypted at rest (`predictions.csv.enc`); only CI with organizer secrets decrypts for scoring.

## Leaderboard

- Public page: `https://mubarraqqq.github.io/gnn-challenge/leaderboard.html`
- Source CSV: `leaderboard/leaderboard.csv`
- Rendered markdown: `leaderboard.md`
- Tie handling: equal scores share rank

## Maintainer Regeneration Command

Use this command to regenerate all leaderboard outputs from the canonical pipeline:

```bash
python update_leaderboard.py && python competition/render_leaderboard.py
```

## Citation

```bibtex
@dataset{gnn_challenge_2026,
  title={GNN Challenge: cfRNA -> Placenta Inductive GNN for Maternal-Fetal Health Prediction},
  author={Mubaraq Onipede},
  year={2026},
  url={https://github.com/Mubarraqqq/gnn-challenge}
}
```

## License

See `LICENSE`.




=== SECTION 2: REPO TREE ===
```
├── .github/
│   ├── keys/
│   │   └── submission_public.asc
│   └── workflows/
│       └── score-submission.yml
├── competition/
│   ├── __pycache__/
│   │   ├── render_leaderboard.cpython-314.pyc
│   │   └── validate_submission.cpython-314.pyc
│   ├── metrics.py
│   ├── render_leaderboard.py
│   └── validate_submission.py
├── data/
│   ├── public/
│   │   ├── adjacency_matrix.csv
│   │   ├── graph_artifacts.pt
│   │   ├── graph_edges.csv
│   │   ├── node_types.csv
│   │   ├── sample_submission.csv
│   │   ├── test_nodes.csv
│   │   ├── test.csv
│   │   └── train.csv
│   ├── expr_df_2_GSE192902.csv
│   ├── expr_df_GSE234729.csv
│   ├── graph_edges.csv
│   ├── metadata_cfRNA.csv
│   ├── metadata_placenta.csv
│   ├── node_types.csv
│   ├── test.csv
│   └── train.csv
├── docs/
│   ├── leaderboard.css
│   ├── leaderboard.csv
│   ├── leaderboard.html
│   └── leaderboard.js
├── images/
│   ├── IMG1.png
│   ├── IMG2.png
│   ├── IMG3.png
│   └── IMG4.jpeg
├── leaderboard/
│   └── leaderboard.csv
├── organizer_scripts/
│   └── build_dataset.ipynb
├── starter_code/
│   ├── advanced_GNN_model.py
│   ├── baseline.py
│   ├── build_adjacency_matrix.py
│   ├── build_graph_artifacts.py
│   └── requirements.txt
├── submissions/
│   ├── inbox/
│   ├── advanced_gnn_preds_with_confidence.csv
│   ├── advanced_gnn_preds.csv
│   ├── baseline_mlp_preds_with_confidence.csv
│   ├── baseline_mlp_preds.csv
│   └── sample_submission.csv
├── .gitignore
├── CONTRIBUTING.md
├── leaderboard.md
├── LICENSE
├── README.md
├── scoring_script.py
├── test_submission_infrastructure.py
└── update_leaderboard.py

```


=== SECTION 3: DATA SAMPLE ===
train.csv : 
Contains labeled training features for cfRNA dataset 
Shape: (nodes * features + [disease_labels , sample_id , node_id] ) = (209 * 6653)  
samples:
ENSG00000169877	ENSG00000176463	ENSG00000205639	ENSG00000244716	....... disease_labels	sample_id	node_id
0.156980	-0.072228	0.514708	-0.401727	..... 0	149	cfRNA_0
0.444778	0.091410	-0.014561	0.108674	..... 0	150	cfRNA_1
0.300879	0.255048	1.440930	0.108674	..... 0	72	cfRNA_2
-0.130817	-0.399504	-0.279196	-0.656928	..... 0	252	cfRNA_3

....
test.csv:
Contains not-labeled testing features for placenta dataset 
shape : (nodes * features + [ sample_id , node_id] ) = (111 * 6652)
samples : 
ENSG00000169877	ENSG00000169877	ENSG00000205639	.... sample_id	node_id
2.001803	2.001803	-0.536747	..... 22	placenta_0
5.482081	5.482081	0.083965	..... 53	placenta_1
3.642971	3.642971	-0.315921	..... 57	placenta_2
0.785889	0.785889	0.419158	..... 61	placenta_3
....

node_types.csv:
Contains the dataset type (cfRNA or placenta) for each node
shape :    (all nodes * 2[node_id	,node_type])  =(320, 2)
samples : 
node_id	node_type
0	cfRNA_0	cfRNA
1	cfRNA_1	cfRNA
2	cfRNA_2	cfRNA
3	cfRNA_3	cfRNA
4	cfRNA_4	cfRNA
...	...	...
315	placenta_106	placenta
316	placenta_107	placenta
317	placenta_108	placenta
318	placenta_109	placenta
319	placenta_110	placenta

.......

graph_edges.csv:
Contains edges information between nodes including source node , destination node and edge type
shape :  (all nodes * 3 [src,dst ,edge_type]) = (3200 *3)
samples : 
src	dst	edge_type
cfRNA_0	cfRNA_193	similarity
cfRNA_0	cfRNA_101	similarity
cfRNA_0	cfRNA_5	similarity
...	...	...	...
placenta_110	placenta_33	similarity
placenta_110	placenta_60	similarity

...........

### Data Splits
- Training set: cfRNA samples
- Test set: placenta samples (unseen during training)
- Labels: binary disease status


### Node Labels
Each node has a label indicating whether the node is `0=Control`or  `1=Preeclampsia`
For training , labels exist in train.csv file in disease_labels column 
For testing , labels are not given and kept hidden for competition evaluation 



=== SECTION 4: SUBMISSION FORMAT ===
After training the model on train.csv , use the trained model to make predictions on test.csv
--
Create a `submission.csv` in the following format and save it in the competition main directory :
This challenge requires: node_id,target columns, 111 rows (one per test node).

```csv
node_id,target
placenta_0,0
placenta_1,0
placenta_2,0
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

import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData


def build_graph(edges_df, node_df, allowed_edge_types):
    node_ids = node_df["node_id"].tolist()
    node_map = {nid: i for i, nid in enumerate(node_ids)}

    data = HeteroData()
    data["node"].num_nodes = len(node_ids)

    for etype in allowed_edge_types:
        df = edges_df[edges_df.edge_type == etype]
        src = torch.tensor([node_map[i] for i in df.src], dtype=torch.long)
        dst = torch.tensor([node_map[i] for i in df.dst], dtype=torch.long)
        data["node", etype, "node"].edge_index = torch.stack([src, dst])

    return data, node_map


def main():
    parser = argparse.ArgumentParser(description="Build and save PyG graph artifacts.")
    parser.add_argument(
        "--data-dir",
        default="data/public",
        help="Directory containing train.csv, test.csv, graph_edges.csv, node_types.csv",
    )
    parser.add_argument(
        "--out",
        default="data/public/graph_artifacts.pt",
        help="Output path for torch.save(...) artifact",
    )
    parser.add_argument(
        "--use-ancestry-in-test",
        action="store_true",
        help="Include ancestry edges in the test graph",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    edges_df = pd.read_csv(os.path.join(data_dir, "graph_edges.csv"))
    node_df = pd.read_csv(os.path.join(data_dir, "node_types.csv"))

    # Identify target column (train uses 'disease_labels' or 'target')
    target_col_train = "disease_labels" if "disease_labels" in train_df.columns else "target"

    # Build graphs
    train_graph, node_map = build_graph(edges_df, node_df, ["similarity"])
    test_edge_types = ["similarity", "ancestry"] if args.use_ancestry_in_test else ["similarity"]
    test_graph, _ = build_graph(edges_df, node_df, test_edge_types)

    # Node features: only shared columns between train and test
    shared_cols = set(train_df.columns).intersection(set(test_df.columns))
    feat_cols = sorted([c for c in shared_cols if c not in ["node_id", target_col_train, "sample_id"]])

    x = torch.zeros((len(node_map), len(feat_cols)))
    train_idx = torch.tensor([node_map[i] for i in train_df.node_id], dtype=torch.long)
    test_idx = torch.tensor([node_map[i] for i in test_df.node_id], dtype=torch.long)
    x[train_idx] = torch.tensor(train_df[feat_cols].values, dtype=torch.float)
    x[test_idx] = torch.tensor(test_df[feat_cols].values, dtype=torch.float)

    train_graph["node"].x = x
    test_graph["node"].x = x

    # Labels (train only; others are -1)
    y = -1 * np.ones(len(node_map), dtype=int)
    y[train_idx] = train_df[target_col_train].values.astype(int)
    y = torch.tensor(y, dtype=torch.long)
    train_graph["node"].y = y
    test_graph["node"].y = y

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    artifact = {
        "train_graph": train_graph,
        "test_graph": test_graph,
        "node_map": node_map,
        "feat_cols": feat_cols,
        "target_col_train": target_col_train,
        "use_ancestry_in_test": args.use_ancestry_in_test,
    }
    torch.save(artifact, args.out)
    print(f"Saved graph artifacts to {args.out}")


if __name__ == "__main__":
    main()



=== SECTION 7: IMPORTANT NOTES  ===

- Data paths: 'gnn-challenge/data/'.
- The final saved file should be submission.csv 
- predictions should be classes (0 or 1) not probabilities.
- The model MUST be a Graph Neural Network (GNN) using torch_geometric.
- Always use float32 instead of float64
- Delete large objects immediately from memory after they are no longer needed.
- Make use of the baseline code if provided for you 
