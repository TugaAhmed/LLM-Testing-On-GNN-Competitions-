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
# Cora Node Classification Challenge (GCN-Based)

## 📌 Overview

This repository hosts the **Cora Node Classification Challenge**, a graph machine learning competition based on the **Cora citation network**. Participants are required to design and train **Graph Neural Network (GNN)** models to classify scientific papers into research topics using node features and graph structure.

## Difficulty level
This implementation does not follow the standard Cora benchmark. 

To increase task difficulty, Gaussian noise (σ = 0.4) has been applied to the node features.


## 🏆 Leaderboard
- Leaderboard scores are automatically updated based on accuracy.
- View the live leaderboard:  
👉 **[Leaderboard](https://tasneem-mselim.github.io/GNN_CoRA/final_leaderboard.html)**

---

## 🧠 Task Description

* Each node represents a **scientific publication**.
* Edges represent **citation relationships** between papers.
* Each node belongs to **one of 7 classes**.

### Objective

Train a model that accurately predicts the class labels of **unlabeled test nodes**, using:

* Node features
* Graph connectivity

---

## 📊 Dataset Details


The dataset is derived from the **Cora citation network**.

| Property      | Value                |
| ------------- | -------------------- |
| Nodes         | 2,708                |
| Edges         | 5,429 (undirected)   |
| Node features | 1,433 (bag-of-words) |
| Classes       | 7                    |

### Data Splits (Standard Cora Protocol)

* **Training nodes**: 140 (20 per class)
* **Validation nodes**: 500 
* **Test nodes**: 1,000 (labels hidden)

---

### Public Files:

##### These files are available in the data/ folder for participants:

1- edge_index.csv — edges between nodes

2- x.csv — node features

3- y_train.csv — labels for the training nodes

4- y_val.csv — labels for validation nodes

5- test_ID _ id of testing nodes 


### Private Files:
- Test_label → Hidden ground-truth data used for automatic evaluation  

---
## 📝 How to Submit Your Results

Follow the steps below to submit your predictions to the competition leaderboard.

---

### Step 1: Fork the Repository

### Step 2: Navigate to Your Forked Repository

### Step 3: Go to the Submission Folder

### Step 4: Prepare the submission .csv locally 

#### 📝 Submission Format

Participants must submit a CSV file named **`submission.csv`** with the following format:

```csv
id,target
1708,3
1709,1
1710,6
...
```

#### Rules

* `id` must match the provided test node IDs
* `target` must be an integer in `{0, 1, 2, 3, 4, 5, 6}`


### Step 5: Encrypt Your Submission locally 

**Make sure you have:**

1- `encrypt_submission.py`
2- `public_key.pem`
3- `Your CSV file submission.csv`

**Open CMD/terminal in the folder containing these files and run the command:**

`python encrypt_submission.py submission.csv submission.enc public_key.pem`

**This will generate two files:**

`submission.enc` → the encrypted submission

`submission.enc.key` → encryption key

Both files are required for submission. Do not submit the original CSV.


### Step 6: Place Encrypted Files in Submission Folder

Upload these files to the submission folder in your forked repo

### Step 7: Create a Pull Request

✅ Your submission will be reviewed and evaluated, and the results will be added to the leaderboard.

**Only one submission** is allowed for each participant. Subsequent submissions will be automatically rejected

---



## 📈 Evaluation Metric

Submissions are evaluated using:

* **Accuracy** 

Evaluation is performed on a **hidden test set** to prevent data leakage.

---

## ✅ Allowed Methods

* Any **Graph Neural Network** architecture 
* Feature preprocessing and normalization
* Hyperparameter tuning

## ❌ Not Allowed

* Using test labels
* Modifying test node IDs
* Training on test nodes

---

## 🏆 Baseline

Participants are encouraged to improve the provided starter code upon this baseline using:

* Deeper architectures
* Attention mechanisms
* Regularization techniques

---



## 📚 References

* Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.
  
- **GNNs Tutorials (YouTube) – BASIRA Lab**:  
  [https://www.youtube.com/@BASIRALab](https://www.youtube.com/playlist?list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T)
  

- **GNN Tutorials (GitHub) – BASIRA Lab**:  
  https://github.com/basiralab
---

## 👩‍💻 Organizer

**Tasneem Selim**
Teaching Assistant & Researcher in Computer Vision and Graph Machine Learning
If you face issues with the repository or evaluation: 
- Contact me at tasneem.mselim@gmail.com 

---

Good luck, and happy graph learning 🚀




=== SECTION 2: REPO TREE ===
./data/edge_index.csv
./data/x.csv
./data/y_train.csv
./data/y_val.csv
./data/test_ID.csv
./LICENSE
./generate_leaderboard_html.py
./final_leaderboard.csv
./evaluate.py
./final_leaderboard.html
./README.md
./starter_code.py
./submission/readme
./.github/workflows/e
./.github/workflows/evaluate_submission.yml
./encryption/decrypt_submission.py
./encryption/encrypt_submission.py
./encryption/public_key.pem
./encryption/generateKeys.py

=== SECTION 3: DATA SAMPLE ===
`x.csv`

Contains the features for each node (trainig, validation and testing nodes)
Number of nodes : 2708
Number of features: 1433
Shape: (2708,1433)
Dtype: float64
The index of each node corresponds to its row index in this file

`y_train.csv`

Contains the labels of the training nodes

| Column  | Dtype | Description           |
| ------- | ----- | --------------------- |
| `index` | int64 | The index of the node |
| `label` | int64 | The label of the node |

**Total rows:** 140

`y_val.csv`

Contains the labels of the validation nodes

| Column  | Dtype | Description           |
| ------- | ----- | --------------------- |
| `index` | int64 | The index of the node |
| `label` | int64 | The label of the node |

**Total rows:** 500

`test_ID.csv`

Contains the ids of the testing nodes

| Column | Dtype | Description           |
| ------ | ----- | --------------------- |
| `id`   | int64 | The index of the node |

**Total rows:** 1000

`edge_index.csv`

Contains the source node and target node for each edge in the graph

| Column   | Dtype | Description                  |
| -------- | ----- | ---------------------------- |
| `source` | int64 | The index of the source node |
| `target` | int64 | The index of the target node |

**Total rows:** 10556

=== SECTION 4: SUBMISSION FORMAT ===
Participants must train on training nodes, validate on validation nodes and generate predictions for testing nodes (their ids are in `test_ID.csv`)

The prediction file should have the following format:

```csv
id,target
1708,3
1709,1
1710,6
```

=== SECTION 5: REQUIREMENTS ===

=== SECTION 6: BASELINE CODE ===

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.utils.class_weight import compute_class_weight


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
import random

features = pd.read_csv('data/x.csv').values  # (200, 14)
edges = pd.read_csv('data/edge_index.csv').values  # (500, 2)

train_df = pd.read_csv('data/y_val.csv')
val_df = pd.read_csv('data/y_train.csv')
test_ids = pd.read_csv('data/test_ID.csv').values.flatten()

num_nodes = features.shape[0]
