You are solving a GNN coding competition.

You will receive:
  (1) the competition README,
  (2) the repository file tree,
  (3) a data-sample summary,
  (4) the required submission format,
  (5) allowed libraries.
  (6) baseline code
  (7) important notes.

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
# ✨ GNN Challenge: Liar Nodes✨

##  The Problem


For most neural networks, classification tasks are made individually based on the embedding of each data point. Samples that belong to the same class tend to have similar embeddings and therefore lie close to each other in the embedding space.

**Graph Neural Networks (GNNs)** fundamentally alter this paradigm by performing **collective decision making**. Rather than relying solely on the representation (embedding) of an individual node, the classification decision is influenced by the structure of the graph and by aggregated information from neighboring nodes.



### 💡 The Core Question

**This raises an important question: what should a model do when there is a contradiction between the information provided by a node's embedding and the information coming from its neighborhood? aka adversary learning**

---
![Alt text](picture.png)

## 🎮 Challenge Overview

In this challenge, the goal is to implement mechanisms that balance these two sources of information that might contradict each other. Node features are manually corrupted, while graph connectivity encodes contextual relationships that may either reinforce or contradict a node's individual features.

We apply this concept to Cancer biology! A cell's environment can give misleading cues , a malignant cell might “look normal” in isolation, or neighboring immune cells might influence its signaling,the node’s own features (gene expression) might conflict with neighborhood information (similar cells in the microenvironment), mimicking real biological “contradictions.”

Participants must adapt GNN models using techniques such as **neighborhood sampling** and **aggregation** to learn when to trust the node embedding, when to trust the neighborhood, and how to effectively combine both in order to perform node classification and detect the cancerous cells.



###  Critical Constraint

**The provided node embeddings are fixed and cannot be modified or recomputed**; participants must rely solely on graph-based aggregation and sampling strategies to resolve conflicting signals.

---

## 💭 Hint

> **This challenge is not about inventing a new GNN, it is about choosing the right way to listen.**
> 
> You can’t tell which cells are sending corrupted signals , you must be clever about how you gather and trust information from the tumor microenvironment!

---

##  What Data You Will Use

The Cancer Single-cell Expression Map (CancerSCEM) dataset provides comprehensive single-cell RNA-seq data across multiple cancer types. Instead of raw sequencing reads, the dataset offers UMI count matrices in standard .tsv format for downstream analysis. Each dataset includes:
Gene Expression Matrices per cell  enabling gene descriptio/Cell-Type Annotations and Components /Functional Molecules /Cell-Cell Interaction Data .
You can read more about this data source on https://ngdc.cncb.ac.cn/cancerscem/index .
As for this task, we have altered samples from this data to present it for a  GNN training tasks, you will be given:

### 🔗 **A (Adjacency Matrix)**
- Encodes the connections and structure of your input graph.
- Most connections reflect real interactions between cells, but some edges may have been altered, introducing noise.
  
### 📈 **X (Node Embeddings)**
- Embeddings of nodes == Genes encodings
- ⚠️ **Some nodes have been corrupted to trick you!**
- You cannot modify these embeddings

### 🎭 **Mask (Corruption Indicator)**
- Gives you a hint about which nodes have been corrupted
- Available in `train.csv`
- **NOT available in `test.csv`** (this is what you need to test on)
- The corruption mask may be used only during training to modulate aggregation, weighting, or attention — not to filter nodes or labels.

### 📁 **Data Files**
- **`train.npz`** - Contains features AND corruption mask.
- **`test.csv`** - Contains features only (no labels, no mask)
- **`labels.csv`** - Contains labels for the full training data
- **`edges.csv`** - Describes the communication between the cells
Please review the baseline code to help you better navigate the data.
---
## 🚫 Constraints

###  You CANNOT
- Remove the noisy nodes based on the mask from the training data
- Alter the embedding of the nodes
- Use external data

###  You CAN:
- Use the mask to guide your aggregation strategy
- Implement creative neighborhood sampling
- Design custom aggregation functions
- Use attention mechanisms
- Combine multiple GNN layers strategically
- Look up additional resources to understand the behavior of the cancerous cells.
---

##  Your Mission ✨✨

Build a GNN that can navigate the noise and make accurate predictions by:

1. **Learning when to trust** individual node features
2. **Learning when to trust** neighborhood aggregation
3. **Build** robust GNNs can detect anomalies or corrupted signals — similar to spotting a misbehaving cell in a tumor.

## Resources

Convex Adversarial Collective Classification ;MohamadAli Torkamani, Daniel Lowd Proceedings of the 30th International Conference on Machine Learning, PMLR 28(1):642-650, 2013.

Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann

The Reason Why Cancer is so Hard to Beat -Kurzgesagt – In a Nutshell

A Multimodal Graph Neural Network Framework of Cancer Molecular Subtype Classification

Cancer Single-cell Expression Map

Graph Neural Networks in Cancer and Oncology Research: Emerging and Future Trends Grigoriy Gogoshin 1,*, Andrei S Rodin 1,


=== SECTION 2: REPO TREE ===
./update_leaderboard.py
./data/edges.csv
./data/test.csv
./data/labels.csv
./data/train_compressed.csv
./LICENSE
./docs/leaderboard.js
./docs/css/leaderboard.css
./docs/css/base.css
./docs/index.html
./render_leaderboard.py
./Challenge Picture!.png
./picture.png
./.gitignore
./README.md
./scoring_script.py
./.github/CODEOWNERS
./.github/workflows/validate_submission.yml
./.github/workflows/score_and_update.yml
./leaderboard/leaderboard.csv
./leaderboard/submitted_teams.txt
./leaderboard/leaderboard.md
./starter_code/baselinecode.ipynb
./starter_code/requirements.txt

=== SECTION 3: DATA SAMPLE ===
`train_compressed.csv`

Contains the features of the training nodes.
shape: (nodes * features) = (3760, 1717)
samples : 
cell_type     MMP12      C1QB    TYROBP   ....... is_perturbed
0          2162  0.000000  0.000000  0.000000           0.0
1          1609  0.000000  0.000000  0.000000           0.0
2          2564  0.000000  0.000000  0.000000           0.0
...         ...       ...       ...       ...           ...
3757       1873  0.022549 -0.029889 -0.015222           1.0
3758        255  0.000000  0.000000  0.000000           0.0
3759       2548  0.000000  0.000000  0.000000           0.0

cell_type is node id 

........................

`test.csv`

Contains the features of the testing nodes (the information is_perturbed and label is hidden)
shape : (nodes * features) = (940, 1716)
samples : 
cell_type     MMP12      C1QB    TYROBP
3611        0.000000  0.000000  0.000000
2259        -0.008862  0.013003  0.002834
4440        0.000000  0.000000  0.000000
..         ...       ...       ...       ...
2098       0.000000  0.000000  0.000000
958        0.000000  0.000000  0.000000

........................

`labels.csv`

Contains the labels of the training nodes (labels for testing set is hidden for competition evaluation)

shape : (training nodes * 2) = (3760, 2)
samples:
cell_type	label
2162	 0
1609	 1
2564	 2

labels.csv has TWO regular columns, no index:
    cell_type  → node IDs (int)
    label      → class labels (0, 1, 2) — THIS IS A 3-CLASS PROBLEM

Correct loading:
    labels_df  = pd.read_csv('data/labels.csv')
    cell_ids   = labels_df['cell_type'].values
    label_vals = labels_df['label'].values
    num_classes = len(np.unique(label_vals))  # = 3

.......................

`edges.csv`

Contains information about the edges for training and testing nodes:
shape : (edges * 4) = (27025, 4)

num	source	target	weight
0	   0	    422   	0.848251
1	   0	    304	    0.848072
2	   0	    1187	  0.846953
3	   0	    748	    0.846359
4	   0	    568	    0.843545
5	   1	    1763	  0.961374




=== SECTION 4: SUBMISSION FORMAT ===
After training the model on training data, use the trained model to make predictions on testing_data
--
Create a `submission.csv` in the following format and save it in the competition main directory :

```csv
id,cell_type
3611,0
2259,1
4440,2
359,1
3647,0
...
number of classes = 3

=== SECTION 5: REQUIREMENTS ===
torch>=2.0.0
torch_geometric>=2.4.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0

=== SECTION 6: BASELINE CODE ===

## ⚠️ CRITICAL — READ BEFORE WRITING ANY CODE
The code below is ALREADY CORRECT and ALREADY TESTED.
It works without any modifications.

YOU MUST:
- Copy it CHARACTER FOR CHARACTER into solution.py
- Not change a single line, character, or argument

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.loader import NeighborSampler , NeighborLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.utils.class_weight import compute_class_weight


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
import random

train = pd.read_csv(data\train_compressed.csv")
labels = pd.read_csv(data\labels.csv")
edges = pd.read_csv(data\edges.csv")
test = pd.read_csv(data\test.csv")

y_train = labels['label'].values  # or your node labels
classes = np.unique(y_train)
num_nodes = max(
    train["cell_type"].max(),
    test["cell_type"].max()
) + 1
num_features = train.shape[1] - 1
print(num_nodes , num_features)

x = torch.zeros((num_nodes, num_features), dtype=torch.float)
x[train["cell_type"].values] = torch.tensor(
    train.iloc[:, 1:].values, dtype=torch.float)

x[test["cell_type"].values] = torch.tensor(
    test.iloc[:, 1:].values, dtype=torch.float)
edge_index = torch.tensor(
    edges[["source", "target"]].values.T,
    dtype=torch.long
)
edge_weight = torch.tensor(edges["weight"].values, dtype=torch.float)

edge_index.shape , edge_weight.shape
y = torch.full((num_nodes,), -1, dtype=torch.long)

y[labels["cell_type"].values] = torch.tensor(
    labels["label"].values, dtype=torch.long)
y.shape , y , y[2259]

train_nodes = labels["cell_type"].values
train_ids, val_ids = train_test_split(train_nodes, test_size=0.2, random_state=42)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_ids] = True
val_mask[val_ids] = True

# train_mask[train["cell_type"].values] = True
test_mask[test["cell_type"].values] = True


data = Data(
    x=x,
    edge_index=edge_index,
    edge_weight=edge_weight,
    y=y)

# use NeighborLoader for train_loader , val_loader and test_loader

model.load_state_dict(best_model_state)
model.eval()

all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        out = model(batch.x, batch.edge_index, edge_weight=batch.edge_weight)
        preds = out[:batch.batch_size].argmax(dim=1)
        all_test_preds.append(preds)

all_test_preds = torch.cat(all_test_preds)

=== SECTION 7: IMPORTANT NOTES  ===

- Data paths: 'Challenge/data/'.
- The final saved file should be submission.csv 
- predictions should be classes (0 , 1 or 2) not probabilities.
- The model MUST be a Graph Neural Network (GNN) using torch_geometric.
- Always use float32 instead of float64
- Delete large objects immediately from memory after they are no longer needed.
- Make use of the baseline code if provided for you
