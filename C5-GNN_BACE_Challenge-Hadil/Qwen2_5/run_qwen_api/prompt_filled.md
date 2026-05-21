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

# GNN Mini-Challenge: BACE Inhibition Prediction

Welcome to the GNN Mini-Challenge! Your task is to build a Graph Neural Network that can predict whether a molecule will inhibit the BACE-1 enzyme, a key target in Alzheimer's disease research.
This challenge is not just about getting the highest score, but about exploring how GNNs can learn meaningful chemical properties directly from a molecule's structure.


## Competition Workflow

```mermaid
graph TD
    subgraph Data_Preparation
        Raw[Raw BACE Data] --> Split[Scaffold Splitter]
        Split -->|Train Set| Imbalance[Class Imbalance Downsampling]
        Split -->|Test Set| Test[Test Set Evaluation]
        Imbalance --> Train[Train Set Imbalanced]
    end

    subgraph Participant_Workflow
        Train --> Model[Train GNN Model]
        Test --> Pred[Generate Predictions]
        Model --> Pred
        Pred --> CSV[Submission.csv]
        CSV --> PR[GitHub Pull Request]
    end

    subgraph Automated_Evaluation
        PR --> Check[Eligibility Check]
        Check -->|Pass| Score[Score Submission]
        Score --> Board[Update Leaderboard]
        Board --> Close[Auto-Close PR]
    end
```

## The Challenge Philosophy: Why Only SMILES?

In many machine learning tasks, you are given a rich table of pre-calculated features. In this challenge, we are doing something different.

You are provided with only one primary piece of information for each molecule: its SMILES string. We have intentionally removed all other molecular descriptors (like PSA, MolLogP, etc.).

**Why?**

Our goal is to test the power of representation learning. Instead of giving the model the "answers" (pre-calculated features), we want your GNN to discover them on its own. The model must learn to infer chemical properties—like polarity, reactivity, and steric effects—by analyzing the graph of atoms and bonds. This is a much harder but more powerful and generalizable approach.

Your challenge is to build a GNN that can act like a computational chemist, deriving its own insights from the raw molecular blueprint.

## About the Data

The dataset for this challenge is derived from the BACE dataset from the widely-used MoleculeNet benchmark suite.

    Original Dataset: BACE Dataset on MoleculeNet
    Biological Context: BACE-1 (β-secretase 1) is an enzyme involved in the production of amyloid-beta peptides, which are a key component of the plaques found in the brains of Alzheimer's patients. Inhibiting BACE-1 is a major strategy in developing treatments for Alzheimer's disease.
    Task: Binary classification. Predict if a molecule will inhibit BACE-1 (1 for active, 0 for inactive).

### Provided Files

     

The data is located in the data/public/ directory and is split into node and edge files for both training and testing sets. 

     train_nodes.csv & train_edges.csv: The graph structure for the training molecules.
     train_labels.csv: The target labels for the training set.
     test_nodes.csv & test_edges.csv: The graph structure for the test molecules.
     sample_submission.csv: An example of the submission format.
     



### Graph Specification

To align with standard GNN formulation:

-   **Node Feature Matrix ($X$)**:
    -   Provided in `train_nodes.csv` and `test_nodes.csv`.
    -   Each row corresponds to a node $v_i$.
    -   Columns `nf_0` to `nf_7` represent the feature vector $x_i$ (One-hot encoded atom type, dim=8).

-   **Adjacency Matrix ($A$)**:
    -   Provided in `train_edges.csv` and `test_edges.csv`.
    -   Represented as a sparse Edge List (COO format).
    -   Columns `src` and `dst` define the edges $(u, v)$ such that $A_{uv} = 1$.
    -   The graph is undirected.


### Data Challenges

To make this a realistic challenge, the data includes: 

     Class Imbalance: The training set is heavily imbalanced (~10% active) to mimic real-world screening. The test set follows a different distribution.
     Distribution Shift: The dataset uses a **Scaffold Split**, meaning the test set contains molecules with different chemical backbones than the training set. Your model must generalize to new chemical space!
     Variable Graph Sizes: Molecules have different numbers of atoms and bonds, so your model must handle batched graphs of varying sizes.
     

## Problem Type

Graph Classification: Given a molecule (represented by its SMILES string), predict if it is active (1) or inactive (0) against BACE-1.


## The Task: Graph Classification 

Your goal is to solve a graph classification problem. Your GNN must analyze the provided node and edge data to predict the target class for each graph.

## Evaluation: The Macro F1-Score 

Submissions will be ranked by the Macro F1-Score. This metric is essential here because it calculates the F1-score for each class independently and then averages them, giving equal importance to both the minority (active) and majority (inactive) classes.

**Ranking Policy**: We follow Kaggle guidelines—tied scores share the same rank (e.g., if two teams have the top score, both are Rank 1, and the next team is Rank 3).

> [!NOTE]
> **Baseline Score (0.3908)**: A Macro F1 score of approximately **0.390879** is the result of predicting "0" (inactive) for all molecules in the test set. Due to class imbalance, this is the baseline performance to beat!

## 📊 Leaderboard 

The leaderboard is automatically updated after each submission is scored. It ranks all submissions by their Macro F1-Score in descending order. 

Click here to see the current leaderboard : [Leaderboard](https://hadilaff.github.io/GNN_BACE_Challenge/leaderboard.html) 
How to Get Started 

Ready to dive in? It only takes a few minutes to get a baseline model running. 

   1. Clone the Repository:
``` code
git clone https://github.com/hadilaff/GNN_BACE_Challenge
cd gnn-challenge
```
   2. Set Up Your Environment (a virtual environment is highly recommended):
   ``` code
   pip install -r requirements.txt
``` 
   3. Run the Baseline Model: 
   ``` code
   cd starter_code
   python baseline.py
   ``` 
This will train a simple Graph Convolutional Network (GCN) and generate a submission.csv in the submissions/inbox/ directory. This score is the one to beat!

## Submission Format 

To submit your results, you must **encrypt** your submission file before creating a pull request. This ensures the test set predictions remain private.

### Step-by-Step Submission Process

1. **Generate your predictions** using the baseline or your own model:
   ```bash
   cd starter_code
   python baseline.py
   ```
   This creates `submissions/inbox/submission.csv` with two columns: `id` and `target`.

2. **Encrypt your submission**:
   ```bash
   python competition/encrypt_submission.py submissions/inbox/submission.csv
   ```
   This creates `submissions/inbox/submission.enc` (encrypted file).

3. **Create a new branch and commit the encrypted file**:
   ```bash
   git checkout -b my-submission
   git add submissions/inbox/submission.enc
   git commit -m "Add encrypted submission"
   git push -u origin my-submission
   ```

4. **Open a Pull Request** on GitHub.

> [!NOTE]
> **Automated Scoring**: Your PR will be automatically scored by GitHub Actions. The workflow will:
> - Decrypt your submission securely
> - Score it against the private test set
> - Update the leaderboard
> - **Close the PR automatically** (it will not be merged)
> - **Delete your branch** to keep the repository clean

> [!IMPORTANT]
> **Security**: Only submit the `.enc` file, never the `.csv` file. The encryption ensures test set predictions remain confidential.

### Submission Policy

> [!WARNING]
> **One Submission Rule**: To encourage thoughtful model design, each participant is allowed **only one successful submission**.

Once your submission is scored and appears on the leaderboard, you cannot submit again. Please verify your model locally before submitting!




### Repository Structure 
``` text
gnn-challenge/
├── data/
│   └── public/             # All public challenge data
│       ├── train_nodes.csv
│       ├── train_edges.csv
│       ├── train_labels.csv
│       ├── test_nodes.csv
│       ├── test_edges.csv
│       └── sample_submission.csv
├── competition/             # Scoring and validation logic
│   ├── config.yaml
│   ├── evaluate.py
│   └── render_leaderboard.py
    └── check_eligibility.py
├── starter_code/           # Baseline model & dependencies
│   ├── baseline.py
├── submissions/
│   └── inbox/             # Place your submission CSVs here
├── leaderboard/           # Auto-updated rankings
│   ├── leaderboard.csv
│   └── leaderboard.md
├── .github/workflows/     # Automation for scoring & leaderboard
├── README.md
└── requirements.txt
└── requirements.txt
└── LICENSE                
```

### Tips for Success 

     Looking for an edge? Here are some ideas to explore: 

     Experiment with Architectures: The baseline uses a GCN. Try different GNN layers like GAT (Graph Attention Network) or GraphSAGE to see how different message-passing strategies perform.
     Feature Engineering: The baseline uses one-hot encoded atomic numbers. Can you add more expressive features for atoms (e.g., formal charge, hybridization) and bonds (e.g., aromaticity)?
     Tune Hyperparameters: Systematically tune the learning rate, number of layers, hidden dimensions, and dropout rate.
     Prevent Overfitting: With a small dataset, overfitting is a real risk. Use techniques like dropout and early stopping.
     
     

Good luck, and happy graph learning!     

=== SECTION 2: REPO TREE ===``` text
gnn-challenge/
├── data/
│   └── public/             # All public challenge data
│       ├── train_nodes.csv
│       ├── train_edges.csv
│       ├── train_labels.csv
│       ├── test_nodes.csv
│       ├── test_edges.csv
│       └── sample_submission.csv
├── competition/             # Scoring and validation logic
│   ├── config.yaml
│   ├── evaluate.py
│   └── render_leaderboard.py
    └── check_eligibility.py
├── starter_code/           # Baseline model & dependencies
│   ├── baseline.py
├── submissions/
│   └── inbox/             # Place your submission CSVs here
├── leaderboard/           # Auto-updated rankings
│   ├── leaderboard.csv
│   └── leaderboard.md
├── .github/workflows/     # Automation for scoring & leaderboard
├── README.md
└── requirements.txt
└── requirements.txt
└── LICENSE                


```


=== SECTION 3: DATA SAMPLE ===
train_nodes.csv :
Contains Node Feature Matrix for training data graphs 
Shape: (nodes * [graph_id, node_id]+features  ) = (20360, 10)
columns : ['graph_id', 'node_id', 'nf_0', 'nf_1', 'nf_2', 'nf_3', 'nf_4', 'nf_5','nf_6', 'nf_7']
samples:
graph_id  node_id  nf_0  nf_1  nf_2  nf_3  nf_4  nf_5  nf_6  nf_7
BACE_1042        0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0
BACE_1042        1   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
BACE_1042        2   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
BACE_1042        3   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
BACE_1042        4   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
...........

train_edges.csv : 
Contains the Adjacency Matrix. Represented as a sparse Edge List (COO format) for training graphs. Columns `src` and `dst` define the edges (u, v) such that A{uv} = 1.
shape :  (edges * [ graph_id , src  , dst , ef_0])   = (43998, 4)
samples:
BACE_1042    0    1   1.0
BACE_1042    1    0   1.0
BACE_1042    1    2   1.5
BACE_1042    2    1   1.5
BACE_1042    2    3   1.5
............................

train_labels.csv: 
Contains the target labels for the training set. Each row is the class(label) for the graph
shape :  (graphs * [id , target] ) =  (642, 2)
samples : 
id           target
BACE_1042       0
BACE_1022       0
BACE_1275       0
BACE_36         1
BACE_851        0
...........................

test_nodes.csv:
Contains Node Feature Matrix for testing data graphs 
Shape: (nodes * [graph_id, node_id]+features  ) =  (12943, 10)
columns : ['graph_id', 'node_id', 'nf_0', 'nf_1', 'nf_2', 'nf_3', 'nf_4', 'nf_5','nf_6', 'nf_7']
samples: 
graph_id  node_id  nf_0  nf_1  nf_2  nf_3  nf_4  nf_5  nf_6  nf_7
BACE_231        0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0
BACE_231        1   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
BACE_231        2   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
BACE_231        3   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0
BACE_231        4   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
.....................

test_edges.csv:
Contains the Adjacency Matrix. Represented as a sparse Edge List (COO format) for testing graphs. Columns `src` and `dst` define the edges (u, v) such that A{uv} = 1.
shape :  (edges * [ graph_id , src  , dst , ef_0])   = (28202, 4)
samples:
graph_id  src  dst  ef_0
BACE_231    0    1   1.0
BACE_231    1    0   1.0
BACE_231    1    2   1.0
BACE_231    2    1   1.0
BACE_231    2    3   1.0
.............


### Data Splits
- Training set: included in train_nodes.csv,train_edges.csv and train_labels.csv
- Test set:  included in test_nodes.csv and test_edges.csv (labels are hidden for competition fair assessment)
- Labels: binary classification (0 or 1 )

## Problem Type
Graph Classification: Given a molecule (represented by its SMILES string), predict if it is active (1) or inactive (0) against BACE-1.

## The Task: Graph Classification 
Your goal is to solve a graph classification problem. Your GNN must analyze the provided node and edge data to predict the target class for each graph.



=== SECTION 4: SUBMISSION FORMAT ===
After training the model using train_nodes.csv , train_edges.csv and train_labels.csv, use the trained model to make predictions on testing set exist in test_nodes.csv and test_edges.csv
--
Create a `submission.csv` in the following format and save it in the competition main directory :
This challenge requires: id,y_pred columns, 374 rows (one per test graph).

```csv
id,target
BACE_231,0
BACE_232,0
BACE_234,1
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
import pandas as pd
import torch
import torch.nn.functional as F
# FIX 1: Updated the import for DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
from torch_geometric.nn import SAGEConv,GCNConv, GATConv, global_max_pool , global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score , accuracy_score
import numpy as np
import os

# --- 1. Load Graphs from CSVs ---
def load_graphs_from_csv(nodes_path, edges_path, labels_path=None):
    """Loads graph data from node and edge CSV files."""
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    data_list = []
    graph_labels = {}
    
    if labels_path:
        labels_df = pd.read_csv(labels_path)
        graph_labels = dict(zip(labels_df['id'], labels_df['target']))

    graph_ids = nodes_df['graph_id'].unique()

    for graph_id in graph_ids:
        graph_nodes = nodes_df[nodes_df['graph_id'] == graph_id]
        graph_edges = edges_df[edges_df['graph_id'] == graph_id]

        node_features = graph_nodes.drop(columns=['graph_id', 'node_id']).values
        x = torch.tensor(node_features, dtype=torch.float)

        if not graph_edges.empty:
            edge_index = torch.tensor(graph_edges[['src', 'dst']].values, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(graph_edges[['ef_0']].values, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        if graph_id in graph_labels:
            data.y = torch.tensor([graph_labels[graph_id]], dtype=torch.long)
        
        data.graph_id = graph_id
        data_list.append(data)
        
    return data_list

all_train_graphs = load_graphs_from_csv(
        '../data/public/train_nodes.csv',
        '../data/public/train_edges.csv',
        '../data/public/train_labels.csv')

print("len all_train_graphs",len(all_train_graphs) , "all_train_graphs[0]",all_train_graphs[0])

train_graphs, val_graphs = train_test_split(all_train_graphs, test_size=0.2, random_state=42)
print("len train_graphs",len(train_graphs) , "train_graphs[0]",train_graphs[0])
print("len val_graphs",len(val_graphs) , "val_graphs[0]",val_graphs[0])

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)



=== SECTION 7: IMPORTANT NOTES  ===

- Data path: 'GNN_BACE_Challenge/data/public'.
- Submission path should be in the competition main directory  'GNN_BACE_Challenge/submission.csv'.
- The final saved file should be submission.csv 
- predictions should be classes (0 or 1) not probabilities.
- The model MUST be a Graph Neural Network (GNN) using torch_geometric.
- Always use float32 instead of float64
- Delete large objects immediately from memory after they are no longer needed.
- Make use of the baseline code if provided for you 
