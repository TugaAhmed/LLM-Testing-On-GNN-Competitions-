You are solving a GNN coding competition.

You will receive:
  (1) the competition README,
  (2) the repository file tree,
  (3) a data-sample summary,
  (4) the required submission format,
  (5) allowed libraries.
  (6) baseline code

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
# 🧠 PARK-GNN Challenge: Parkinson's Disease Detection using Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Challenge Status](https://img.shields.io/badge/status-active-success.svg)](https://aiikram.github.io/gnn-parkinsons-challenge/)

**[🏆 View Live Leaderboard](https://aiikram.github.io/gnn-parkinsons-challenge/leaderboard.html)** | **[📖 View Challenge Website](https://aiikram.github.io/gnn-parkinsons-challenge/)**

---

## 🎯 Challenge Overview

Welcome to the **PARK-GNN Challenge** (**P**arkinson's **A**coustic **R**epresentation & **K**nowledge with **G**raph **N**eural **N**etworks).

This mini-competition focuses on detecting **Parkinson's Disease (PD)** from **acoustic voice measurements** using **Graph Neural Networks (GNNs)**.

### Why Graph Neural Networks?

Parkinson's Disease affects multiple vocal biomarkers **simultaneously and interdependently**. Traditional machine learning models treat samples as independent, ignoring these relationships.

In this challenge, the problem is framed as a **graph learning task**, where:

- **Nodes** represent individual voice recordings (or patients)
- **Edges** encode similarity between patients or shared subject-level information
- **Node features** consist of acoustic voice measurements  
  (e.g., jitter, shimmer, pitch, harmonics, nonlinear features)

By leveraging GNNs, participants can model **relational structure** in the data and capture patterns that classical tabular approaches may miss.

---

### 🏆 Competition Details

| **Aspect**     | **Details**                                  |
| -------------- | -------------------------------------------- |
| **Task Type**  | Node Classification (Binary)                 |
| **Difficulty** | ⭐⭐⭐⭐ (Challenging)                           |
| **Metric**     | **Macro F1-Score** (handles class imbalance) |
| **Dataset**    | UCI Parkinson's Dataset with graph structure |
| **Deadline**   | Open-ended (rolling leaderboard)             |

---

### 🎓 Learning Objectives

This challenge covers concepts from **DGL Lectures 1.1-4.6**:
- Graph construction from tabular data
- Message passing neural networks (MPNN)
- Graph attention mechanisms (GAT)
- Sampling methods for large graphs
- Node classification with GNNs

---

## 📊 Dataset Description

### Source
- **Original Dataset**: [UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Citation**: Little et al. (2008), 'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease'

### Features (22 acoustic measurements)
- **Vocal fundamental frequency measures**: MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)
- **Jitter variations**: MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP
- **Shimmer variations**: MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA
- **Harmonics & noise ratios**: NHR, HNR
- **Nonlinear measures**: RPDE, DFA, spread1, spread2, D2, PPE

### Graph Structure
- **Nodes**: 195 voice recordings from 31 subjects (23 PD, 8 healthy)
- **Edges**: K-nearest neighbors (k=5) + subject connections
- **Training**: 156 nodes (80%) — labels provided
- **Test**: 39 nodes (20%) — labels hidden

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/AiIkram/gnn-parkinsons-challenge.git
cd gnn-parkinsons-challenge
```

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r starter_code/requirements.txt
```

### 3. Run Baseline Model
```bash
cd starter_code
python baseline.py
```

Expected baseline F1-score: **~0.72-0.78**

### 4. Encrypt Your Submission
```bash
python encryption/encrypt_submission.py submissions/your_name.csv
# Automatically saves to submissions/encrypted/your_name.enc
```

---

## 📁 Repository Structure

```
gnn-parkinsons-challenge/
├── data/
│   ├── train_graph.pkl          # Training graph with labels
│   ├── test_graph.pkl           # Test graph without labels
│   └── feature_names.txt        # Feature descriptions
├── submissions/
│   └── sample_submission.csv    # Example submission
├── starter_code/
│   ├── baseline.py              # GCN baseline
│   ├── baseline_gat.py          # GAT baseline
│   └── requirements.txt         # Dependencies
├── scripts/
│   ├── generate_graph_data.py   # Data preprocessing
│   ├── scoring_script.py        # Evaluation
│   └── update_leaderboard.py    # Leaderboard management
├── .github/workflows/
│   └── score_submission.yml     # Auto-scoring
├── leaderboard.html             # Live leaderboard page
├── leaderboard.json             # Leaderboard data
├── index.html                   # Challenge homepage
├── _config.yml                  # GitHub Pages config
├── LEADERBOARD.md
├── RULES.md
└── README.md
```

---

## 📤 Making a Submission

### Submission Format

CSV with exactly 39 rows:

```csv
node_id,prediction
0,1
1,0
2,1
...
```

### How to Submit

1. **Fork this repository**
2. **Train your model** and generate predictions
3. **Save your predictions** to `submissions/your_name.csv`
4. **Encrypt your submission**:

```bash
python encryption/encrypt_submission.py submissions/your_name.csv
# Automatically saves the encrypted file to submissions/encrypted/your_name.enc
```

5. **Add and commit the encrypted file**:

```bash
git add submissions/encrypted/your_name.enc
git commit -m "Add my submission"
git push
```

6. **Create a Pull Request**
7. **GitHub Actions scores automatically** — decrypts securely, scores against hidden labels, then deletes everything
8. **Results posted** as a comment and added to the leaderboard

> ⚠️ **Only submit the `.enc` file** — never commit your raw CSV predictions. The automated pipeline will reject unencrypted submissions.

---

## 📈 Evaluation Metric

**Macro F1-Score** = (F1_Healthy + F1_Parkinson's) / 2

**Why?**
- ✅ Handles class imbalance
- ✅ Equal importance to both classes
- ✅ More challenging than accuracy
- ✅ Better reflects real-world performance

---

## 💡 Tips & Tricks

### For Beginners
1. ✅ Start with baseline GCN
2. ✅ Try different hidden sizes (32, 64, 128)
3. ✅ Vary number of layers (2-4)
4. ✅ Add dropout for regularization (0.3-0.5)
5. ✅ Use cross-validation

### Advanced
- 🔥 Experiment with k in KNN graphs (3, 5, 7, 10)
- 🔥 Add edge weights based on similarity
- 🔥 Try GAT attention mechanisms
- 🔥 Use skip connections / residual connections
- 🔥 Handle class imbalance (weighted loss, oversampling)
- 🔥 Ensemble multiple models
- 🔥 Try GraphSAGE, GIN, or other architectures

### Common Pitfalls
⚠️ **Overfitting** (small dataset - use regularization!)  
⚠️ **Over-smoothing** (too many layers collapse node representations)  
⚠️ **Ignoring class imbalance** (use weighted metrics)  
⚠️ **Data leakage** (don't use test labels!)

---

## 🎯 Challenge Rules

### ✅ Must Do
- Use at least one GNN layer
- Only use provided dataset
- Complete inference within 5 minutes
- Set random seeds for reproducibility
- Provide code with submission

### ❌ Cannot Do
- Use test labels (obviously!)
- Use external Parkinson's datasets
- Use pure non-GNN models (e.g., just MLP)

**See [RULES.md](RULES.md) for complete details.**

---

## 🤝 Contributing

- **Bug?** [Open an issue](https://github.com/AiIkram/gnn-parkinsons-challenge/issues)
- **Question?** [Start a discussion](https://github.com/AiIkram/gnn-parkinsons-challenge/discussions)
- **Improvement?** Submit a PR

---

## 📝 Citation

```bibtex
@misc{gnn_parkinsons_challenge2025,
  title={GNN Mini-Challenge: Parkinson's Disease Detection},
  author={Aissiou Ikram},
  year={2025},
  url={https://github.com/AiIkram/gnn-parkinsons-challenge}
}
```

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/AiIkram/gnn-parkinsons-challenge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AiIkram/gnn-parkinsons-challenge/discussions)
- **Email**: [aissiouikram47@gmail.com](mailto:aissiouikram47@gmail.com)

---

## 📜 License

MIT License — see [LICENSE](LICENSE) file.

**Dataset License**: UCI Parkinson's Dataset — CC BY 4.0

---

<div align="center">

### 🚀 Ready to start?

**[View Leaderboard](https://aiikram.github.io/gnn-parkinsons-challenge/leaderboard.html)** | **[Fork Repo](https://github.com/AiIkram/gnn-parkinsons-challenge/fork)** | **[Submit Solution](https://github.com/AiIkram/gnn-parkinsons-challenge/pulls)**

**Good luck! 🎉**

</div>


=== SECTION 2: REPO TREE ===
./data/public/node_id_mapping.csv
./data/public/test_graph.pkl
./data/public/feature_names.txt
./data/public/test_nodes.csv
./data/public/train_graph_free.pkl
./data/public/train_graph.pkl
./data/public/test_graph_free.pkl
./LICENSE
./RULES.md
./docs/leaderboard.html
./docs/leaderboard.csv
./docs/leaderboard.json
./docs/leaderboard.css
./docs/leaderboard.js
./docs/index.html
./leaderboard.json
./_config.yml
./submissions/encrypted/submission_5.enc
./submissions/encrypted/sanae_submissions.enc
./submissions/encrypted/Ik_submission.enc
./submissions/encrypted/tugasubmission.enc
./submissions/encrypted/llm_gpt5_codex.enc
./submissions/encrypted/llm_codex_gpt5-4.enc
./submissions/encrypted/Abderrahmane_submission.enc
./submissions/encrypted/llm_gemini-3-flash.enc
./submissions/encrypted/gat_submission.enc
./submissions/encrypted/gururgg.enc
./submissions/encrypted/Muhammad_Isah.enc
./submissions/encrypted/rpy_submission.enc
./submissions/encrypted/gcn_submission.enc
./submissions/encrypted/test_pred.enc
./submissions/encrypted/peguy_baseline_gcn.enc
./submissions/encrypted/bijay.enc
./submissions/encrypted/sanae_submission.enc
./submissions/encrypted/sargam.enc
./submissions/encrypted/Iksubmission.enc
./submissions/encrypted/ElIkram_submission.enc
./submissions/encrypted/emmanuel_owusu.enc
./submissions/encrypted/test_submission.enc
./submissions/encrypted/samuel.enc
./submissions/encrypted/claude_opus_4_6.csv
./submissions/encrypted/.DS_Store
./submissions/encrypted/claude_opus_4_6.enc
./.gitignore
./scripts/update_leaderboard.py
./scripts/convert_to_dgl_free.py
./scripts/fix_test_labels.py
./scripts/generate_graph_data.py
./scripts/scoring_script.py
./competition/render_leaderboard.py
./competition/evaluate.py
./competition/metrics.py
./index.html
./README.md
./scoring_script.py
./.github/workflows/score_submission.yml
./.github/workflows/update_leaderboard.yml
./leaderboard/leaderboard.md
./encryption/decrypt_submission.py
./encryption/generate_keys.py
./encryption/encrypt_submission.py
./encryption/public_key.pem
./.nojekyll
./starter_code/best_gcn_model.pt
./starter_code/generate_submission.py
./starter_code/baseline_gat.py
./starter_code/requirements.txt
./starter_code/baseline.py
./starter_code/best_model.pt
./starter_code/submissions/baseline_gcn.csv

=== SECTION 3: DATA SAMPLE ===
`train_graph_free.pkl`

Contains information about training and validation nodes in a form of python dict structured as follows:

| Key          | type         | Dtype   | Shape    | Description                                                      |
| ------------ | ------------ | ------- | -------- | ---------------------------------------------------------------- |
| `features`   | torch.Tensor | float32 | 195, 22  | 22 features for all 195 nodes                                    |
| `labels`     | torch.Tensor | int64   | 195      | label of each node (-1 value indicate testing nodes)             |
| `train_mask` | torch.Tensor | bool    | 195      | indicate with `True` training nodes and with `False` otherwise   |
| `val_mask`   | torch.Tensor | bool    | 195      | indicate with `True` validation nodes and with `False` otherwise |
| `edge_index` | torch.Tensor | int64   | 2, 39780 | source and target node ids for each edge                         |

`test_graph_free.pkl`

Contains information about testing nodes in a form of python dict structured as follows:

| Key          | type          | Dtype   | Shape    | Description                                                                      |
| ------------ | ------------- | ------- | -------- | -------------------------------------------------------------------------------- |
| `features`   | torch.Tensor  | float32 | 195, 22  | 22 features for all 195 nodes                                                    |
| `node_ids`   | numpy.ndarray | int32   | 39       | indicate with `True` validation nodes and with `False` otherwise                 |
| `edge_index` | torch.Tensor  | int64   | 2, 39780 | source and target node ids for each edge (same edges in `train_graph_free.pkl` ) |

`train_graph.pkl` and `test_graph.pkl` are the same as `train_graph_free.pkl` and `test_graph_free.pkl` respectively but in the form of dgl.heterograph.DGLGraph


=== SECTION 4: SUBMISSION FORMAT ===
After training the model on training data and validate on validation data , use the trained model to make predictions on testings
--
Create a `submission.csv` in the following format and save it in the competition main directory :

```csv
node_id,prediction
0,1
1,0
2,1
...
...


=== SECTION 5: REQUIREMENTS ===
torch==2.0.1
dgl==1.1.2
numpy==1.24.3
pandas>=1.3.0
scikit-learn>=1.0.0
networkx>=2.6.0
matplotlib>=3.4.0
scipy>=1.7.0
torchdata==0.6.1





=== SECTION 6: BASELINE CODE  ===
import numpy as np
from torchmetrics.classification import MulticlassF1Score

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import pickle
import pandas as pd
# import numpy as np
from sklearn.metrics import f1_score
from dgl.nn import 

def load_data():
    print("Loading data...")

    # ─────────────────────────────────────────────────────────────
    # CHOOSE YOUR DATA FORMAT:
    #   "free" → train_graph_free.pkl  (no DGL needed to load data)
    #   "dgl"  → train_graph.pkl       (requires DGL installed)
    DATA_FORMAT = "free"   # ← change to "dgl" if you prefer
    # ─────────────────────────────────────────────────────────────

    if DATA_FORMAT == "free":
        with open('../data/public/train_graph_free.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('../data/public/test_graph_free.pkl', 'rb') as f:
            test_data = pickle.load(f)

        def rebuild_dgl_graph(d):
            src, dst = d["edge_index"]
            g = dgl.graph((src, dst), num_nodes=d["num_nodes"])
            d["graph"] = g
            return d

        train_data = rebuild_dgl_graph(train_data)
        test_data  = rebuild_dgl_graph(test_data)
        print("  Loaded DGL-free format (rebuilt graph at runtime)")

    elif DATA_FORMAT == "dgl":
        with open('../data/public/train_graph.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('../data/public/test_graph.pkl', 'rb') as f:
            test_data = pickle.load(f)
        print("  Loaded DGL format")

    else:
        raise ValueError(f"Unknown DATA_FORMAT '{DATA_FORMAT}'. Choose 'free' or 'dgl'.")

    return train_data, test_data


train_data, test_data = load_data()

g          = train_data['graph']
features   = train_data['features']
labels     = train_data['labels']
train_mask = train_data['train_mask']
val_mask   = train_data['val_mask']





Important notes : 
- This is Binary Graph Classification not node classification , so you need to Aggregate node embeddings
- Data paths: 'gnn-parkinsons-challenge/data/public'.
- The final saved file should be submission.csv 
- predictions should be classes not probabilities.
- The model MUST be a Graph Neural Network (GNN) using torch_geometric.
- Always use float32 instead of float64
- Delete large objects immediately from memory after they are no longer needed.
- Make use of the baseline code if provided for you
- This is a NODE CLASSIFICATION task with a SINGLE graph.
- The graph contains 195 nodes, but ONLY a subset of nodes (test nodes) must be predicted.
- The test nodes are provided in:
    test_graph_free.pkl → key: "node_ids"

- After training on training set , You MUST Extract predictions ONLY for test node indices
- DO NOT output predictions for all 195 nodes.