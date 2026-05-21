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
# 🛡️ PROVEN-GNN

## PROgram Vulnerability Examination Network using Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Challenge Status](https://img.shields.io/badge/status-active-success.svg)](https://abdksm.github.io/PROVEN-GNN/docs/leaderboard.html)

**[🏆 View Live Leaderboard](https://abdksm.github.io/PROVEN-GNN/docs/leaderboard.html)**

---

## 🎯 Challenge Overview

Welcome to **PROVEN-GNN** (**PRO**gram **V**ulnerability **E**xamination **N**etwork using **G**raph **N**eural **N**etworks), a competition designed to compare human-built Graph Neural Network (GNN) solutions against Large Language Model (LLM)-based approaches.

The objective is to classify code functions as **vulnerable** or **non-vulnerable** using graph representations of source code.

---

## 🏆 Competition Details

| **Aspect**            | **Details**                                                          |
| --------------------- | -------------------------------------------------------------------- |
| **Task Type**         | Binary Graph Classification                                          |
| **Evaluation Metric** | **Macro F1-Score**                                                   |
| **Dataset**           | Inspired by [DiverseVul](https://github.com/wagner-group/diversevul) |

---

## 📊 Dataset Description

### Source

* **Original Dataset**: [DiverseVul Dataset](https://github.com/wagner-group/diversevul)
* **Citation**:
  Chen, Yizheng, et al. *"DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning-Based Vulnerability Detection."*
  Proceedings of the 26th International Symposium on Research in Attacks, Intrusions and Defenses (RAID), 2023.

---

### Graph Construction

The dataset was adapted by generating graph representations for a subset of code functions. It includes **2,487 training graphs** and **622 test graphs**. The class distribution is imbalanced, with vulnerable code representing only **29%** of the data.


We used **Joern** to construct **Code Property Graphs (CPG)**, which combine:

* Abstract Syntax Tree (AST)
* Control Flow Graph (CFG)
* Program Dependence Graph (PDG)
* Control Dependence Graph (CDG)
* Data Dependence Graph (DDG)

The following figure shows an example of a generated CPG:

![CPG](extra/CPG.png)

---

### Node Features (527 Dimensions)

* **Node Type (15 features)**
  Encodes the type of node (e.g., method, function call, variable declaration, etc.)

* **Code Embedding (512 features)**
  Dense embedding representing the code snippet associated with the node.

---

### Edge Features (12 Types)

Edge features represent the type of relationship between nodes (e.g., AST, CFG, PDG, CDG, DDG).

---

## 📁 Repository Structure

```
PROVEN-GNN
├── data/
│   ├── public/
│   │   ├── train_data.csv    (to be downloaded)
│   │   ├── test_data.csv     (to be downloaded)
│   │   ├── test_ids.csv
│   │   ├── sample_submission.csv
│   │   └── README.md
├── competition/
│   ├── config.yaml
│   ├── update_leaderboard.py
│   ├── evaluate.py
│   ├── render_leaderboard.py
│   └── validate_submission.py
├── submissions/             (place submission files here)
│   └── README.md
├── leaderboard/
│   ├── leaderboard.csv
│   └── leaderboard.md
├── docs/
│   ├── leaderboard.html
│   ├── leaderboard.css
│   └── leaderboard.js
└── .github/workflows/
    ├── score_submission.yml
    └── publish_leaderboard.yml
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/abdksm/PROVEN-GNN.git
cd PROVEN-GNN
```

### 2️⃣ Install Dependencies

```bash
pip install -r starter_code/requirements.txt
```

### 3️⃣ Download the Data

```bash
cd data/public
pip install gdown

gdown --id 1kUNwo7WjVpJ2D1GPsotiNO5FJnCqt--9 -O train_data.parquet
gdown --id 1xhg62LTAJm5ityBKiXKv8Rsg0eSl9TJC -O test_data.parquet
```

### 4️⃣ Run the Baseline Model

```bash
cd ../../starter_code
python baseline.py
```

---

## 📤 Making a Submission

### Submission Format

#### `predictions.csv`

```csv
id,y_pred
0,1
1,0
2,1
...
```

#### `metadata.json`

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

* `"human"`
* `"llm-only"`
* `"human+llm"`

### Encryption Process

You should be in the root directory, and the file `predictions.csv` should be inside the `submissions/` directory
```bash
# For Windows users
.\extra\encrypt_win.ps1 submissions\predictions.csv

# For Linux users
bash extra/encrypt_linux.sh submissions/predictions.csv
```

---

### Submission Steps

1. Fork this repository
2. Train your model and generate predictions
3. Add `predictions.csv` and `metadata.json` to the `submissions/` directory
4. Run the encryption script (**A critical step !**)
5. Create a Pull Request
6. GitHub Actions automatically evaluate your submission
7. Results are posted as a comment and added to the leaderboard

---

## 📈 Evaluation Metric

The evaluation metric is **Macro F1-Score**:

$$
\text{Macro F1} = \frac{F1_{Vulnerable} + F1_{Non\text{-}Vulnerable}}{2}
$$

Rankings are sorted by **descending score**.

---

## 🏆 Leaderboard

After submission, scores are automatically added to:

* `leaderboard/leaderboard.csv`
* `leaderboard/leaderboard.md`

An interactive leaderboard is available here:

👉 [Live Leaderboard](https://abdksm.github.io/PROVEN-GNN/docs/leaderboard.html)

---

## 🎯 Challenge Rules

* ❌ No external or private datasets
* ❌ No manual labeling of test data
* ❌ No modification of evaluation scripts
* ✅ Unlimited offline training is allowed
* ⚠️ Only one submission per user is allowed

Violations may result in disqualification.

---

## 📝 Citation

```bibtex
@misc{proven_gnn_2025,
  title={PROVEN-GNN: Program Vulnerability Examination Network},
  author={Abderrahmane Kasmi},
  year={2025},
  url={https://github.com/abdksm/PROVEN-GNN}
}
```

---

## 📜 License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.




=== SECTION 2: REPO TREE ===
./extra/CPG.png
./extra/encrypt_win.ps1
./extra/public_key.pem
./extra/encrypt_linux.sh
./data/README.md
./data/public/sample_submission.csv
./data/public/train_data.parquet
./data/public/test_data.parquet
./data/public/test_ids.csv
./docs/leaderboard.html
./docs/leaderboard.css
./docs/leaderboard.js
./submissions/README.md
./competition/update_leaderboard.py
./competition/render_leaderboard.py
./competition/validate_submission.py
./competition/evaluate.py
./competition/config.yaml
./README.md
./.github/workflows/score_submission.yml
./.github/workflows/publish_leaderboard.yml
./leaderboard/leaderboard.csv
./leaderboard/leaderboard.md
./starter_code/requirements.txt
./starter_code/baseline.py

=== SECTION 3: DATA SAMPLE ===
`train_data.parquet`

| Column       | Runtime Type               | Inner Elements    | Logical Shape                    | Description                    |
| ------------ | -------------------------- | ----------------- | -------------------------------- | ------------------------------ |
| `id`         | `int64`                    | —                 | scalar                           | Unique graph/sample identifier |
| `node_feat`  | `np.ndarray(dtype=object)` | `float64 ndarray` | object array of `(527,)` vectors | Node feature vectors           |
| `edge_index` | `np.ndarray(dtype=object)` | `int64 ndarray`   | 2 arrays of shape `(N_edges,)`   | COO graph connectivity         |
| `edge_attr`  | `np.ndarray(dtype=object)` | `int64 ndarray`   | object array of `(12,)` vectors  | Edge feature vectors           |
| `label`      | `int64`                    | —                 | scalar                           | Target label                   |


**Total rows:** 2487

---

`test_data.parquet`

| Column       | Runtime Type               | Inner Elements    | Logical Shape                    | Description                    |
| ------------ | -------------------------- | ----------------- | -------------------------------- | ------------------------------ |
| `id`         | `int64`                    | —                 | scalar                           | Unique graph/sample identifier |
| `node_feat`  | `np.ndarray(dtype=object)` | `float64 ndarray` | object array of `(527,)` vectors | Node feature vectors           |
| `edge_index` | `np.ndarray(dtype=object)` | `int64 ndarray`   | 2 arrays of shape `(N_edges,)`   | COO graph connectivity         |
| `edge_attr`  | `np.ndarray(dtype=object)` | `int64 ndarray`   | object array of `(12,)` vectors  | Edge feature vectors           |

**Total rows:** 622

---

`test_ids.csv`

| Column | Data Type |
| ------ | --------- |
| `id`   | `int64`   |

**Total rows:** 622

### Graph Labels
Each graph in the dataset has a label (0 or 1 ) 

=== SECTION 4: SUBMISSION FORMAT ===
Participants must train on `train_data.parquet` and generate predictions for `test_data.parquet`.

The prediction file should have the following format:

```csv
id,y_pred
9419,1
2905,1
7712,0
13344,1
18760,1
```
## Graph-Level Classification — CRITICAL RULES

- This is a GRAPH classification task, not a NODE classification task.
- The model must output ONE prediction per GRAPH, not one per node.


=== SECTION 5: REQUIREMENTS ===
pandas==2.3.3
pyarrow==22.0.0
numpy==2.3.5
scikit-learn==1.8.0
torch==2.5.1
torch-geometric==2.7.0



=== SECTION 6: BASELINE CODE ===

## ⚠️ CRITICAL — READ BEFORE WRITING ANY CODE

The function below is ALREADY CORRECT and ALREADY TESTED.
It works without any modifications.

YOU MUST:
- Copy it CHARACTER FOR CHARACTER into solution.py
- Not change a single line, character, or argument

```python
def build_dataloader(df, batch_size=32, shuffle=False, has_labels=True):
    graph_list = []

    for _, row in df.iterrows():
        x = torch.tensor(np.vstack(row["node_feat"]).astype(np.float32))
        edge_index = torch.tensor(np.vstack(row["edge_index"]).astype(np.int64))
        edge_attr = torch.tensor(np.vstack(row["edge_attr"]).astype(np.float32))

        if has_labels:
            y = torch.tensor(row["label"], dtype=torch.long)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
            )
        else:
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

        graph_list.append(data)

    return DataLoader(graph_list, batch_size=batch_size, shuffle=shuffle)
```

=== SECTION 7: IMPORTANT NOTES  ===

- Data paths: 'PROVEN-GNN/data/public/'.
- The final saved file should be submission.csv 
- Your task is Graph classification not node classification.  
- predictions should be classes (0 or 1) not probabilities.
- The model MUST be a Graph Neural Network (GNN) using torch_geometric.
- Always use float32 instead of float64
- Delete large objects immediately from memory after they are no longer needed.
- Use the baseline code provided to you exactly as it is.
- Don't encrypt submissions.csv . 
