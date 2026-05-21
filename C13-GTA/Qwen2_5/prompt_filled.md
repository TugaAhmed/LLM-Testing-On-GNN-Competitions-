You are solving a GNN coding competition.

You will receive:
  (1) the competition README,
  (2) the repository file tree,
  (3) a data-sample summary,
  (4) the required submission format,
  (5) allowed libraries.

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
# GTA (Graph Topology Ablation) Challenge

This repository hosts the official evaluation system for the **Graph Topology Ablation (GTA) challenge**. Participants submit predictions for ideal and perturbed graph topologies. All submissions are encrypted, automatically evaluated, and ranked on a public leaderboard.

📊 **Live leaderboard**: [Open leaderboard](https://idrees11.github.io/GTA-Graph-Topology-Ablation/index.html)  
---

## 🎯 Objective ##          

Participants must generate predictions for two settings:

- ✅ **Ideal graph topology** – clean, unmodified node features.
- ✅ **Perturbed graph topology** – node features corrupted by a combination of distribution shift and Gaussian noise.

The goal is to build a Graph Neural Network (GNN) that is both accurate on clean data and robust to realistic feature corruptions.

---

## 📌 Dataset Description

We use the **MUTAG dataset**, a classic benchmark for graph classification from chemical informatics.

**🔗 Official source**:  
[https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip](https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip)

### Core Statistics

| Property             | Value                                             |
| -------------------- | ------------------------------------------------- |
| Task                 | Binary graph classification                       |
| Domain               | Chemical compounds (mutagenic vs non‑mutagenic)   |
| Number of graphs     | 188                                               |
| Avg. nodes per graph | ~18                                               |
| Avg. edges per graph | ~40                                               |
| Node features        | Categorical atom labels (interpreted as features) |
| Number of classes    | 2                                                 |

Each graph represents a molecule:
- **Nodes** – atoms
- **Edges** – chemical bonds
- **Graph label** – indicates whether the molecule is mutagenic to *Salmonella typhimurium*.

### Data Split

The dataset is split **70/30** with stratification by class:

- `data/train.csv` – labeled training graphs (70%)
- `data/test.csv`  – unlabeled test graphs (30%)

Training labels are provided in `data/train.csv` with columns `graph_index` and `label`.  
Test graphs are listed in `data/test.csv` with only `graph_index` (labels are hidden for scoring).

---

## ⚙️ Perturbation Mechanism

Two types of feature corruption are applied to the test graphs to generate the **perturbed** setting:

1. **Distribution Shift**  
   A constant offset is added to each node feature:  
   `x ← x + δ`  
   where `δ = feature_shift` (default `0.3`).  
   *Simulates systematic measurement bias or domain shift.*

2. **Gaussian Noise Injection**  
   Random noise is added to each feature:  
   `x ← x + ϵ,  ϵ ~ N(0, σ²)`  
   where `σ = noise_std` (default `0.05`).  
   *Simulates noisy feature extraction.*

**Purpose**:  
This setup evaluates whether a GNN:
- relies on exact feature values,
- generalizes under feature distribution shift,
- remains stable under noisy topological descriptors.

The model is trained on clean features and evaluated on corrupted features to measure robustness.

---

## 📊 Evaluation Metrics

Performance is measured using the **F1 score** (macro‑averaged) because it balances precision and recall, providing a more reliable measure than accuracy alone.

Each submission is evaluated under two conditions:

- **F1 Score (Ideal)** – performance on clean topological features.
- **F1 Score (Perturbed)** – performance on corrupted features.

To quantify robustness, we compute the **Robustness Gap**:

`Robustness Gap = F1_ideal − F1_perturbed`

A smaller gap indicates a more stable and reliable model.

### 🏁 Ranking Priority

1. **Highest Perturbed F1 Score** (primary)
2. **Lowest Robustness Gap** (secondary)
3. **Most recent submission** (tie‑breaker)

Only the **best perturbed score** per participant is kept in the leaderboard.


-----------------------
## 🚀 Getting Started
-----------------------
 
### Environment Setup

Create a Python virtual environment and install dependencies:

```
Starter Code
A baseline GIN model is provided in starter_code/baseline.py. You can modify it or build your own model. 
```
The script Loads the MUTAG dataset:

```
Reads train.csv and test.csv.

Trains on clean graphs.

Generates predictions for both ideal and perturbed test graphs.

Saves submission files in the required format.
```
---------------------
To run the baseline:
---------------------
```
cd starter_code
starter_code/python baseline.py
```

This will create ideal_submission.csv and perturbed_submission.csv in the submissions/ folder (which is git‑ignored).

-------------------------
📤 Submission Procedure
-------------------------

Submissions must be encrypted and placed inside a folder named after your team.

**Step 1:** Prepare your submission files

Your CSV files must have the following format (example for 38 test graphs):

```
graph_index,label
160,1
62,0
48,0
173,1
.....
.....

ideal_submission.csv- predictions on clean test graphs.

perturbed_submission.csv – predictions on perturbed test graphs.
```

**Step 2:** Encrypt your files

From the project root, run the encryption script:

```
cd submissions
submissions/python encrypt_submissions.py
cd ..
```
This script will:

Look for ideal_submission.csv and perturbed_submission.csv in 'submissions' folder.

1.   Encrypt them using the public key (submissions/python encrypt_submissions.py).

2.   It will Produce ideal.enc and perturbed.enc in the same folder (submissions/<Team_Name>/*.enc).

Only the .enc files should be committed; the raw .csv files remain local (they are git‑ignored).
Note: Please dont forget to create submisions/<Team_Name>/  (two *enc files will be placed here)

**Step 3:** Commit and push

1.   Fork the repository.

2.   Create a folder submissions/<YourTeamName>/ and place the .enc files inside.
```
project_root/
├── encryption/
│   ├── __init__.py
│   └── encrypt.py           # contains encrypt_file function
├── submissions/
│   ├── ideal_submission.csv
│   ├── perturbed_submission.csv
│   └── team_name/           # <-- folder to be created
│       ├── ideal.enc        # For your geenrated Ideal_submission.csv.enc change name to ideal.enc (mandatory) see, submissions/Idrees_Bhat/
│       └── perturbed.enc    # or your geenrated perturbed_submission.csv.enc change name to perturbed.enc (mandatory) see, submissions/Idrees_Bhat/
└── encrypt_submissions.py   # script that encrypts files
```

3.   Create a new branch, commit only the .enc files, and open a Pull Request (PR) against the main branch.


**Important:**

Do not commit any raw .csv files.

Ensure your team folder name does not contain spaces.

**Step 4:** Automatic evaluation

1.   Once the PR is opened, the automated workflow will:

2.   Decrypt your files (using the organiser’s private key, stored as a secret).

3.   Compute F1 scores for both ideal and perturbed submissions.

4.   Calculate the robustness gap.

5.   Update the leaderboard (only your best perturbed score is retained).

6.   The live leaderboard will reflect the new results within minutes.

--------------------------
📁 Repository Structure
--------------------------
```
gnn-topology-ablation/
│
├── .github/                       # GitHub Actions workflows
│   ├── scripts/                   # Helper scripts for evaluation
│   └── workflows/                 # CI/CD pipeline definition
│
├── data/                          # Dataset files
│   └── MUTAG/
│       ├── test.csv               # Test graph indices
│       └── train.csv              # Training labels
│
├── docs/                          # Live leaderboard website
│   ├── index.html
│   ├── leaderboard.css
│   ├── leaderboard.csv            # Auto‑generated ranking
│   ├── leaderboard.js
│   └── readme
│
├── encryption/                    # Encryption/decryption utilities
│   ├── __init__.py
│   ├── decrypt.py
│   ├── encrypt.py
│   ├── generate_keys.py
│   └── public_key.pem             # Public key for participants
│
├── leaderboard/                   # Scoring and ranking logic
│   ├── __init__.py
│   ├── calculate_scores.py
│   ├── hidden_labels_reader.py
│   ├── render_leaderboard.py
│   ├── score_submission.py
│   └── update_leaderboard.py
│
├── starter_code/                   # Participant starter kit
│   ├── baseline.py
│   └── requirements.txt
│
├── submissions/                    # Encrypted submissions (git‑tracked)
│   └──<Team_Name>
│   └── encrypt_submissions.py      # encrypts the *.CSVs
│
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── leaderboard.md
└── requirements.txt
└── scoring_scripts.py
└── utils.py
```

-----------------------
🔒 Security Guarantee
-----------------------

1.   Predictions are encrypted locally using a symmetric key, which is then encrypted with the organiser’s RSA public key.

2.   Only the organiser (with the corresponding private key stored as a GitHub secret) can decrypt the submissions.

3.   Encrypted files are visible in the repository but completely unreadable without the private key.

4.   This ensures blind evaluation – participants cannot see each other’s predictions, and the organiser cannot see them until after the submission
     deadline (if desired).
     
-------------
📜 License
------------
This project is released under the MIT License. See the LICENSE file for details.


=== SECTION 2: REPO TREE ===
./data/train.csv
./data/MUTAG/processed/pre_transform.pt
./data/MUTAG/processed/data.pt
./data/MUTAG/processed/pre_filter.pt
./data/MUTAG/raw/MUTAG_edge_labels.txt
./data/MUTAG/raw/MUTAG_graph_indicator.txt
./data/MUTAG/raw/MUTAG_node_labels.txt
./data/MUTAG/raw/README.txt
./data/MUTAG/raw/MUTAG_A.txt
./data/MUTAG/raw/MUTAG_graph_labels.txt
./data/test.csv
./LICENSE
./docs/leaderboard.csv
./docs/leaderboard.css
./docs/leaderboard.js
./docs/index.html
./docs/readme
./.env.example
./utils.py
./requirements.txt
./submissions/Sargam_Goyal/metadata.json
./submissions/Sargam_Goyal/perturbed.enc
./submissions/Sargam_Goyal/ideal.enc
./.gitignore
./leaderboard.md
./README.md
./scoring_script.py
./.github/scripts/process_submission.py
./.github/workflows/process_submission.yml
./leaderboard/update_leaderboard.py
./leaderboard/render_leaderboard.py
./leaderboard/score_submission.py
./leaderboard/hidden_labels_reader.py
./leaderboard/calculate_scores.py
./leaderboard/__init__.py
./encryption/encrypt.py
./encryption/generate_keys.py
./encryption/decrypt.py
./encryption/__init__.py
./encryption/public_key.pem
./starter_code/requirements.txt
./starter_code/baseline.py

=== SECTION 3: DATA SAMPLE ===
`train.csv`

Contains information about training nodes, it has this format:

| Column        | Dtype | Description     |
| ------------- | ----- | --------------- |
| `graph_index` | int64 | The graph id    |
| `label`       | int64 | The graph label |

**Total_rows**: 131 

`test.csv`

Contain information about testing nodes, it has this format:

| Column        | Dtype | Description  |
| ------------- | ----- | ------------ |
| `graph_index` | int64 | The graph id |

**Total_rows**: 38

`MUTAG` is a folder containing features of nodes, it is available in the folder `data` or can be downloaded using the code:

```python
dataset = TUDataset(root=DATA_DIR, name="MUTAG")
```

Note that the graph from MUTAG that are concerned by the competition are those with ids available in the `graph_index` column in both `train.csv` and `test.csv`.

Also predictions should be made on both testing graph without modification and testing graphs after perturbation, the following function should be used for perturbation:

```python
def perturb_graph(data, feature_shift=0.3, noise_std=0.05):
    """
    Apply distribution shift and Gaussian noise to node features.

    Args:
        data (torch_geometric.data.Data): Original graph data.
        feature_shift (float): Constant offset added to each feature (default 0.3).
        noise_std (float): Standard deviation of Gaussian noise (default 0.05).

    Returns:
        torch_geometric.data.Data: Perturbed graph data.
    """
    data = data.clone()
    if data.x is not None:
        # 1️⃣ Distribution shift
        shift = torch.full_like(data.x, feature_shift)
        data.x = data.x + shift
        # 2️⃣ Gaussian noise
        noise = torch.randn_like(data.x) * noise_std
        data.x = data.x + noise
    return data
```



=== SECTION 4: SUBMISSION FORMAT ===
After training the model on training data, use the trained model to make predictions on testing data
--
Create `submission.csv` (predictions on clean test graphs) and `perturbed_submission.csv` (predictions on perturbed test graphs) both should be in the following format and save them it in the competition main directory :

```csv
graph_index,label
160,1
62,0
48,0
173,1
...


=== SECTION 5: REQUIREMENTS ===
torch
torch-geometric
numpy
pandas
scikit-learn