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
# 🎗️ The THInC Challenge: Tumor Histopathology Inductive Classification

## Welcome to the Tumor Diagnosis Challenge 🚀

This competition bridges **Biomedical Engineering** and **Graph Machine Learning**. Your task is to build a model that can diagnose the type of a cell (Tumor, Stroma, Immune, etc.) based on its features and its spatial neighbors in a tissue biopsy. Standard AI often fails in clinics because it memorizes training data, so our challenge forces models to predict cell types inductively, meaning they must generalize to completely unseen patients.

---

## 🎯 The Task: Inductive Node Classification

You are provided with **cell graphs** constructed from H&E stained histology images.

- **Training Phase:** You receive full graphs (nodes, edges, and labels) from a set of patients (e.g., Patient A, Patient B).
- **Testing Phase:** You must predict the cell types for **completely unseen patients** (e.g., Patient C).

### 🔍 Why "Inductive"?

Unlike standard transductive tasks (like Cora), the test nodes belong to **entirely new graphs** (new tissue slides) that were not present during training. Your model must learn **general rules** about tissue organization, not just memorize a specific graph structure.

---

## 📂 The Dataset (NuCLS-Based)

The data is derived from the [NuCLS dataset](https://nucls.grand-challenge.org/) (breast cancer).

<img width="1799" height="692" alt="1800w" src="https://github.com/user-attachments/assets/68b847f7-af4c-4383-9e8b-a38d244c7a03" />



### 🏗️ Graph Construction Pipeline

The graph was built using the following inductive pipeline to ensure biological realism:
```
[ Histology Image ]  -->  [ Cell Detection ]  -->  [ Graph Construction ]
       🖼️                       📍                        🕸️
   (Raw Pixels)            (Centroids x,y)           (Nodes + Edges)
                                                            |
                                                    (k-NN Neighbors)
```

| Component                    | Description                                                                                                                                                                                                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Node Definition**          | Raw bounding boxes from NuCLS were converted into centroids `(x, y)`. Each node represents a single cell nucleus.                                                                                                                                                         |
| **Edge Construction (k-NN)** | For every cell, we computed its **5 nearest spatial neighbors** within the same tissue image. Edges represent the local tissue microenvironment (e.g., cell–cell interactions). Edges strictly connect cells within the same image — no edges between different patients. |
| **Inductive Split**          | Dataset was split by **Image ID**, not by random cells. Training graph = 80% of tissue images. Test graph = remaining 20% (completely unseen patients).                                                                                                                   |
| **Feature Normalization**    | Node features `(x, y, width, height)` are standardized (zero mean, unit variance) for stable GNN training.                                                                                                                                                                |

---

## 1. Graph Components

- **Nodes:** Individual cells (nuclei)
- **Edges:** Spatial proximity via k-Nearest Neighbors (`k=5`). Physically close cells are connected.

### Node Features (X)

| Feature           | Description                           |
| ----------------- | ------------------------------------- |
| `x`, `y`          | Normalized spatial coordinates        |
| `width`, `height` | Morphological features of the nucleus |

### Labels (Y)

| Label | Cell Type  | Description       |
| ----- | ---------- | ----------------- |
| `0`   | Tumor      | Malignant cells   |
| `1`   | Stromal    | Connective tissue |
| `2`   | Lymphocyte | Immune cells      |
| `3`   | Macrophage | Immune cells      |

---

## 2. File Structure (`data/public/`)

| File                    | Description                   | Columns                          |
| ----------------------- | ----------------------------- | -------------------------------- |
| `train.csv`             | Training nodes with labels    | `id, x, y, width, height, label` |
| `edge_list.csv`         | Edges for the training graph  | `source, target`                 |
| `test_nodes.csv`        | Unseen test nodes (no labels) | `id, x, y, width, height`        |
| `test_edges.csv`        | Edges for the test graph      | `source, target`                 |
| `sample_submission.csv` | Example submission format     | —                                |

---

## 📝 Submission Format

Create a single CSV file named strictly using your team name: `<team_name>.csv` (e.g., `emmanuel_owusu.csv`).

```csv
id,y_pred
41269,0
41270,2
...
```

**Requirements:**
- Header must be exactly: `id,y_pred`
- One row per test node
- Labels must be integers in `[0–3]`

---

## 🚀 How to Participate

### 1️⃣ Clone the Repository & Install Dependencies
```bash
git clone https://github.com/emmakowu3579-ui/inductive-class-challenge.git
cd inductive-class-challenge
pip install -r starter_code/requirements.txt
```

### 2️⃣ Run the Baseline Model

A simple PyTorch GCN baseline is provided in the `starter_code/` directory.
```bash
python starter_code/baseline.py
```

This will:
- Train a basic GCN on the training graph
- Generate a submission file at `submissions/baseline_submission.csv`


### 3️⃣ Encrypt Your Submission

To preserve privacy, you **must encrypt** your CSV file before uploading. Do not upload raw CSV files.

```bash
# Usage: python starter_code/encrypt.py <path_to_your_team_csv>
python starter_code/encrypt.py submissions/<team_name>.csv

# Example:
# python starter_code/encrypt.py submissions/emmanuel_owusu.csv
```



### 4️⃣ Submit via GitHub

> ⚠️ **IMPORTANT:** Do **NOT** commit the raw `.csv` file. Ensure you are only committing the `.enc` file with your team name.

```bash
git add submissions/<team_name>.csv.enc
git commit -m "Add encrypted submission for <team_name>"
git push origin <your-branch-name>

# Example:
# git add submissions/emmanuel_owusu.csv.enc
# git commit -m "Add encrypted submission for emmanuel_owusu"
# git push origin main
```
Then open a **Pull Request** against the `main` branch on GitHub.

---

## 🤖 Instant Grading

Once your PR is opened:

- ✅ An **Auto-Grader Bot** runs automatically (it decrypts your file securely)
- 📊 Your **Macro F1-Score** is computed
- 💬 The score is posted as a **comment on your PR**

If the submission is valid, the PR will be merged by an admin and 🎉 your name appears on the **Leaderboard**.

---

## 📏 Rules & Restrictions

| Rule                   | Detail                                                                                             |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| **Evaluation Metric**  | Macro F1-Score                                                                                     |
| **Inductive Setting**  | No access to test labels during training. No memorization of node IDs or embeddings.               |
| **Message Passing**    | Allowed only on the training graph during training. Test edges may be used only at inference time. |
| **External Data**      | ❌ Strictly forbidden                                                                               |
| **Runtime Constraint** | Training must finish in **< 5 minutes** on Google Colab (CPU/GPU)                                  |
| **Libraries**          | Any standard GNN library (PyTorch, PyTorch Geometric, DGL, etc.)                                   |

---

## 🏆 Leaderboard
Normal table leaderboard: [📈 View Leaderboard](LEADERBOARD.md)

Interactive web leaderboard that updates automatically whenever a new submission is graded!

[🌐 View the Interactive THInC Leaderboard Here](https://emmakowu3579-ui.github.io/inductive-class-challenge/)


=== SECTION 2: REPO TREE ===
./data/README.md
./data/public/edge_list.csv
./data/public/sample_submission.csv
./data/public/train.csv
./data/public/test_nodes.csv
./data/public/test_edges.csv
./submissions/LLM_Claude_Opus_47.csv.enc
./submissions/Ignatius.csv.enc
./submissions/gururgg.csv.enc
./submissions/sample_submission.csv
./submissions/llm_gemini-3-flash.csv.enc
./submissions/Peguy_v2_code.py
./submissions/predictions.csv.enc
./submissions/Llm_Codex.csv.enc
./submissions/VinitSingroha.enc
./submissions/RPYpredictions.csv.enc
./submissions/Tasneem.csv.enc
./submissions/hadil.csv.enc
./submissions/el_Ikram.csv.enc
./submissions/Peguy_v2.csv.enc
./submissions/Bijay.csv.enc
./submissions/murad.csv.enc
./submissions/faranbutt.csv.enc
./submissions/Mubaraq.csv.enc
./submissions/README.md
./submissions/MahaTeam.csv.enc
./submissions/.DS_Store
./submissions/predictionns.csv.enc
./.gitignore
./competition/update_leaderboard.py
./competition/render_leaderboard.py
./competition/validate_submission.py
./competition/evaluate.py
./competition/decrypt.py
./index.html
./README.md
./.github/workflows/update_leaderboard.yml
./.github/workflows/scoring.yml
./.github/workflows/admin_refresh.yml
./leaderboard/leaderboard.csv
./.DS_Store
./LEADERBOARD.md
./starter_code/baseline_gcn.py
./starter_code/encrypt.py
./starter_code/README.md
./starter_code/public_key.pem

=== SECTION 3: DATA SAMPLE ===
`train.csv`

Contains information about training nodes, it has this format:

| Column   | Dtype   | Description         |
| -------- | ------- | ------------------- |
| `id`     | int64   | The node id         |
| `x`      | float64 | The first features  |
| `y`      | float64 | The second features |
| `width`  | float64 | The third features  |
| `height` | float64 | The fourth features |
| `label`  | int64   | The node label      |

**Total_rows**: 41266

`test_nodes.csv`

Contains information about testing nodes, it has this format:

| Column   | Dtype   | Description         |
| -------- | ------- | ------------------- |
| `id`     | int64   | The node id         |
| `x`      | float64 | The first features  |
| `y`      | float64 | The second features |
| `width`  | float64 | The third features  |
| `height` | float64 | The fourth features |

**Total_rows**: 10471

`edge_list.csv`

Contains information about training edges, it has this format:

| Column   | Dtype | Description        |
| -------- | ----- | ------------------ |
| `source` | int64 | The source node id |
| `target` | int64 | The target node id |

**Total_rows**: 206238 

`test_edges.csv`

Contains information about training edges, it has this format:

| Column   | Dtype | Description        |
| -------- | ----- | ------------------ |
| `source` | int64 | The source node id |
| `target` | int64 | The target node id |

**Total_rows**: 52332

=== SECTION 4: SUBMISSION FORMAT ===
After training the model on training data, use the trained model to make predictions on testing data
--
Create a `predictions.csv` in the following format and save it in the competition main directory :

```csv
id,y_pred
47532,0
47533,0
47534,0
47535,0
47536,0
...


=== SECTION 5: REQUIREMENTS ===
