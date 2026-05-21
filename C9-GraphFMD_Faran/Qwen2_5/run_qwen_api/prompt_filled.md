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
# 🪙 GraphFMD: Graph based Financial Misconduct Detection 

**GraphFMD** is a temporal graph learning benchmark for financial misconduct detection in the Bitcoin transaction network.  
Participants must classify transactions as illicit (fraudulent) or licit (legitimate).

This repository is designed for **Human vs. LLM** task.

---

<img src="images/outputs.png" width="800" />




## 🏆 Leaderboard

View the real-time rankings here: **[https://faranbutt.github.io/GraphFMD/](https://faranbutt.github.io/GraphFMD/)**


## 🚀 How to Participate

To ensure the secrecy of the test labels and participant data, we use a **Secure Submission Portal**.

### Step 1: Prepare your Files
You must prepare two files:

1. **`predictions.csv`**: Must contain exactly two columns: `id` and `y_pred`.
   - `1`: Illicit (Fraudulent)
   - `2`: Licit (Legal)
2. **`metadata.json`**: A short description of your approach.

```json
{
  "team": "Your_Team_Name",
  "run_id": "run_01/run_02.... etc",
  "author_type": "human / llm / hybrid",
  "model": "GCN / GraphSAGE / etc.",
  "notes": "Briefly describe your layers/hyperparameters"
}


```

### Step 2: Upload to the Submission Portal
Submit your files via the official Google Form:  
👉 **[Official Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSdyqFftCdnH3SAn6AFBVu3qUpG_MUXNSyRssUcw9RKqh9V-nw/viewform?usp=dialog)**

### Step 3: Automated Scoring
Once you submit the form:
* A **GitHub Action** is triggered automatically.
* Your model is scored against the **Hidden Ground Truth**.
* The **Leaderboard** is updated instantly.


## 1. Task Overview

* **Task:** Temporal Inductive Node Classification (Licit vs. Illicit).
* **Domain:** Cryptocurrency (Bitcoin) Forensics.
* **Target:** Predict the class label of each transaction (Illicit = 1, Licit = 2).
* **Metric:** **Macro-F1 across both classes (Illicit and Licit)**.  

## 2. The Data
* **Nodes (Node Feature Matrix (X)):** Bitcoin transactions.165 local and aggregate features. (Train =  16658 , Test = 8896)
* **Edges (adjacency matrix (A)) :** The flow of BTC between transactions.


## 3. Difficulty level:
- **Feature Noise** Gaussian noise was added to make the features simulate real world noisy data. 
- **Temporal Shifting:** Time-based split (Train: 1–34, Test: 35+)
-  **Class Imbalance & Graph Sparsity:** All illicit transactions are preserved while only 50% of licit transactions are retained (unknown nodes removed)



## 4. Submission Policy:
For maintaining fairness and competition competency
- One submission policy is enforced so you are only allowed to do one form submission


## 6. Submission Format

To enter the competition, you must submit a CSV file named exactly prediction.csv inside the submissions/ folder.

```csv
submissions/participant1/prediction.csv
id,y_pred
6418,1
7952,2
.....
.....
```
id: Transaction ID (must match test_nodes.csv).

y_pred: The predicted class label:

 1: Illicit (Fraudulent)
 2: Licit (Legal)



## 7. Automated Validation Checks:
When a Pull Request is opened the bot will
- Check identity (Verify if you have already submitted)
- Check Formats (Ensure your JSON and CSV files are structured properly)



## 8. Repository Structure

```text
.
├── data/
│   ├── public/            
│   │   ├── train_nodes.csv
│   │   ├── train_labels.csv
│   │   ├── test_nodes.csv
│   │   └── edgelist.csv
├── competition/
│   ├── baseline.py         # Starter GCN model
│   ├── evaluate.py         # Scoring logic
│   ├── metrics.py          # F1-Score calculation
│   └── update_leaderboard.py
├── submissions/            # Submission directory
│   └── participant1
│   │  └── predictions.csv
├── leaderboard/            # CSV/Markdown rankings
└── docs/                   # Interactive Leaderboard
└── images/                   
```

## 📝 Citation

If you use this challenge, dataset, or repository in your research, please cite:

```bibtex
@dataset{graphfmd_2026,
  title={GraphFMD: Graph-based Financial Misconduct Detection Benchmark},
  author={Faran Taimoor Butt},
  year={2026},
  url = {https://github.com/faranbutt/GraphFMD}
}
```

## Organizer
**Faran Taimoor Butt** Software Engineer and Researcher in Computer Vision, NLP & Graph ML.
* **Email:** [faranbutt789@gmail.com](mailto:faranbutt789@gmail.com)
* **GitHub:** [@faranbutt](https://github.com/faranbutt)

For questions regarding the competition setup, data preprocessing or automated scoring issues, please open an **Issue** in this repository or contact me directly.

## 📚 References

### Learning Resources
* **[Basira Lab]** [Deep Graph Learning Playlist](https://www.youtube.com/watch?v=gQRV_jUyaDw&list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T) – Essential video tutorials for GNN fundamentals.
* **[Basira Lab]** [Deep Graph Learning GitHub](https://github.com/basiralab/DGL) – Codebase and implementations for graph-based models.
---

### Datasets

* **[1]** Elliptic, [www.elliptic.co](http://www.elliptic.co).
* **[2]** M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, C. E. Leiserson, "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics", KDD ’19 Workshop on Anomaly Detection in Finance, August 2019, Anchorage, AK, USA.



=== SECTION 2: REPO TREE ===
./data/public/sample_submission.csv
./data/public/train_nodes.csv
./data/public/edgelist.csv
./data/public/test_nodes.csv
./data/public/train_labels.csv
./docs/leaderboard.html
./docs/leaderboard.css
./docs/leaderboard.js
./docs/index.html
./requirements.txt
./submissions/.gitkeep
./submissions/inbox/team_alpha/run_01/predictions.csv
./.gitignore
./images/outputs.png
./competition/update_leaderboard.py
./competition/render_leaderboard.py
./competition/baseline.py
./competition/validate_submission.py
./competition/evaluate.py
./competition/config.yaml
./competition/metrics.py
./README.md
./.github/workflows/score_submission.yml
./.github/workflows/publish_leaderboard.yml
./leaderboard/leaderboard.csv
./leaderboard/leaderboard.md

=== SECTION 3: DATA SAMPLE ===
`train_nodes.csv`

Contains information about training nodes, it has this format:

| Column       | Dtype   | Description       |
| ------------ | ------- | ----------------- |
| `id`         | int64   | The node id       |
| `timestep`   | int64   | The timestep      |
| `2` to `166` | float64 | The node features |

**Total_rows**: 16658 

`test_nodes.csv`

Contains information about testing nodes, it has this format:

| Column       | Dtype   | Description       |
| ------------ | ------- | ----------------- |
| `id`         | int64   | The node id       |
| `timestep`   | int64   | The timestep      |
| `2` to `166` | float64 | The node features |

**Total_rows**: 8896

`train_labels.csv`

Contains labels of the training nodes, it has this format:

| Column | Dtype | Description    |
| ------ | ----- | -------------- |
| `id`   | int64 | The node id    |
| `y`    | int64 | The node label |

**Total_rows**: 16658

`edgelist.csv`

Contains information about all edges, it has this format:

| Column  | Dtype | Description        |
| ------- | ----- | ------------------ |
| `txId1` | int64 | The source node id |
| `txId2` | int64 | The target node id |

**Total_rows**: 10197

=== SECTION 4: SUBMISSION FORMAT ===
After training the model on training data, use the trained model to make predictions on testing data
--
Create a `predictions.csv` in the following format and save it in the competition main directory :

```csv
id,y_pred
6418,1
7952,2
9363,1
9466,2
9551,1
10280,2
...


=== SECTION 5: REQUIREMENTS ===
pandas>=2.0.0
scikit-learn>=1.3.0
torch
torch-geometric



Important notes : 
- Labels are not zero-based (classes are 1 , 2 instead of 0 , 1 )
- Data paths: 'GraphFMD/data/public'.
- The final saved file should be submission.csv 
- predictions should be classes not probabilities.
- The model MUST be a Graph Neural Network (GNN) using torch_geometric.
- Always use float32 instead of float64
- Delete large objects immediately from memory after they are no longer needed.
- Make use of the baseline code if provided for you
- Identifier columns MUST be treated as integers, not floats.
- When creating the submission DataFrame, Explicitly cast 'id' → int
