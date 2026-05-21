#  Mix2Print: Learning Material Interaction Physics for identifying parameters of 3D Bioprinting


A challenge for predicting 3D bioprinting parameters using Graph Neural Networks. 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Leaderboard: https://vinitsingroha.github.io/Mix2Print/leaderboard.html

---

## 🧪 What is Bioprinting?
Bioprinting is an additive manufacturing process that functions similarly to 3D printing but uses **"bio-inks"**—materials combined with living cells. Instead of printing plastic or metal, we print tissue-like structures layer-by-layer. This technology is at the forefront of regenerative medicine, aiming to create functional organs, skin grafts, and disease models for drug testing without animal subjects.

The most common method is **Extrusion-based Bioprinting**, where a syringe-like printhead pushes bio-ink through a needle. Success depends on the perfect balance between material viscosity, cell viability, and the mechanical parameters of the printer.

![Bioprinting Flow](assets/overall%20flow.png)

## 🍳 Think of Bioprinting Like Cooking (Seriously)
If you’ve ever cooked a complex dish, you already understand the core problem in bioprinting. 

You start with **ingredients** (biomaterials like Gelatin, Alginate, or Fibrinogen) in specific proportions. You choose **how to cook**: the heat level, the pressure applied to the "piping bag," and the speed of your hand. If you get it right, the structure holds its shape. If you don't, it’s a mess—either too runny, too stiff, or the "cells" (the biological garnish) simply don't survive.

Currently, these "recipes" are scattered across thousands of research papers. This challenge is about learning the **recipe logic** behind bioprinting using the power of Graph Machine Learning.

---

## 📋 Challenge Overview


### Task
Predict **three continuous targets** from bioink formulation graphs:

- **Pressure** (kPa): Extrusion force
- **Temperature** (°C): Printing temperature  
- **Speed** (mm/s): Print head velocity

## 📐 Graph Specification

### Graph Definition
Each formulation is a graph $G_i = (V_i, E_i, X_i)$ where:
- $V_i$: Biomaterials in formulation $i$
- $E_i$: Fully connected edges between all materials
- $X_i \in R^{n_i \times D}$: Node feature matrix (Dimension $D \approx 31$)

Target $y_i \in R^3$: (pressure, temperature, speed)

![Graph Data Structure](assets/graph%20data%20structure.png)

### 1️⃣ Adjacency Matrix (Mandatory)
For formulation $i$ with $n$ materials:
$A_i \in R^{n_i \times n_i}$

- **Binary connectivity**: $A_{ij} = 1$ for all $i, j$ (Fully connected clique).
- **Topology**: Represents a mixture where all components potentially interact.
- **Note**: While the provided $A$ is binary, participants are encouraged to explore weighted adjacency strategies (e.g., based on concentration differences) closer to the physical reality of mixture interactions.

Files: `data/public/train_graphs/graph_{id}_A.npy`

### 2️⃣ Node Feature Matrix X
Each node corresponds to one biomaterial in the formulation.
$X_i$ shape: $(n_i \times D)$ where $D = N_{materials} + 1$.

| Feature | Description | Dim |
|---------|-------------|-----|
| Material Identity | One-Hot Encoding of material type | ~30 |
| Concentration | Normalized concentration in formulation | 1 |

Files: `data/public/train_graphs/graph_{id}_X.npy`

### 3️⃣ Targets
Graph-level regression targets:
- **Pressure** (kPa)
- **Temperature** (°C)
- **Speed** (mm/s)

Files: `data/public/train_graphs/graph_{id}_y.npy` (Train only)

### 📂 Dataset Provided
The processed graph dataset (`.npy` matrices) is already generated and available in:
- `data/public/train_graphs/`
- `data/public/test_graphs/`

For transparency, the generation script is included as `scripts/build_graph.py`.

### Dataset
- **423 formulations** from peer-reviewed publications
- **30 biomaterials** (appearing ≥5 times each)
- **303 training** / **120 test** samples (70/30 stratified group split)
- Real-world scientific data with natural complexity

### Evaluation Metric

NMAE = (1/3) × [MAE_pressure/1496 + MAE_temperature/228 + MAE_speed/90]

Lower is better. Range: 0.0 (perfect) to 1.0+ (poor).

### Baseline Performance
- **Random Forest:** NMAE = 0.060

---

## 🚀 Quick Start

### 1. Get the Data

```bash
git clone <this-repo>
cd bioink-gnn-challenge
pip install -r requirements.txt
```

Graph data (ready to use) is in `data/public/`:
- `train_graphs/` — `.npy` files: `graph_{id}_A.npy`, `graph_{id}_X.npy`, `graph_{id}_y.npy`
- `test_graphs/` — `.npy` files: `graph_{id}_A.npy`, `graph_{id}_X.npy`
- `node_vocabulary.txt` — Material index mapping
- `train.csv` — Original CSV (for reference)
- `test_nodes.csv` — Test IDs
- `sample_submission.csv` — Example submission format

### 2. Train Your Model

Train on `train.csv`. Since there is no official validation set, you should create your own split (e.g., 80/20) from the training data to evaluate your model locally.

### 3. Generate Predictions

Create `predictions.csv` for test set:

```csv
id,pressure,temperature,speed
340,150.5,25.0,5.0
341,800.0,155.0,1.2
...
399,45.0,23.0,8.5
```

### 4. Submit (Secure)

Since PRs are public, **you must encrypt your submission** to keep your predictions private.

1.  **Encrypt your CSV**:
    ```bash
    python scripts/encrypt_submission.py predictions.csv --team YourTeamName
    # Output: submission.enc (This file is safe to share)
    ```

2.  **Upload to GitHub**:
    Create a folder structure with your encrypted file:
    ```
    submissions/inbox/<YourTeamName>/
    └── submission.enc
    ```

3.  **Open Pull Request**:
    Target the `master` branch. The bot will decrypt it securely, score it, and close the PR.

**Submission Policy (Strict)**
- 🚨 **One Submission Only**: Each participant (GitHub user) is allowed exactly ONE submission.
- **Privacy**: Your `submission.enc` is decrypted only by the scoring server. The plaintext CSV is never stored in the repo.
- **Format**: Submit only `submission.enc`. Do NOT upload `predictions.csv`.

---

## 📊 Leaderboard

View the leaderboard:
- **Static:** [leaderboard/leaderboard.md](leaderboard/leaderboard.md)
- **Interactive:** Enable GitHub Pages → `/docs/leaderboard.html`

Rankings are by **NMAE (ascending)** - lower is better.

---

## 🔬 Data Details

### Bioink Components

30 common biomaterials across categories:
- **Alginates:** Alginate, Alginate Methacrylated, Alginate Dialdehyde
- **Gelatins:** Gelatin, Gelatin Methacrylated (GelMA)
- **Polymers:** PCL, PLGA, PEG derivatives
- **Natural:** Collagen, Chitosan, Hyaluronic Acid
- **Ceramics:** Hydroxyapatite, β-TCP, Bioactive Glass

### Target Distributions

| Target | Min | Max | Distribution |
|--------|-----|-----|--------------|
| Pressure | 4 kPa | 1500 kPa | Log-distributed, bimodal |
| Temperature | 2°C | 230°C | Bimodal (room temp vs melt) |
| Speed | 0.02 mm/s | 90 mm/s | Many near-zero values |

### Data Preprocessing

- **Ranges converted to means:** "70-80 kPa" → 75.0 kPa
- **Unit standardization:** All pressure in kPa, temp in °C, speed in mm/s
- **Stratified split:** By temperature regime (hydrogel vs thermoplastic)

---

## 🏗️ Repository Structure

```
bioink-gnn-challenge/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Excludes private data
│
├── data/
│   ├── public/                  # Visible to participants
│   │   ├── train.csv
│   │   ├── test_features.csv
│   │   ├── test_nodes.csv
│   │   ├── train_graphs/        # A, X, y matrices (npy)
│   │   ├── test_graphs/         # A, X matrices (npy)
│   │   └── node_vocabulary.txt  # Material list
│
├── scripts/
│   └── build_graph.py          # Script used to generate graphs
│
├── competition/                 # Evaluation code
│   ├── data_utils.py           # Parsing & preprocessing
│   ├── metrics.py              # NMAE calculation
│   ├── validation.py           # Format checking
│   ├── evaluate.py             # Scoring script
│   └── render_leaderboard.py   # Generate markdown
│
├── baselines/                   # Reference implementations
│   ├── README.md
│   ├── gnn_utils.py            # Graph data loader (npy → PyG)
│   ├── mlp_baseline.py         # MLP (ignores graph structure)
│   ├── gcn_baseline.py         # Graph Convolutional Network
│   ├── gat_baseline.py         # Graph Attention Network
│   └── random_forest_baseline.py # Tabular baseline
│
├── submissions/
│   └── inbox/                   # PR submissions go here
│
├── leaderboard/
│   ├── leaderboard.csv         # Authoritative scores
│   └── leaderboard.md          # Auto-generated table
│
├── docs/                        # GitHub Pages
│   ├── leaderboard.html
│   ├── leaderboard.css
│   └── leaderboard.js
│
└── .github/workflows/
    ├── score_submission.yml     # Auto-score PRs
    └── update_leaderboard.yml   # Update on merge
```
---


## 📖 Dataset link

The raw dataset for the data used in this challenge can be found at [https://cect.umd.edu/database]

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙋 Support

- **Issues:** Use GitHub Issues for bugs/questions
- **Discussions:** Use GitHub Discussions for general chat
- **Email:** [vineet10338@gmail.com] for private inquiries

---

Good luck! 🚀
