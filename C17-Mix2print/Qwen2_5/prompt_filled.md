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

| Feature           | Description                             | Dim |
| ----------------- | --------------------------------------- | --- |
| Material Identity | One-Hot Encoding of material type       | ~30 |
| Concentration     | Normalized concentration in formulation | 1   |

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

| Target      | Min       | Max      | Distribution                |
| ----------- | --------- | -------- | --------------------------- |
| Pressure    | 4 kPa     | 1500 kPa | Log-distributed, bimodal    |
| Temperature | 2°C       | 230°C    | Bimodal (room temp vs melt) |
| Speed       | 0.02 mm/s | 90 mm/s  | Many near-zero values       |

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

=== SECTION 2: REPO TREE ===
./data/public/sample_submission.csv
./data/public/train.csv
./data/public/train_graphs/graph_410_X.npy
./data/public/train_graphs/graph_63_A.npy
./data/public/train_graphs/graph_32_A.npy
./data/public/train_graphs/graph_320_y.npy
./data/public/train_graphs/graph_112_X.npy
./data/public/train_graphs/graph_319_A.npy
./data/public/train_graphs/graph_293_X.npy
./data/public/train_graphs/graph_291_y.npy
./data/public/train_graphs/graph_312_X.npy
./data/public/train_graphs/graph_153_y.npy
./data/public/train_graphs/graph_234_A.npy
./data/public/train_graphs/graph_196_A.npy
./data/public/train_graphs/graph_332_A.npy
./data/public/train_graphs/graph_298_X.npy
./data/public/train_graphs/graph_101_y.npy
./data/public/train_graphs/graph_147_X.npy
./data/public/train_graphs/graph_312_y.npy
./data/public/train_graphs/graph_275_y.npy
./data/public/train_graphs/graph_222_X.npy
./data/public/train_graphs/graph_279_y.npy
./data/public/train_graphs/graph_367_y.npy
./data/public/train_graphs/graph_169_y.npy
./data/public/train_graphs/graph_221_y.npy
./data/public/train_graphs/graph_215_X.npy
./data/public/train_graphs/graph_115_A.npy
./data/public/train_graphs/graph_38_X.npy
./data/public/train_graphs/graph_66_A.npy
./data/public/train_graphs/graph_362_y.npy
./data/public/train_graphs/graph_294_y.npy
./data/public/train_graphs/graph_153_X.npy
./data/public/train_graphs/graph_400_y.npy
./data/public/train_graphs/graph_254_A.npy
./data/public/train_graphs/graph_226_A.npy
./data/public/train_graphs/graph_67_X.npy
./data/public/train_graphs/graph_25_y.npy
./data/public/train_graphs/graph_266_y.npy
./data/public/train_graphs/graph_131_X.npy
./data/public/train_graphs/graph_209_A.npy
./data/public/train_graphs/graph_374_y.npy
./data/public/train_graphs/graph_355_A.npy
./data/public/train_graphs/graph_22_y.npy
./data/public/train_graphs/graph_333_y.npy
./data/public/train_graphs/graph_297_y.npy
./data/public/train_graphs/graph_353_A.npy
./data/public/train_graphs/graph_323_X.npy
./data/public/train_graphs/graph_422_y.npy
./data/public/train_graphs/graph_2_A.npy
./data/public/train_graphs/graph_414_X.npy
./data/public/train_graphs/graph_2_y.npy
./data/public/train_graphs/graph_313_A.npy
./data/public/train_graphs/graph_218_y.npy
./data/public/train_graphs/graph_176_X.npy
./data/public/train_graphs/graph_313_X.npy
./data/public/train_graphs/graph_277_A.npy
./data/public/train_graphs/graph_407_A.npy
./data/public/train_graphs/graph_179_X.npy
./data/public/train_graphs/graph_7_A.npy
./data/public/train_graphs/graph_208_y.npy
./data/public/train_graphs/graph_79_y.npy
./data/public/train_graphs/graph_320_A.npy
./data/public/train_graphs/graph_334_y.npy
./data/public/train_graphs/graph_331_y.npy
./data/public/train_graphs/graph_30_y.npy
./data/public/train_graphs/graph_93_A.npy
./data/public/train_graphs/graph_128_y.npy
./data/public/train_graphs/graph_98_A.npy
./data/public/train_graphs/graph_339_y.npy
./data/public/train_graphs/graph_416_A.npy
./data/public/train_graphs/graph_46_y.npy
./data/public/train_graphs/graph_27_A.npy
./data/public/train_graphs/graph_148_X.npy
./data/public/train_graphs/graph_390_A.npy
./data/public/train_graphs/graph_414_y.npy
./data/public/train_graphs/graph_401_A.npy
./data/public/train_graphs/graph_316_A.npy
./data/public/train_graphs/graph_243_A.npy
./data/public/train_graphs/graph_267_y.npy
./data/public/train_graphs/graph_394_A.npy
./data/public/train_graphs/graph_210_A.npy
./data/public/train_graphs/graph_297_A.npy
./data/public/train_graphs/graph_175_X.npy
./data/public/train_graphs/graph_283_X.npy
./data/public/train_graphs/graph_49_A.npy
./data/public/train_graphs/graph_359_X.npy
./data/public/train_graphs/graph_46_X.npy
./data/public/train_graphs/graph_215_y.npy
./data/public/train_graphs/graph_140_X.npy
./data/public/train_graphs/graph_154_A.npy
./data/public/train_graphs/graph_28_A.npy
./data/public/train_graphs/graph_366_y.npy
./data/public/train_graphs/graph_242_y.npy
./data/public/train_graphs/graph_344_X.npy
./data/public/train_graphs/graph_404_y.npy
./data/public/train_graphs/graph_180_y.npy
./data/public/train_graphs/graph_64_X.npy
./data/public/train_graphs/graph_185_A.npy
./data/public/train_graphs/graph_409_A.npy
./data/public/train_graphs/graph_255_A.npy
./data/public/train_graphs/graph_59_X.npy
./data/public/train_graphs/graph_265_y.npy
./data/public/train_graphs/graph_217_y.npy
./data/public/train_graphs/graph_96_X.npy
./data/public/train_graphs/graph_208_A.npy
./data/public/train_graphs/graph_297_X.npy
./data/public/train_graphs/graph_115_y.npy
./data/public/train_graphs/graph_71_X.npy
./data/public/train_graphs/graph_211_X.npy
./data/public/train_graphs/graph_195_A.npy
./data/public/train_graphs/graph_345_A.npy
./data/public/train_graphs/graph_72_y.npy
./data/public/train_graphs/graph_351_y.npy
./data/public/train_graphs/graph_226_y.npy
./data/public/train_graphs/graph_1_y.npy
./data/public/train_graphs/graph_397_X.npy
./data/public/train_graphs/graph_257_y.npy
./data/public/train_graphs/graph_241_X.npy
./data/public/train_graphs/graph_336_y.npy
./data/public/train_graphs/graph_290_A.npy
./data/public/train_graphs/graph_415_y.npy
./data/public/train_graphs/graph_289_A.npy
./data/public/train_graphs/graph_48_A.npy
./data/public/train_graphs/graph_93_y.npy
./data/public/train_graphs/graph_28_X.npy
./data/public/train_graphs/graph_234_X.npy
./data/public/train_graphs/graph_278_y.npy
./data/public/train_graphs/graph_76_y.npy
./data/public/train_graphs/graph_408_y.npy
./data/public/train_graphs/graph_279_A.npy
./data/public/train_graphs/graph_406_A.npy
./data/public/train_graphs/graph_240_A.npy
./data/public/train_graphs/graph_303_X.npy
./data/public/train_graphs/graph_387_X.npy
./data/public/train_graphs/graph_264_X.npy
./data/public/train_graphs/graph_135_A.npy
./data/public/train_graphs/graph_68_y.npy
./data/public/train_graphs/graph_12_X.npy
./data/public/train_graphs/graph_18_X.npy
./data/public/train_graphs/graph_101_X.npy
./data/public/train_graphs/graph_413_A.npy
./data/public/train_graphs/graph_306_A.npy
./data/public/train_graphs/graph_57_A.npy
./data/public/train_graphs/graph_32_X.npy
./data/public/train_graphs/graph_226_X.npy
./data/public/train_graphs/graph_0_y.npy
./data/public/train_graphs/graph_396_y.npy
./data/public/train_graphs/graph_131_A.npy
./data/public/train_graphs/graph_267_X.npy
./data/public/train_graphs/graph_29_y.npy
./data/public/train_graphs/graph_291_X.npy
./data/public/train_graphs/graph_45_A.npy
./data/public/train_graphs/graph_344_A.npy
./data/public/train_graphs/graph_187_A.npy
./data/public/train_graphs/graph_91_y.npy
./data/public/train_graphs/graph_296_A.npy
./data/public/train_graphs/graph_317_A.npy
./data/public/train_graphs/graph_310_X.npy
./data/public/train_graphs/graph_49_y.npy
./data/public/train_graphs/graph_383_y.npy
./data/public/train_graphs/graph_360_y.npy
./data/public/train_graphs/graph_141_A.npy
./data/public/train_graphs/graph_148_y.npy
./data/public/train_graphs/graph_335_A.npy
./data/public/train_graphs/graph_166_y.npy
./data/public/train_graphs/graph_405_A.npy
./data/public/train_graphs/graph_61_y.npy
./data/public/train_graphs/graph_38_A.npy
./data/public/train_graphs/graph_202_X.npy
./data/public/train_graphs/graph_173_A.npy
./data/public/train_graphs/graph_74_y.npy
./data/public/train_graphs/graph_306_y.npy
./data/public/train_graphs/graph_275_X.npy
./data/public/train_graphs/graph_182_y.npy
./data/public/train_graphs/graph_66_y.npy
./data/public/train_graphs/graph_324_A.npy
./data/public/train_graphs/graph_64_y.npy
./data/public/train_graphs/graph_352_X.npy
./data/public/train_graphs/graph_320_X.npy
./data/public/train_graphs/graph_288_X.npy
./data/public/train_graphs/graph_93_X.npy
./data/public/train_graphs/graph_315_A.npy
./data/public/train_graphs/graph_374_A.npy
./data/public/train_graphs/graph_393_y.npy
./data/public/train_graphs/graph_149_A.npy
./data/public/train_graphs/graph_255_X.npy
./data/public/train_graphs/graph_187_X.npy
./data/public/train_graphs/graph_15_y.npy
./data/public/train_graphs/graph_356_y.npy
./data/public/train_graphs/graph_110_y.npy
./data/public/train_graphs/graph_151_X.npy
./data/public/train_graphs/graph_61_X.npy
./data/public/train_graphs/graph_187_y.npy
./data/public/train_graphs/graph_367_X.npy
./data/public/train_graphs/graph_332_y.npy
./data/public/train_graphs/graph_225_A.npy
./data/public/train_graphs/graph_61_A.npy
./data/public/train_graphs/graph_126_y.npy
./data/public/train_graphs/graph_402_X.npy
./data/public/train_graphs/graph_364_A.npy
./data/public/train_graphs/graph_260_X.npy
./data/public/train_graphs/graph_42_y.npy
./data/public/train_graphs/graph_155_A.npy
./data/public/train_graphs/graph_167_A.npy
./data/public/train_graphs/graph_154_y.npy
./data/public/train_graphs/graph_391_X.npy
./data/public/train_graphs/graph_96_y.npy
./data/public/train_graphs/graph_79_X.npy
./data/public/train_graphs/graph_244_y.npy
./data/public/train_graphs/graph_402_A.npy
./data/public/train_graphs/graph_127_A.npy
./data/public/train_graphs/graph_232_y.npy
./data/public/train_graphs/graph_70_A.npy
./data/public/train_graphs/graph_71_A.npy
./data/public/train_graphs/graph_358_y.npy
./data/public/train_graphs/graph_360_X.npy
./data/public/train_graphs/graph_352_A.npy
./data/public/train_graphs/graph_400_A.npy
./data/public/train_graphs/graph_44_A.npy
./data/public/train_graphs/graph_314_A.npy
./data/public/train_graphs/graph_406_X.npy
./data/public/train_graphs/graph_97_X.npy
./data/public/train_graphs/graph_245_X.npy
./data/public/train_graphs/graph_257_A.npy
./data/public/train_graphs/graph_340_y.npy
./data/public/train_graphs/graph_340_A.npy
./data/public/train_graphs/graph_220_A.npy
./data/public/train_graphs/graph_232_A.npy
./data/public/train_graphs/graph_294_A.npy
./data/public/train_graphs/graph_329_y.npy
./data/public/train_graphs/graph_361_X.npy
./data/public/train_graphs/graph_110_A.npy
./data/public/train_graphs/graph_394_y.npy
./data/public/train_graphs/graph_48_X.npy
./data/public/train_graphs/graph_339_X.npy
./data/public/train_graphs/graph_245_y.npy
./data/public/train_graphs/graph_67_A.npy
./data/public/train_graphs/graph_138_y.npy
./data/public/train_graphs/graph_92_X.npy
./data/public/train_graphs/graph_266_A.npy
./data/public/train_graphs/graph_326_X.npy
./data/public/train_graphs/graph_63_y.npy
./data/public/train_graphs/graph_43_y.npy
./data/public/train_graphs/graph_358_A.npy
./data/public/train_graphs/graph_16_y.npy
./data/public/train_graphs/graph_151_y.npy
./data/public/train_graphs/graph_408_X.npy
./data/public/train_graphs/graph_285_y.npy
./data/public/train_graphs/graph_95_A.npy
./data/public/train_graphs/graph_342_y.npy
./data/public/train_graphs/graph_304_y.npy
./data/public/train_graphs/graph_114_y.npy
./data/public/train_graphs/graph_43_A.npy
./data/public/train_graphs/graph_168_A.npy
./data/public/train_graphs/graph_397_A.npy
./data/public/train_graphs/graph_313_y.npy
./data/public/train_graphs/graph_305_X.npy
./data/public/train_graphs/graph_87_y.npy
./data/public/train_graphs/graph_17_y.npy
./data/public/train_graphs/graph_256_y.npy
./data/public/train_graphs/graph_225_y.npy
./data/public/train_graphs/graph_289_X.npy
./data/public/train_graphs/graph_42_A.npy
./data/public/train_graphs/graph_350_A.npy
./data/public/train_graphs/graph_90_y.npy
./data/public/train_graphs/graph_332_X.npy
./data/public/train_graphs/graph_213_X.npy
./data/public/train_graphs/graph_65_y.npy
./data/public/train_graphs/graph_13_y.npy
./data/public/train_graphs/graph_8_A.npy
./data/public/train_graphs/graph_403_y.npy
./data/public/train_graphs/graph_80_X.npy
./data/public/train_graphs/graph_173_X.npy
./data/public/train_graphs/graph_42_X.npy
./data/public/train_graphs/graph_31_A.npy
./data/public/train_graphs/graph_261_y.npy
./data/public/train_graphs/graph_262_A.npy
./data/public/train_graphs/graph_136_y.npy
./data/public/train_graphs/graph_401_y.npy
./data/public/train_graphs/graph_138_A.npy
./data/public/train_graphs/graph_50_A.npy
./data/public/train_graphs/graph_111_A.npy
./data/public/train_graphs/graph_174_X.npy
./data/public/train_graphs/graph_346_y.npy
./data/public/train_graphs/graph_175_y.npy
./data/public/train_graphs/graph_315_y.npy
./data/public/train_graphs/graph_222_y.npy
./data/public/train_graphs/graph_368_X.npy
./data/public/train_graphs/graph_221_A.npy
./data/public/train_graphs/graph_353_y.npy
./data/public/train_graphs/graph_60_A.npy
./data/public/train_graphs/graph_170_A.npy
./data/public/train_graphs/graph_262_X.npy
./data/public/train_graphs/graph_299_X.npy
./data/public/train_graphs/graph_334_X.npy
./data/public/train_graphs/graph_0_A.npy
./data/public/train_graphs/graph_37_y.npy
./data/public/train_graphs/graph_181_X.npy
./data/public/train_graphs/graph_13_A.npy
./data/public/train_graphs/graph_181_y.npy
./data/public/train_graphs/graph_214_A.npy
./data/public/train_graphs/graph_68_X.npy
./data/public/train_graphs/graph_99_y.npy
./data/public/train_graphs/graph_362_A.npy
./data/public/train_graphs/graph_327_y.npy
./data/public/train_graphs/graph_292_X.npy
./data/public/train_graphs/graph_24_A.npy
./data/public/train_graphs/graph_45_X.npy
./data/public/train_graphs/graph_154_X.npy
./data/public/train_graphs/graph_26_A.npy
./data/public/train_graphs/graph_17_A.npy
./data/public/train_graphs/graph_227_y.npy
./data/public/train_graphs/graph_389_y.npy
./data/public/train_graphs/graph_213_y.npy
./data/public/train_graphs/graph_398_y.npy
./data/public/train_graphs/graph_239_X.npy
./data/public/train_graphs/graph_303_A.npy
./data/public/train_graphs/graph_342_A.npy
./data/public/train_graphs/graph_285_X.npy
./data/public/train_graphs/graph_73_X.npy
./data/public/train_graphs/graph_298_A.npy
./data/public/train_graphs/graph_223_X.npy
./data/public/train_graphs/graph_394_X.npy
./data/public/train_graphs/graph_21_A.npy
./data/public/train_graphs/graph_227_X.npy
./data/public/train_graphs/graph_213_A.npy
./data/public/train_graphs/graph_348_y.npy
./data/public/train_graphs/graph_398_X.npy
./data/public/train_graphs/graph_354_X.npy
./data/public/train_graphs/graph_70_X.npy
./data/public/train_graphs/graph_38_y.npy
./data/public/train_graphs/graph_328_A.npy
./data/public/train_graphs/graph_10_X.npy
./data/public/train_graphs/graph_353_X.npy
./data/public/train_graphs/graph_346_X.npy
./data/public/train_graphs/graph_164_X.npy
./data/public/train_graphs/graph_383_A.npy
./data/public/train_graphs/graph_261_X.npy
./data/public/train_graphs/graph_188_A.npy
./data/public/train_graphs/graph_307_X.npy
./data/public/train_graphs/graph_227_A.npy
./data/public/train_graphs/graph_29_A.npy
./data/public/train_graphs/graph_409_y.npy
./data/public/train_graphs/graph_18_y.npy
./data/public/train_graphs/graph_152_X.npy
./data/public/train_graphs/graph_385_A.npy
./data/public/train_graphs/graph_3_X.npy
./data/public/train_graphs/graph_351_A.npy
./data/public/train_graphs/graph_290_y.npy
./data/public/train_graphs/graph_340_X.npy
./data/public/train_graphs/graph_27_X.npy
./data/public/train_graphs/graph_44_y.npy
./data/public/train_graphs/graph_24_X.npy
./data/public/train_graphs/graph_310_A.npy
./data/public/train_graphs/graph_386_X.npy
./data/public/train_graphs/graph_325_A.npy
./data/public/train_graphs/graph_233_y.npy
./data/public/train_graphs/graph_166_A.npy
./data/public/train_graphs/graph_243_X.npy
./data/public/train_graphs/graph_62_A.npy
./data/public/train_graphs/graph_281_X.npy
./data/public/train_graphs/graph_135_X.npy
./data/public/train_graphs/graph_90_A.npy
./data/public/train_graphs/graph_23_y.npy
./data/public/train_graphs/graph_402_y.npy
./data/public/train_graphs/graph_356_A.npy
./data/public/train_graphs/graph_176_y.npy
./data/public/train_graphs/graph_77_y.npy
./data/public/train_graphs/graph_357_y.npy
./data/public/train_graphs/graph_244_X.npy
./data/public/train_graphs/graph_281_y.npy
./data/public/train_graphs/graph_312_A.npy
./data/public/train_graphs/graph_265_X.npy
./data/public/train_graphs/graph_266_X.npy
./data/public/train_graphs/graph_97_y.npy
./data/public/train_graphs/graph_295_A.npy
./data/public/train_graphs/graph_308_y.npy
./data/public/train_graphs/graph_223_A.npy
./data/public/train_graphs/graph_96_A.npy
./data/public/train_graphs/graph_242_X.npy
./data/public/train_graphs/graph_26_y.npy
./data/public/train_graphs/graph_174_A.npy
./data/public/train_graphs/graph_309_y.npy
./data/public/train_graphs/graph_77_A.npy
./data/public/train_graphs/graph_356_X.npy
./data/public/train_graphs/graph_184_A.npy
./data/public/train_graphs/graph_211_A.npy
./data/public/train_graphs/graph_389_X.npy
./data/public/train_graphs/graph_134_A.npy
./data/public/train_graphs/graph_208_X.npy
./data/public/train_graphs/graph_314_X.npy
./data/public/train_graphs/graph_138_X.npy
./data/public/train_graphs/graph_105_y.npy
./data/public/train_graphs/graph_382_y.npy
./data/public/train_graphs/graph_260_y.npy
./data/public/train_graphs/graph_254_X.npy
./data/public/train_graphs/graph_263_A.npy
./data/public/train_graphs/graph_268_y.npy
./data/public/train_graphs/graph_357_A.npy
./data/public/train_graphs/graph_193_X.npy
./data/public/train_graphs/graph_411_y.npy
./data/public/train_graphs/graph_136_X.npy
./data/public/train_graphs/graph_101_A.npy
./data/public/train_graphs/graph_19_y.npy
./data/public/train_graphs/graph_163_X.npy
./data/public/train_graphs/graph_171_y.npy
./data/public/train_graphs/graph_167_X.npy
./data/public/train_graphs/graph_190_y.npy
./data/public/train_graphs/graph_296_y.npy
./data/public/train_graphs/graph_240_X.npy
./data/public/train_graphs/graph_404_X.npy
./data/public/train_graphs/graph_147_A.npy
./data/public/train_graphs/graph_330_y.npy
./data/public/train_graphs/graph_73_A.npy
./data/public/train_graphs/graph_89_X.npy
./data/public/train_graphs/graph_0_X.npy
./data/public/train_graphs/graph_217_X.npy
./data/public/train_graphs/graph_9_A.npy
./data/public/train_graphs/graph_141_y.npy
./data/public/train_graphs/graph_20_A.npy
./data/public/train_graphs/graph_129_y.npy
./data/public/train_graphs/graph_291_A.npy
./data/public/train_graphs/graph_341_X.npy
./data/public/train_graphs/graph_106_A.npy
./data/public/train_graphs/graph_22_A.npy
./data/public/train_graphs/graph_192_X.npy
./data/public/train_graphs/graph_413_y.npy
./data/public/train_graphs/graph_16_X.npy
./data/public/train_graphs/graph_164_A.npy
./data/public/train_graphs/graph_328_X.npy
./data/public/train_graphs/graph_174_y.npy
./data/public/train_graphs/graph_244_A.npy
./data/public/train_graphs/graph_25_A.npy
./data/public/train_graphs/graph_67_y.npy
./data/public/train_graphs/graph_126_X.npy
./data/public/train_graphs/graph_66_X.npy
./data/public/train_graphs/graph_31_y.npy
./data/public/train_graphs/graph_307_A.npy
./data/public/train_graphs/graph_106_y.npy
./data/public/train_graphs/graph_19_X.npy
./data/public/train_graphs/graph_152_y.npy
./data/public/train_graphs/graph_382_A.npy
./data/public/train_graphs/graph_139_X.npy
./data/public/train_graphs/graph_409_X.npy
./data/public/train_graphs/graph_194_A.npy
./data/public/train_graphs/graph_23_X.npy
./data/public/train_graphs/graph_125_y.npy
./data/public/train_graphs/graph_404_A.npy
./data/public/train_graphs/graph_310_y.npy
./data/public/train_graphs/graph_133_A.npy
./data/public/train_graphs/graph_183_X.npy
./data/public/train_graphs/graph_113_y.npy
./data/public/train_graphs/graph_113_A.npy
./data/public/train_graphs/graph_280_X.npy
./data/public/train_graphs/graph_327_A.npy
./data/public/train_graphs/graph_184_X.npy
./data/public/train_graphs/graph_49_X.npy
./data/public/train_graphs/graph_92_A.npy
./data/public/train_graphs/graph_399_y.npy
./data/public/train_graphs/graph_211_y.npy
./data/public/train_graphs/graph_18_A.npy
./data/public/train_graphs/graph_316_X.npy
./data/public/train_graphs/graph_390_X.npy
./data/public/train_graphs/graph_350_X.npy
./data/public/train_graphs/graph_77_X.npy
./data/public/train_graphs/graph_293_A.npy
./data/public/train_graphs/graph_299_A.npy
./data/public/train_graphs/graph_366_X.npy
./data/public/train_graphs/graph_342_X.npy
./data/public/train_graphs/graph_111_X.npy
./data/public/train_graphs/graph_400_X.npy
./data/public/train_graphs/graph_171_A.npy
./data/public/train_graphs/graph_48_y.npy
./data/public/train_graphs/graph_292_A.npy
./data/public/train_graphs/graph_90_X.npy
./data/public/train_graphs/graph_330_X.npy
./data/public/train_graphs/graph_336_A.npy
./data/public/train_graphs/graph_359_y.npy
./data/public/train_graphs/graph_386_y.npy
./data/public/train_graphs/graph_398_A.npy
./data/public/train_graphs/graph_128_A.npy
./data/public/train_graphs/graph_163_y.npy
./data/public/train_graphs/graph_168_X.npy
./data/public/train_graphs/graph_254_y.npy
./data/public/train_graphs/graph_64_A.npy
./data/public/train_graphs/graph_70_y.npy
./data/public/train_graphs/graph_125_X.npy
./data/public/train_graphs/graph_268_X.npy
./data/public/train_graphs/graph_279_X.npy
./data/public/train_graphs/graph_391_y.npy
./data/public/train_graphs/graph_368_y.npy
./data/public/train_graphs/graph_351_X.npy
./data/public/train_graphs/graph_281_A.npy
./data/public/train_graphs/graph_47_A.npy
./data/public/train_graphs/graph_328_y.npy
./data/public/train_graphs/graph_283_y.npy
./data/public/train_graphs/graph_421_y.npy
./data/public/train_graphs/graph_132_X.npy
./data/public/train_graphs/graph_46_A.npy
./data/public/train_graphs/graph_210_y.npy
./data/public/train_graphs/graph_373_y.npy
./data/public/train_graphs/graph_180_X.npy
./data/public/train_graphs/graph_275_A.npy
./data/public/train_graphs/graph_57_y.npy
./data/public/train_graphs/graph_155_X.npy
./data/public/train_graphs/graph_21_y.npy
./data/public/train_graphs/graph_149_y.npy
./data/public/train_graphs/graph_284_A.npy
./data/public/train_graphs/graph_218_A.npy
./data/public/train_graphs/graph_352_y.npy
./data/public/train_graphs/graph_209_y.npy
./data/public/train_graphs/graph_15_X.npy
./data/public/train_graphs/graph_240_y.npy
./data/public/train_graphs/graph_311_A.npy
./data/public/train_graphs/graph_9_y.npy
./data/public/train_graphs/graph_72_X.npy
./data/public/train_graphs/graph_341_y.npy
./data/public/train_graphs/graph_384_y.npy
./data/public/train_graphs/graph_321_X.npy
./data/public/train_graphs/graph_124_y.npy
./data/public/train_graphs/graph_32_y.npy
./data/public/train_graphs/graph_11_A.npy
./data/public/train_graphs/graph_169_X.npy
./data/public/train_graphs/graph_133_X.npy
./data/public/train_graphs/graph_59_y.npy
./data/public/train_graphs/graph_336_X.npy
./data/public/train_graphs/graph_80_A.npy
./data/public/train_graphs/graph_263_y.npy
./data/public/train_graphs/graph_225_X.npy
./data/public/train_graphs/graph_422_A.npy
./data/public/train_graphs/graph_27_y.npy
./data/public/train_graphs/graph_410_y.npy
./data/public/train_graphs/graph_21_X.npy
./data/public/train_graphs/graph_360_A.npy
./data/public/train_graphs/graph_411_X.npy
./data/public/train_graphs/graph_388_A.npy
./data/public/train_graphs/graph_74_X.npy
./data/public/train_graphs/graph_65_X.npy
./data/public/train_graphs/graph_59_A.npy
./data/public/train_graphs/graph_31_X.npy
./data/public/train_graphs/graph_392_X.npy
./data/public/train_graphs/graph_68_A.npy
./data/public/train_graphs/graph_133_y.npy
./data/public/train_graphs/graph_137_X.npy
./data/public/train_graphs/graph_373_A.npy
./data/public/train_graphs/graph_234_y.npy
./data/public/train_graphs/graph_339_A.npy
./data/public/train_graphs/graph_319_y.npy
./data/public/train_graphs/graph_8_y.npy
./data/public/train_graphs/graph_326_y.npy
./data/public/train_graphs/graph_188_y.npy
./data/public/train_graphs/graph_293_y.npy
./data/public/train_graphs/graph_91_X.npy
./data/public/train_graphs/graph_258_y.npy
./data/public/train_graphs/graph_100_y.npy
./data/public/train_graphs/graph_69_y.npy
./data/public/train_graphs/graph_308_X.npy
./data/public/train_graphs/graph_305_y.npy
./data/public/train_graphs/graph_283_A.npy
./data/public/train_graphs/graph_319_X.npy
./data/public/train_graphs/graph_393_A.npy
./data/public/train_graphs/graph_129_X.npy
./data/public/train_graphs/graph_65_A.npy
./data/public/train_graphs/graph_47_y.npy
./data/public/train_graphs/graph_3_y.npy
./data/public/train_graphs/graph_113_X.npy
./data/public/train_graphs/graph_185_y.npy
./data/public/train_graphs/graph_71_y.npy
./data/public/train_graphs/graph_25_X.npy
./data/public/train_graphs/graph_11_y.npy
./data/public/train_graphs/graph_308_A.npy
./data/public/train_graphs/graph_329_X.npy
./data/public/train_graphs/graph_277_X.npy
./data/public/train_graphs/graph_87_A.npy
./data/public/train_graphs/graph_324_X.npy
./data/public/train_graphs/graph_405_y.npy
./data/public/train_graphs/graph_17_X.npy
./data/public/train_graphs/graph_289_y.npy
./data/public/train_graphs/graph_78_A.npy
./data/public/train_graphs/graph_135_y.npy
./data/public/train_graphs/graph_366_A.npy
./data/public/train_graphs/graph_155_y.npy
./data/public/train_graphs/graph_323_y.npy
./data/public/train_graphs/graph_89_A.npy
./data/public/train_graphs/graph_219_y.npy
./data/public/train_graphs/graph_256_A.npy
./data/public/train_graphs/graph_148_A.npy
./data/public/train_graphs/graph_62_X.npy
./data/public/train_graphs/graph_263_X.npy
./data/public/train_graphs/graph_233_A.npy
./data/public/train_graphs/graph_185_X.npy
./data/public/train_graphs/graph_150_A.npy
./data/public/train_graphs/graph_255_y.npy
./data/public/train_graphs/graph_106_X.npy
./data/public/train_graphs/graph_140_y.npy
./data/public/train_graphs/graph_314_y.npy
./data/public/train_graphs/graph_37_X.npy
./data/public/train_graphs/graph_164_y.npy
./data/public/train_graphs/graph_241_A.npy
./data/public/train_graphs/graph_330_A.npy
./data/public/train_graphs/graph_1_X.npy
./data/public/train_graphs/graph_259_X.npy
./data/public/train_graphs/graph_298_y.npy
./data/public/train_graphs/graph_321_y.npy
./data/public/train_graphs/graph_215_A.npy
./data/public/train_graphs/graph_79_A.npy
./data/public/train_graphs/graph_261_A.npy
./data/public/train_graphs/graph_406_y.npy
./data/public/train_graphs/graph_384_A.npy
./data/public/train_graphs/graph_392_A.npy
./data/public/train_graphs/graph_188_X.npy
./data/public/train_graphs/graph_149_X.npy
./data/public/train_graphs/graph_385_y.npy
./data/public/train_graphs/graph_209_X.npy
./data/public/train_graphs/graph_321_A.npy
./data/public/train_graphs/graph_100_A.npy
./data/public/train_graphs/graph_132_A.npy
./data/public/train_graphs/graph_304_X.npy
./data/public/train_graphs/graph_20_X.npy
./data/public/train_graphs/graph_295_y.npy
./data/public/train_graphs/graph_384_X.npy
./data/public/train_graphs/graph_387_y.npy
./data/public/train_graphs/graph_344_y.npy
./data/public/train_graphs/graph_311_X.npy
./data/public/train_graphs/graph_299_y.npy
./data/public/train_graphs/graph_156_y.npy
./data/public/train_graphs/graph_69_X.npy
./data/public/train_graphs/graph_195_y.npy
./data/public/train_graphs/graph_45_y.npy
./data/public/train_graphs/graph_325_X.npy
./data/public/train_graphs/graph_75_A.npy
./data/public/train_graphs/graph_278_A.npy
./data/public/train_graphs/graph_241_y.npy
./data/public/train_graphs/graph_307_y.npy
./data/public/train_graphs/graph_139_y.npy
./data/public/train_graphs/graph_361_A.npy
./data/public/train_graphs/graph_29_X.npy
./data/public/train_graphs/graph_345_y.npy
./data/public/train_graphs/graph_50_y.npy
./data/public/train_graphs/graph_258_A.npy
./data/public/train_graphs/graph_325_y.npy
./data/public/train_graphs/graph_10_y.npy
./data/public/train_graphs/graph_76_A.npy
./data/public/train_graphs/graph_280_y.npy
./data/public/train_graphs/graph_114_X.npy
./data/public/train_graphs/graph_296_X.npy
./data/public/train_graphs/graph_183_A.npy
./data/public/train_graphs/graph_194_X.npy
./data/public/train_graphs/graph_183_y.npy
./data/public/train_graphs/graph_407_y.npy
./data/public/train_graphs/graph_167_y.npy
./data/public/train_graphs/graph_193_y.npy
./data/public/train_graphs/graph_22_X.npy
./data/public/train_graphs/graph_9_X.npy
./data/public/train_graphs/graph_156_A.npy
./data/public/train_graphs/graph_13_X.npy
./data/public/train_graphs/graph_24_y.npy
./data/public/train_graphs/graph_170_X.npy
./data/public/train_graphs/graph_346_A.npy
./data/public/train_graphs/graph_69_A.npy
./data/public/train_graphs/graph_257_X.npy
./data/public/train_graphs/graph_331_A.npy
./data/public/train_graphs/graph_268_A.npy
./data/public/train_graphs/graph_396_X.npy
./data/public/train_graphs/graph_44_X.npy
./data/public/train_graphs/graph_322_y.npy
./data/public/train_graphs/graph_388_X.npy
./data/public/train_graphs/graph_386_A.npy
./data/public/train_graphs/graph_219_A.npy
./data/public/train_graphs/graph_411_A.npy
./data/public/train_graphs/graph_152_A.npy
./data/public/train_graphs/graph_290_X.npy
./data/public/train_graphs/graph_131_y.npy
./data/public/train_graphs/graph_100_X.npy
./data/public/train_graphs/graph_26_X.npy
./data/public/train_graphs/graph_129_A.npy
./data/public/train_graphs/graph_16_A.npy
./data/public/train_graphs/graph_302_X.npy
./data/public/train_graphs/graph_309_X.npy
./data/public/train_graphs/graph_362_X.npy
./data/public/train_graphs/graph_405_X.npy
./data/public/train_graphs/graph_150_X.npy
./data/public/train_graphs/graph_222_A.npy
./data/public/train_graphs/graph_50_X.npy
./data/public/train_graphs/graph_14_A.npy
./data/public/train_graphs/graph_99_A.npy
./data/public/train_graphs/graph_350_y.npy
./data/public/train_graphs/graph_202_A.npy
./data/public/train_graphs/graph_37_A.npy
./data/public/train_graphs/graph_303_y.npy
./data/public/train_graphs/graph_111_y.npy
./data/public/train_graphs/graph_137_A.npy
./data/public/train_graphs/graph_89_y.npy
./data/public/train_graphs/graph_171_X.npy
./data/public/train_graphs/graph_192_A.npy
./data/public/train_graphs/graph_341_A.npy
./data/public/train_graphs/graph_63_X.npy
./data/public/train_graphs/graph_140_A.npy
./data/public/train_graphs/graph_210_X.npy
./data/public/train_graphs/graph_333_X.npy
./data/public/train_graphs/graph_355_y.npy
./data/public/train_graphs/graph_141_X.npy
./data/public/train_graphs/graph_368_A.npy
./data/public/train_graphs/graph_97_A.npy
./data/public/train_graphs/graph_288_y.npy
./data/public/train_graphs/graph_347_X.npy
./data/public/train_graphs/graph_88_A.npy
./data/public/train_graphs/graph_94_A.npy
./data/public/train_graphs/graph_73_y.npy
./data/public/train_graphs/graph_60_X.npy
./data/public/train_graphs/graph_305_A.npy
./data/public/train_graphs/graph_78_y.npy
./data/public/train_graphs/graph_12_y.npy
./data/public/train_graphs/graph_223_y.npy
./data/public/train_graphs/graph_94_y.npy
./data/public/train_graphs/graph_316_y.npy
./data/public/train_graphs/graph_280_A.npy
./data/public/train_graphs/graph_169_A.npy
./data/public/train_graphs/graph_192_y.npy
./data/public/train_graphs/graph_264_y.npy
./data/public/train_graphs/graph_304_A.npy
./data/public/train_graphs/graph_166_X.npy
./data/public/train_graphs/graph_30_X.npy
./data/public/train_graphs/graph_256_X.npy
./data/public/train_graphs/graph_57_X.npy
./data/public/train_graphs/graph_124_A.npy
./data/public/train_graphs/graph_137_y.npy
./data/public/train_graphs/graph_295_X.npy
./data/public/train_graphs/graph_399_A.npy
./data/public/train_graphs/graph_367_A.npy
./data/public/train_graphs/graph_392_y.npy
./data/public/train_graphs/graph_196_X.npy
./data/public/train_graphs/graph_327_X.npy
./data/public/train_graphs/graph_347_y.npy
./data/public/train_graphs/graph_176_A.npy
./data/public/train_graphs/graph_355_X.npy
./data/public/train_graphs/graph_15_A.npy
./data/public/train_graphs/graph_333_A.npy
./data/public/train_graphs/graph_348_X.npy
./data/public/train_graphs/graph_233_X.npy
./data/public/train_graphs/graph_7_X.npy
./data/public/train_graphs/graph_245_A.npy
./data/public/train_graphs/graph_391_A.npy
./data/public/train_graphs/graph_259_A.npy
./data/public/train_graphs/graph_76_X.npy
./data/public/train_graphs/graph_285_A.npy
./data/public/train_graphs/graph_413_X.npy
./data/public/train_graphs/graph_326_A.npy
./data/public/train_graphs/graph_11_X.npy
./data/public/train_graphs/graph_242_A.npy
./data/public/train_graphs/graph_8_X.npy
./data/public/train_graphs/graph_180_A.npy
./data/public/train_graphs/graph_127_y.npy
./data/public/train_graphs/graph_374_X.npy
./data/public/train_graphs/graph_357_X.npy
./data/public/train_graphs/graph_393_X.npy
./data/public/train_graphs/graph_179_A.npy
./data/public/train_graphs/graph_264_A.npy
./data/public/train_graphs/graph_382_X.npy
./data/public/train_graphs/graph_173_y.npy
./data/public/train_graphs/graph_364_y.npy
./data/public/train_graphs/graph_387_A.npy
./data/public/train_graphs/graph_214_X.npy
./data/public/train_graphs/graph_317_y.npy
./data/public/train_graphs/graph_95_y.npy
./data/public/train_graphs/graph_258_X.npy
./data/public/train_graphs/graph_163_A.npy
./data/public/train_graphs/graph_414_A.npy
./data/public/train_graphs/graph_190_A.npy
./data/public/train_graphs/graph_99_X.npy
./data/public/train_graphs/graph_74_A.npy
./data/public/train_graphs/graph_114_A.npy
./data/public/train_graphs/graph_194_y.npy
./data/public/train_graphs/graph_335_X.npy
./data/public/train_graphs/graph_134_X.npy
./data/public/train_graphs/graph_112_A.npy
./data/public/train_graphs/graph_284_y.npy
./data/public/train_graphs/graph_395_X.npy
./data/public/train_graphs/graph_421_X.npy
./data/public/train_graphs/graph_43_X.npy
./data/public/train_graphs/graph_396_A.npy
./data/public/train_graphs/graph_150_y.npy
./data/public/train_graphs/graph_19_A.npy
./data/public/train_graphs/graph_220_y.npy
./data/public/train_graphs/graph_151_A.npy
./data/public/train_graphs/graph_179_y.npy
./data/public/train_graphs/graph_75_y.npy
./data/public/train_graphs/graph_322_A.npy
./data/public/train_graphs/graph_408_A.npy
./data/public/train_graphs/graph_232_X.npy
./data/public/train_graphs/graph_153_A.npy
./data/public/train_graphs/graph_92_y.npy
./data/public/train_graphs/graph_302_A.npy
./data/public/train_graphs/graph_397_y.npy
./data/public/train_graphs/graph_288_A.npy
./data/public/train_graphs/graph_175_A.npy
./data/public/train_graphs/graph_139_A.npy
./data/public/train_graphs/graph_87_X.npy
./data/public/train_graphs/graph_88_y.npy
./data/public/train_graphs/graph_390_y.npy
./data/public/train_graphs/graph_98_y.npy
./data/public/train_graphs/graph_182_A.npy
./data/public/train_graphs/graph_331_X.npy
./data/public/train_graphs/graph_354_y.npy
./data/public/train_graphs/graph_294_X.npy
./data/public/train_graphs/graph_14_X.npy
./data/public/train_graphs/graph_75_X.npy
./data/public/train_graphs/graph_95_X.npy
./data/public/train_graphs/graph_78_X.npy
./data/public/train_graphs/graph_395_y.npy
./data/public/train_graphs/graph_23_A.npy
./data/public/train_graphs/graph_324_y.npy
./data/public/train_graphs/graph_196_y.npy
./data/public/train_graphs/graph_156_X.npy
./data/public/train_graphs/graph_403_A.npy
./data/public/train_graphs/graph_1_A.npy
./data/public/train_graphs/graph_322_X.npy
./data/public/train_graphs/graph_335_y.npy
./data/public/train_graphs/graph_383_X.npy
./data/public/train_graphs/graph_403_X.npy
./data/public/train_graphs/graph_399_X.npy
./data/public/train_graphs/graph_132_y.npy
./data/public/train_graphs/graph_10_A.npy
./data/public/train_graphs/graph_195_X.npy
./data/public/train_graphs/graph_105_X.npy
./data/public/train_graphs/graph_181_A.npy
./data/public/train_graphs/graph_354_A.npy
./data/public/train_graphs/graph_80_y.npy
./data/public/train_graphs/graph_317_X.npy
./data/public/train_graphs/graph_147_y.npy
./data/public/train_graphs/graph_348_A.npy
./data/public/train_graphs/graph_190_X.npy
./data/public/train_graphs/graph_284_X.npy
./data/public/train_graphs/graph_401_X.npy
./data/public/train_graphs/graph_127_X.npy
./data/public/train_graphs/graph_12_A.npy
./data/public/train_graphs/graph_345_X.npy
./data/public/train_graphs/graph_134_y.npy
./data/public/train_graphs/graph_334_A.npy
./data/public/train_graphs/graph_3_A.npy
./data/public/train_graphs/graph_422_X.npy
./data/public/train_graphs/graph_239_A.npy
./data/public/train_graphs/graph_220_X.npy
./data/public/train_graphs/graph_385_X.npy
./data/public/train_graphs/graph_415_A.npy
./data/public/train_graphs/graph_416_X.npy
./data/public/train_graphs/graph_302_y.npy
./data/public/train_graphs/graph_359_A.npy
./data/public/train_graphs/graph_407_X.npy
./data/public/train_graphs/graph_62_y.npy
./data/public/train_graphs/graph_267_A.npy
./data/public/train_graphs/graph_259_y.npy
./data/public/train_graphs/graph_28_y.npy
./data/public/train_graphs/graph_88_X.npy
./data/public/train_graphs/graph_72_A.npy
./data/public/train_graphs/graph_278_X.npy
./data/public/train_graphs/graph_311_y.npy
./data/public/train_graphs/graph_20_y.npy
./data/public/train_graphs/graph_125_A.npy
./data/public/train_graphs/graph_221_X.npy
./data/public/train_graphs/graph_98_X.npy
./data/public/train_graphs/graph_170_y.npy
./data/public/train_graphs/graph_128_X.npy
./data/public/train_graphs/graph_193_A.npy
./data/public/train_graphs/graph_292_y.npy
./data/public/train_graphs/graph_115_X.npy
./data/public/train_graphs/graph_415_X.npy
./data/public/train_graphs/graph_218_X.npy
./data/public/train_graphs/graph_47_X.npy
./data/public/train_graphs/graph_105_A.npy
./data/public/train_graphs/graph_168_y.npy
./data/public/train_graphs/graph_388_y.npy
./data/public/train_graphs/graph_373_X.npy
./data/public/train_graphs/graph_214_y.npy
./data/public/train_graphs/graph_361_y.npy
./data/public/train_graphs/graph_389_A.npy
./data/public/train_graphs/graph_2_X.npy
./data/public/train_graphs/graph_136_A.npy
./data/public/train_graphs/graph_260_A.npy
./data/public/train_graphs/graph_243_y.npy
./data/public/train_graphs/graph_364_X.npy
./data/public/train_graphs/graph_410_A.npy
./data/public/train_graphs/graph_14_y.npy
./data/public/train_graphs/graph_239_y.npy
./data/public/train_graphs/graph_7_y.npy
./data/public/train_graphs/graph_202_y.npy
./data/public/train_graphs/graph_94_X.npy
./data/public/train_graphs/graph_184_y.npy
./data/public/train_graphs/graph_110_X.npy
./data/public/train_graphs/graph_421_A.npy
./data/public/train_graphs/graph_395_A.npy
./data/public/train_graphs/graph_416_y.npy
./data/public/train_graphs/graph_126_A.npy
./data/public/train_graphs/graph_217_A.npy
./data/public/train_graphs/graph_91_A.npy
./data/public/train_graphs/graph_112_y.npy
./data/public/train_graphs/graph_358_X.npy
./data/public/train_graphs/graph_265_A.npy
./data/public/train_graphs/graph_315_X.npy
./data/public/train_graphs/graph_329_A.npy
./data/public/train_graphs/graph_60_y.npy
./data/public/train_graphs/graph_347_A.npy
./data/public/train_graphs/graph_262_y.npy
./data/public/train_graphs/graph_323_A.npy
./data/public/train_graphs/graph_309_A.npy
./data/public/train_graphs/graph_306_X.npy
./data/public/train_graphs/graph_124_X.npy
./data/public/train_graphs/graph_277_y.npy
./data/public/train_graphs/graph_182_X.npy
./data/public/train_graphs/graph_30_A.npy
./data/public/train_graphs/graph_219_X.npy
./data/public/test_nodes.csv
./data/public/test_features.csv
./data/public/test_graphs/graph_119_X.npy
./data/public/test_graphs/graph_301_A.npy
./data/public/test_graphs/graph_276_X.npy
./data/public/test_graphs/graph_378_X.npy
./data/public/test_graphs/graph_300_X.npy
./data/public/test_graphs/graph_161_A.npy
./data/public/test_graphs/graph_419_A.npy
./data/public/test_graphs/graph_271_X.npy
./data/public/test_graphs/graph_116_X.npy
./data/public/test_graphs/graph_228_X.npy
./data/public/test_graphs/graph_162_A.npy
./data/public/test_graphs/graph_365_A.npy
./data/public/test_graphs/graph_103_A.npy
./data/public/test_graphs/graph_172_X.npy
./data/public/test_graphs/graph_51_A.npy
./data/public/test_graphs/graph_84_X.npy
./data/public/test_graphs/graph_206_A.npy
./data/public/test_graphs/graph_236_X.npy
./data/public/test_graphs/graph_338_A.npy
./data/public/test_graphs/graph_365_X.npy
./data/public/test_graphs/graph_300_A.npy
./data/public/test_graphs/graph_197_X.npy
./data/public/test_graphs/graph_287_X.npy
./data/public/test_graphs/graph_207_A.npy
./data/public/test_graphs/graph_143_X.npy
./data/public/test_graphs/graph_199_X.npy
./data/public/test_graphs/graph_418_A.npy
./data/public/test_graphs/graph_238_X.npy
./data/public/test_graphs/graph_343_X.npy
./data/public/test_graphs/graph_235_X.npy
./data/public/test_graphs/graph_377_X.npy
./data/public/test_graphs/graph_270_A.npy
./data/public/test_graphs/graph_301_X.npy
./data/public/test_graphs/graph_370_X.npy
./data/public/test_graphs/graph_236_A.npy
./data/public/test_graphs/graph_84_A.npy
./data/public/test_graphs/graph_270_X.npy
./data/public/test_graphs/graph_216_X.npy
./data/public/test_graphs/graph_39_X.npy
./data/public/test_graphs/graph_369_X.npy
./data/public/test_graphs/graph_375_A.npy
./data/public/test_graphs/graph_117_A.npy
./data/public/test_graphs/graph_142_A.npy
./data/public/test_graphs/graph_272_A.npy
./data/public/test_graphs/graph_123_A.npy
./data/public/test_graphs/graph_189_A.npy
./data/public/test_graphs/graph_419_X.npy
./data/public/test_graphs/graph_363_X.npy
./data/public/test_graphs/graph_117_X.npy
./data/public/test_graphs/graph_376_X.npy
./data/public/test_graphs/graph_178_A.npy
./data/public/test_graphs/graph_237_X.npy
./data/public/test_graphs/graph_165_A.npy
./data/public/test_graphs/graph_249_X.npy
./data/public/test_graphs/graph_205_X.npy
./data/public/test_graphs/graph_276_A.npy
./data/public/test_graphs/graph_251_A.npy
./data/public/test_graphs/graph_109_X.npy
./data/public/test_graphs/graph_282_X.npy
./data/public/test_graphs/graph_229_A.npy
./data/public/test_graphs/graph_212_A.npy
./data/public/test_graphs/graph_145_A.npy
./data/public/test_graphs/graph_165_X.npy
./data/public/test_graphs/graph_371_A.npy
./data/public/test_graphs/graph_6_X.npy
./data/public/test_graphs/graph_177_A.npy
./data/public/test_graphs/graph_118_X.npy
./data/public/test_graphs/graph_41_X.npy
./data/public/test_graphs/graph_252_A.npy
./data/public/test_graphs/graph_53_A.npy
./data/public/test_graphs/graph_145_X.npy
./data/public/test_graphs/graph_160_X.npy
./data/public/test_graphs/graph_204_X.npy
./data/public/test_graphs/graph_246_X.npy
./data/public/test_graphs/graph_121_X.npy
./data/public/test_graphs/graph_158_X.npy
./data/public/test_graphs/graph_104_X.npy
./data/public/test_graphs/graph_200_X.npy
./data/public/test_graphs/graph_56_A.npy
./data/public/test_graphs/graph_250_X.npy
./data/public/test_graphs/graph_251_X.npy
./data/public/test_graphs/graph_102_A.npy
./data/public/test_graphs/graph_85_X.npy
./data/public/test_graphs/graph_372_A.npy
./data/public/test_graphs/graph_204_A.npy
./data/public/test_graphs/graph_35_X.npy
./data/public/test_graphs/graph_205_A.npy
./data/public/test_graphs/graph_107_X.npy
./data/public/test_graphs/graph_247_X.npy
./data/public/test_graphs/graph_420_A.npy
./data/public/test_graphs/graph_248_X.npy
./data/public/test_graphs/graph_103_X.npy
./data/public/test_graphs/graph_34_X.npy
./data/public/test_graphs/graph_36_A.npy
./data/public/test_graphs/graph_102_X.npy
./data/public/test_graphs/graph_369_A.npy
./data/public/test_graphs/graph_253_A.npy
./data/public/test_graphs/graph_198_A.npy
./data/public/test_graphs/graph_273_X.npy
./data/public/test_graphs/graph_191_X.npy
./data/public/test_graphs/graph_160_A.npy
./data/public/test_graphs/graph_418_X.npy
./data/public/test_graphs/graph_380_X.npy
./data/public/test_graphs/graph_237_A.npy
./data/public/test_graphs/graph_144_A.npy
./data/public/test_graphs/graph_250_A.npy
./data/public/test_graphs/graph_34_A.npy
./data/public/test_graphs/graph_224_A.npy
./data/public/test_graphs/graph_81_X.npy
./data/public/test_graphs/graph_108_X.npy
./data/public/test_graphs/graph_379_X.npy
./data/public/test_graphs/graph_159_X.npy
./data/public/test_graphs/graph_35_A.npy
./data/public/test_graphs/graph_338_X.npy
./data/public/test_graphs/graph_349_A.npy
./data/public/test_graphs/graph_130_X.npy
./data/public/test_graphs/graph_273_A.npy
./data/public/test_graphs/graph_271_A.npy
./data/public/test_graphs/graph_107_A.npy
./data/public/test_graphs/graph_178_X.npy
./data/public/test_graphs/graph_142_X.npy
./data/public/test_graphs/graph_199_A.npy
./data/public/test_graphs/graph_212_X.npy
./data/public/test_graphs/graph_207_X.npy
./data/public/test_graphs/graph_246_A.npy
./data/public/test_graphs/graph_230_A.npy
./data/public/test_graphs/graph_318_A.npy
./data/public/test_graphs/graph_162_X.npy
./data/public/test_graphs/graph_55_A.npy
./data/public/test_graphs/graph_231_X.npy
./data/public/test_graphs/graph_104_A.npy
./data/public/test_graphs/graph_5_X.npy
./data/public/test_graphs/graph_161_X.npy
./data/public/test_graphs/graph_82_A.npy
./data/public/test_graphs/graph_41_A.npy
./data/public/test_graphs/graph_186_X.npy
./data/public/test_graphs/graph_378_A.npy
./data/public/test_graphs/graph_248_A.npy
./data/public/test_graphs/graph_51_X.npy
./data/public/test_graphs/graph_56_X.npy
./data/public/test_graphs/graph_376_A.npy
./data/public/test_graphs/graph_228_A.npy
./data/public/test_graphs/graph_143_A.npy
./data/public/test_graphs/graph_412_A.npy
./data/public/test_graphs/graph_247_A.npy
./data/public/test_graphs/graph_172_A.npy
./data/public/test_graphs/graph_54_X.npy
./data/public/test_graphs/graph_229_X.npy
./data/public/test_graphs/graph_36_X.npy
./data/public/test_graphs/graph_287_A.npy
./data/public/test_graphs/graph_189_X.npy
./data/public/test_graphs/graph_86_X.npy
./data/public/test_graphs/graph_108_A.npy
./data/public/test_graphs/graph_381_X.npy
./data/public/test_graphs/graph_286_X.npy
./data/public/test_graphs/graph_231_A.npy
./data/public/test_graphs/graph_381_A.npy
./data/public/test_graphs/graph_272_X.npy
./data/public/test_graphs/graph_203_A.npy
./data/public/test_graphs/graph_119_A.npy
./data/public/test_graphs/graph_157_X.npy
./data/public/test_graphs/graph_52_X.npy
./data/public/test_graphs/graph_274_X.npy
./data/public/test_graphs/graph_379_A.npy
./data/public/test_graphs/graph_83_X.npy
./data/public/test_graphs/graph_371_X.npy
./data/public/test_graphs/graph_380_A.npy
./data/public/test_graphs/graph_186_A.npy
./data/public/test_graphs/graph_159_A.npy
./data/public/test_graphs/graph_370_A.npy
./data/public/test_graphs/graph_203_X.npy
./data/public/test_graphs/graph_375_X.npy
./data/public/test_graphs/graph_81_A.npy
./data/public/test_graphs/graph_238_A.npy
./data/public/test_graphs/graph_4_A.npy
./data/public/test_graphs/graph_197_A.npy
./data/public/test_graphs/graph_417_A.npy
./data/public/test_graphs/graph_201_X.npy
./data/public/test_graphs/graph_157_A.npy
./data/public/test_graphs/graph_85_A.npy
./data/public/test_graphs/graph_123_X.npy
./data/public/test_graphs/graph_122_A.npy
./data/public/test_graphs/graph_86_A.npy
./data/public/test_graphs/graph_377_A.npy
./data/public/test_graphs/graph_33_A.npy
./data/public/test_graphs/graph_177_X.npy
./data/public/test_graphs/graph_109_A.npy
./data/public/test_graphs/graph_363_A.npy
./data/public/test_graphs/graph_372_X.npy
./data/public/test_graphs/graph_55_X.npy
./data/public/test_graphs/graph_230_X.npy
./data/public/test_graphs/graph_146_A.npy
./data/public/test_graphs/graph_200_A.npy
./data/public/test_graphs/graph_53_X.npy
./data/public/test_graphs/graph_82_X.npy
./data/public/test_graphs/graph_198_X.npy
./data/public/test_graphs/graph_40_A.npy
./data/public/test_graphs/graph_120_X.npy
./data/public/test_graphs/graph_120_A.npy
./data/public/test_graphs/graph_122_X.npy
./data/public/test_graphs/graph_269_A.npy
./data/public/test_graphs/graph_286_A.npy
./data/public/test_graphs/graph_252_X.npy
./data/public/test_graphs/graph_130_A.npy
./data/public/test_graphs/graph_118_A.npy
./data/public/test_graphs/graph_349_X.npy
./data/public/test_graphs/graph_54_A.npy
./data/public/test_graphs/graph_146_X.npy
./data/public/test_graphs/graph_121_A.npy
./data/public/test_graphs/graph_191_A.npy
./data/public/test_graphs/graph_33_X.npy
./data/public/test_graphs/graph_58_A.npy
./data/public/test_graphs/graph_52_A.npy
./data/public/test_graphs/graph_318_X.npy
./data/public/test_graphs/graph_39_A.npy
./data/public/test_graphs/graph_249_A.npy
./data/public/test_graphs/graph_253_X.npy
./data/public/test_graphs/graph_412_X.npy
./data/public/test_graphs/graph_269_X.npy
./data/public/test_graphs/graph_417_X.npy
./data/public/test_graphs/graph_224_X.npy
./data/public/test_graphs/graph_343_A.npy
./data/public/test_graphs/graph_83_A.npy
./data/public/test_graphs/graph_235_A.npy
./data/public/test_graphs/graph_337_A.npy
./data/public/test_graphs/graph_58_X.npy
./data/public/test_graphs/graph_4_X.npy
./data/public/test_graphs/graph_216_A.npy
./data/public/test_graphs/graph_420_X.npy
./data/public/test_graphs/graph_5_A.npy
./data/public/test_graphs/graph_6_A.npy
./data/public/test_graphs/graph_40_X.npy
./data/public/test_graphs/graph_201_A.npy
./data/public/test_graphs/graph_144_X.npy
./data/public/test_graphs/graph_337_X.npy
./data/public/test_graphs/graph_116_A.npy
./data/public/test_graphs/graph_282_A.npy
./data/public/test_graphs/graph_274_A.npy
./data/public/test_graphs/graph_158_A.npy
./data/public/test_graphs/graph_206_X.npy
./data/public/node_vocabulary.txt
./data/public/submission.key
./LICENSE
./assets/graph data structure.png
./assets/overall flow.png
./docs/leaderboard.html
./docs/leaderboard.csv
./docs/leaderboard.css
./docs/leaderboard.js
./docs/leaderboard_data.js
./requirements.txt
./submissions/inbox/.gitkeep
./submissions/README.md
./.gitignore
./scripts/build_graph.py
./scripts/generate_keys.py
./scripts/encrypt_submission.py
./competition/data_utils.py
./competition/render_leaderboard.py
./competition/validation.py
./competition/evaluate.py
./competition/config.yaml
./competition/__init__.py
./competition/decrypt_workflow.py
./competition/metrics.py
./competition/crypto_utils.py
./README.md
./baselines/random_forest_baseline.py
./baselines/gat_baseline.py
./baselines/mlp_baseline.py
./baselines/gnn_utils.py
./baselines/README.md
./baselines/gcn_baseline.py
./.github/PULL_REQUEST_TEMPLATE.md
./.github/workflows/score_submission.yml
./.github/workflows/update_leaderboard.yml
./leaderboard/leaderboard.csv
./leaderboard/leaderboard.md

=== SECTION 3: DATA SAMPLE ===
`graph_i_A.npy`:
`i` represents the graph id, it is a numpy array with shape (N_nodes, N_nodes) and dtype float32. It represents the adjacency matrix of graph `i`
`N_nodes` is the number of nodes in the graph

`graph_i_X.npy`:
`i` represents the graph id, it is a numpy array with shape (N_nodes, 37) and dtype float32. It represents the feature matrix of graph `i`
`N_nodes` is the number of nodes in the graph

`graph_i_y.npy`
`i` represents the graph id, it is a numpy array with shape (3,) and dtype float32. It represents the true labels of graph `i` (pressure,temperature,speed)
`N_nodes` is the number of nodes in the graph

Note that `graph_i_y.npy` is not provided for testing graphs

=== SECTION 4: SUBMISSION FORMAT ===
After training the model on training data, use the trained model to make predictions on testing data
--
Create a `predictions.csv` in the following format and save it in the competition main directory :

```csv
id,pressure,temperature,speed
340,150.5,25.0,5.0
341,800.0,155.0,1.2
...
399,45.0,23.0,8.5
...


=== SECTION 5: REQUIREMENTS ===
pandas>=2.0.0
numpy>=1.24.0
scipy
scikit-learn>=1.3.0
torch
torch-geometric




=== SECTION 6: BASELINE CODE === 

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Load the dataset
def load_dataset(data_dir):
    train_ids = sorted([int(f.split('_')[1]) for f in os.listdir(os.path.join(data_dir, 'train_graphs')) if f.endswith('_X.npy')])
    test_ids = sorted([int(f.split('_')[1]) for f in os.listdir(os.path.join(data_dir, 'test_graphs')) if f.endswith('_X.npy')])
    
    train_data_list = []
    for graph_id in train_ids:
        X = np.load(os.path.join(data_dir, 'train_graphs', f'graph_{graph_id}_X.npy'))
        A = np.load(os.path.join(data_dir, 'train_graphs', f'graph_{graph_id}_A.npy'))
        y = np.load(os.path.join(data_dir, 'train_graphs', f'graph_{graph_id}_y.npy'))
        edge_index = torch.tensor(np.array((A > 0).nonzero()), dtype=torch.long)
        data = Data(x=torch.from_numpy(X).float(), edge_index=edge_index, y=torch.from_numpy(y).float().unsqueeze(0))
        train_data_list.append(data)
    
    test_data_list = []
    for graph_id in test_ids:
        X = np.load(os.path.join(data_dir, 'test_graphs', f'graph_{graph_id}_X.npy'))
        A = np.load(os.path.join(data_dir, 'test_graphs', f'graph_{graph_id}_A.npy'))
        edge_index = torch.tensor(np.array((A > 0).nonzero()), dtype=torch.long)
        data = Data(x=torch.from_numpy(X).float(), edge_index=edge_index)
        test_data_list.append(data)
    
    return train_data_list, test_data_list, train_ids, test_ids


data_dir = './data/public'
train_data_list, test_data_list, train_ids, test_ids = load_dataset(data_dir)

# Split into train and validation sets
train_indices, val_indices = train_test_split(range(len(train_data_list)), test_size=0.2, random_state=42)
train_loader = DataLoader([train_data_list[i] for i in train_indices], batch_size=32, shuffle=True)
val_loader = DataLoader([train_data_list[i] for i in val_indices], batch_size=32, shuffle=False)




CRITICAL OUTPUT CONSTRAINTS (MUST FOLLOW EXACTLY):
1. The final file MUST be named exactly: submission.csv
2. The CSV MUST have EXACTLY 4 columns in this order:
   id,pressure,temperature,speed
3. The number of rows MUST equal the number of test samples.
4. Each row MUST correspond to exactly one test graph.
5. Predictions MUST be a 2D array of shape (N, 3):
   - N = number of test samples
   - 3 = [pressure, temperature, speed]
6. You MUST ensure predictions are NOT:
   - shape (N, 1, 3)
   - shape (N,)
   - shape (3, N)


