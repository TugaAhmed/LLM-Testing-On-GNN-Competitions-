# Baseline Models

Reference implementations to help you get started.

## Baselines

### 1. Random Forest (NMAE ≈ 0.060)
Simple multi-output Random Forest using basic features:
- Material class concentrations
- Needle diameter and geometry
- Formulation complexity metrics

**File:** `random_forest_baseline.py`

## Running Baselines

### Random Forest

```bash
python baselines/random_forest_baseline.py
```

This will:
1. Load `train.csv` and `test_features.csv`
2. Create an internal 80/20 train/validation split from `train.csv`
3. Engineer basic features
4. Train Random Forest on the 80% split
5. Evaluate on the 20% internal validation set
6. Generate predictions for the official test set
7. Save to `baselines/outputs/rf_predictions.csv`


## Feature Engineering Ideas

### Material Features
- One-hot encoding of material classes
- Concentration values
- Number of components
- Material diversity (entropy)

### Graph Features
- Node features: material embeddings + concentrations
- Edge features: component interactions
- Graph topology: fully connected vs sparse

### Physical Features
- Needle diameter (affects flow)
- Needle geometry (cylindrical vs conical)
- Total solid content
- Viscosity proxies

### Advanced Ideas
- Material property databases (molecular weight, melting point)
- Chemical similarity graphs
- Temporal features (publication year trends)
- Multi-scale graphs (material → formulation → batch)

## Tips

1. **Start simple:** Beat Random Forest first
2. **Feature importance:** Analyze which features matter
3. **Cross-validation:** Use validation set properly
4. **Multi-task learning:** Share representations across targets
5. **Ensemble:** Combine multiple models

Good luck! 🚀
