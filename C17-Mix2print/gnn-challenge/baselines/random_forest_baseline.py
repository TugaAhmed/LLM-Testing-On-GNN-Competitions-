"""
Random Forest Baseline for Bioink GNN Challenge

Expected NMAE: ~0.060

This baseline uses simple feature engineering:
- Material class one-hot encoding
- Concentration sums per material class
- Needle diameter
- Number of components
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from competition.data_utils import parse_components, parse_needle


def extract_features(df):
    """Extract features from dataframe."""
    features = []
    
    for _, row in df.iterrows():
        feat = {}
        
        # Parse components
        components = parse_components(row['Components'])
        
        # Material class concentrations (simplified)
        material_classes = {
            'Alginate': 0, 'Gelatin': 0, 'PCL': 0, 'PLGA': 0,
            'Hyaluronic Acid': 0, 'Collagen': 0, 'Chitosan': 0,
            'Cellulose': 0, 'Hydroxyapatite': 0, 'Other': 0
        }
        
        for comp in components:
            name = comp['name']
            conc = comp['concentration']
            
            # Classify material
            if 'Alginate' in name:
                material_classes['Alginate'] += conc
            elif 'Gelatin' in name or 'GelMA' in name:
                material_classes['Gelatin'] += conc
            elif 'PCL' in name or 'Polycaprolactone' in name:
                material_classes['PCL'] += conc
            elif 'PLGA' in name:
                material_classes['PLGA'] += conc
            elif 'Hyaluronic' in name or 'HA' in name:
                material_classes['Hyaluronic Acid'] += conc
            elif 'Collagen' in name:
                material_classes['Collagen'] += conc
            elif 'Chitosan' in name:
                material_classes['Chitosan'] += conc
            elif 'Cellulose' in name or 'CNF' in name or 'CNC' in name:
                material_classes['Cellulose'] += conc
            elif 'Hydroxyapatite' in name or 'TCP' in name:
                material_classes['Hydroxyapatite'] += conc
            else:
                material_classes['Other'] += conc
        
        feat.update(material_classes)
        
        # Needle features
        needle = parse_needle(row['Needle'])
        feat['needle_diameter'] = needle['diameter'] if needle['diameter'] else 400.0
        feat['needle_cylindrical'] = 1 if needle['geometry'] == 'cylindrical' else 0
        feat['needle_conical'] = 1 if needle['geometry'] == 'conical' else 0
        
        # Complexity features
        feat['num_components'] = len(components)
        feat['total_concentration'] = sum(c['concentration'] for c in components)
        
        features.append(feat)
    
    return pd.DataFrame(features)


def main():
    print("=" * 60)
    print("RANDOM FOREST BASELINE")
    print("=" * 60)
    
    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'public')
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    # Load data
    print("\n[1/5] Loading data...")
    train_full = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
    
    val_path = os.path.join(data_dir, 'val.csv')
    if os.path.exists(val_path):
        print("  [INFO] Found val.csv")
        val_df = pd.read_csv(val_path)
        train_df = train_full
    else:
        print("  [INFO] No val.csv found. Creating internal validation split (80/20)...")
        from sklearn.model_selection import train_test_split
        # Simple random split for baseline purposes
        train_df, val_df = train_test_split(train_full, test_size=0.2, random_state=42)
        print(f"  Internal Split -> Train: {len(train_df)}, Val: {len(val_df)}")
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Extract features
    print("\n[2/5] Engineering features...")
    X_train = extract_features(train_df)
    X_val = extract_features(val_df)
    X_test = extract_features(test_df)
    
    y_train = train_df[['pressure', 'temperature', 'speed']].values
    y_val = val_df[['pressure', 'temperature', 'speed']].values
    
    print(f"  Feature dimension: {X_train.shape[1]}")
    
    # Train model
    print("\n[3/5] Training Random Forest...")
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
    
    model.fit(X_train, y_train)
    print("  [OK] Training complete")
    
    # Evaluate on validation
    print("\n[4/5] Evaluating on internal validation set...")
    y_val_pred = model.predict(X_val)
    
    from competition.metrics import compute_scores
    
    # Save val predictions temporarily
    val_pred_df = val_df[['id']].copy()
    val_pred_df['pressure'] = y_val_pred[:, 0]
    val_pred_df['temperature'] = y_val_pred[:, 1]
    val_pred_df['speed'] = y_val_pred[:, 2]
    val_pred_path = os.path.join(output_dir, 'val_predictions_temp.csv')
    val_pred_df.to_csv(val_pred_path, index=False)
    
    # Save val truth
    val_truth_df = val_df[['id', 'pressure', 'temperature', 'speed']].copy()
    val_truth_df.columns = ['id', 'pressure_true', 'temperature_true', 'speed_true']
    val_truth_path = os.path.join(output_dir, 'val_truth_temp.csv')
    val_truth_df.to_csv(val_truth_path, index=False)
    
    scores = compute_scores(val_pred_path, val_truth_path)
    
    print(f"  Pressure NMAE:    {scores['pressure_nmae']:.6f}")
    print(f"  Temperature NMAE: {scores['temperature_nmae']:.6f}")
    print(f"  Speed NMAE:       {scores['speed_nmae']:.6f}")
    print(f"  Combined NMAE:    {scores['combined_nmae']:.6f} ({scores['combined_pct']:.2f}%)")
    
    # Clean up temp files
    os.remove(val_pred_path)
    os.remove(val_truth_path)
    
    # Generate test predictions
    print("\n[5/5] Generating test predictions...")
    y_test_pred = model.predict(X_test)
    
    test_pred_df = test_df[['id']].copy()
    test_pred_df['pressure'] = y_test_pred[:, 0]
    test_pred_df['temperature'] = y_test_pred[:, 1]
    test_pred_df['speed'] = y_test_pred[:, 2]
    
    output_path = os.path.join(output_dir, 'rf_predictions.csv')
    test_pred_df.to_csv(output_path, index=False)
    print(f"  [OK] Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("[OK] BASELINE COMPLETE")
    print("=" * 60)
    print(f"\nValidation NMAE: {scores['combined_nmae']:.6f}")
    print(f"\nNext steps:")
    print(f"  1. Review predictions: {output_path}")
    print(f"  2. Try to beat this baseline!")
    print(f"  3. Submit your improved model")


if __name__ == '__main__':
    main()
