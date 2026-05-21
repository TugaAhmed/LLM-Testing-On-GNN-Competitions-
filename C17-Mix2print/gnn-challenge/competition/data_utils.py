"""
Data utilities for bioink GNN challenge.
Handles parsing of components, targets, and preprocessing.
"""

import re
import pandas as pd
from typing import List, Dict, Optional, Tuple


# Material frequency threshold
MIN_MATERIAL_FREQUENCY = 5

# Normalization ranges for NMAE
PRESSURE_RANGE = 1496.0  # kPa
TEMPERATURE_RANGE = 228.0  # °C
SPEED_RANGE = 90.0  # mm/s


def parse_components(comp_str: str) -> List[Dict]:
    """
    Parse component string into structured format.
    
    Example:
        "Alginate [3 wt%] Gelatin [10 wt%]" ->
        [{'name': 'Alginate', 'concentration': 3.0, 'unit': 'wt%'},
         {'name': 'Gelatin', 'concentration': 10.0, 'unit': 'wt%'}]
    """
    if pd.isna(comp_str) or not comp_str.strip():
        return []
    
    # Pattern: Material Name [concentration unit]
    pattern = r'([A-Za-z0-9\s\-\(\),]+?)\s*\[([0-9.]+)\s*([a-zA-Z%/]+)\]'
    matches = re.findall(pattern, comp_str)
    
    components = []
    for name, conc, unit in matches:
        components.append({
            'name': name.strip(),
            'concentration': float(conc),
            'unit': unit.strip()
        })
    
    return components


def parse_range_to_mean(value_str: str) -> Optional[float]:
    """
    Parse a value that might be a range or single number.
    
    Examples:
        "70-80" -> 75.0
        "100" -> 100.0
        "50 - 80" -> 65.0
    """
    if pd.isna(value_str):
        return None
    
    value_str = str(value_str).strip()
    if not value_str or value_str.upper() in ['N/A', 'NA', '']:
        return None
    
    # Check for range pattern
    range_match = re.search(r'([0-9.]+)\s*-\s*([0-9.]+)', value_str)
    if range_match:
        low, high = float(range_match.group(1)), float(range_match.group(2))
        return (low + high) / 2.0
    
    # Extract first number
    num_match = re.search(r'([0-9.]+)', value_str)
    if num_match:
        return float(num_match.group(1))
    
    return None


def parse_pressure(pressure_str: str) -> Optional[float]:
    """Parse pressure and convert to kPa."""
    value = parse_range_to_mean(pressure_str)
    if value is None:
        return None
    
    pressure_str_lower = str(pressure_str).lower()
    
    # Convert bar to kPa
    if 'bar' in pressure_str_lower:
        value = value * 100.0  # 1 bar = 100 kPa
    elif 'psi' in pressure_str_lower:
        value = value * 6.89476  # 1 psi = 6.89476 kPa
    # Assume kPa if no unit specified
    
    return value


def parse_temperature(temp_str: str) -> Optional[float]:
    """Parse temperature (assume Celsius)."""
    return parse_range_to_mean(temp_str)


def parse_speed(speed_str: str) -> Optional[float]:
    """Parse speed (assume mm/s)."""
    return parse_range_to_mean(speed_str)


def parse_needle(needle_str: str) -> Dict:
    """
    Parse needle specification.
    
    Returns dict with 'diameter' (µm) and 'geometry' (cylindrical/conical).
    """
    if pd.isna(needle_str) or not needle_str.strip():
        return {'diameter': None, 'geometry': None}
    
    needle_str = str(needle_str).strip()
    
    # Extract diameter
    diameter = None
    
    # Try µm pattern
    um_match = re.search(r'([0-9.]+)\s*[µu]m', needle_str, re.IGNORECASE)
    if um_match:
        diameter = float(um_match.group(1))
    else:
        # Try gauge conversion (approximate)
        gauge_match = re.search(r'([0-9]+)\s*[Gg]auge', needle_str)
        if gauge_match:
            gauge = int(gauge_match.group(1))
            # Approximate gauge to diameter conversion
            gauge_to_um = {
                18: 838, 19: 686, 20: 603, 21: 514, 22: 413,
                23: 337, 24: 311, 25: 260, 26: 260, 27: 210,
                30: 159, 32: 108
            }
            diameter = gauge_to_um.get(gauge)
    
    # Extract geometry
    geometry = None
    if 'cylindrical' in needle_str.lower():
        geometry = 'cylindrical'
    elif 'conical' in needle_str.lower():
        geometry = 'conical'
    
    return {'diameter': diameter, 'geometry': geometry}


def get_material_frequencies(df: pd.DataFrame) -> Dict[str, int]:
    """Count how many times each material appears in the dataset."""
    material_counts = {}
    
    for _, row in df.iterrows():
        components = parse_components(row.get('Components', ''))
        for comp in components:
            name = comp['name']
            material_counts[name] = material_counts.get(name, 0) + 1
    
    return material_counts


def filter_common_materials(df: pd.DataFrame, min_freq: int = MIN_MATERIAL_FREQUENCY) -> pd.DataFrame:
    """
    Keep only formulations where ALL components appear >= min_freq times.
    """
    material_counts = get_material_frequencies(df)
    common_materials = {name for name, count in material_counts.items() if count >= min_freq}
    
    def has_only_common_materials(comp_str):
        components = parse_components(comp_str)
        if not components:
            return False
        return all(comp['name'] in common_materials for comp in components)
    
    mask = df['Components'].apply(has_only_common_materials)
    return df[mask].copy()


def preprocess_dataset(csv_path: str, output_dir: str = None) -> pd.DataFrame:
    """
    Main preprocessing pipeline.
    
    Steps:
    1. Load raw data
    2. Parse targets (pressure, temperature, speed)
    3. Filter to common materials (>=5 occurrences)
    4. Remove rows with missing targets
    5. Assign IDs
    """
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} samples")
    
    # Parse targets
    df['pressure'] = df['Pressure'].apply(parse_pressure)
    df['temperature'] = df['Temperature (C)'].apply(parse_temperature)
    df['speed'] = df['Speed (mm/s)'].apply(parse_speed)
    
    # Remove rows with any missing target
    df = df.dropna(subset=['pressure', 'temperature', 'speed'])
    print(f"After removing missing targets: {len(df)} samples")
    
    # Filter to common materials
    df = filter_common_materials(df, MIN_MATERIAL_FREQUENCY)
    print(f"After filtering to common materials (>={MIN_MATERIAL_FREQUENCY}): {len(df)} samples")
    
    # Assign IDs
    df = df.reset_index(drop=True)
    df['id'] = df.index
    
    return df


def create_train_val_test_split(df: pd.DataFrame, 
                                  train_ratio: float = 0.70,
                                  val_ratio: float = 0.15,
                                  test_ratio: float = 0.15,
                                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create STRATIFIED GROUP split to avoid data leakage.
    
    Rules:
    1. Group by DOI: All experiments from one paper must be in the same split.
    2. Stratify by Temp Regime: Maintain Hydrogel/Thermoplastic balance across splits.
    
    Hydrogel: temp < 50°C
    Thermoplastic: temp >= 50°C
    """
    from sklearn.model_selection import train_test_split
    
    # Create stratification variable
    df['temp_regime'] = (df['temperature'] >= 50).astype(int)
    
    # Check if DOI exists (it should from preprocess_dataset, assuming it was loaded)
    # If DOI is not in columns (e.g. older csv), fallback to simple stratification
    if 'DOI' not in df.columns:
        print("WARNING: 'DOI' column not found. Falling back to simple stratified split (potential leakage).")
        # Fallback to original logic
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_ratio + test_ratio),
            stratify=df['temp_regime'],
            random_state=random_state
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_df['temp_regime'],
            random_state=random_state
        )
    else:
        print("Performing DOI-based Stratified Group Split...")
        
        # 1. aggregate by DOI to get group-level labels
        # We classify a DOI as 'High Temp' if >50% of its samples are High Temp
        doi_groups = df.groupby('DOI')['temp_regime'].mean()
        doi_labels = (doi_groups > 0.5).astype(int)
        
        unique_dois = doi_groups.index.tolist()
        group_labels = doi_labels.values
        
        # 2. Split DOIs (Groups)
        train_dois, temp_dois, _, temp_labels = train_test_split(
            unique_dois, 
            group_labels,
            test_size=(val_ratio + test_ratio),
            stratify=group_labels, # Stratify the GROUPS
            random_state=random_state
        )
        
        if val_ratio > 0:
            val_dois, test_dois = train_test_split(
                temp_dois,
                test_size=test_ratio / (val_ratio + test_ratio),
                stratify=temp_labels,
                random_state=random_state
            )
            val_df = df[df['DOI'].isin(val_dois)].copy()
            test_df = df[df['DOI'].isin(test_dois)].copy()
        else:
            # If val_ratio is 0, all temp (30%) goes to test
            val_df = pd.DataFrame(columns=df.columns)
            test_df = df[df['DOI'].isin(temp_dois)].copy()
        
        # 3. Map back to dataframe (Already handled for val/test above)
        train_df = df[df['DOI'].isin(train_dois)].copy()
    
    # Drop temporary column
    if 'temp_regime' in train_df.columns:
        train_df = train_df.drop('temp_regime', axis=1)
    if 'temp_regime' in val_df.columns:
        val_df = val_df.drop('temp_regime', axis=1)
    if 'temp_regime' in test_df.columns:
        test_df = test_df.drop('temp_regime', axis=1)
    
    print(f"\nSplit sizes (by samples):")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify no DOI leakage
    train_dois_set = set(train_df['DOI']) if 'DOI' in train_df.columns else set()
    val_dois_set = set(val_df['DOI']) if 'DOI' in val_df.columns else set()
    test_dois_set = set(test_df['DOI']) if 'DOI' in test_df.columns else set()
    
    leakage = train_dois_set.intersection(val_dois_set).union(
              train_dois_set.intersection(test_dois_set)).union(
              val_dois_set.intersection(test_dois_set))
              
    if leakage:
        print(f"CRITICAL WARNING: Data leakage detected! DOIs in multiple splits: {leakage}")
    else:
        print("  [OK] No DOI leakage detected (experiment-level split)")
        
    return train_df, val_df, test_df
