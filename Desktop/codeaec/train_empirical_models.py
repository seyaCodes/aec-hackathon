"""
Empirical Parameter Extraction & CatBoost Model Training

This script:
1. Calculates data-derived Monte Carlo parameters by zone
2. Trains CatBoost classifier for underwriting decisions
3. Exports everything for dashboard integration
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
import json

print("=" * 80)
print("EMPIRICAL PARAMETER EXTRACTION & MODEL TRAINING")
print("=" * 80)

# Load portfolio data
try:
    df = pd.read_csv('gam_master_data.csv', low_memory=False)
    for col in ['CAPITAL_ASSURE', 'PRIME_NETTE', 'PML_EXPOSE', 'RISK_SCORE']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"\n[OK] Loaded {len(df):,} policies from portfolio")
except Exception as e:
    print(f"[ERROR] Could not load portfolio: {e}")
    exit(1)

# ════════════════════════════════════════════════════════════════
# STEP 1: CALCULATE EMPIRICAL LOSS RATIOS BY ZONE
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("STEP 1: EMPIRICAL MONTE CARLO PARAMETERS BY ZONE")
print("=" * 80)

# Calculate loss ratio for each policy
df['loss_ratio'] = df['PML_EXPOSE'] / df['CAPITAL_ASSURE']
df['loss_ratio'] = df['loss_ratio'].fillna(0).clip(0, 1)  # 0-100%

# Group by zone
zone_stats = df.groupby('ZONE_RPA', dropna=False).agg({
    'CAPITAL_ASSURE': 'sum',
    'PML_EXPOSE': 'sum',
    'loss_ratio': ['mean', 'std', 'count'],
    'NUMERO_POLICE': 'count'
}).reset_index()

zone_stats.columns = ['zone', 'capital_sum', 'pml_sum', 'loss_mean', 'loss_std', 'loss_count', 'policy_count']

# Calculate probability (% of policies with significant loss)
df['has_significant_loss'] = df['loss_ratio'] > 0.05  # loss > 5% of capital
loss_prob = df.groupby('ZONE_RPA', dropna=False)['has_significant_loss'].mean().reset_index()
loss_prob.columns = ['zone', 'loss_probability']

zone_stats = zone_stats.merge(loss_prob, on='zone', how='left')

print("\nZone Statistics:")
print(zone_stats.to_string())

# Export as JSON for dashboard
monte_carlo_params = {}
for _, row in zone_stats.iterrows():
    zone = row['zone'] if pd.notna(row['zone']) else 'Unknown'
    monte_carlo_params[zone] = {
        'total_capital_B': round(row['capital_sum'] / 1e9, 2),
        'total_pml_B': round(row['pml_sum'] / 1e9, 2),
        'loss_probability': round(row['loss_probability'], 4),  # Prob of loss > 5%
        'mean_loss_ratio': round(row['loss_mean'], 4),          # Mean loss as % of capital
        'std_loss_ratio': round(row['loss_std'], 4),            # Std dev of losses
        'num_policies': int(row['policy_count'])
    }

with open('empirical_monte_carlo_params.json', 'w') as f:
    json.dump(monte_carlo_params, f, indent=2)
    print(f"\n[OK] Saved: empirical_monte_carlo_params.json")

# ════════════════════════════════════════════════════════════════
# STEP 2: TRAIN CATBOOST UNDERWRITING CLASSIFIER
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("STEP 2: TRAIN CATBOOST UNDERWRITING CLASSIFIER")
print("=" * 80)

# Load decision output to get ground truth labels
try:
    decisions_df = pd.read_csv('decision_output_final.csv')
    print(f"[OK] Loaded {len(decisions_df):,} decision labels")
except Exception as e:
    print(f"[WARNING] Could not load decisions, creating synthetic labels...")
    # Create synthetic labels based on heuristics
    df['label'] = 'ACCEPT'
    df.loc[df['ZONE_RPA'] == 'Zone III', 'label'] = 'REJECT'
    df.loc[(df['ZONE_RPA'] == 'Zone IIb') & (df['CAPITAL_ASSURE'] > 2e9), 'label'] = 'ADJUST'
    df.loc[df['loss_ratio'] > 0.15, 'label'] = 'ADJUST'
    decisions_df = None

# Merge decisions with portfolio if available
if decisions_df is not None:
    # Decisions are by commune, so we need to map
    df_train = df.merge(
        decisions_df[['COMMUNE', 'ZONE_RPA', 'DECISION']],
        on=['COMMUNE', 'ZONE_RPA'],
        how='left'
    )
    df_train['label'] = df_train['DECISION'].fillna('ACCEPT')
else:
    df_train = df[['COMMUNE', 'WILAYA', 'ZONE_RPA', 'TYPE', 'CAPITAL_ASSURE', 'loss_ratio', 'label']].copy()

# Remove rows without labels
df_train = df_train.dropna(subset=['label'])

print(f"\nTraining set: {len(df_train):,} policies")
print(f"Label distribution:\n{df_train['label'].value_counts()}")

# Prepare features
feature_cols = ['CAPITAL_ASSURE', 'loss_ratio', 'WILAYA', 'COMMUNE', 'TYPE']
cat_cols = ['WILAYA', 'COMMUNE', 'TYPE']

X = df_train[feature_cols].copy()
y = df_train['label'].copy()

# Fill missing categories
for col in cat_cols:
    X[col] = X[col].fillna('Unknown')

# Encode labels to numeric
label_mapping = {'ACCEPT': 0, 'ADJUST': 1, 'REJECT': 2}
y_numeric = y.map(label_mapping)

print(f"\nTraining CatBoost classifier...")
print(f"Features: {feature_cols}")
print(f"Categorical: {cat_cols}")

# Train CatBoost
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_cols,
    verbose=False,
    random_state=42
)

model.fit(X, y_numeric)

# Save model
model.save_model('catboost_underwriting_model.cbm')
print(f"[OK] Model saved: catboost_underwriting_model.cbm")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.get_feature_importance()
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:\n{feature_importance.to_string(index=False)}")

# Save label mapping for app
with open('catboost_label_mapping.json', 'w') as f:
    json.dump({
        'label_mapping': label_mapping,
        'reverse_mapping': {v: k for k, v in label_mapping.items()},
        'feature_importance': feature_importance.to_dict('records')
    }, f, indent=2)
    print(f"\n[OK] Saved: catboost_label_mapping.json")

# ════════════════════════════════════════════════════════════════
# STEP 3: EVALUATE MODEL ON PORTFOLIO
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("STEP 3: MODEL EVALUATION")
print("=" * 80)

# Predict on training data
y_pred_numeric = model.predict(X)
if y_pred_numeric.ndim > 1:
    y_pred_numeric = y_pred_numeric.flatten()
y_pred = pd.Series(y_pred_numeric).map({v: k for k, v in label_mapping.items()})

accuracy = (y_pred.values == y.values).mean()
print(f"\nAccuracy: {accuracy:.1%}")

# Per-class metrics
print(f"\nPer-Class Breakdown:")
for label in ['ACCEPT', 'ADJUST', 'REJECT']:
    count = (y_pred == label).sum()
    total = len(y_pred)
    pct = count / total * 100 if total > 0 else 0
    print(f"  {label}: {count:,} policies ({pct:.1f}%)")

# ════════════════════════════════════════════════════════════════
# STEP 4: CREATE INFERENCE WRAPPER FOR DASHBOARD
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("STEP 4: CREATE INFERENCE FUNCTION")
print("=" * 80)

inference_code = '''
import pandas as pd
import json
from catboost import CatBoostClassifier

def load_underwriting_model():
    """Load trained CatBoost model and metadata"""
    model = CatBoostClassifier()
    model.load_model('catboost_underwriting_model.cbm')
    with open('catboost_label_mapping.json', 'r') as f:
        metadata = json.load(f)
    return model, metadata

def evaluate_policy(commune, capital_B, wilaya, policy_type):
    """
    Evaluate a new policy using CatBoost.
    
    Args:
        commune: Commune name
        capital_B: Capital insured in billions DZD
        wilaya: Wilaya name
        policy_type: Policy type (Residential, Commercial, etc.)
    
    Returns:
        {
            'decision': 'ACCEPT' | 'ADJUST' | 'REJECT',
            'confidence': 0-1,
            'reasoning': explanation
        }
    """
    model, metadata = load_underwriting_model()
    
    # Load portfolio to estimate loss ratio
    df_port = pd.read_csv('gam_master_data.csv', low_memory=False)
    df_port['loss_ratio'] = pd.to_numeric(df_port['PML_EXPOSE'], errors='coerce') / pd.to_numeric(df_port['CAPITAL_ASSURE'], errors='coerce')
    
    # Estimate loss ratio from similar policies
    similar = df_port[df_port['COMMUNE'] == commune]
    if len(similar) > 0:
        loss_ratio = similar['loss_ratio'].median() or 0.1
    else:
        # Use zone average
        similar = df_port[df_port['ZONE_RPA'] == df_port[df_port['COMMUNE']==commune]['ZONE_RPA'].values[0]] if len(df_port[df_port['COMMUNE']==commune]) > 0 else df_port
        loss_ratio = similar['loss_ratio'].median() or 0.1
    
    # Prepare input
    X_test = pd.DataFrame({
        'CAPITAL_ASSURE': [capital_B * 1e9],
        'loss_ratio': [loss_ratio],
        'WILAYA': [wilaya],
        'COMMUNE': [commune],
        'TYPE': [policy_type]
    })
    
    # Predict
    pred_class = model.predict(X_test)[0]
    pred_proba = model.predict_proba(X_test)[0]
    
    label_map = metadata['reverse_mapping']
    decision = label_map[str(pred_class)]
    confidence = float(pred_proba[pred_class])
    
    # Generate reasoning
    reasoning_map = {
        'REJECT': f"High risk: {commune} in high-risk zone. Exposure already concentrated.",
        'ADJUST': f"Medium risk: {commune} requires premium adjustment (+15-25%). Monitor concentration.",
        'ACCEPT': f"Low risk: {commune} is suitable. Standard underwriting applies."
    }
    
    return {
        'decision': decision,
        'confidence': round(confidence, 2),
        'reasoning': reasoning_map[decision],
        'loss_ratio_estimate': round(loss_ratio, 3)
    }
'''

with open('underwriting_inference.py', 'w') as f:
    f.write(inference_code)
    print("[OK] Created: underwriting_inference.py")

print("\n" + "=" * 80)
print("[SUCCESS] ALL STEPS COMPLETE")
print("=" * 80)
print("""
Generated Files:
  1. empirical_monte_carlo_params.json - Data-derived Monte Carlo parameters by zone
  2. catboost_underwriting_model.cbm - Trained classifier for ACCEPT/ADJUST/REJECT
  3. catboost_label_mapping.json - Label mapping and feature importance
  4. underwriting_inference.py - Inference wrapper for dashboard

Next Steps:
  1. Update app.py to use empirical_monte_carlo_params.json in Monte Carlo
  2. Integrate CatBoost model in AI Underwriting Assistant section
  3. Test dashboard with new data-driven parameters
""")
