
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
