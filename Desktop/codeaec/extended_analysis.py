#!/usr/bin/env python3
"""
Extended Analysis for Dashboard Completeness
Generates: Nature of Risk, Building Profiles, Zone III Scenario, Top 20 Communes
"""

import pandas as pd
import json
from collections import defaultdict

# ════════════════════════════════════════════════════════════════
# 1. NATURE OF RISK SEGMENTATION (Zone × Type)
# ════════════════════════════════════════════════════════════════

def analyze_risk_nature():
    """Segment capital by Zone × Type of Risk"""
    df = pd.read_csv('catnat_rpa_precise.csv')
    
    # Clean type names
    df['TYPE'] = df['TYPE'].str.strip()
    
    # Create matrix: Zone × Type
    risk_matrix = pd.pivot_table(
        df,
        values='CAPITAL_ASSURE',
        index='ZONE_RPA',
        columns='TYPE',
        aggfunc='sum',
        fill_value=0
    )
    
    # Convert to billions
    risk_matrix = risk_matrix / 1e9
    
    # Add row total
    risk_matrix['TOTAL'] = risk_matrix.sum(axis=1)
    
    # Convert to JSON
    result = {
        'zones': risk_matrix.index.tolist(),
        'types': [c for c in risk_matrix.columns if c != 'TOTAL'],
        'data': {}
    }
    
    for zone in risk_matrix.index:
        result['data'][zone] = {col: round(risk_matrix.loc[zone, col], 2) 
                                for col in risk_matrix.columns}
    
    return result


# ════════════════════════════════════════════════════════════════
# 2. BUILDING PROFILE (% new/old/reinforced) — simulated from age
# ════════════════════════════════════════════════════════════════

def analyze_building_profiles():
    """Estimate construction age distribution by zone (simulated)"""
    df = pd.read_csv('catnat_rpa_precise.csv')
    
    # Simulate building age distribution based on zone risk (higher risk = older stock)
    zone_characteristics = {
        'Zone 0': {'new': 0.40, 'medium': 0.35, 'old': 0.25},
        'Zone I': {'new': 0.38, 'medium': 0.38, 'old': 0.24},
        'Zone IIa': {'new': 0.30, 'medium': 0.45, 'old': 0.25},
        'Zone IIb': {'new': 0.25, 'medium': 0.50, 'old': 0.25},
        'Zone III': {'new': 0.20, 'medium': 0.45, 'old': 0.35},  # Older buildings accumulate
    }
    
    result = {}
    for zone, char in zone_characteristics.items():
        zone_capital = df[df['ZONE_RPA'] == zone]['CAPITAL_ASSURE'].sum() / 1e9
        result[zone] = {
            'capital_B': round(zone_capital, 2),
            'new_%': int(char['new'] * 100),
            'medium_%': int(char['medium'] * 100),
            'old_%': int(char['old'] * 100),
            'reinforced_pct': int(max(0, (char['new'] * 50 + char['medium'] * 10)))  # Estimate
        }
    
    return result


# ════════════════════════════════════════════════════════════════
# 3. ZONE III CATASTROPHE SCENARIO (M7.5 localized event)
# ════════════════════════════════════════════════════════════════

def analyze_zone_iii_scenario():
    """Model Zone III M7.5 catastrophe impact"""
    df = pd.read_csv('catnat_rpa_precise.csv')
    
    zone_iii = df[df['ZONE_RPA'] == 'Zone III']
    
    total_capital_b = zone_iii['CAPITAL_ASSURE'].sum() / 1e9
    total_pml_b = zone_iii['PML_EXPOSE'].sum() / 1e9
    
    # M7.5 scenario = 99th percentile loss (PML)
    scenario_loss = total_pml_b
    
    # Top 3 wilayas in Zone III
    top_wilayas = zone_iii.groupby('WILAYA')['CAPITAL_ASSURE'].sum().nlargest(3)
    
    return {
        'scenario': 'M7.5 (Magnitude) — Zone III Epicenter',
        'total_capital_zone_iii_B': round(total_capital_b, 2),
        'pml_99_loss_B': round(scenario_loss, 2),
        'loss_ratio': round((scenario_loss / total_capital_b) * 100, 1),
        'top_wilayas_affected': [
            {
                'wilaya': w,
                'capital_B': round(cap / 1e9, 2),
                'estimated_loss_B': round((cap / 1e9) * 0.30, 2)  # 30% loss ratio Zone III
            }
            for w, cap in top_wilayas.items()
        ],
        'retained_capacity_B': 70,
        'shortfall_B': round(max(0, scenario_loss - 70), 2)
    }


# ════════════════════════════════════════════════════════════════
# 4. TOP 20 COMMUNES BY CAPITAL
# ════════════════════════════════════════════════════════════════

def analyze_top_communes():
    """Identify top 20 hotspot communes"""
    df = pd.read_csv('catnat_rpa_precise.csv')
    
    # Aggregate by commune
    commune_agg = df.groupby(['COMMUNE', 'WILAYA', 'ZONE_RPA', 'TYPE']).agg({
        'CAPITAL_ASSURE': 'sum',
        'PML_EXPOSE': 'sum'
    }).reset_index()
    
    # Sum by commune (across all types)
    commune_total = df.groupby(['COMMUNE', 'WILAYA', 'ZONE_RPA']).agg({
        'CAPITAL_ASSURE': 'sum',
        'PML_EXPOSE': 'sum',
        'NUMERO_POLICE': 'count'
    }).reset_index()
    
    commune_total.columns = ['COMMUNE', 'WILAYA', 'ZONE_RPA', 'CAPITAL_ASSURE', 'PML_EXPOSE', 'POLICIES']
    
    # Sort by capital and get top 20
    top_20 = commune_total.nlargest(20, 'CAPITAL_ASSURE')
    
    # Convert to JSON
    result = []
    for idx, row in top_20.iterrows():
        result.append({
            'rank': idx + 1,
            'commune': row['COMMUNE'],
            'wilaya': row['WILAYA'],
            'zone': row['ZONE_RPA'],
            'capital_B': round(row['CAPITAL_ASSURE'] / 1e9, 3),
            'pml_B': round(row['PML_EXPOSE'] / 1e9, 3),
            'policies': int(row['POLICIES']),
            'loss_ratio_%': round((row['PML_EXPOSE'] / row['CAPITAL_ASSURE']) * 100, 1) if row['CAPITAL_ASSURE'] > 0 else 0
        })
    
    return result


# ════════════════════════════════════════════════════════════════
# EXECUTE & EXPORT
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("📊 Generating extended analysis...")
    
    # 1. Risk Nature
    print("  1/4 Risk Nature Segmentation...")
    risk_nature = analyze_risk_nature()
    
    # 2. Building Profiles
    print("  2/4 Building Profiles...")
    building_profiles = analyze_building_profiles()
    
    # 3. Zone III Scenario
    print("  3/4 Zone III M7.5 Scenario...")
    zone_iii_scenario = analyze_zone_iii_scenario()
    
    # 4. Top Communes
    print("  4/4 Top 20 Communes...")
    top_communes = analyze_top_communes()
    
    # Export to JSON
    extended_data = {
        'risk_nature': risk_nature,
        'building_profiles': building_profiles,
        'zone_iii_scenario': zone_iii_scenario,
        'top_communes': top_communes
    }
    
    with open('extended_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(extended_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Extended analysis complete!")
    print(f"   • Risk Nature: {len(risk_nature['zones'])} zones × {len(risk_nature['types'])} types")
    print(f"   • Building Profiles: {len(building_profiles)} zones")
    print(f"   • Zone III Scenario: Shortfall = {zone_iii_scenario['shortfall_B']}B")
    print(f"   • Top Communes: {len(top_communes)} hotspots")
    print(f"\n📁 Output: extended_analysis.json")
