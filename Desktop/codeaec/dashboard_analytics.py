"""
Dashboard Analytics Module — Additional KPIs and Scenarios

Implements missing features:
1. Vulnerability Ratio (Capital Exposed / Retention Capacity) by zone
2. Geographic Growth Strategy (opportunity zones vs moratoriums)
3. Named Historical Scenarios (Boumerdes M6.5)
4. Building Type Vulnerability Integration
5. Hotspot Identification with Retention Thresholds
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json


# ════════════════════════════════════════════════════════════════
# FEATURE 1: VULNERABILITY RATIO KPI TABLE
# ════════════════════════════════════════════════════════════════

def calculate_zone_vulnerability_ratios(df: pd.DataFrame, empirical_params: Dict) -> pd.DataFrame:
    """
    Calculate vulnerability ratio per zone: Capital Exposé / Capacité Retention.
    
    Retention Capacity = Capital * (1 - loss_probability) for conservative estimate
    Or use insurance industry standard: 1-2% of total capital for retention
    
    Args:
        df: Portfolio dataframe with ZONE_RPA and CAPITAL_ASSURE columns
        empirical_params: Zone-wise loss statistics
    
    Returns:
        DataFrame with columns:
        - Zone
        - Capital_Exposé_B
        - Capacité_Retention_B (estimated max loss in worst-case)
        - Ratio (Capital / Retention)
        - Status (OK / Alert)
    """
    zones = ['Zone 0', 'Zone I', 'Zone IIa', 'Zone IIb', 'Zone III']
    results = []
    
    for zone in zones:
        zone_data = df[df['ZONE_RPA'] == zone]
        if len(zone_data) == 0:
            continue
        
        capital_B = zone_data['CAPITAL_ASSURE'].sum() / 1e9
        
        # Get empirical parameters
        params = empirical_params.get(zone, {})
        pml_B = params.get('total_pml_B', capital_B * 0.3)  # Default to 30% if missing
        loss_prob = params.get('loss_probability', 0.5)
        mean_loss = params.get('mean_loss_ratio', 0.15)
        
        # Retention capacity = PML (Probable Maximum Loss)
        # Conservative: use 95th percentile which is approximately PML
        # Standard insurance approach: retention = capital * (1 - loss_prob) is too optimistic
        # Use empirical PML directly as retention threshold
        retention_B = pml_B
        
        # Vulnerability ratio
        if retention_B > 0:
            ratio = capital_B / retention_B
        else:
            ratio = float('inf')
        
        # Status: Ratio > 3 is alert (capital 3x larger than retention = risky concentration)
        if ratio > 3:
            status = "🔴 Alert"
            color = "red"
        elif ratio > 2:
            status = "🟡 Caution"
            color = "orange"
        else:
            status = "🟢 OK"
            color = "green"
        
        results.append({
            'Zone': zone,
            'Capital_Exposé_B': round(capital_B, 2),
            'Capacité_Retention_B': round(retention_B, 2),
            'Ratio': round(ratio, 2),
            'Status': status,
            'Color': color,
            'Num_Policies': len(zone_data),
            'Mean_Loss_Ratio': round(mean_loss, 3),
            'PML_Percent': round(pml_B / capital_B * 100 if capital_B > 0 else 0, 1)
        })
    
    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════
# FEATURE 2: GEOGRAPHIC GROWTH STRATEGY (Opportunity vs Moratorium)
# ════════════════════════════════════════════════════════════════

def identify_growth_zones(df: pd.DataFrame, empirical_params: Dict) -> Dict:
    """
    Identify geographic growth strategy: where to expand vs where to stop.
    
    Green zones (Opportunity): Zone 0, Zone I, Zone IIa with low concentration
    Red zones (Moratorium): Zone III overconcentrated
    
    Args:
        df: Portfolio dataframe
        empirical_params: Zone-wise statistics
    
    Returns:
        {
            'opportunity_zones': list of zones to grow,
            'moratorium_zones': list of zones to stop,
            'strategic_recommendations': dict of actions
        }
    """
    
    zone_capitals = df.groupby('ZONE_RPA')['CAPITAL_ASSURE'].sum() / 1e9
    total_capital = zone_capitals.sum()
    
    zone_pcts = (zone_capitals / total_capital * 100).to_dict()
    
    opportunity_zones = []
    moratorium_zones = []
    strategic_actions = {}
    
    for zone in ['Zone 0', 'Zone I', 'Zone IIa', 'Zone IIb', 'Zone III']:
        if zone not in zone_pcts:
            continue
        
        pct = zone_pcts[zone]
        params = empirical_params.get(zone, {})
        loss_prob = params.get('loss_probability', 0)
        mean_loss = params.get('mean_loss_ratio', 0)
        
        # Strategy: Opportunity if low risk AND underrepresented (< 15% of portfolio)
        # Moratorium if high risk AND overconcentrated (> 30% of portfolio for high-risk zones)
        if zone == 'Zone 0':
            if pct < 5:
                opportunity_zones.append(zone)
                strategic_actions[zone] = {
                    'action': 'EXPAND',
                    'target_pct': 10,
                    'reason': 'Zero seismic risk',
                    'priority': 'HIGH'
                }
        elif zone == 'Zone I':
            if pct < 20:
                opportunity_zones.append(zone)
                strategic_actions[zone] = {
                    'action': 'EXPAND',
                    'target_pct': 25,
                    'reason': 'Very low seismic probability (0.13%)',
                    'priority': 'HIGH'
                }
        elif zone == 'Zone IIa':
            if pct < 40:
                opportunity_zones.append(zone)
                strategic_actions[zone] = {
                    'action': 'EXPAND',
                    'target_pct': 45,
                    'reason': f'Moderate risk, target {45-pct:.0f}% more growth',
                    'priority': 'MEDIUM'
                }
        elif zone == 'Zone IIb':
            strategic_actions[zone] = {
                'action': 'HOLD',
                'target_pct': pct,
                'reason': 'Maintain current position',
                'priority': 'LOW'
            }
        elif zone == 'Zone III':
            if pct > 30:
                moratorium_zones.append(zone)
                strategic_actions[zone] = {
                    'action': 'REDUCE',
                    'target_pct': 25,
                    'reason': f'Overconcentrated: {pct:.1f}% (target: 25%). Universal loss probability.',
                    'priority': 'CRITICAL'
                }
            else:
                strategic_actions[zone] = {
                    'action': 'HOLD',
                    'target_pct': pct,
                    'reason': 'Within acceptable risk bounds',
                    'priority': 'LOW'
                }
    
    return {
        'opportunity_zones': opportunity_zones,
        'moratorium_zones': moratorium_zones,
        'strategic_recommendations': strategic_actions,
        'current_distribution': zone_pcts
    }


# ════════════════════════════════════════════════════════════════
# FEATURE 3: NAMED HISTORICAL SCENARIOS (Boumerdes M6.5)
# ════════════════════════════════════════════════════════════════

def simulate_boumerdes_scenario(df: pd.DataFrame, empirical_params: Dict) -> Dict:
    """
    Simulate 2003 Boumerdes earthquake (M6.5) impact on current portfolio.
    
    Historical: May 21, 2003 - Magnitude 6.8, killed 2,266 people
    Epicenter: Near Boumerdes (latitude 36.74, longitude 3.65)
    Zone III affected: ALGER, TIPAZA, BOUMERDES
    
    Intensity map approximation:
    - Epicenter zone: Intensity IX (violent) → 85% loss ratio
    - 30km radius: Intensity VIII (severe) → 65% loss ratio
    - 60km radius: Intensity VII (very strong) → 35% loss ratio
    - >60km: Intensity VI → 15% loss ratio
    
    Args:
        df: Portfolio dataframe
        empirical_params: Zone-wise loss statistics
    
    Returns:
        {
            'scenario_name': 'Boumerdes M6.5 (2003 Analogue)',
            'gross_loss_B': Total loss in billions DZD,
            'insured_loss_B': Loss from insured portfolio,
            'pml_coverage_pct': How much of PML this represents,
            'impact_by_zone': Zone-wise breakdown,
            'impact_by_wilaya': Wilaya-wise breakdown (focus on hotspots)
        }
    """
    
    # Define affected zones and intensity zones
    boumerdes_wilayas = {
        'ALGER': {'distance_km': 20, 'intensity': 'IX', 'intensity_damage': 0.85},
        'BOUMERDES': {'distance_km': 0, 'intensity': 'IX', 'intensity_damage': 0.85},
        'TIPAZA': {'distance_km': 40, 'intensity': 'VIII', 'intensity_damage': 0.65},
        'BLIDA': {'distance_km': 50, 'intensity': 'VII', 'intensity_damage': 0.35},
    }
    
    scenario_results = {
        'scenario_name': 'Boumerdes M6.5 (2003 Analogue)',
        'date': 'May 21, 2003',
        'magnitude': 6.8,
        'epicenter': 'Boumerdes (36.74°N, 3.65°E)',
        'total_loss_B': 0,
        'insured_loss_B': 0,
        'by_zone': {},
        'by_wilaya': {}
    }
    
    # Calculate losses
    for zone in ['Zone 0', 'Zone I', 'Zone IIa', 'Zone IIb', 'Zone III']:
        zone_data = df[df['ZONE_RPA'] == zone]
        if len(zone_data) == 0:
            continue
        
        # Base loss from empirical params
        params = empirical_params.get(zone, {})
        base_loss_ratio = params.get('mean_loss_ratio', 0.15)
        
        # For Boumerdes scenario: enhance loss ratio for affected wilayas
        zone_loss = 0
        zone_policies = 0
        
        for wilaya in zone_data['WILAYA'].unique():
            wilaya_data = zone_data[zone_data['WILAYA'] == wilaya]
            wilaya_capital = wilaya_data['CAPITAL_ASSURE'].sum()
            
            # Apply intensity modification if affected
            if wilaya in boumerdes_wilayas:
                intensity_damage = boumerdes_wilayas[wilaya]['intensity_damage']
                # Boumerdes is in Zone III, so assume vulnerability = 1.0
                loss_ratio = intensity_damage * 1.0  # Full vulnerability
            else:
                # Unaffected areas: use baseline (10% of normal, or assume unaffected)
                loss_ratio = base_loss_ratio * 0.1
            
            wilaya_loss = wilaya_capital * loss_ratio
            zone_loss += wilaya_loss
            zone_policies += len(wilaya_data)
            
            if wilaya not in scenario_results['by_wilaya']:
                scenario_results['by_wilaya'][wilaya] = {
                    'loss_ratio': 0,
                    'loss_B': 0,
                    'capital_B': 0,
                    'policies': 0
                }
            
            scenario_results['by_wilaya'][wilaya]['capital_B'] += wilaya_capital / 1e9
            scenario_results['by_wilaya'][wilaya]['loss_B'] += wilaya_loss / 1e9
            scenario_results['by_wilaya'][wilaya]['policies'] += len(wilaya_data)
        
        scenario_results['by_zone'][zone] = {
            'loss_B': zone_loss / 1e9,
            'capital_B': zone_data['CAPITAL_ASSURE'].sum() / 1e9,
            'policies': zone_policies,
            'loss_ratio': (zone_loss / zone_data['CAPITAL_ASSURE'].sum()) if len(zone_data) > 0 else 0
        }
        
        scenario_results['insured_loss_B'] += zone_loss / 1e9
    
    # Calculate PML coverage (what % of total PML is this event)
    total_pml = sum([v.get('total_pml_B', 0) for v in empirical_params.values()])
    scenario_results['pml_coverage_pct'] = round(
        (scenario_results['insured_loss_B'] / total_pml * 100) if total_pml > 0 else 0, 1
    )
    
    # Reinsurance impact (assume 3-layer program from Phase 4)
    # Layer 1: Ground-up (own retention 1B)
    # Layer 2: 1B-10B xs 1B (reinsured)
    # Layer 3: 10B+ xs 10B (reinsured)
    net_loss = scenario_results['insured_loss_B']
    if net_loss > 10:
        net_loss = 1 + min(9, net_loss - 1)  # Cap at 10B
    
    scenario_results['net_loss_after_reinsurance_B'] = round(net_loss, 2)
    scenario_results['impact_statement'] = (
        f"Boumerdes M6.5 simulation: {scenario_results['insured_loss_B']:.1f}B DZD insured loss "
        f"({scenario_results['pml_coverage_pct']:.0f}% of annual PML), "
        f"{scenario_results['net_loss_after_reinsurance_B']:.1f}B DZD net loss after reinsurance."
    )
    
    return scenario_results


# ════════════════════════════════════════════════════════════════
# FEATURE 4: BUILDING TYPE VULNERABILITY INTEGRATION
# ════════════════════════════════════════════════════════════════

def calculate_building_type_vulnerability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment portfolio by building type vulnerability.
    
    Uses TYPE field and applies VulnerabilityModel multipliers.
    
    Returns:
        DataFrame with building type risk breakdown
    """
    from advanced_analytics import VulnerabilityModel
    
    building_type_stats = []
    
    for btype in df['TYPE'].unique():
        if pd.isna(btype):
            continue
        
        btype_data = df[df['TYPE'] == btype]
        mult = VulnerabilityModel.BUILDING_TYPES.get(
            btype, VulnerabilityModel.BUILDING_TYPES['Unknown']
        )['multiplier']
        
        capital = btype_data['CAPITAL_ASSURE'].sum() / 1e9
        pml = btype_data['PML_EXPOSE'].sum() / 1e9
        num_policies = len(btype_data)
        avg_pml_ratio = pml / capital if capital > 0 else 0
        
        # Adjusted PML with vulnerability multiplier
        adjusted_pml = pml * mult
        
        building_type_stats.append({
            'Building_Type': btype,
            'Vulnerability_Multiplier': round(mult, 2),
            'Num_Policies': num_policies,
            'Capital_B': round(capital, 2),
            'PML_B': round(pml, 2),
            'Adjusted_PML_B': round(adjusted_pml, 2),
            'PML_Ratio': round(avg_pml_ratio, 3),
            'Risk_Level': 'HIGH' if mult > 1.0 else 'MEDIUM' if mult > 0.8 else 'LOW'
        })
    
    return pd.DataFrame(building_type_stats).sort_values('Adjusted_PML_B', ascending=False)


# ════════════════════════════════════════════════════════════════
# FEATURE 5: HOTSPOT IDENTIFICATION WITH RETENTION THRESHOLDS
# ════════════════════════════════════════════════════════════════

def identify_hotspots_with_retention(
    df: pd.DataFrame,
    hotspots_csv: str = 'hotspots_communes_precise.csv',
    retention_capacity_B: float = 1.0
) -> pd.DataFrame:
    """
    Identify hotspots (communes where capital > retention capacity).
    
    A hotspot is defined as: cumul capital dépasse la capacité de rétention
    Retention capacity = estimated maximum loss the company can sustain
    
    Args:
        df: Portfolio dataframe
        hotspots_csv: Path to hotspots CSV
        retention_capacity_B: Max loss in billions DZD
    
    Returns:
        DataFrame with hotspot classification
    """
    try:
        hotspots = pd.read_csv(hotspots_csv, low_memory=False)
    except:
        # Fallback: calculate from df
        hotspots = df.groupby(['WILAYA', 'COMMUNE', 'ZONE_RPA']).agg({
            'CAPITAL_ASSURE': 'sum',
            'PML_EXPOSE': 'sum',
        }).reset_index()
        hotspots.columns = ['WILAYA', 'COMMUNE', 'ZONE_RPA', 'capital_B', 'pml_B']
        hotspots['capital_B'] = hotspots['capital_B'] / 1e9
        hotspots['pml_B'] = hotspots['pml_B'] / 1e9
    
    # Ensure capital_B column exists
    if 'capital_B' not in hotspots.columns:
        hotspots['capital_B'] = hotspots.get('capital_B', 0)
    
    # Calculate if commune exceeds retention
    hotspots['exceeds_retention'] = hotspots['capital_B'] > retention_capacity_B
    
    # Classify hotspots
    def classify_hotspot(row):
        if row['capital_B'] > retention_capacity_B * 2:
            return '🔴 CRITICAL'
        elif row['capital_B'] > retention_capacity_B * 1.5:
            return '🟠 HIGH'
        elif row['capital_B'] > retention_capacity_B:
            return '🟡 MEDIUM'
        else:
            return '🟢 OK'
    
    hotspots['Hotspot_Status'] = hotspots.apply(classify_hotspot, axis=1)
    hotspots['Retention_Threshold_B'] = retention_capacity_B
    hotspots['Excess_Over_Retention_B'] = (hotspots['capital_B'] - retention_capacity_B).clip(lower=0)
    
    return hotspots.sort_values('capital_B', ascending=False)
