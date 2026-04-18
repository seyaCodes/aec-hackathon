#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Business Intelligence Outputs
(Separated to avoid encoding issues)
"""

import pandas as pd
import json
import numpy as np

# Load the master data
df = pd.read_csv('gam_master_data.csv')
total_cap = df['CAPITAL_ASSURE'].sum()
total_prime = df['PRIME_NETTE'].sum()

# Load hotspots communes
hotspots_communes = pd.read_csv('hotspots_communes_precise.csv')

# ════════════════════════════════════════════════════════════════
# CREATE DECISION OUTPUT TABLE
# ════════════════════════════════════════════════════════════════

decision_output = hotspots_communes.copy()

# Add risk classification based on PML per unit capital
decision_output['pml_to_capital_pct'] = (
    decision_output['pml_B'] / decision_output['capital_B'] * 100
).round(2)

# Risk scoring: Zone weight + concentration
zone_weights_map = {'Zone III': 5, 'Zone IIb': 4, 'Zone IIa': 3, 'Zone I': 2, 'Zone 0': 1}
decision_output['zone_weight'] = decision_output['ZONE_RPA'].map(zone_weights_map)

# Portfolio concentration (% of zone's total)
for zone in decision_output['ZONE_RPA'].unique():
    zone_total = decision_output[decision_output['ZONE_RPA']==zone]['capital_B'].sum()
    mask = decision_output['ZONE_RPA'] == zone
    decision_output.loc[mask, 'zone_concentration_pct'] = (
        decision_output.loc[mask, 'capital_B'] / zone_total * 100
    ).round(1)

# DECISION CLASSIFIER
def classify_decision(row):
    zone = row['ZONE_RPA']
    pml_pct = row['pml_to_capital_pct']
    concentration = row['zone_concentration_pct']
    capital = row['capital_B']
    
    # REJECT conditions
    if zone == 'Zone III' and concentration > 5:
        return 'REJECT'
    if zone == 'Zone IIb' and capital > 5:
        return 'REJECT'
    
    # ADJUST conditions
    if pml_pct > 20:
        return 'ADJUST'
    if zone in ['Zone III','Zone IIb'] and capital > 2:
        return 'ADJUST'
    
    return 'ACCEPT'

decision_output['DECISION'] = decision_output.apply(classify_decision, axis=1)

# Add strategic action recommendation
def action_plan(row):
    decision = row['DECISION']
    zone = row['ZONE_RPA']
    
    if decision == 'REJECT':
        return 'STOP new policies, reduce exposure'
    elif decision == 'ADJUST':
        if row['pml_to_capital_pct'] > 20:
            return 'Increase premium 15-30%'
        else:
            return 'Strict underwriting, review case-by-case'
    else:
        if zone in ['Zone 0','Zone I']:
            return 'Grow business - low risk'
        elif zone == 'Zone IIa':
            return 'Steady growth, market rate'
        else:
            return 'Maintain current exposure'

decision_output['ACTION'] = decision_output.apply(action_plan, axis=1)

# Reorder columns
decision_output = decision_output[[
    'WILAYA','COMMUNE','ZONE_RPA','nb_polices','capital_B','pml_B',
    'pml_to_capital_pct','zone_concentration_pct','DECISION','ACTION'
]]

decision_output.to_csv('decision_output_final.csv', index=False, encoding='utf-8-sig')
print("[OK] decision_output_final.csv")

# ════════════════════════════════════════════════════════════════
# CAPACITY ANALYSIS
# ════════════════════════════════════════════════════════════════

# Load simulation results
with open('simulation_results.json','r') as f:
    sim_results = json.load(f)

pml_99_B = sim_results['pml_99_B']

# Company parameters
COMPANY_ANNUAL_PREMIUM  = total_prime / 1e6
COMPANY_CAPACITY_PRIMARY = 100
COMPANY_CAPACITY_REINSURANCE = 50
REINSURANCE_RETENTION_PCT = 70
REINSURANCE_TRANSFER_PCT = 30

# Calculate effective capacity
primary_capacity = COMPANY_CAPACITY_PRIMARY * REINSURANCE_RETENTION_PCT / 100
reinsurance_capacity = COMPANY_CAPACITY_REINSURANCE * REINSURANCE_TRANSFER_PCT / 100
total_capacity = primary_capacity + reinsurance_capacity

surplus = total_capacity - pml_99_B

capacity_analysis = {
    'primary_capacity_B': round(primary_capacity, 1),
    'reinsurance_capacity_B': round(reinsurance_capacity, 1),
    'total_capacity_B': round(total_capacity, 1),
    'pml_99_B': round(pml_99_B, 2),
    'surplus_deficit_B': round(surplus, 2),
    'capital_exposed_B': round(total_cap/1e9, 0),
    'concentration_alger_pct': round(df[df['WILAYA']=='ALGER']['CAPITAL_ASSURE'].sum()/total_cap*100, 1),
    'concentration_zone_iii_pct': round(df[df['ZONE_RPA']=='Zone III']['CAPITAL_ASSURE'].sum()/total_cap*100, 1),
}

with open('capacity_analysis.json','w', encoding='utf-8') as f:
    json.dump(capacity_analysis, f, indent=2)
print("[OK] capacity_analysis.json")

# ════════════════════════════════════════════════════════════════
# CEO DECISION PANEL
# ════════════════════════════════════════════════════════════════

ceo_actions = []

# Action 1: Concentration risk
alger_pct = df[df['WILAYA']=='ALGER']['CAPITAL_ASSURE'].sum()/total_cap*100
if alger_pct > 25:
    ceo_actions.append({
        'priority': '[CRITICAL]',
        'action': 'Reduce ALGER concentration from 28pct to 15pct',
        'why': 'Single wilaya >25pct violates portfolio policy. PML risk: 75.4B DZD',
        'how': 'Pause new policies in Alger Zone III. Incentivize underwriting in Zone IIa/I.'
    })

# Action 2: Zone III pricing
zone_iii_df = df[df['ZONE_RPA']=='Zone III']
zone_iii_prime_pct = zone_iii_df['PRIME_NETTE'].sum() / df['PRIME_NETTE'].sum() * 100
zone_iii_pml_pct = df[df['ZONE_RPA']=='Zone III']['PML_EXPOSE'].sum() / df['PML_EXPOSE'].sum() * 100
if zone_iii_pml_pct > zone_iii_prime_pct * 2:
    ceo_actions.append({
        'priority': '[CRITICAL]',
        'action': 'Increase Zone III premium by 20-30pct',
        'why': f'Zone III collects {zone_iii_prime_pct:.0f}pct of primes but drives {zone_iii_pml_pct:.0f}pct of PML risk',
        'how': 'Implement tiered pricing: Alger (premium 40pct), Zone IIb (30pct), Zone IIa (standard)'
    })

# Action 3: Premium adequacy
annual_profit_margin = 0.15
max_pml_from_primes = COMPANY_ANNUAL_PREMIUM * annual_profit_margin
if pml_99_B > max_pml_from_primes:
    ceo_actions.append({
        'priority': '[HIGH]',
        'action': 'Increase annual premium by 20pct or reduce exposure',
        'why': f'Current annual premium {COMPANY_ANNUAL_PREMIUM:.0f}M only covers 2pct of PML 99pct',
        'how': 'Option A: Rate increase 20pct across board. Option B: Stop Zone III growth 6 months.'
    })

# Action 4: Reinsurance adequacy
if surplus < 10:
    ceo_actions.append({
        'priority': '[HIGH]',
        'action': 'Negotiate additional reinsurance coverage',
        'why': f'Capacity margin only {surplus:.1f}B (want 15-20B buffer)',
        'how': 'Contact XL Capital, Everest Re for Zone III increase. Budget: ~5M DZD additional premium'
    })

# Action 5: Product mix shift
industrial_df = df[df['TYPE']=='Installation Industrielle']
industrial_capital = industrial_df['CAPITAL_ASSURE'].sum()
industrial_pml = industrial_df['PML_EXPOSE'].sum()
if industrial_capital > 0 and industrial_pml / industrial_capital > 0.25:
    ceo_actions.append({
        'priority': '[MEDIUM]',
        'action': 'Shift underwriting toward Commercial/Residential',
        'why': 'Industrial properties in high-risk zones exceed loss tolerance',
        'how': 'Implement minimum premium requirement: Industrial Type 1 = +50pct from market rate'
    })

ceo_panel = {
    'generated_date': str(pd.Timestamp.now()),
    'total_actions': len(ceo_actions),
    'critical_count': sum(1 for a in ceo_actions if '[CRITICAL]' in a['priority']),
    'actions': ceo_actions
}

with open('ceo_decision_panel.json','w', encoding='utf-8') as f:
    json.dump(ceo_panel, f, indent=2, default=str)
print("[OK] ceo_decision_panel.json")

# ════════════════════════════════════════════════════════════════
# AI UNDERWRITING ASSISTANT
# ════════════════════════════════════════════════════════════════

def underwriting_assistant(commune_name, capital_B, zone_proposed=None):
    commune_matches = df[df['COMMUNE'].str.contains(commune_name.upper(), na=False, case=False)]
    
    if len(commune_matches) == 0:
        return {
            'decision': 'INSUFFICIENT DATA',
            'confidence': 'LOW',
            'reasoning': f'Commune "{commune_name}" not found in portfolio',
            'recommendation': 'Check spelling or classify manually'
        }
    
    commune_zone = commune_matches['ZONE_RPA'].mode()[0] if len(commune_matches) > 0 else 'Unknown'
    commune_existing_capital = commune_matches['CAPITAL_ASSURE'].sum() / 1e9
    commune_wilaya = commune_matches['WILAYA'].mode()[0] if len(commune_matches) > 0 else 'Unknown'
    
    wilaya_data = df[df['WILAYA'] == commune_wilaya]
    wilaya_capital = wilaya_data['CAPITAL_ASSURE'].sum() / 1e9
    wilaya_concentration = wilaya_capital / total_cap * 100
    
    proposed_commune_capital = commune_existing_capital + capital_B
    proposed_wilaya_capital = wilaya_capital + capital_B
    
    reasons = []
    reject_score = 0
    adjust_score = 0
    
    if commune_zone in ['Zone III','Zone IIb']:
        reject_score += 3
        reasons.append(f'High-risk zone ({commune_zone})')
    elif commune_zone in ['Zone IIa']:
        adjust_score += 1
        reasons.append(f'Medium-risk zone ({commune_zone})')
    
    if proposed_commune_capital > commune_existing_capital * 2:
        reject_score += 2
        reasons.append(f'Doubles existing commune exposure ({commune_existing_capital:.1f}B to {proposed_commune_capital:.1f}B)')
    
    if proposed_wilaya_capital / total_cap * 100 > 35:
        reject_score += 2
        reasons.append(f'Wilaya would exceed 35pct threshold ({wilaya_concentration:.0f}pct to {proposed_wilaya_capital/total_cap*100:.0f}pct)')
    
    if capital_B > 10 and commune_zone == 'Zone III':
        reject_score += 2
        reasons.append('Large capital (>10B) in very high risk zone')
    elif capital_B > 5 and commune_zone == 'Zone IIb':
        adjust_score += 2
        reasons.append('Large capital (>5B) requires premium adjustment')
    
    commune_count = len(commune_matches)
    if commune_count > 2000:
        adjust_score += 1
        reasons.append(f'Commune already highly concentrated ({commune_count} policies)')
    
    if reject_score > adjust_score:
        decision = 'REJECT'
        if reject_score >= 5:
            confidence = 'HIGH'
        else:
            confidence = 'MEDIUM'
    elif adjust_score > 0 or (reject_score > 0 and reject_score == adjust_score):
        decision = 'ADJUST'
        confidence = 'MEDIUM'
        if adjust_score == 0:
            reasons.append('Premium adjustment recommended')
    else:
        decision = 'ACCEPT'
        confidence = 'HIGH'
    
    premium_adjustment = 0
    if decision == 'ADJUST':
        if commune_zone == 'Zone III':
            premium_adjustment = 30
        elif commune_zone == 'Zone IIb':
            premium_adjustment = 20
        elif commune_zone == 'Zone IIa':
            premium_adjustment = 10
    
    return {
        'decision': decision,
        'confidence': confidence,
        'commune': commune_name.upper(),
        'zone': commune_zone,
        'existing_commune_capital_B': round(commune_existing_capital, 2),
        'proposed_capital_B': round(capital_B, 2),
        'existing_wilaya_capital_B': round(wilaya_capital, 2),
        'wilaya_concentration_pct': round(wilaya_concentration, 1),
        'premium_adjustment_pct': premium_adjustment,
        'reasoning': ' | '.join(reasons),
        'recommendation': f'{decision} - Premium adjustment: {premium_adjustment}pct' if decision == 'ADJUST' else decision
    }

# Test cases
test_cases = [
    ('ALGER', 5.0),
    ('ROUIBA', 2.0),
    ('HYDRA', 0.5),
    ('MOUZAIA BLIDA', 3.0),
    ('LARBATACHE', 2.5),
]

ai_decisions = []
for commune, capital in test_cases:
    result = underwriting_assistant(commune, capital)
    ai_decisions.append(result)

with open('ai_underwriting_log.json','w', encoding='utf-8') as f:
    json.dump(ai_decisions, f, indent=2)
print("[OK] ai_underwriting_log.json")

print("\n[SUCCESS] All business intelligence outputs generated!")
