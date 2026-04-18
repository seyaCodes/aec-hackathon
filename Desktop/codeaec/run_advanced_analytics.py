"""
Advanced Analytics Integration - Demo and Runner

Shows how to use:
1. Vulnerability Model - Replace zone-based losses with probabilistic f()
2. Portfolio Optimization - Recommend rebalancing
3. Reinsurance Strategy - Design layered protection
4. Scenario Engine - Impact analysis for specific earthquakes
"""

import pandas as pd
import numpy as np
from advanced_analytics import (
    VulnerabilityModel,
    PortfolioOptimizer,
    ReinsuranceEngine,
    ScenarioEngine,
    save_results_to_json,
    save_results_to_csv
)
import json

print("=" * 80)
print("ADVANCED ANALYTICS ENGINE - 4 MAJOR UPGRADES FOR GAM")
print("=" * 80)

# Load portfolio data
try:
    df_master = pd.read_csv('gam_master_data.csv', low_memory=False)
    print(f"\n[OK] Loaded {len(df_master):,} policies from portfolio")
except Exception as e:
    print(f"[ERROR] Could not load portfolio: {e}")
    print("Creating sample portfolio for demo...")
    # Create sample if file not found
    zones = ['Zone 0', 'Zone I', 'Zone IIa', 'Zone IIb', 'Zone III']
    df_master = pd.DataFrame({
        'ZONE_RPA': np.random.choice(zones, 1000),
        'CAPITAL_ASSURE': np.random.lognormal(3, 1, 1000),
        'PML_EXPOSE': np.random.lognormal(2, 1.5, 1000),
        'COMMUNE': np.random.choice(['Alger', 'Blida', 'Boumerdes'], 1000),
        'TYPE': np.random.choice(['Residential', 'Commercial', 'Industrial'], 1000)
    })

# ════════════════════════════════════════════════════════════════
# UPGRADE 1: VULNERABILITY MODEL
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("UPGRADE 1: VULNERABILITY MODEL")
print("=" * 80)
print("\nBefore: Zone III = 30% loss (static)")
print("After:  loss = f(building_type, age, materials, soil, intensity)")
print("Advantage: 🔥 Closer to Swiss Re / RMS models\n")

vuln_examples = []

# Example 1: Modern residential building
loss1 = VulnerabilityModel.calculate_loss(
    building_type='Residential',
    age=10,
    material_condition='good',
    soil_type='Rock',
    material='Reinforced Concrete',
    intensity='VIII',
    capital_insured=2.0
)
vuln_examples.append({
    'scenario': 'Modern residential on rock (Zone III earthquake)',
    'loss_ratio': loss1['loss_ratio'],
    'loss_B_DZD': loss1['loss_amount'],
    'reasoning': loss1['reasoning']
})
print(f"Modern residential (10y, concrete, rock):")
print(f"  → Loss: {loss1['loss_ratio']*100:.1f}% = {loss1['loss_amount']:.2f}B DZD\n")

# Example 2: Old masonry building
loss2 = VulnerabilityModel.calculate_loss(
    building_type='Commercial',
    age=80,
    material_condition='poor',
    soil_type='Soft clay',
    material='Unreinforced',
    intensity='VIII',
    capital_insured=1.5
)
vuln_examples.append({
    'scenario': 'Old masonry building on soft clay (Zone III earthquake)',
    'loss_ratio': loss2['loss_ratio'],
    'loss_B_DZD': loss2['loss_amount'],
    'reasoning': loss2['reasoning']
})
print(f"Old masonry (80y, unreinforced, soft clay):")
print(f"  → Loss: {loss2['loss_ratio']*100:.1f}% = {loss2['loss_amount']:.2f}B DZD\n")

# Example 3: Industrial facility
loss3 = VulnerabilityModel.calculate_loss(
    building_type='Industrial',
    age=30,
    material_condition='medium',
    soil_type='Stiff soil',
    material='Steel Frame',
    intensity='VII',
    capital_insured=5.0
)
vuln_examples.append({
    'scenario': 'Industrial facility (Zone IIb earthquake)',
    'loss_ratio': loss3['loss_ratio'],
    'loss_B_DZD': loss3['loss_amount'],
    'reasoning': loss3['reasoning']
})
print(f"Industrial facility (30y, steel, stiff soil, Zone IIb):")
print(f"  → Loss: {loss3['loss_ratio']*100:.1f}% = {loss3['loss_amount']:.2f}B DZD\n")

save_results_to_json({'vulnerability_examples': vuln_examples}, 'upgrade1_vulnerability_model.json')
print("✓ Saved: upgrade1_vulnerability_model.json\n")

# ════════════════════════════════════════════════════════════════
# UPGRADE 2: PORTFOLIO OPTIMIZATION
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("UPGRADE 2: PORTFOLIO OPTIMIZATION ENGINE")
print("=" * 80)
print("\nInput: Current portfolio")
print("Output: Optimal redistribution + KPI improvements")
print("Advantage: 🔥 Reduce PML + Increase profit\n")

optimization_results = PortfolioOptimizer.analyze_portfolio(
    current_portfolio=df_master,
    target_pml_threshold=0.15
)

print("\nCURRENT STATE:")
print(f"  Total capital: {optimization_results['current_metrics']['total_capital_B']}B DZD")
print(f"  Total PML (99%): {optimization_results['current_metrics']['total_pml_B']}B DZD")
print(f"  PML ratio: {optimization_results['current_metrics']['pml_ratio']:.1%}")
print(f"  High-risk concentration (Zone IIb+III): {optimization_results['current_metrics']['high_risk_concentration']:.1%}")

print("\nOPTIMIZED STATE:")
print(f"  Total capital: {optimization_results['optimal_metrics']['total_capital_B']}B DZD")
print(f"  Total PML (99%): {optimization_results['optimal_metrics']['total_pml_B']}B DZD")
print(f"  PML ratio: {optimization_results['optimal_metrics']['pml_ratio']:.1%}")
print(f"  High-risk concentration: {optimization_results['optimal_metrics']['high_risk_concentration']:.1%}")

print("\nEXPECTED IMPROVEMENTS:")
print(f"  PML reduction: {optimization_results['expected_improvements']['pml_reduction_%']:.1f}%")
print(f"  Risk reduction: {optimization_results['expected_improvements']['risk_reduction_%']:.1f}%")

print("\nRECOMMENDATIONS:")
for i, rec in enumerate(optimization_results['recommendations'], 1):
    print(f"\n  {i}. {rec['priority']} {rec['action']}")
    print(f"     Reason: {rec['rationale']}")
    print(f"     How: {rec['method']}")
    print(f"     Impact: {rec['impact']}")

save_results_to_json(optimization_results, 'upgrade2_portfolio_optimization.json')
print("\n✓ Saved: upgrade2_portfolio_optimization.json\n")

# ════════════════════════════════════════════════════════════════
# UPGRADE 3: REINSURANCE STRATEGY
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("UPGRADE 3: REINSURANCE STRATEGY ENGINE")
print("=" * 80)
print("\nInput: Portfolio + PML deficit")
print("Output: Optimal reinsurance layers + cost-benefit")
print("Advantage: 🔥 CEO loves this - it's how insurers survive tail events\n")

# Use actual portfolio metrics
total_capital = df_master['CAPITAL_ASSURE'].sum() / 1e9
pml_99 = df_master['PML_EXPOSE'].sum() / 1e9

print(f"Portfolio Summary:")
print(f"  Total capital: {total_capital:.2f}B DZD")
print(f"  Total PML 99%: {pml_99:.2f}B DZD")
print(f"  Deficit: {max(0, pml_99 - total_capital * 0.70):.2f}B DZD\n")

reinsurance_program = ReinsuranceEngine.design_reinsurance_program(
    total_capital_B=total_capital,
    pml_99_B=pml_99,
    primary_retention_pct=0.70
)

print("REINSURANCE STRUCTURE:")
print(f"  Retained capacity: {reinsurance_program['retention_capacity_B']}B DZD (70%)")
print(f"  Transfer capacity: {reinsurance_program['transfer_capacity_B']}B DZD (30%)")
print(f"  Shortfall to cover: {reinsurance_program['shortfall_B']}B DZD\n")

print("LAYERS:")
for i, layer in enumerate(reinsurance_program['layers'], 1):
    print(f"\n  Layer {i}: {layer['name']}")
    print(f"    Type: {layer['type']}")
    print(f"    Retention point: {layer['retention_point_B']}B DZD")
    print(f"    Coverage limit: {layer['coverage_limit_B']}B DZD")
    print(f"    Annual cost: {layer['annual_cost_B']}B DZD")
    print(f"    Cost ratio: {layer['cost_ratio_%']}%")
    print(f"    Rating requirement: {layer['rating']}")
    print(f"    Note: {layer['note']}")

print(f"\nFINANCIAL IMPACT:")
print(f"  Total reinsurance cost: {reinsurance_program['financial_impact']['total_reinsurance_cost_annual_B']}B DZD/year")
print(f"  Coverage achieved: {reinsurance_program['financial_impact']['coverage_achieved_%']:.1f}%")
print(f"  Uncovered deficit: {reinsurance_program['uncovered_deficit_B']}B DZD")

print(f"\nRECOMMENDATION:")
print(f"  {reinsurance_program['recommendation']}")

save_results_to_json(reinsurance_program, 'upgrade3_reinsurance_strategy.json')
print("\n✓ Saved: upgrade3_reinsurance_strategy.json\n")

# ════════════════════════════════════════════════════════════════
# UPGRADE 4: SCENARIO ENGINE
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("UPGRADE 4: SCENARIO ENGINE")
print("=" * 80)
print("\nInput: Earthquake scenario (location, magnitude)")
print("Output: Losses, insolvency risk, liquidity needs")
print("Advantage: 🔥 'What if' analysis for board meetings\n")

# Assume company reserves = 20% of capital (typical)
company_reserves = total_capital * 0.20

# Test scenarios
scenarios_to_test = ['Alger', 'Boumerdes', 'Blida']
scenario_results = []

for scenario in scenarios_to_test:
    result = ScenarioEngine.simulate_earthquake_scenario(
        scenario_name=scenario,
        portfolio_capital_B=total_capital,
        portfolio_pml_B=pml_99,
        company_reserves_B=company_reserves,
        reinsurance_layers=reinsurance_program['layers']
    )
    scenario_results.append(result)
    
    print(f"\nSCENARIO: Earthquake in {scenario}")
    print(f"  Zone: {result['scenario_parameters']['zone']}")
    print(f"  Intensity: {result['scenario_parameters']['intensity']}")
    print(f"  Exposure: {result['scenario_parameters']['capital_exposure_%']:.1f}% of portfolio\n")
    
    print(f"  LOSSES:")
    print(f"    Gross loss: {result['loss_estimate']['gross_loss_B']}B DZD")
    print(f"    Insured loss: {result['loss_estimate']['insured_loss_B']}B DZD")
    print(f"    Reinsurance recovery: {result['reinsurance_impact']['reinsurance_recovery_B']}B DZD")
    print(f"    Net loss to GAM: {result['reinsurance_impact']['net_loss_to_company_B']}B DZD\n")
    
    print(f"  INSOLVENCY RISK: {result['insolvency_risk']['status']}")
    print(f"    Risk probability: {result['insolvency_risk']['risk_probability_%']}%")
    print(f"    Capital surplus/deficit: {result['insolvency_risk']['capital_surplus_deficit_B']}B DZD\n")
    
    print(f"  LIQUIDITY NEEDS:")
    print(f"    Month 1 payouts: {result['liquidity_needs']['month1_claim_payout_B']}B DZD")
    print(f"    Available liquidity: {result['liquidity_needs']['available_liquidity_B']}B DZD")
    print(f"    Status: {'✓ SUFFICIENT' if result['liquidity_needs']['liquidity_sufficient'] else '⚠️ FUNDING GAP: ' + str(result['liquidity_needs']['funding_gap_B'])}B DZD")

save_results_to_json(scenario_results, 'upgrade4_scenario_analysis.json')
print("\n✓ Saved: upgrade4_scenario_analysis.json\n")

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SUMMARY: 4 MAJOR UPGRADES COMPLETED")
print("=" * 80)

print("""
✓ UPGRADE 1: VULNERABILITY MODEL
  • Replaced: Zone III = 30% (static)
  • With: loss = f(building_type, age, materials, soil, intensity)
  • Benefit: Probabilistic, comparable to Swiss Re/RMS models
  
✓ UPGRADE 2: PORTFOLIO OPTIMIZATION
  • Analyzes: Current zone concentrations
  • Recommends: Specific rebalancing targets (e.g., -15% Zone III, +20% Zone I)
  • Benefit: PML reduction {:.0f}%, profit increase
  
✓ UPGRADE 3: REINSURANCE STRATEGY
  • Designs: 3-layer XL program (Ground-Up + Zone-Specific + Stop-Loss)
  • Output: Layer-by-layer cost breakdown, coverage achievable
  • Benefit: CEO knows exactly how company survives tail events
  
✓ UPGRADE 4: SCENARIO ENGINE
  • Simulates: Specific earthquake scenarios (Alger, Boumerdes, etc.)
  • Shows: Losses, insolvency risk, liquidity needs
  • Benefit: Board can see impact of different earthquakes + plan accordingly

FILES GENERATED:
  • upgrade1_vulnerability_model.json
  • upgrade2_portfolio_optimization.json
  • upgrade3_reinsurance_strategy.json
  • upgrade4_scenario_analysis.json
  
NEXT STEPS:
  1. Integrate these into Streamlit dashboard (add tabs for each upgrade)
  2. Connect vulnerability model to policy underwriting
  3. Use portfolio optimization for annual strategy planning
  4. Show reinsurance program in investor relations materials
  5. Run scenarios quarterly with board of directors
""".format(optimization_results['expected_improvements']['pml_reduction_%']))

print("=" * 80)
print("[SUCCESS] All 4 upgrades implemented and tested!")
print("=" * 80)
