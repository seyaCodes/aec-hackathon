#!/usr/bin/env python
"""
Test script for dashboard_analytics module
"""

import pandas as pd
import json
from dashboard_analytics import (
    calculate_zone_vulnerability_ratios,
    identify_growth_zones,
    simulate_boumerdes_scenario,
    calculate_building_type_vulnerability,
    identify_hotspots_with_retention
)

print("=" * 80)
print("TESTING DASHBOARD ANALYTICS MODULE")
print("=" * 80)

# Load test data
print("\n[1] Loading test data...")
df = pd.read_csv('gam_master_data.csv', low_memory=False)
with open('empirical_monte_carlo_params.json') as f:
    params = json.load(f)
print(f"✓ Loaded {len(df)} policies")

# Test 1: Vulnerability ratios
print("\n[2] Testing Vulnerability Ratios...")
vuln = calculate_zone_vulnerability_ratios(df, params)
print("✓ Vulnerability ratios computed")
print(vuln[['Zone', 'Ratio', 'Status']].to_string())

# Test 2: Growth zones
print("\n[3] Testing Growth Zones Strategy...")
growth = identify_growth_zones(df, params)
print("✓ Growth strategy computed")
print(f"  Opportunity zones: {growth['opportunity_zones']}")
print(f"  Moratorium zones: {growth['moratorium_zones']}")

# Test 3: Boumerdes scenario
print("\n[4] Testing Boumerdes M6.5 Scenario...")
boumerdes = simulate_boumerdes_scenario(df, params)
print("✓ Boumerdes scenario computed")
print(f"  Insured Loss: {boumerdes['insured_loss_B']:.1f}B DZD")
print(f"  PML Coverage: {boumerdes['pml_coverage_pct']:.0f}%")
print(f"  Net Loss (after reinsurance): {boumerdes['net_loss_after_reinsurance_B']:.1f}B DZD")

# Test 4: Building type vulnerability
print("\n[5] Testing Building Type Vulnerability...")
building_vuln = calculate_building_type_vulnerability(df)
print(f"✓ Building vulnerability computed for {len(building_vuln)} types")
print(building_vuln[['Building_Type', 'Vulnerability_Multiplier', 'Capital_B']].head().to_string())

# Test 5: Hotspots
print("\n[6] Testing Hotspot Identification...")
hotspots = identify_hotspots_with_retention(df)
print(f"✓ Hotspots computed: {len(hotspots)} communes analyzed")
critical = len(hotspots[hotspots['Hotspot_Status'] == '🔴 CRITICAL'])
high = len(hotspots[hotspots['Hotspot_Status'] == '🟠 HIGH'])
medium = len(hotspots[hotspots['Hotspot_Status'] == '🟡 MEDIUM'])
print(f"  Critical hotspots: {critical}")
print(f"  High hotspots: {high}")
print(f"  Medium hotspots: {medium}")

print("\n" + "=" * 80)
print("ALL TESTS PASSED ✓")
print("=" * 80)
