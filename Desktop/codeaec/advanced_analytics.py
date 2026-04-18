"""
Advanced Analytics Engine for GAM Risk Management

Implements 4 upgrades:
1. Vulnerability Model - Probabilistic loss based on building characteristics
2. Portfolio Optimization Engine - Recommend optimal portfolio rebalancing
3. Reinsurance Strategy Engine - Design layered reinsurance structures
4. Scenario Engine - Detailed earthquake impact analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json

# ════════════════════════════════════════════════════════════════
# UPGRADE 1: VULNERABILITY MODEL
# ════════════════════════════════════════════════════════════════

class VulnerabilityModel:
    """
    Probabilistic loss function based on building characteristics.
    
    Replaces: Zone III = 30% loss (static)
    Does: loss = f(building_type, age, materials, soil, intensity)
    
    Based on: Swiss Re / RMS vulnerability curves
    """
    
    # Vulnerability weights by building type (0-1)
    BUILDING_TYPES = {
        'Residential': {'multiplier': 0.8, 'description': 'Standard residential'},
        'Commercial': {'multiplier': 1.0, 'description': 'Office/retail'},
        'Hospital': {'multiplier': 0.9, 'description': 'Medical facility'},
        'School': {'multiplier': 0.75, 'description': 'Educational'},
        'Industrial': {'multiplier': 1.2, 'description': 'Factory/warehouse'},
        'Government': {'multiplier': 0.85, 'description': 'Public building'},
        'Unknown': {'multiplier': 0.95, 'description': 'Generic building'}
    }
    
    # Age degradation factor (building gets more vulnerable with age)
    # Age 0 = new (lower damage), Age 100 = very old (higher damage)
    AGE_CURVES = {
        'poor': lambda age: 0.5 + (age / 100) * 0.5,      # 0.5-1.0
        'medium': lambda age: 0.6 + (age / 100) * 0.3,    # 0.6-0.9
        'good': lambda age: 0.3 + (age / 100) * 0.5,      # 0.3-0.8
        'excellent': lambda age: 0.1 + (age / 100) * 0.4  # 0.1-0.5
    }
    
    # Material resilience (lower = more resilient)
    MATERIALS = {
        'Reinforced Concrete': 0.4,
        'Steel Frame': 0.35,
        'Masonry': 0.8,
        'Timber': 1.0,
        'Unreinforced': 1.1,
        'Unknown': 0.7
    }
    
    # Soil conditions amplify shaking (higher = worse)
    SOIL_AMPLIFICATION = {
        'Rock': 0.8,
        'Stiff soil': 1.0,
        'Soft soil': 1.3,
        'Soft clay': 1.5,
        'Very soft clay': 1.7,
        'Unknown': 1.1
    }
    
    # Intensity scale (European Macroseismic Scale)
    # I = not felt, II = weak, ..., XII = total destruction
    INTENSITY_DAMAGE = {
        'I': 0.0,      # Not felt
        'II': 0.0,     # Weak
        'III': 0.01,   # Weak
        'IV': 0.02,    # Moderate
        'V': 0.05,     # Strong
        'VI': 0.15,    # Strong
        'VII': 0.35,   # Very strong
        'VIII': 0.65,  # Severe
        'IX': 0.85,    # Violent
        'X': 0.95,     # Extreme
        'XI': 0.98,    # Extreme
        'XII': 1.0     # Total destruction
    }
    
    @staticmethod
    def calculate_loss(
        building_type: str = 'Unknown',
        age: int = 30,
        material_condition: str = 'medium',
        soil_type: str = 'Unknown',
        material: str = 'Unknown',
        intensity: str = 'VIII',
        capital_insured: float = 1.0
    ) -> Dict:
        """
        Calculate probabilistic loss for a building.
        
        Args:
            building_type: Type of building (Residential, Commercial, etc.)
            age: Building age in years (0-150)
            material_condition: 'poor', 'medium', 'good', 'excellent'
            soil_type: Soil classification (Rock, Stiff soil, Soft soil, etc.)
            material: Construction material
            intensity: Earthquake intensity (I-XII)
            capital_insured: Capital insured in billions DZD
            
        Returns:
            {
                'loss_ratio': 0.0-1.0 (% of capital),
                'loss_amount': amount in billions,
                'vulnerability_index': 0-1,
                'reasoning': explanation,
                'components': breakdown by factor
            }
        """
        
        # Get base factors
        building_mult = VulnerabilityModel.BUILDING_TYPES.get(building_type, VulnerabilityModel.BUILDING_TYPES['Unknown'])['multiplier']
        age_factor = VulnerabilityModel.AGE_CURVES[material_condition](min(age, 100))
        material_mult = VulnerabilityModel.MATERIALS.get(material, VulnerabilityModel.MATERIALS['Unknown'])
        soil_amp = VulnerabilityModel.SOIL_AMPLIFICATION.get(soil_type, VulnerabilityModel.SOIL_AMPLIFICATION['Unknown'])
        intensity_damage = VulnerabilityModel.INTENSITY_DAMAGE.get(intensity, VulnerabilityModel.INTENSITY_DAMAGE['VIII'])
        
        # Combined vulnerability index (0-1)
        vulnerability_index = min(1.0, building_mult * age_factor * material_mult * soil_amp)
        
        # Loss ratio combines intensity with vulnerability
        # Intensity provides baseline; vulnerability modulates it
        loss_ratio = intensity_damage * vulnerability_index
        
        # Add uncertainty factor (±10%)
        uncertainty = np.random.normal(1.0, 0.1)
        loss_ratio_final = min(1.0, max(0.0, loss_ratio * uncertainty))
        
        loss_amount = capital_insured * loss_ratio_final
        
        return {
            'loss_ratio': round(loss_ratio_final, 3),
            'loss_amount': round(loss_amount, 3),
            'vulnerability_index': round(vulnerability_index, 3),
            'components': {
                'building_type': f"{building_type} (×{building_mult})",
                'age_factor': f"{age} years → {round(age_factor, 3)}",
                'material': f"{material} (×{material_mult})",
                'soil_amplification': f"{soil_type} (×{soil_amp})",
                'intensity_damage': f"Intensity {intensity} → {round(intensity_damage, 3)}"
            },
            'reasoning': f"{building_type} age {age}y ({material_condition} condition) · {material} on {soil_type} soil · Intensity {intensity} → Loss {loss_ratio_final*100:.1f}%"
        }


# ════════════════════════════════════════════════════════════════
# UPGRADE 2: PORTFOLIO OPTIMIZATION ENGINE
# ════════════════════════════════════════════════════════════════

class PortfolioOptimizer:
    """
    Portfolio rebalancing recommendations.
    
    Input: Current portfolio
    Output: Optimal redistribution + KPI improvements
    
    Example: "Reduce Zone III by 15%, Increase Zone I by 20%"
    """
    
    @staticmethod
    def analyze_portfolio(
        current_portfolio: pd.DataFrame,
        target_pml_threshold: float = 0.15,  # 15% of total capital
        max_zone_concentration: Dict[str, float] = None
    ) -> Dict:
        """
        Analyze portfolio and recommend optimization.
        
        Args:
            current_portfolio: DataFrame with ZONE_RPA, CAPITAL_ASSURE, PML_EXPOSE columns
            target_pml_threshold: Target PML as % of total capital
            max_zone_concentration: Max allowed % capital per zone
            
        Returns:
            {
                'current_metrics': {...},
                'optimal_metrics': {...},
                'recommendations': [...],
                'expected_improvements': {...},
                'rebalancing_plan': {...}
            }
        """
        
        if max_zone_concentration is None:
            max_zone_concentration = {
                'Zone 0': 0.15,    # Max 15%
                'Zone I': 0.25,    # Max 25%
                'Zone IIa': 0.25,  # Max 25%
                'Zone IIb': 0.20,  # Max 20%
                'Zone III': 0.15   # Max 15% (critical)
            }
        
        # Current state
        zone_summary = current_portfolio.groupby('ZONE_RPA').agg({
            'CAPITAL_ASSURE': 'sum',
            'PML_EXPOSE': 'sum',
            'NUMERO_POLICE': 'count'
        }).reset_index()
        zone_summary.columns = ['ZONE', 'Capital_B', 'PML_B', 'Policies']
        
        total_capital = zone_summary['Capital_B'].sum()
        total_pml = zone_summary['PML_B'].sum()
        zone_summary['Current_%'] = (zone_summary['Capital_B'] / total_capital * 100).round(1)
        zone_summary['PML_Ratio_%'] = (zone_summary['PML_B'] / zone_summary['Capital_B'] * 100).round(1)
        
        # Identify problem zones (over-concentrated or high PML)
        current_metrics = {
            'total_capital_B': round(total_capital, 2),
            'total_pml_B': round(total_pml, 2),
            'pml_ratio': round(total_pml / total_capital, 3),
            'high_risk_concentration': round(
                zone_summary[zone_summary['ZONE'].isin(['Zone IIb', 'Zone III'])]['Capital_B'].sum() / total_capital,
                3
            ),
            'zone_breakdown': zone_summary.to_dict('records')
        }
        
        # Calculate optimal allocation
        optimal_allocation = {}
        for zone, max_pct in max_zone_concentration.items():
            current_pct = zone_summary[zone_summary['ZONE'] == zone]['Current_%'].values
            current_pct = current_pct[0] if len(current_pct) > 0 else 0
            optimal_allocation[zone] = {
                'current_%': current_pct,
                'target_%': max_pct * 100,
                'change': (max_pct * 100) - current_pct
            }
        
        # Generate recommendations
        recommendations = []
        
        # Check Zone III over-concentration
        zone3_current = zone_summary[zone_summary['ZONE'] == 'Zone III']['Current_%'].values[0] if len(zone_summary[zone_summary['ZONE'] == 'Zone III']) > 0 else 0
        if zone3_current > max_zone_concentration['Zone III'] * 100:
            reduction = zone3_current - (max_zone_concentration['Zone III'] * 100)
            reduction_capital = (reduction / 100) * total_capital
            recommendations.append({
                'priority': '[CRITICAL]',
                'action': f"Reduce Zone III by {reduction:.1f}% ({reduction_capital:.1f}B DZD)",
                'rationale': f"Zone III is {reduction:.1f}% above target. High seismic risk.",
                'method': 'Stop new policies + selective renewal cancellations',
                'impact': f"Reduce PML by ~{reduction_capital * 0.30:.1f}B (est. 30% loss rate)"
            })
        
        # Check Zone I under-utilization
        zone1_current = zone_summary[zone_summary['ZONE'] == 'Zone I']['Current_%'].values[0] if len(zone_summary[zone_summary['ZONE'] == 'Zone I']) > 0 else 0
        zone1_target = max_zone_concentration['Zone I'] * 100
        if zone1_current < zone1_target * 0.7:  # If well below target
            increase = zone1_target - zone1_current
            increase_capital = (increase / 100) * total_capital
            recommendations.append({
                'priority': '[HIGH]',
                'action': f"Expand Zone I by {increase:.1f}% ({increase_capital:.1f}B DZD)",
                'rationale': 'Zone I is low-risk with profitable margins. Growth opportunity.',
                'method': 'Targeted acquisition + aggressive marketing in low-risk regions',
                'impact': f"Increase profit by ~{increase_capital * 0.005:.1f}B (est. 0.5% margin)"
            })
        
        # PML threshold warning
        current_pml_ratio = current_metrics['pml_ratio']
        if current_pml_ratio > target_pml_threshold:
            recommendations.append({
                'priority': '[HIGH]',
                'action': f"Reduce PML from {current_pml_ratio:.1%} to {target_pml_threshold:.1%} of capital",
                'rationale': 'PML exceeds safe threshold. Risk of insolvency in tail event.',
                'method': 'Purchase additional reinsurance + reduce high-risk exposures',
                'impact': f"PML reduction of {(current_pml_ratio - target_pml_threshold) * total_capital:.1f}B DZD"
            })
        
        # Calculate optimized metrics
        optimal_capital_allocation = {}
        for zone in optimal_allocation.keys():
            optimal_capital_allocation[zone] = total_capital * (optimal_allocation[zone]['target_%'] / 100)
        
        # Estimate optimized PML (rough: zone3 has ~30% loss rate, zone1 has ~5%)
        optimized_pml = (
            optimal_capital_allocation.get('Zone III', 0) * 0.30 +
            optimal_capital_allocation.get('Zone IIb', 0) * 0.20 +
            optimal_capital_allocation.get('Zone IIa', 0) * 0.12 +
            optimal_capital_allocation.get('Zone I', 0) * 0.05 +
            optimal_capital_allocation.get('Zone 0', 0) * 0.01
        )
        
        optimal_metrics = {
            'total_capital_B': round(total_capital, 2),
            'total_pml_B': round(optimized_pml, 2),
            'pml_ratio': round(optimized_pml / total_capital, 3),
            'high_risk_concentration': round(
                (optimal_capital_allocation.get('Zone IIb', 0) + optimal_capital_allocation.get('Zone III', 0)) / total_capital,
                3
            ),
            'estimated_profit_impact': f"+{(total_capital * 0.003):.1f}B DZD" if zone1_current < zone1_target * 0.7 else "Neutral"
        }
        
        return {
            'current_metrics': current_metrics,
            'optimal_metrics': optimal_metrics,
            'recommendations': recommendations,
            'expected_improvements': {
                'pml_reduction_%': round((1 - (optimal_metrics['pml_ratio'] / current_metrics['pml_ratio'])) * 100, 1),
                'risk_reduction_%': round((1 - (optimal_metrics['high_risk_concentration'] / current_metrics['high_risk_concentration'])) * 100, 1) if current_metrics['high_risk_concentration'] > 0 else 0
            },
            'rebalancing_plan': optimal_allocation
        }


# ════════════════════════════════════════════════════════════════
# UPGRADE 3: REINSURANCE STRATEGY ENGINE
# ════════════════════════════════════════════════════════════════

class ReinsuranceEngine:
    """
    Design layered reinsurance structures (excess-of-loss, stop-loss).
    
    Input: Portfolio + PML deficit
    Output: Reinsurance layers + cost-benefit analysis
    
    CEO cares A LOT about this - it's how insurers survive tail events.
    """
    
    @staticmethod
    def design_reinsurance_program(
        total_capital_B: float,
        pml_99_B: float,
        primary_retention_pct: float = 0.70,
        reinsurer_capacity_cost_pct: float = 0.015  # 1.5% per $1B capacity
    ) -> Dict:
        """
        Design optimal reinsurance program.
        
        Args:
            total_capital_B: Total capital in billions
            pml_99_B: 99% PML in billions
            primary_retention_pct: % of risk retained by primary insurer [DEPRECATED - uses fixed reserve]
            reinsurer_capacity_cost_pct: Cost per $1B of reinsurance capacity
            
        Returns:
            {
                'retention_capacity_B': amount retained,
                'transfer_capacity_B': amount transferred,
                'shortfall_B': uninsured deficit,
                'layers': [layer1, layer2, layer3, ...],
                'total_reinsurance_cost_B': annual cost,
                'financial_impact': {...}
            }
        """
        
        # Calculate capacities — use fixed 70B internal reserve
        retained_capacity = 70  # GAM proprietary retention limit (internal capital reserves)
        transfer_capacity = total_capital_B - retained_capacity
        shortfall = max(0, pml_99_B - retained_capacity)
        
        layers = []
        cumulative_coverage = 0
        remaining_shortfall = shortfall
        
        # Layer 1: XL Layer 1 (Excess-of-Loss, relatively cheap, high retention point)
        # Covers losses above retention point, up to 1x coverage
        if remaining_shortfall > 0:
            layer1_limit = min(transfer_capacity * 0.5, remaining_shortfall)
            layer1_cost = layer1_limit * reinsurer_capacity_cost_pct
            layers.append({
                'name': 'XL Layer 1 (Ground-Up Coverage)',
                'type': 'Excess-of-Loss',
                'retention_point_B': retained_capacity,
                'coverage_limit_B': layer1_limit,
                'deductible_pct': 0.05,
                'annual_cost_B': round(layer1_cost, 3),
                'cost_ratio_%': round((layer1_cost / layer1_limit) * 100, 2),
                'rating': 'Standard (A+ rated reinsurers)',
                'availability': 'Guaranteed',
                'note': 'Primary protection layer. Most essential.'
            })
            cumulative_coverage += layer1_limit
            remaining_shortfall -= layer1_limit
        
        # Layer 2: XL Layer 2 (Focused on Zone III exposure)
        if remaining_shortfall > 0:
            layer2_limit = min(transfer_capacity * 0.3, remaining_shortfall)
            layer2_cost = layer2_limit * reinsurer_capacity_cost_pct * 1.3  # 30% more expensive
            layers.append({
                'name': 'XL Layer 2 (Zone III Specific)',
                'type': 'Excess-of-Loss',
                'retention_point_B': retained_capacity + layers[0]['coverage_limit_B'],
                'coverage_limit_B': layer2_cost,
                'deductible_pct': 0.10,
                'annual_cost_B': round(layer2_cost, 3),
                'cost_ratio_%': round((layer2_cost / layer2_limit) * 100 if layer2_limit > 0 else 0, 2),
                'rating': 'AA/AAA rated reinsurers only',
                'availability': 'Competitive market. Requires rate lock.',
                'note': 'Focuses on highest-risk zone. More expensive but targeted.'
            })
            cumulative_coverage += layer2_limit
            remaining_shortfall -= layer2_limit
        
        # Layer 3: Stop-Loss (Aggregate, covers catastrophic years)
        if remaining_shortfall > 0:
            stopless_limit = min(transfer_capacity * 0.2, remaining_shortfall)
            stopless_cost = stopless_limit * reinsurer_capacity_cost_pct * 2.5  # 2.5x more expensive
            layers.append({
                'name': 'Stop-Loss (Portfolio Protection)',
                'type': 'Stop-Loss/Aggregate XL',
                'retention_point_B': total_capital_B * 0.10,  # Company absorbs 10% loss year
                'coverage_limit_B': stopless_limit,
                'deductible_pct': 0.20,
                'annual_cost_B': round(stopless_cost, 3),
                'cost_ratio_%': round((stopless_cost / stopless_limit) * 100 if stopless_limit > 0 else 0, 2),
                'rating': 'AAA only. Lloyd\'s syndicates.',
                'availability': 'Scarce. Limited capacity. 6-9 month placement.',
                'note': 'Last-resort protection. Prevents insolvency in tail event.'
            })
            cumulative_coverage += stopless_limit
            remaining_shortfall -= stopless_limit
        
        # Calculate financials
        total_reinsurance_cost = sum([layer['annual_cost_B'] for layer in layers])
        total_coverage = cumulative_coverage
        coverage_pct = (total_coverage / shortfall * 100) if shortfall > 0 else 100
        
        # Cost-benefit: What is protection worth?
        # Value = Probability(loss > retained) × average loss when it occurs
        loss_probability_99 = 0.01  # 1% chance per year
        protection_value = loss_probability_99 * remaining_shortfall  # Expected loss not covered
        roi = (protection_value / total_reinsurance_cost) if total_reinsurance_cost > 0 else 0
        
        return {
            'retention_capacity_B': round(retained_capacity, 2),
            'transfer_capacity_B': round(transfer_capacity, 2),
            'shortfall_B': round(shortfall, 2),
            'covered_by_layers_B': round(total_coverage, 2),
            'uncovered_deficit_B': round(remaining_shortfall, 2),
            'layers': layers,
            'financial_impact': {
                'total_reinsurance_cost_annual_B': round(total_reinsurance_cost, 3),
                'cost_as_%_of_premium': 'TBD (depends on premium volume)',
                'protection_value_annual_B': round(protection_value, 3),
                'roi_of_reinsurance': round(roi, 2),
                'coverage_achieved_%': round(coverage_pct, 1)
            },
            'recommendation': f"3-layer XL program covers {coverage_pct:.0f}% of {shortfall:.1f}B shortfall. Annual cost: {total_reinsurance_cost:.3f}B. ROI: {roi:.2f}x."
        }


# ════════════════════════════════════════════════════════════════
# UPGRADE 4: SCENARIO ENGINE
# ════════════════════════════════════════════════════════════════

class ScenarioEngine:
    """
    Detailed earthquake impact analysis.
    
    Input: Earthquake scenario (location, magnitude)
    Output: Losses, insolvency risk, liquidity needs
    
    Example: "If earthquake hits Alger → Loss XB, Impact Y%, Liquidity need Z"
    """
    
    WILAYA_EXPOSURE_MAP = {
        'Alger': {'capital_pct': 0.15, 'zone': 'Zone III', 'intensity': 'VIII'},
        'Boumerdes': {'capital_pct': 0.08, 'zone': 'Zone III', 'intensity': 'VIII'},
        'Blida': {'capital_pct': 0.06, 'zone': 'Zone IIb', 'intensity': 'VII'},
        'Setif': {'capital_pct': 0.05, 'zone': 'Zone IIb', 'intensity': 'VI'},
        'Algiers-Region': {'capital_pct': 0.25, 'zone': 'Zone III', 'intensity': 'VIII'}
    }
    
    @staticmethod
    def simulate_earthquake_scenario(
        scenario_name: str,
        portfolio_capital_B: float,
        portfolio_pml_B: float,
        company_reserves_B: float,
        reinsurance_layers: List[Dict] = None
    ) -> Dict:
        """
        Simulate specific earthquake scenario.
        
        Args:
            scenario_name: 'Alger', 'Boumerdes', etc.
            portfolio_capital_B: Total insured capital
            portfolio_pml_B: 99% PML
            company_reserves_B: Cash reserves + investment portfolio
            reinsurance_layers: Reinsurance program structure
            
        Returns:
            {
                'scenario': name,
                'exposure_B': capital exposed,
                'estimated_loss_B': direct loss,
                'insured_loss_B': after deductibles,
                'reinsurance_recovery_B': recovery from reinsurers,
                'net_loss_B': GAM's loss after reinsurance,
                'financial_impact': {...},
                'insolvency_risk': {...},
                'liquidity_needs': {...}
            }
        """
        
        scenario = ScenarioEngine.WILAYA_EXPOSURE_MAP.get(scenario_name, {})
        if not scenario:
            return {'error': f"Scenario '{scenario_name}' not found"}
        
        # Exposure in affected zone
        exposure_capital = portfolio_capital_B * scenario['capital_pct']
        
        # Loss estimation based on zone and intensity
        vulnerability = {
            'Zone III': {'base_loss': 0.30, 'variance': 0.10},
            'Zone IIb': {'base_loss': 0.20, 'variance': 0.08},
            'Zone IIa': {'base_loss': 0.12, 'variance': 0.05},
            'Zone I': {'base_loss': 0.05, 'variance': 0.02}
        }
        
        zone_vuln = vulnerability.get(scenario['zone'], vulnerability['Zone IIa'])
        # Simulate loss with uncertainty
        loss_ratio = np.random.normal(zone_vuln['base_loss'], zone_vuln['variance'])
        loss_ratio = max(0, min(1.0, loss_ratio))
        estimated_loss = exposure_capital * loss_ratio
        
        # Apply deductible/franchise (typical: 5% of loss)
        deductible = estimated_loss * 0.05
        insured_loss = estimated_loss - deductible
        
        # Reinsurance recovery (if program exists)
        reinsurance_recovery = 0
        if reinsurance_layers:
            for layer in reinsurance_layers:
                if insured_loss > layer.get('retention_point_B', 0):
                    layer_payout = min(
                        layer['coverage_limit_B'],
                        insured_loss - layer.get('retention_point_B', 0)
                    )
                    reinsurance_recovery += layer_payout
        
        # Net loss for company
        net_loss = insured_loss - reinsurance_recovery
        
        # Financial impact assessment
        loss_as_pct_capital = (net_loss / portfolio_capital_B) * 100
        loss_as_pct_pml = (net_loss / portfolio_pml_B) * 100
        capital_after_loss = company_reserves_B - net_loss
        
        # Insolvency risk
        solvency_capital_requirement = portfolio_capital_B * 0.12  # Regulatory minimum 12%
        insolvency_risk_pct = 0
        if capital_after_loss < solvency_capital_requirement:
            insolvency_risk_pct = ((solvency_capital_requirement - capital_after_loss) / solvency_capital_requirement) * 100
            insolvency_status = 'HIGH RISK'
        elif capital_after_loss < solvency_capital_requirement * 1.5:
            insolvency_risk_pct = 30
            insolvency_status = 'ELEVATED RISK'
        else:
            insolvency_risk_pct = 5
            insolvency_status = 'ACCEPTABLE'
        
        # Liquidity needs
        claim_payment_rate = 0.20  # Pay 20% of claims in first month
        liquidity_needed_month1 = insured_loss * claim_payment_rate
        available_liquidity = company_reserves_B
        liquidity_sufficient = available_liquidity > liquidity_needed_month1
        
        return {
            'scenario': scenario_name,
            'scenario_parameters': {
                'zone': scenario['zone'],
                'intensity': scenario['intensity'],
                'capital_exposure_%': scenario['capital_pct'] * 100
            },
            'loss_estimate': {
                'gross_loss_B': round(estimated_loss, 2),
                'deductible_B': round(deductible, 2),
                'insured_loss_B': round(insured_loss, 2),
                'loss_as_%_portfolio': round(loss_as_pct_capital, 1),
                'loss_as_%_pml': round(loss_as_pct_pml, 1)
            },
            'reinsurance_impact': {
                'reinsurance_recovery_B': round(reinsurance_recovery, 2),
                'net_loss_to_company_B': round(net_loss, 2),
                'reinsurance_effectiveness_%': round((reinsurance_recovery / insured_loss * 100) if insured_loss > 0 else 0, 1)
            },
            'financial_impact': {
                'company_reserves_before_B': round(company_reserves_B, 2),
                'company_reserves_after_B': round(capital_after_loss, 2),
                'capital_erosion_B': round(net_loss, 2)
            },
            'insolvency_risk': {
                'status': insolvency_status,
                'risk_probability_%': round(insolvency_risk_pct, 1),
                'solvency_capital_requirement_B': round(solvency_capital_requirement, 2),
                'capital_surplus_deficit_B': round(capital_after_loss - solvency_capital_requirement, 2)
            },
            'liquidity_needs': {
                'month1_claim_payout_B': round(liquidity_needed_month1, 2),
                'available_liquidity_B': round(available_liquidity, 2),
                'liquidity_sufficient': str(liquidity_sufficient),  # Convert bool to string for JSON
                'funding_gap_B': max(0, round(liquidity_needed_month1 - available_liquidity, 2))
            }
        }


# ════════════════════════════════════════════════════════════════
# HELPER: Export to JSON/CSV
# ════════════════════════════════════════════════════════════════

def save_results_to_json(results: Dict, filename: str):
    """Save analysis results to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved: {filename}")

def save_results_to_csv(results: List[Dict], filename: str):
    """Save analysis results to CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"[OK] Saved: {filename}")
