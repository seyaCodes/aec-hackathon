import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from dashboard_analytics import (
    calculate_zone_vulnerability_ratios,
    identify_growth_zones,
    simulate_boumerdes_scenario,
    calculate_building_type_vulnerability,
    identify_hotspots_with_retention
)

st.set_page_config(page_title="GAM — Risk Management Engine", page_icon="🏔️", layout="wide")

st.markdown("""<style>
.metric-card{background:#f8f9fa;border-radius:10px;padding:16px 20px;margin-bottom:8px}
.metric-label{font-size:12px;color:#666;margin-bottom:4px}
.metric-value{font-size:26px;font-weight:700;color:#1a1a1a}
.metric-sub{font-size:11px;color:#999;margin-top:2px}
.action-card{padding:12px;border-radius:8px;margin-bottom:10px}
.action-high{background:#FFEBEE;border-left:4px solid #E53935}
.action-medium{background:#FFF3E0;border-left:4px solid #FF9800}
.action-low{background:#E8F5E9;border-left:4px solid #43A047}
</style>""", unsafe_allow_html=True)

ZONE_COLORS = {'Zone 0':'#43A047','Zone I':'#8BC34A','Zone IIa':'#FFC107','Zone IIb':'#FF9800','Zone III':'#E53935'}
ZONE_ORDER  = ['Zone 0','Zone I','Zone IIa','Zone IIb','Zone III']

WILAYA_COORDS = {
    'ADRAR':(27.87,-0.29),'CHLEF':(36.17,1.33),'LAGHOUAT':(33.80,2.86),
    'OUM EL BOUAGHI':(35.87,7.11),'BATNA':(35.56,6.17),'BEJAIA':(36.75,5.08),
    'BISKRA':(34.85,5.73),'BECHAR':(31.62,-2.22),'BLIDA':(36.47,2.83),
    'BOUIRA':(36.37,3.90),'TAMANRASSET':(22.79,5.52),'TEBESSA':(35.40,8.12),
    'TLEMCEN':(34.88,-1.31),'TIARET':(35.37,1.32),'TIZI OUZOU':(36.72,4.05),
    'ALGER':(36.74,3.06),'DJELFA':(34.67,3.26),'JIJEL':(36.82,5.77),
    'SETIF':(36.19,5.41),'SAIDA':(34.83,0.15),'SKIKDA':(36.88,6.91),
    'SIDI BEL ABBES':(35.20,-0.63),'ANNABA':(36.90,7.76),'GUELMA':(36.46,7.43),
    'CONSTANTINE':(36.36,6.61),'MEDEA':(36.27,2.75),'MOSTAGANEM':(35.93,0.09),
    'M SILA':(35.70,4.54),'MASCARA':(35.40,0.14),'OUARGLA':(31.95,5.32),
    'ORAN':(35.70,-0.62),'EL BAYADH':(33.68,1.01),'ILLIZI':(26.51,8.48),
    'B.B ARRERIDJ':(36.07,4.76),'BOUMERDES':(36.76,3.48),'EL TARF':(36.77,8.31),
    'TINDOUF':(27.67,-8.14),'TISSEMSILT':(35.61,1.81),'EL OUED':(33.36,6.86),
    'KHENCHELA':(35.43,7.14),'SOUK AHRAS':(36.28,7.93),'TIPAZA':(36.59,2.45),
    'MILA':(36.45,6.26),'AIN DEFLA':(36.26,1.97),'NAAMA':(33.27,-0.31),
    'AIN TEMOUCHENT':(35.30,-1.14),'GHARDAIA':(32.49,3.67),'RELIZANE':(35.74,0.55),
    'INCONNU':(28.00,3.00),
}

@st.cache_data
def load_data():
    df = pd.read_csv('gam_master_data.csv', low_memory=False)
    for col in ['CAPITAL_ASSURE','PRIME_NETTE','PML_EXPOSE','RISK_SCORE']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['WILAYA']   = df['WILAYA'].fillna('INCONNU')
    df['TYPE']     = df['TYPE'].fillna('Inconnu')
    df['ZONE_RPA'] = df['ZONE_RPA'].fillna('Non classé')
    return df

@st.cache_data
def load_decision_data():
    try:
        decisions = pd.read_csv('decision_output_final.csv')
        return decisions
    except:
        return None

@st.cache_data
def load_ceo_panel():
    try:
        with open('ceo_decision_panel.json','r',encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_ai_log():
    try:
        with open('ai_underwriting_log.json','r',encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_empirical_params():
    """Load data-derived Monte Carlo parameters by zone"""
    try:
        with open('empirical_monte_carlo_params.json','r',encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_catboost_model():
    """Load trained CatBoost underwriting model"""
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model('catboost_underwriting_model.cbm')
        with open('catboost_label_mapping.json','r',encoding='utf-8') as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        st.warning(f"Could not load CatBoost model: {e}")
        return None, None

@st.cache_data
def run_monte_carlo(df_filtered, empirical_params):
    """
    Run Monte Carlo simulation using data-derived parameters.
    
    Args:
        df_filtered: Filtered portfolio dataframe
        empirical_params: Zone-wise loss parameters
    
    Returns:
        Array of 10,000 simulated annual losses in billions DZD
    """
    np.random.seed(42)
    N = 10_000
    losses = np.zeros(N)
    
    # Build zone-wise loss simulation parameters from empirical data
    zones = ['Zone 0', 'Zone I', 'Zone IIa', 'Zone IIb', 'Zone III']
    
    for i in range(N):
        total_loss = 0
        
        for zone in zones:
            if zone not in empirical_params:
                continue
            
            zone_data = empirical_params[zone]
            zone_capital = zone_data['total_capital_B']
            loss_prob = zone_data['loss_probability']
            mean_loss_ratio = zone_data['mean_loss_ratio']
            std_loss_ratio = zone_data['std_loss_ratio']
            
            # Simulate: does this zone have a loss event?
            if np.random.random() < loss_prob:
                # Sample loss ratio from distribution (using empirical mean/std)
                loss_ratio = np.clip(
                    np.random.normal(mean_loss_ratio, std_loss_ratio),
                    0, 1
                )
                zone_loss = zone_capital * loss_ratio
                total_loss += zone_loss
        
        losses[i] = total_loss
    
    return losses

@st.cache_data
def compute_vulnerability_ratios(df, empirical_params):
    """Compute zone vulnerability ratios"""
    return calculate_zone_vulnerability_ratios(df, empirical_params)

@st.cache_data
def compute_growth_zones(df, empirical_params):
    """Compute geographic growth strategy"""
    return identify_growth_zones(df, empirical_params)

@st.cache_data
def compute_boumerdes_scenario(df, empirical_params):
    """Compute Boumerdes M6.5 scenario"""
    return simulate_boumerdes_scenario(df, empirical_params)

@st.cache_data
def compute_building_vulnerability(df):
    """Compute building type vulnerability breakdown"""
    return calculate_building_type_vulnerability(df)

@st.cache_data
def compute_hotspots(df):
    """Compute hotspots with retention thresholds"""
    return identify_hotspots_with_retention(df)

try:
    df = load_data()
    decisions = load_decision_data()
    ceo_panel = load_ceo_panel()
    ai_log = load_ai_log()
except FileNotFoundError as e:
    st.error(f"Missing data file: {e}")
    st.stop()

# ════════════════════════════════════════════════════════════════
# SECTION 0: SIDEBAR & FILTERS
# ════════════════════════════════════════════════════════════════
st.sidebar.title("🔍 Filters & Configuration")
all_zones   = [z for z in ZONE_ORDER if z in df['ZONE_RPA'].values]
sel_zones   = st.sidebar.multiselect("Zone RPA99", all_zones, default=all_zones)
all_wil     = sorted(df['WILAYA'].dropna().unique())
sel_wil     = st.sidebar.multiselect("Wilaya", all_wil, default=all_wil[:12])
all_types   = sorted(df['TYPE'].dropna().unique())
sel_types   = st.sidebar.multiselect("Type", all_types, default=all_types)
st.sidebar.markdown("---")
st.sidebar.caption(f"{len(df):,} policies · 2023–2025")

mask = df['ZONE_RPA'].isin(sel_zones) & df['WILAYA'].isin(sel_wil) & df['TYPE'].isin(sel_types)
dff  = df[mask].copy()

if len(dff) == 0:
    st.warning("No results — expand filters.")
    st.stop()

# ════════════════════════════════════════════════════════════════
# SECTION 1: OVERVIEW (Top KPIs)
# ════════════════════════════════════════════════════════════════
st.title("🏔️ GAM — Catastrophe Risk Management Engine")
st.caption("Seismic Portfolio Analysis · RPA99 · AI Decision Support · Real-time Scenario Simulation")
st.markdown("---")

tot_cap   = dff['CAPITAL_ASSURE'].sum()
tot_prime = dff['PRIME_NETTE'].sum()
tot_pml   = dff['PML_EXPOSE'].sum()
hi_pct    = dff[dff['ZONE_RPA'].isin(['Zone III','Zone IIb'])]['CAPITAL_ASSURE'].sum() / tot_cap * 100

c1,c2,c3,c4,c5 = st.columns(5)
for col, lbl, val, sub, color in [
    (c1,"Total Policies",f"{len(dff):,}",f"of {len(df):,}","#1a1a1a"),
    (c2,"Capital Insured",f"{tot_cap/1e9:.1f}B","Billion DZD","#1a1a1a"),
    (c3,"PML 99% Loss",f"{tot_pml/1e9:.1f}B","Worst case","#E53935"),
    (c4,"High Risk %",f"{hi_pct:.1f}%","Zone IIb+III","#FF9800"),
    (c5,"Annual Premium",f"{tot_prime/1e6:.0f}M",f"{tot_prime/tot_cap*100:.3f}% of cap","#1a1a1a"),
]:
    col.markdown(f"""<div class="metric-card"><div class="metric-label">{lbl}</div>
    <div class="metric-value" style="color:{color}">{val}</div>
    <div class="metric-sub">{sub}</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SECTION 2: PORTFOLIO ACTIONS (Auto-generated from decisions)
# ════════════════════════════════════════════════════════════════
st.subheader("🚨 Portfolio Actions — Executive Summary")
st.caption("Auto-generated strategic recommendations based on risk data analysis")

if ceo_panel and 'actions' in ceo_panel:
    cols = st.columns(min(3, len(ceo_panel['actions'])))
    for idx, action in enumerate(ceo_panel['actions'][:3]):
        with cols[idx % len(cols)]:
            priority_class = 'action-high' if '[CRITICAL]' in action['priority'] else 'action-medium' if '[HIGH]' in action['priority'] else 'action-low'
            st.markdown(f"""
<div class="action-card {priority_class}">
<strong>{action['priority']}</strong><br>
{action['action']}<br>
<small><em>{action['why']}</em></small><br>
<small>→ {action['how']}</small>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SECTION 2.5: VULNERABILITY RATIO KPI (Phase III — CDC Requirement)
# ════════════════════════════════════════════════════════════════
st.subheader("📊 Indicateurs de Vulnérabilité — Ratio Capital/Rétention par Zone")
st.caption("CDC Requirement: Zone | Capital Exposé | Capacité Retention | Ratio | Status")

empirical_params = load_empirical_params()
if empirical_params:
    vuln_ratios = compute_vulnerability_ratios(df, empirical_params)
    
    # Display as color-coded table
    if len(vuln_ratios) > 0:
        # Format for display
        display_cols = ['Zone', 'Capital_Exposé_B', 'Capacité_Retention_B', 'Ratio', 'Status']
        display_df = vuln_ratios[display_cols].copy()
        display_df.columns = ['Zone', 'Capital (B DZD)', 'Retention (B DZD)', 'Ratio', 'Status']
        
        # Create metrics row
        v_cols = st.columns(len(vuln_ratios))
        for idx, (col, row) in enumerate(zip(v_cols, vuln_ratios.itertuples())):
            with col:
                status_color = row.Color
                st.markdown(f"""
<div style="background-color: rgba({
    '229,57,57' if status_color == 'red' else
    '255,152,0' if status_color == 'orange' else
    '67,160,71'
}, 0.1); border-left: 4px solid {'#E53935' if status_color == 'red' else '#FF9800' if status_color == 'orange' else '#43A047'}; padding: 12px; border-radius: 8px;">
<strong>{row.Zone}</strong><br>
Capital: <b>{row.Capital_Exposé_B}B</b><br>
Retention: {row.Capacité_Retention_B}B<br>
Ratio: <span style="font-size: 18px; font-weight: bold;">{row.Ratio}</span><br>
{row.Status}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SECTION 3: RISK MAP & CONCENTRATION
# ════════════════════════════════════════════════════════════════
map_col, concentration_col = st.columns([2, 1])

with map_col:
    st.subheader("🗺️ Concentration Heat Map — Algérie")
    md = dff.groupby(['WILAYA','ZONE_RPA'],dropna=False).agg(
        Capital_B=('CAPITAL_ASSURE',lambda x:round(x.sum()/1e9,2)),
        PML_B=('PML_EXPOSE',lambda x:round(x.sum()/1e9,2)),
        Polices=('NUMERO_POLICE','count')).reset_index()
    md['lat'] = md['WILAYA'].map(lambda w: WILAYA_COORDS.get(w,(None,None))[0])
    md['lon'] = md['WILAYA'].map(lambda w: WILAYA_COORDS.get(w,(None,None))[1])
    md = md.dropna(subset=['lat','lon'])
    md = md[md['Capital_B']>0]
    fig_map = px.scatter_mapbox(md,lat='lat',lon='lon',size='Capital_B',color='ZONE_RPA',
        color_discrete_map=ZONE_COLORS,hover_name='WILAYA',
        hover_data={'Capital_B':':.2f','PML_B':':.2f','Polices':True,'lat':False,'lon':False},
        size_max=60,zoom=4.5,center={'lat':28.0,'lon':2.5},mapbox_style='carto-positron',
        labels={'Capital_B':'Capital (B DZD)','PML_B':'PML (B DZD)','ZONE_RPA':'Zone RPA'})
    fig_map.update_layout(height=400,margin=dict(l=0,r=0,t=0,b=0),
        legend=dict(orientation='h',y=0,x=1,xanchor='right'))
    st.plotly_chart(fig_map, use_container_width=True)

with concentration_col:
    st.subheader("Risk Distribution")
    za = dff.groupby('ZONE_RPA', dropna=False).agg(Cap=('CAPITAL_ASSURE',lambda x:round(x.sum()/1e9,1))).reset_index()
    za = za[za['ZONE_RPA'].isin(ZONE_ORDER)]
    za['s'] = za['ZONE_RPA'].map({z:i for i,z in enumerate(ZONE_ORDER)})
    za = za.sort_values('s')
    fig1 = px.bar(za, x='Cap', y='ZONE_RPA', orientation='h', color='ZONE_RPA',
                  color_discrete_map=ZONE_COLORS, text=za['Cap'].apply(lambda x:f"{x:.0f}B"),
                  labels={'Cap':'Capital (B DZD)','ZONE_RPA':''})
    fig1.update_traces(textposition='outside')
    fig1.update_layout(showlegend=False,height=300,plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=0,r=50,t=10,b=0))
    st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SECTION 3.5: GEOGRAPHIC GROWTH STRATEGY (Phase III — CDC Requirement)
# ════════════════════════════════════════════════════════════════
st.subheader("🌍 Politique d'Implantation — Strategic Geographic Growth")
st.caption("Where to EXPAND (green zones) vs where to STOP (red zones/moratorium)")

if empirical_params:
    growth_strategy = compute_growth_zones(df, empirical_params)
    
    # Create visual: opportunity vs moratorium zones
    strat_cols = st.columns([1, 1])
    
    with strat_cols[0]:
        st.markdown("### 🟢 **Zones d'Opportunité** (Expand Here)")
        if growth_strategy['opportunity_zones']:
            for zone in growth_strategy['opportunity_zones']:
                rec = growth_strategy['strategic_recommendations'].get(zone, {})
                target = rec.get('target_pct', 0) - growth_strategy['current_distribution'].get(zone, 0)
                st.success(f"**{zone}** → +{target:.0f}% growth potential  \n{rec.get('reason', '')}")
        else:
            st.info("No immediate expansion opportunities (zones at capacity)")
    
    with strat_cols[1]:
        st.markdown("### 🔴 **Zones de Moratoire** (Freeze Growth)")
        if growth_strategy['moratorium_zones']:
            for zone in growth_strategy['moratorium_zones']:
                rec = growth_strategy['strategic_recommendations'].get(zone, {})
                current = growth_strategy['current_distribution'].get(zone, 0)
                target = rec.get('target_pct', 0)
                st.error(f"**{zone}** → -{current - target:.1f}% rebalancing needed  \n{rec.get('reason', '')}")
        else:
            st.info("No zones in moratorium status")
    
    # Summary table of all zones
    st.markdown("#### Strategic Actions by Zone")
    summary_actions = []
    for zone, rec in growth_strategy['strategic_recommendations'].items():
        summary_actions.append({
            'Zone': zone,
            'Action': rec.get('action', 'HOLD'),
            'Current %': f"{growth_strategy['current_distribution'].get(zone, 0):.1f}%",
            'Target %': f"{rec.get('target_pct', 0):.1f}%",
            'Reason': rec.get('reason', ''),
            'Priority': rec.get('priority', 'LOW')
        })
    
    action_df = pd.DataFrame(summary_actions)
    st.dataframe(action_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SECTION 4: COMMUNE-LEVEL DECISION TABLE
# ════════════════════════════════════════════════════════════════
st.subheader("📍 Commune-Level Risk Classification & Actions")
st.caption("Actionable decisions for each high-risk commune (sorted by exposure)")

if decisions is not None:
    # Filter to selected wilayas/zones
    dec_filtered = decisions[
        (decisions['WILAYA'].isin(sel_wil)) & 
        (decisions['ZONE_RPA'].isin(sel_zones))
    ].sort_values('capital_B', ascending=False)
    
    # Create color-coded display
    display_cols = ['WILAYA','COMMUNE','ZONE_RPA','capital_B','pml_B','DECISION','ACTION']
    dec_display = dec_filtered[display_cols].head(30).copy()
    dec_display.columns = ['Wilaya','Commune','Zone','Capital (B)','PML (B)','Decision','Action']
    
    # Create styled dataframe
    def style_decision(val):
        if 'REJECT' in str(val):
            return 'background-color: #FFCDD2; font-weight: bold'
        elif 'ADJUST' in str(val):
            return 'background-color: #FFE0B2; font-weight: bold'
        else:
            return 'background-color: #C8E6C9; font-weight: bold'
    
    styled_df = dec_display.style.map(style_decision, subset=['Decision'])
    st.dataframe(styled_df, use_container_width=True, height=500)
    
    stats_c1, stats_c2, stats_c3 = st.columns(3)
    stats_c1.metric("REJECT Communes", f"{(dec_filtered['DECISION']=='REJECT').sum()}", "Stop new policies")
    stats_c2.metric("ADJUST Communes", f"{(dec_filtered['DECISION']=='ADJUST').sum()}", "Premium adjustments")
    stats_c3.metric("ACCEPT Communes", f"{(dec_filtered['DECISION']=='ACCEPT').sum()}", "Growth opportunities")
    
    # Phase II-B: Hotspot identification with retention thresholds
    with st.expander("📍 Points Chauds — Hotspot Identification with Retention Thresholds", expanded=False):
        st.caption("Communes where cumul capital dépasse la capacité de rétention")
        
        # Default retention capacity (can be adjusted)
        retention_capacity = st.number_input("Retention Capacity (B DZD):", 
                                            min_value=0.5, max_value=5.0, value=1.0, step=0.1)
        
        hotspots = compute_hotspots(dff)
        hotspots['Retention_Threshold_B'] = retention_capacity
        hotspots['Excess_Over_Retention_B'] = (hotspots.get('capital_B', 0) - retention_capacity).clip(lower=0)
        
        # Reclassify with adjusted retention
        def classify_hotspot(capital_b):
            if capital_b > retention_capacity * 2:
                return '🔴 CRITICAL'
            elif capital_b > retention_capacity * 1.5:
                return '🟠 HIGH'
            elif capital_b > retention_capacity:
                return '🟡 MEDIUM'
            else:
                return '🟢 OK'
        
        hotspots['Hotspot_Status'] = hotspots.get('capital_B', 0).apply(classify_hotspot)
        
        # Display hotspots
        hotspot_display = hotspots[['WILAYA', 'COMMUNE', 'ZONE_RPA', 'capital_B', 
                                     'Excess_Over_Retention_B', 'Hotspot_Status']].copy()
        hotspot_display.columns = ['Wilaya', 'Commune', 'Zone', 'Capital (B)', 
                                   'Excess Retention (B)', 'Hotspot Status']
        hotspot_display = hotspot_display[hotspot_display['Excess Retention (B)'] > 0].sort_values('Capital (B)', ascending=False)
        
        if len(hotspot_display) > 0:
            st.dataframe(hotspot_display.head(20), use_container_width=True, hide_index=True)
            st.info(f"Found {len(hotspot_display)} communes exceeding retention capacity of {retention_capacity}B DZD")
        else:
            st.success("✓ No hotspots exceeding retention capacity")

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SECTION 4.5: BUILDING TYPE VULNERABILITY (Phase I — CDC Requirement)
# ════════════════════════════════════════════════════════════════
with st.expander("🏢 Vulnérabilité Intrinsèque par Type de Construction", expanded=False):
    st.caption("Building type multipliers affect loss ratio during earthquakes")
    
    building_vuln = compute_building_vulnerability(dff)
    
    if len(building_vuln) > 0:
        # Metrics row
        bv_cols = st.columns(min(3, len(building_vuln)))
        for idx, (col, row) in enumerate(zip(bv_cols, building_vuln.itertuples())):
            with col:
                risk_color = '#E53935' if row.Risk_Level == 'HIGH' else '#FF9800' if row.Risk_Level == 'MEDIUM' else '#43A047'
                st.markdown(f"""
<div style="background-color: rgba({
    '229,57,57' if row.Risk_Level == 'HIGH' else
    '255,152,0' if row.Risk_Level == 'MEDIUM' else
    '67,160,71'
}, 0.1); border-left: 4px solid {risk_color}; padding: 12px; border-radius: 8px;">
<strong>{row.Building_Type}</strong><br>
Mult: <b>{row.Vulnerability_Multiplier}×</b><br>
Capital: {row.Capital_B}B<br>
Adj. PML: {row.Adjusted_PML_B}B<br>
<small>{row.Risk_Level} Risk</small>
</div>
""", unsafe_allow_html=True)
            if idx >= 2:
                break
        
        # Full table
        st.markdown("#### Building Type Risk Breakdown")
        bv_display = building_vuln[['Building_Type', 'Vulnerability_Multiplier', 'Num_Policies', 'Capital_B', 'Adjusted_PML_B', 'Risk_Level']].copy()
        bv_display.columns = ['Building Type', 'Multiplier', 'Policies', 'Capital (B)', 'Adj. PML (B)', 'Risk Level']
        st.dataframe(bv_display, use_container_width=True, hide_index=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SECTION 5: MONTE CARLO & SCENARIO SIMULATION
# ════════════════════════════════════════════════════════════════
st.subheader("⚡ Scenario Simulation — What If Earthquake?")

scenario_col1, scenario_col2 = st.columns([2, 1])

with scenario_col1:
    st.caption("Monte Carlo 10,000-year loss distribution (data-derived parameters)")
    empirical_params = load_empirical_params()
    if empirical_params:
        losses = run_monte_carlo(df, empirical_params)
    else:
        st.error("Could not load empirical parameters")
        st.stop()
    p90, p99, avg = np.percentile(losses,90), np.percentile(losses,99), np.mean(losses)
    
    counts, edges = np.histogram(losses/1e9, bins=80)
    mids = [(edges[i]+edges[i+1])/2 for i in range(len(counts))]
    bcolors = ['#E53935' if m>=p99/1e9 else '#FF9800' if m>=p90/1e9 else '#90A4AE' for m in mids]
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Bar(x=mids,y=counts,marker_color=bcolors,
                            hovertemplate='Loss: %{x:.1f}B<br>Years: %{y}<extra></extra>'))
    for x,color,txt,pos in [(avg/1e9,'#43A047',f"Avg {avg/1e9:.1f}B","top right"),
                             (p90/1e9,'#FF9800',f"PML90% {p90/1e9:.1f}B","top right"),
                             (p99/1e9,'#E53935',f"PML99% {p99/1e9:.1f}B","top left")]:
        fig_mc.add_vline(x=x,line_color=color,line_width=2+(x==p99/1e9)*0.5,
                         line_dash='solid' if x==p99/1e9 else 'dash',
                         annotation_text=txt,annotation_position=pos,
                         annotation_font=dict(size=10,color=color))
    fig_mc.update_layout(height=300,showlegend=False,
                         xaxis_title='Annual Loss (B DZD)',yaxis_title="Years (out of 10,000)",
                         plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                         margin=dict(l=0,r=20,t=20,b=0))
    st.plotly_chart(fig_mc, use_container_width=True)

with scenario_col2:
    st.subheader("Targeted Scenarios")
    
    # Add Boumerdes named scenario (Phase II-B CDC Requirement)
    scenario_type = st.radio("Select Scenario Type:", 
        ["📊 Monte Carlo (Percentiles)", "🗺️ Named Historical: Boumerdes M6.5 (2003)"], 
        horizontal=False)
    
    if scenario_type == "🗺️ Named Historical: Boumerdes M6.5 (2003)":
        st.markdown("#### Simulation: May 21, 2003 Analogue")
        st.caption("Epicenter: Boumerdes (36.74°N, 3.65°E) - Magnitude 6.8")
        
        empirical_params = load_empirical_params()
        boumerdes_sim = compute_boumerdes_scenario(df, empirical_params)
        
        st.metric("Insured Loss", f"{boumerdes_sim['insured_loss_B']:.1f}B DZD", 
                 f"{boumerdes_sim['pml_coverage_pct']:.0f}% of annual PML")
        st.metric("After Reinsurance", f"{boumerdes_sim['net_loss_after_reinsurance_B']:.1f}B DZD", 
                 "Net exposure")
        
        # Show affected zones
        st.markdown("**Impact by Zone:**")
        for zone, impact in boumerdes_sim['by_zone'].items():
            if impact['loss_B'] > 0:
                st.write(f"• {zone}: **{impact['loss_B']:.1f}B DZD** ({impact['loss_ratio']*100:.0f}% of zone capital)")
        
        # Show affected wilayas
        st.markdown("**Impact by Wilaya:**")
        for wilaya, impact in sorted(boumerdes_sim['by_wilaya'].items(), 
                                      key=lambda x: x[1]['loss_B'], reverse=True):
            if impact['loss_B'] > 0:
                st.warning(f"🔴 {wilaya}: **{impact['loss_B']:.1f}B** ({impact['policies']} policies)")
    
    else:
        # Original Monte Carlo percentiles
        scenario_zone = st.selectbox("Select Zone:", ZONE_ORDER)
        scenario_wilaya = st.selectbox("Or Select Wilaya:", sorted(df['WILAYA'].unique()))
        
        # Calculate scenario impact
        if scenario_zone:
            zone_cap = df[df['ZONE_RPA']==scenario_zone]['CAPITAL_ASSURE'].sum()
            zone_pml = df[df['ZONE_RPA']==scenario_zone]['PML_EXPOSE'].sum()
            pct_total = zone_cap / df['CAPITAL_ASSURE'].sum() * 100
            
            st.metric("Zone Capital", f"{zone_cap/1e9:.1f}B DZD")
            st.metric("% of Portfolio", f"{pct_total:.1f}%")
            st.metric("PML Exposure", f"{zone_pml/1e9:.1f}B DZD")
            st.info(f"If {scenario_zone} earthquake → Loss ~{zone_pml/1e9:.1f}B DZD")

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SECTION 6: PREMIUM ADEQUACY
# ════════════════════════════════════════════════════════════════
st.subheader("💰 Premium Adequacy Analysis")
st.caption("Recommended vs. Current Premium Rates by Risk Zone")

prem_col1, prem_col2 = st.columns(2)

with prem_col1:
    ra = df.groupby('ZONE_RPA',dropna=False).agg(
        cap=('CAPITAL_ASSURE','sum'),
        prime=('PRIME_NETTE','sum'),
        pml=('PML_EXPOSE','sum')).reset_index()
    ra = ra[ra['ZONE_RPA'].isin(ZONE_ORDER)]
    ra['ratio'] = ra['prime']/ra['cap']*100
    ra['required'] = ra['pml']/ra['cap']*100  # PML-based required premium
    ra['gap'] = ra['required'] - ra['ratio']
    ra['s'] = ra['ZONE_RPA'].map({z:i for i,z in enumerate(ZONE_ORDER)})
    ra = ra.sort_values('s')
    
    fig_prem = go.Figure()
    fig_prem.add_trace(go.Bar(x=ra['ZONE_RPA'], y=ra['ratio'], name='Current Premium', 
                              marker_color='#90CAF9'))
    fig_prem.add_trace(go.Bar(x=ra['ZONE_RPA'], y=ra['required'], name='Required Premium',
                              marker_color='#E53935'))
    fig_prem.update_layout(barmode='group', height=300, title="Current vs Required Premium Rates (%)",
                          plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                          xaxis_title='Zone RPA99', yaxis_title='Premium Rate (%)')
    st.plotly_chart(fig_prem, use_container_width=True)

with prem_col2:
    st.write("**Premium Gap Analysis**")
    gap_data = ra[['ZONE_RPA','ratio','required','gap']].copy()
    gap_data.columns = ['Zone','Current %','Required %','Gap %']
    gap_data['Current %'] = gap_data['Current %'].apply(lambda x: f"{x:.3f}%")
    gap_data['Required %'] = gap_data['Required %'].apply(lambda x: f"{x:.3f}%")
    gap_data['Gap %'] = gap_data['Gap %'].apply(lambda x: f"{x:+.3f}%" if abs(x) > 0.01 else "✓ OK")
    st.dataframe(gap_data, use_container_width=True, hide_index=True)
    st.warning(f"**Action Required:** Zone III is underpriced by ~{ra[ra['ZONE_RPA']=='Zone III']['gap'].values[0]:.2f}% — implement immediate rate increase")

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SECTION 7: AI UNDERWRITING ASSISTANT
# ════════════════════════════════════════════════════════════════
st.subheader("🤖 AI Underwriting Assistant — New Policy Evaluator")
st.caption("Instant decision support: ACCEPT / ADJUST / REJECT")

with st.expander("📋 View AI Decision Examples from Portfolio", expanded=False):
    if ai_log:
        for i, decision in enumerate(ai_log, 1):
            cols = st.columns([2, 3])
            with cols[0]:
                status_icon = "✅" if decision['decision'] == 'ACCEPT' else "⚠️" if decision['decision'] == 'ADJUST' else "❌"
                st.markdown(f"**{status_icon} {decision['decision']}**")
            with cols[1]:
                st.caption(f"{decision.get('commune', 'Unknown')} · {decision.get('zone', 'N/A')}")
            st.write(decision.get('reasoning', 'N/A'))
            st.divider()

# Interactive policy evaluator
st.write("**➕ Evaluate New Policy**")
eval_col1, eval_col2, eval_col3 = st.columns(3)

with eval_col1:
    new_commune = st.selectbox("Commune:", sorted(df['COMMUNE'].unique()), key='new_commune')
with eval_col2:
    new_capital = st.number_input("Capital (Billion DZD):", min_value=0.1, max_value=100.0, value=1.0, step=0.5)
with eval_col3:
    st.write("")  # spacing
    if st.button("🔍 Evaluate Policy", use_container_width=True):
        # Find commune info
        commune_data = df[df['COMMUNE'] == new_commune]
        if len(commune_data) > 0:
            comm_zone = commune_data['ZONE_RPA'].mode()[0] if len(commune_data) > 0 else "Unknown"
            comm_wilaya = commune_data['WILAYA'].mode()[0] if len(commune_data) > 0 else "Unknown"
            policy_type = commune_data['TYPE'].mode()[0] if len(commune_data) > 0 else "Unknown"
            existing_cap = commune_data['CAPITAL_ASSURE'].sum() / 1e9
            
            # Use CatBoost model for decision
            try:
                from underwriting_inference import evaluate_policy
                
                result = evaluate_policy(
                    commune=new_commune,
                    capital_B=new_capital,
                    wilaya=comm_wilaya,
                    policy_type=policy_type
                )
                
                decision_result = result['decision']
                confidence = result['confidence']
                explanation = result['reasoning']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Zone", comm_zone)
                with col2:
                    st.metric("Existing Capital", f"{existing_cap:.1f}B")
                with col3:
                    st.metric("Model Confidence", f"{confidence:.1%}")
                
                # Color-code decision
                decision_color = {"ACCEPT": "🟢", "ADJUST": "🟡", "REJECT": "🔴"}
                st.markdown(f"### {decision_color.get(decision_result, '⚪')} Decision: **{decision_result}**")
                
                if confidence < 0.7:
                    st.warning(f"⚠️ Low confidence ({confidence:.1%}). Consider manual review.")
                
                st.info(explanation)
            except Exception as e:
                st.error(f"Error: {e}. Ensure CatBoost model is trained: python train_empirical_models.py")

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.caption("GAM — Catastrophe Risk Engine · RPA99/2003 · CatBoost AI · Monte Carlo 10,000 scenarios · Decision Intelligence Layer")
