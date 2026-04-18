

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="GAM — Analyse Sismique CATNAT", page_icon="🏔️", layout="wide")

st.markdown("""<style>
.metric-card{background:#f8f9fa;border-radius:10px;padding:16px 20px;margin-bottom:8px}
.metric-label{font-size:12px;color:#666;margin-bottom:4px}
.metric-value{font-size:26px;font-weight:700;color:#1a1a1a}
.metric-sub{font-size:11px;color:#999;margin-top:2px}
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
def run_monte_carlo(cap3, cap2b, cap2a, cap1, cap0):
    np.random.seed(42)
    N = 10_000
    S = [
        (cap3,  0.07,  0.30, 0.08),
        (cap2b, 0.04,  0.20, 0.06),
        (cap2a, 0.025, 0.12, 0.04),
        (cap1,  0.010, 0.05, 0.02),
        (cap0,  0.002, 0.01, 0.005),
    ]
    losses = np.zeros(N)
    for i in range(N):
        t = 0
        for cap, prob, mean, std in S:
            if np.random.random() < prob:
                t += cap * np.clip(np.random.normal(mean, std), 0, 1)
        losses[i] = t
    return losses

try:
    df = load_data()
except FileNotFoundError:
    st.error(" gam_master_data.csv not found — put it in the SAME folder as app.py")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🔍 Filtres")
all_zones   = [z for z in ZONE_ORDER if z in df['ZONE_RPA'].values]
sel_zones   = st.sidebar.multiselect("Zone RPA99", all_zones, default=all_zones)
all_wil     = sorted(df['WILAYA'].dropna().unique())
sel_wil     = st.sidebar.multiselect("Wilaya", all_wil, default=all_wil[:12])
all_types   = sorted(df['TYPE'].dropna().unique())
sel_types   = st.sidebar.multiselect("Type", all_types, default=all_types)
st.sidebar.markdown("---")
st.sidebar.caption(f" {len(df):,} polices · 2023–2025")

mask = df['ZONE_RPA'].isin(sel_zones) & df['WILAYA'].isin(sel_wil) & df['TYPE'].isin(sel_types)
dff  = df[mask].copy()

# ── Title ─────────────────────────────────────────────────────
st.title(" GAM — Tableau de Bord Sismique CATNAT")
st.caption("Portefeuille garantie Tremblement de Terre · RPA99 · CatBoost AI · Monte Carlo 10 000 simulations")
st.markdown("---")

if len(dff) == 0:
    st.warning("Aucun résultat — élargissez les filtres.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────
tot_cap   = dff['CAPITAL_ASSURE'].sum()
tot_prime = dff['PRIME_NETTE'].sum()
tot_pml   = dff['PML_EXPOSE'].sum()
hi_pct    = dff[dff['ZONE_RPA'].isin(['Zone III','Zone IIb'])]['CAPITAL_ASSURE'].sum() / tot_cap * 100

c1,c2,c3,c4,c5 = st.columns(5)
for col, lbl, val, sub, color in [
    (c1,"Polices",f"{len(dff):,}",f"sur {len(df):,}","#1a1a1a"),
    (c2,"Capital assuré",f"{tot_cap/1e9:.1f} B","Milliards DZD","#1a1a1a"),
    (c3,"PML estimé",f"{tot_pml/1e9:.1f} B","Perte max. probable","#B71C1C"),
    (c4,"Zones risque élevé",f"{hi_pct:.1f}%","Capital IIb + III","#E65100"),
    (c5,"Primes nettes",f"{tot_prime/1e6:.0f} M",f"Ratio {tot_prime/tot_cap*100:.3f}%","#1a1a1a"),
]:
    col.markdown(f"""<div class="metric-card"><div class="metric-label">{lbl}</div>
    <div class="metric-value" style="color:{color}">{val}</div>
    <div class="metric-sub">{sub}</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Chart row 1 ───────────────────────────────────────────────
cl, cr = st.columns(2)

with cl:
    st.subheader("Capital par zone RPA99")
    za = dff.groupby('ZONE_RPA', dropna=False).agg(Cap=('CAPITAL_ASSURE',lambda x:round(x.sum()/1e9,1))).reset_index()
    za = za[za['ZONE_RPA'].isin(ZONE_ORDER)]
    za['s'] = za['ZONE_RPA'].map({z:i for i,z in enumerate(ZONE_ORDER)})
    za = za.sort_values('s')
    fig1 = px.bar(za, x='Cap', y='ZONE_RPA', orientation='h', color='ZONE_RPA',
                  color_discrete_map=ZONE_COLORS, text=za['Cap'].apply(lambda x:f"{x:.0f}B"),
                  labels={'Cap':'Capital (B DZD)','ZONE_RPA':''})
    fig1.update_traces(textposition='outside')
    fig1.update_layout(showlegend=False,height=280,plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=0,r=50,t=10,b=0))
    st.plotly_chart(fig1, use_container_width=True)

with cr:
    st.subheader("Top wilayas par exposition")
    wa = dff.groupby(['WILAYA','ZONE_RPA'], dropna=False).agg(
        Cap=('CAPITAL_ASSURE',lambda x:round(x.sum()/1e9,1))).reset_index()
    wa = wa.sort_values('Cap',ascending=False).head(12)
    fig2 = px.bar(wa, x='WILAYA', y='Cap', color='ZONE_RPA', color_discrete_map=ZONE_COLORS,
                  labels={'Cap':'Capital (B DZD)','WILAYA':'','ZONE_RPA':'Zone'})
    fig2.update_layout(height=280,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                       margin=dict(l=0,r=0,t=10,b=0),
                       xaxis=dict(tickangle=-30,tickfont=dict(size=9)),
                       legend=dict(orientation='h',y=1.05,x=1,xanchor='right',font=dict(size=9)))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Chart row 2: Monte Carlo + Type + Ratio ───────────────────
cm, ct = st.columns([3,2])

with cm:
    st.subheader(" Monte Carlo — Distribution des pertes (10 000 ans)")
    zc = df.groupby('ZONE_RPA')['CAPITAL_ASSURE'].sum()
    losses = run_monte_carlo(
        zc.get('Zone III',0), zc.get('Zone IIb',0), zc.get('Zone IIa',0),
        zc.get('Zone I',0),   zc.get('Zone 0',0))
    p90, p99, avg = np.percentile(losses,90), np.percentile(losses,99), np.mean(losses)
    st.caption(f"**PML 99% = {p99/1e9:.1f} B DZD** · Ratio PML/Primes = {p99/df['PRIME_NETTE'].sum():.0f}x")
    counts, edges = np.histogram(losses/1e9, bins=80)
    mids = [(edges[i]+edges[i+1])/2 for i in range(len(counts))]
    bcolors = ['#E53935' if m>=p99/1e9 else '#FF9800' if m>=p90/1e9 else '#90A4AE' for m in mids]
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Bar(x=mids,y=counts,marker_color=bcolors,
                            hovertemplate='Perte: %{x:.1f}B<br>Années: %{y}<extra></extra>'))
    for x,color,txt,pos in [(avg/1e9,'#43A047',f"Moy {avg/1e9:.1f}B","top right"),
                             (p90/1e9,'#FF9800',f"PML90% {p90/1e9:.1f}B","top right"),
                             (p99/1e9,'#E53935',f"PML99% {p99/1e9:.1f}B","top left")]:
        fig_mc.add_vline(x=x,line_color=color,line_width=2+(x==p99/1e9)*0.5,
                         line_dash='solid' if x==p99/1e9 else 'dash',
                         annotation_text=txt,annotation_position=pos,
                         annotation_font=dict(size=10,color=color))
    fig_mc.update_layout(height=300,showlegend=False,
                         xaxis_title='Perte annuelle (B DZD)',yaxis_title="Nb d'années",
                         plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                         margin=dict(l=0,r=20,t=20,b=0))
    st.plotly_chart(fig_mc, use_container_width=True)
    m1,m2,m3 = st.columns(3)
    m1.metric("PML 99%", f"{p99/1e9:.1f} B DZD")
    m2.metric("Perte moyenne", f"{avg/1e9:.1f} B DZD")
    m3.metric("PML / Primes", f"{p99/df['PRIME_NETTE'].sum():.0f}x", delta="Insuffisant", delta_color='inverse')

with ct:
    st.subheader("Par type de risque")
    ta = dff.groupby('TYPE',dropna=False)['CAPITAL_ASSURE'].sum().reset_index()
    ta.columns = ['Type','Capital']
    ta = ta[ta['Capital']>0]
    ta['Capital_B'] = ta['Capital']/1e9
    fig_t = px.pie(ta,values='Capital_B',names='Type',hole=0.45,
                   color='Type',color_discrete_map={'Bien Immobilier':'#5C6BC0',
                   'Installation Commerciale':'#26A69A','Installation Industrielle':'#FFA726'})
    fig_t.update_traces(textinfo='percent+label',
                        hovertemplate='%{label}<br>%{value:.1f}B<br>%{percent}<extra></extra>')
    fig_t.update_layout(height=230,showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_t, use_container_width=True)

    st.subheader("Ratio Prime / Capital")
    ra = df.groupby('ZONE_RPA',dropna=False).agg(cap=('CAPITAL_ASSURE','sum'),prime=('PRIME_NETTE','sum')).reset_index()
    ra = ra[ra['ZONE_RPA'].isin(ZONE_ORDER)]
    ra['ratio'] = ra['prime']/ra['cap']*100
    ra['s'] = ra['ZONE_RPA'].map({z:i for i,z in enumerate(ZONE_ORDER)})
    ra = ra.sort_values('s')
    fig_r = px.bar(ra,x='ZONE_RPA',y='ratio',color='ZONE_RPA',color_discrete_map=ZONE_COLORS,
                   text=ra['ratio'].apply(lambda x:f"{x:.3f}%"),labels={'ratio':'Ratio P/C (%)','ZONE_RPA':''})
    fig_r.update_traces(textposition='outside',textfont_size=9)
    fig_r.update_layout(height=190,showlegend=False,plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig_r, use_container_width=True)

st.markdown("---")

# ── Map ───────────────────────────────────────────────────────
st.subheader("🗺️ Carte de concentration — Algérie")
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
fig_map.update_layout(height=500,margin=dict(l=0,r=0,t=0,b=0),
    legend=dict(orientation='h',y=0,x=1,xanchor='right'))
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")


st.subheader(" Recommandations stratégiques")
r1,r2,r3 = st.columns(3)
with r1:
    st.error("** Surconcentration — ALGER**\n\nALGER = 251 Mrd DZD en Zone III = 28.4% du portefeuille.\n\n→ Plafonner cumuls par commune à 50 Mrd DZD\n→ Refuser polices industrielles >500M sans réassurance")
with r2:
    st.warning("** Tarification inadaptée**\n\nZone III 0.058% vs Zone I 0.033% — écart = 1.8x seulement, doit être 5x.\n\n→ Appliquer facteur 5.0x en Zone III\n→ Révision tarifaire immédiate")
with r3:
    st.success("** Croissance — zones sous-représentées**\n\nZone 0+I = 11.7% seulement. Adrar, Bechar, Ouargla = risque minimal.\n\n→ Développer ces wilayas\n→ Objectif 25% du portefeuille en 3 ans")

st.markdown("---")
st.caption("GAM · CATNAT 2023–2025 · RPA99/2003 · CatBoost AI · Monte Carlo 10 000 ans")