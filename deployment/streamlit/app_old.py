"""
app.py — Application Streamlit interactive
CEET Smart Grid · Energy Blackout Prediction Project
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CEET Smart Grid",
    page_icon="EDATA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1D3557, #E63946);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 24px;
    }
    .kpi-box {
        background: #1e1e3a; border-radius: 10px; padding: 16px;
        text-align: center; border-left: 4px solid;
    }
    .risk-critical  { background: #3a0000; border-color: #E63946; }
    .risk-high      { background: #2a1500; border-color: #F4A261; }
    .risk-moderate  { background: #2a2200; border-color: #E9C46A; }
    .risk-low       { background: #002a15; border-color: #2A9D8F; }
</style>
""", unsafe_allow_html=True)


# ─── Chargement des données ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    paths = [
        Path(__file__).parent.parent.parent / "data" / "processed" / "ceet_processed.csv",
        Path(__file__).parent.parent.parent / "data" / "raw" / "ceet_togo_smartgrid_dataset.csv",
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p, parse_dates=["datetime"])
            if "load_ratio" not in df.columns:
                df["load_ratio"] = df["total_load_mw"] / df["available_power_mw"].replace(0, np.nan)
            if "power_margin" not in df.columns:
                df["power_margin"] = df["available_power_mw"] - df["total_load_mw"]
            return df
    return pd.DataFrame()

df = load_data()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/68/Flag_of_Togo.svg", width=80)
    st.markdown("## CEET Smart Grid")
    st.markdown("---")

    page = st.radio("Navigation", [
        " Accueil",
        " Analyse EDA",
        " Prédiction Live",
        " Détection Anomalies",
        " Séries Temporelles",
    ])

    if not df.empty:
        st.markdown("---")
        st.markdown("**Filtres Globaux**")
        regions = ["Toutes"] + sorted(df["region"].unique().tolist())
        sel_region = st.selectbox("Région", regions)
        seasons = ["Toutes"] + sorted(df["season"].unique().tolist())
        sel_season = st.selectbox("Saison", seasons)

    st.markdown("---")
    st.markdown("*Données collecte suite a des rapport sur le net · 50 000 observations*")


# ─── Filtrage global ─────────────────────────────────────────────────────────
def get_filtered_df():
    dff = df.copy()
    if not df.empty:
        if sel_region != "Toutes":
            dff = dff[dff["region"] == sel_region]
        if sel_season != "Toutes":
            dff = dff[dff["season"] == sel_season]
    return dff


# ═══════════════════════════════════════════════════════════
# PAGE 1 : ACCUEIL / KPIs
# ═══════════════════════════════════════════════════════════
if page == " Accueil":
    st.markdown("""
    <div class="main-header">
        <h1> CEET Smart Grid Analytics</h1>
        <p>Prédiction des délestages · Monitoring du réseau électrique · Togo</p>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.error("Données non trouvées. Lancez d'abord data_preprocessing.py")
        st.stop()

    dff = get_filtered_df()
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric(" Taux Blackout", f"{dff['blackout'].mean()*100:.2f}%",
                  delta=f"{(dff['blackout'].mean()-df['blackout'].mean())*100:.2f}% vs global")
    with c2:
        st.metric(" Risque Moyen", f"{dff['outage_risk'].mean():.1f}%")
    with c3:
        st.metric(" Délestage Total", f"{dff['load_shedding_mw'].sum():,.0f} MW")
    with c4:
        st.metric(" Charge Moy.", f"{dff['total_load_mw'].mean():.0f} MW")
    with c5:
        st.metric(" Surcharges", f"{dff['overload'].sum():,}")
    with c6:
        st.metric(" Observations", f"{len(dff):,}")

    st.markdown("---")
    c_left, c_right = st.columns([2, 1])

    with c_left:
        st.subheader(" Charge Électrique (Série Temporelle)")
        ts = dff.set_index("datetime")["total_load_mw"].resample("D").mean().reset_index()
        fig = px.area(ts, x="datetime", y="total_load_mw",
                      color_discrete_sequence=["#457B9D"],
                      labels={"total_load_mw": "MW", "datetime": ""},
                      template="plotly_dark")
        fig.update_layout(margin=dict(t=20, b=20), height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.subheader(" Blackouts par Région")
        reg_bl = dff.groupby("region")["blackout"].mean().reset_index()
        reg_bl.columns = ["Région", "Taux (%)"]
        reg_bl["Taux (%)"] = (reg_bl["Taux (%)"] * 100).round(2)
        fig2 = px.bar(reg_bl, x="Taux (%)", y="Région", orientation="h",
                      color="Taux (%)", color_continuous_scale="Reds",
                      template="plotly_dark")
        fig2.update_layout(margin=dict(t=20, b=20), height=350)
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2 : ANALYSE EDA
# ═══════════════════════════════════════════════════════════
elif page == " Analyse EDA":
    st.header(" Analyse Exploratoire des Données")
    dff = get_filtered_df()
    if dff.empty:
        st.warning("Aucune donnée pour ces filtres")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Distributions", "Corrélations", "Temporel"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(dff, x="total_load_mw", nbins=50,
                               title="Distribution de la Charge Totale",
                               template="plotly_dark", color_discrete_sequence=["#E63946"])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(dff, x="season", y="total_load_mw",
                         title="Charge par Saison", template="plotly_dark",
                         color="season")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_cols = ["temperature", "humidity", "total_load_mw", "available_power_mw",
                    "transformer_temp", "voltage", "frequency", "outage_risk", "blackout"]
        avail = [c for c in num_cols if c in dff.columns]
        corr  = dff[avail].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Matrice de Corrélation", template="plotly_dark",
                        aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if "hour" in dff.columns and "day_of_week" in dff.columns:
            pivot = dff.pivot_table("total_load_mw", "hour", "day_of_week", aggfunc="mean")
            pivot.columns = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
            fig = px.imshow(pivot, color_continuous_scale="YlOrRd",
                            title="Charge MW : Heure × Jour de la semaine",
                            template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 3 : PRÉDICTION LIVE
# ═══════════════════════════════════════════════════════════
elif page == " Prédiction Live":
    st.header(" Prédiction en Temps Réel")
    st.info("Entrez les paramètres réseau pour obtenir une prédiction instantanée.")

    col1, col2, col3 = st.columns(3)
    with col1:
        temperature      = st.slider("Température (°C)",      15.0, 50.0, 32.0, 0.5)
        humidity         = st.slider("Humidité (%)",           0.0, 100.0, 70.0, 1.0)
        transformer_temp = st.slider("Temp. Transformateur",  20.0, 120.0, 75.0, 1.0)
    with col2:
        total_load       = st.slider("Charge Totale (MW)",    50.0, 800.0, 320.0, 5.0)
        available_power  = st.slider("Puissance Dispo (MW)",  50.0, 800.0, 350.0, 5.0)
        renewable_power  = st.slider("Énergie Renouvelable",  0.0, 200.0, 30.0, 5.0)
    with col3:
        voltage          = st.slider("Tension (V)",          180.0, 260.0, 220.0, 1.0)
        frequency        = st.slider("Fréquence (Hz)",        47.0, 53.0,  50.0,  0.1)
        outage_risk_in   = st.slider("Risque estimé (%)",      0.0, 100.0, 50.0, 1.0)
        hour             = st.slider("Heure",                  0, 23, 18)

    event = st.selectbox("Événement", ["No Event", "Football Match", "Concert", "Festival", "Political Event"])

    if st.button(" Lancer la Prédiction", type="primary"):
        # Calcul des features
        load_ratio   = total_load / max(available_power, 1)
        power_margin = available_power - total_load
        volt_dev     = abs(voltage - 220)
        freq_dev     = abs(frequency - 50)
        grid_stress  = (0.4 * min(load_ratio, 2)/2 + 0.3*(transformer_temp/120) +
                        0.15*(volt_dev/40) + 0.15*(freq_dev/3))

        event_factor = 1.15 if event != "No Event" else 1.0
        risk_score   = min((outage_risk_in/100 * 0.35 + min(load_ratio,2)/2 * 0.30 +
                           grid_stress * 0.20 + transformer_temp/120 * 0.15) * event_factor, 1.0)

        pred = int(risk_score >= 0.5)

        # Affichage
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)

        risk_css = ("risk-critical" if risk_score >= 0.75 else
                    "risk-high" if risk_score >= 0.5 else
                    "risk-moderate" if risk_score >= 0.25 else "risk-low")
        risk_label = ("CRITIQUE " if risk_score >= 0.75 else
                      "ÉLEVÉ " if risk_score >= 0.5 else
                      "MODÉRÉ " if risk_score >= 0.25 else "FAIBLE ")

        with c1:
            st.metric("Prédiction Blackout", "OUI " if pred else "NON ")
        with c2:
            st.metric("Probabilité", f"{risk_score*100:.1f}%")
        with c3:
            st.metric("Ratio de Charge", f"{load_ratio:.2f}")
        with c4:
            st.metric("Stress Réseau", f"{grid_stress*100:.1f}%")

        st.markdown(f"""
        <div class="kpi-box {risk_css}" style="margin-top:16px;padding:20px;">
            <h2>Niveau de Risque : {risk_label}</h2>
            <p>Marge de puissance : {power_margin:+.1f} MW |
               Tension : {volt_dev:.1f}V d'écart |
               Fréquence : {freq_dev:.2f}Hz d'écart</p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            title={"text": "Score de Risque Global"},
            delta={"reference": 50},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": "#E63946"},
                "steps": [
                    {"range": [0,  25], "color": "#2A9D8F"},
                    {"range": [25, 50], "color": "#E9C46A"},
                    {"range": [50, 75], "color": "#F4A261"},
                    {"range": [75,100], "color": "#E63946"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 75}
            }
        ))
        fig_gauge.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig_gauge, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4 : DÉTECTION ANOMALIES
# ═══════════════════════════════════════════════════════════
elif page == " Détection Anomalies":
    st.header(" Détection d'Anomalies Réseau")
    dff = get_filtered_df()
    if dff.empty or len(dff) < 100:
        st.warning("Pas assez de données")
        st.stop()

    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    feats = [c for c in ["total_load_mw","available_power_mw","transformer_temp",
                          "voltage","frequency","outage_risk"] if c in dff.columns]
    X = dff[feats].fillna(0)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    with st.spinner("Détection en cours (Isolation Forest)..."):
        iso = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
        preds  = iso.fit_predict(X_sc)
        scores = iso.score_samples(X_sc)

    dff = dff.copy()
    dff["is_anomaly"]    = (preds == -1).astype(int)
    dff["anomaly_score"] = scores

    n_anom = dff["is_anomaly"].sum()
    st.success(f" {n_anom} anomalies détectées sur {len(dff)} observations ({n_anom/len(dff)*100:.1f}%)")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(dff, x="anomaly_score", nbins=60,
                           color="is_anomaly", barmode="overlay",
                           title="Distribution des Scores d'Anomalie",
                           template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "region" in dff.columns:
            reg_anom = dff.groupby("region")["is_anomaly"].mean().reset_index()
            fig = px.bar(reg_anom, x="region", y="is_anomaly",
                         title="Taux d'Anomalies par Région",
                         template="plotly_dark", color_discrete_sequence=["#E63946"])
            st.plotly_chart(fig, use_container_width=True)

    st.subheader(" Top Anomalies")
    top_anom = dff[dff["is_anomaly"]==1].sort_values("anomaly_score").head(20)
    st.dataframe(top_anom[["datetime","region","city","total_load_mw",
                             "transformer_temp","voltage","anomaly_score"]].reset_index(drop=True),
                 use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 5 : SÉRIES TEMPORELLES
# ═══════════════════════════════════════════════════════════
elif page == " Séries Temporelles":
    st.header(" Analyse des Séries Temporelles")
    dff = get_filtered_df()

    target = st.selectbox("Variable à analyser",
                          [c for c in ["total_load_mw","outage_risk","energy_price"] if c in dff.columns])
    freq   = st.radio("Fréquence de resampling", ["H", "D", "W"], horizontal=True,
                      format_func=lambda x: {"H":"Horaire","D":"Journalier","W":"Hebdomadaire"}[x])

    ts = dff.set_index("datetime")[target].resample(freq).mean().dropna()

    # Graphique principal
    fig = px.line(x=ts.index, y=ts.values,
                  title=f"{target} – Évolution {freq}",
                  labels={"x": "Date", "y": target},
                  template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Rolling stats
    window = st.slider("Fenêtre de lissage", 3, 30, 7)
    rolled = ts.rolling(window).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Original", opacity=0.4,
                              line=dict(color="#457B9D")))
    fig2.add_trace(go.Scatter(x=rolled.index, y=rolled.values,
                              name=f"Moyenne mobile ({window})",
                              line=dict(color="#E63946", width=2)))
    fig2.update_layout(template="plotly_dark", title="Série lissée vs originale")
    st.plotly_chart(fig2, use_container_width=True)

    # Statistiques
    st.subheader(" Statistiques Descriptives")
    stats = ts.describe()
    st.dataframe(pd.DataFrame(stats).T.round(2), use_container_width=True)
