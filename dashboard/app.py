"""
app.py — Dashboard Dash professionnel
CEET Smart Grid · Energy Blackout Prediction Project
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc

from utils import DATA_PROC, DATA_RAW, COLORS

# ─── Chargement des données ───────────────────────────────────────────────────
def load_data():
    proc = DATA_PROC / "ceet_processed.csv"
    raw  = DATA_RAW  / "ceet_togo_smartgrid_dataset.csv"
    path = proc if proc.exists() else raw
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["date"] = df["datetime"].dt.date
    if "load_ratio" not in df.columns:
        df["load_ratio"] = df["total_load_mw"] / df["available_power_mw"].replace(0, np.nan)
    if "power_margin" not in df.columns:
        df["power_margin"] = df["available_power_mw"] - df["total_load_mw"]
    return df

DF = load_data()
REGIONS = ["Toutes"] + sorted(DF["region"].unique().tolist())
CITIES  = ["Toutes"] + sorted(DF["city"].unique().tolist())
DATE_MIN = str(DF["datetime"].min().date())
DATE_MAX = str(DF["datetime"].max().date())

# ─── Application ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
    title="CEET Smart Grid Dashboard",
    suppress_callback_exceptions=True,
)

# ─── Styles ───────────────────────────────────────────────────────────────────
CARD_STYLE = {
    "borderRadius": "12px",
    "border": "none",
    "boxShadow": "0 4px 15px rgba(0,0,0,0.3)",
    "marginBottom": "16px",
}
KPI_VALUE = {"fontSize": "2.4rem", "fontWeight": "bold", "margin": "0"}
KPI_LABEL = {"fontSize": "0.85rem", "opacity": "0.75", "marginTop": "4px"}

PLOTLY_TEMPLATE = "plotly_dark"

# ─── KPI Cards ────────────────────────────────────────────────────────────────
def kpi_card(title, value, icon, color, subtitle=""):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"fas {icon} fa-2x", style={"color": color, "marginRight": "12px"}),
                html.Div([
                    html.P(value,    style={**KPI_VALUE, "color": color}),
                    html.P(title,    style=KPI_LABEL),
                    html.P(subtitle, style={"fontSize": "0.75rem", "opacity": "0.55"}),
                ])
            ], style={"display": "flex", "alignItems": "center"})
        ])
    ], style=CARD_STYLE)


# ─── Layout ────────────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H2(" CEET Smart Grid – Dashboard de Monitoring",
                    className="text-center my-3",
                    style={"color": "#E63946", "fontWeight": "bold"}),
            html.P("Prédiction des délestages · Analyse du réseau électrique togolais · Temps réel",
                   className="text-center text-muted mb-4"),
        ])
    ]),

    # Filtres
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Région", className="fw-bold"),
                    dcc.Dropdown(REGIONS, value="Toutes", id="filter-region",
                                 clearable=False, className="text-dark"),
                ], md=3),
                dbc.Col([
                    html.Label("Ville", className="fw-bold"),
                    dcc.Dropdown(CITIES, value="Toutes", id="filter-city",
                                 clearable=False, className="text-dark"),
                ], md=3),
                dbc.Col([
                    html.Label("Plage de dates", className="fw-bold"),
                    dcc.DatePickerRange(
                        id="filter-dates",
                        start_date=DATE_MIN, end_date=DATE_MAX,
                        min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX,
                        display_format="DD/MM/YYYY",
                        className="text-dark",
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Saison", className="fw-bold"),
                    dcc.Dropdown(
                        ["Toutes"] + sorted(DF["season"].unique().tolist()),
                        value="Toutes", id="filter-season",
                        clearable=False, className="text-dark"
                    ),
                ], md=2),
            ])
        ])
    ], style=CARD_STYLE),

    # KPIs dynamiques
    html.Div(id="kpi-row"),

    # Graphiques principaux – Ligne 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(" Évolution de la Charge Électrique"),
                dbc.CardBody(dcc.Graph(id="chart-load-ts", style={"height": "320px"}))
            ], style=CARD_STYLE)
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(" Blackouts par Région"),
                dbc.CardBody(dcc.Graph(id="chart-region-bar", style={"height": "320px"}))
            ], style=CARD_STYLE)
        ], md=4),
    ]),

    # Graphiques – Ligne 2
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(" Heatmap Heure × Jour"),
                dbc.CardBody(dcc.Graph(id="chart-heatmap", style={"height": "320px"}))
            ], style=CARD_STYLE)
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(" Risque de Panne par Heure"),
                dbc.CardBody(dcc.Graph(id="chart-hourly-risk", style={"height": "320px"}))
            ], style=CARD_STYLE)
        ], md=6),
    ]),

    # Graphiques – Ligne 3
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(" Impact des Événements"),
                dbc.CardBody(dcc.Graph(id="chart-events", style={"height": "300px"}))
            ], style=CARD_STYLE)
        ], md=5),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(" Charge vs Disponibilité"),
                dbc.CardBody(dcc.Graph(id="chart-scatter", style={"height": "300px"}))
            ], style=CARD_STYLE)
        ], md=7),
    ]),

    # Tableau des événements critiques
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(" Événements Critiques (Risque > 90%)"),
                dbc.CardBody(html.Div(id="table-critical"))
            ], style=CARD_STYLE)
        ])
    ]),

    # Footer
    html.Hr(),
    html.P("© 2024 CEET Smart Grid Analytics · Données simulées à des fins de démonstration",
           className="text-center text-muted small mb-4"),

], fluid=True, style={"backgroundColor": "#1a1a2e", "minHeight": "100vh", "padding": "20px"})


# ─── Callback : filtrage ───────────────────────────────────────────────────────
def filter_df(region, city, start_date, end_date, season):
    dff = DF.copy()
    if region != "Toutes":
        dff = dff[dff["region"] == region]
    if city != "Toutes":
        dff = dff[dff["city"] == city]
    if start_date:
        dff = dff[dff["datetime"] >= start_date]
    if end_date:
        dff = dff[dff["datetime"] <= end_date]
    if season != "Toutes":
        dff = dff[dff["season"] == season]
    return dff


@app.callback(
    Output("kpi-row", "children"),
    Output("chart-load-ts", "figure"),
    Output("chart-region-bar", "figure"),
    Output("chart-heatmap", "figure"),
    Output("chart-hourly-risk", "figure"),
    Output("chart-events", "figure"),
    Output("chart-scatter", "figure"),
    Output("table-critical", "children"),
    Input("filter-region", "value"),
    Input("filter-city", "value"),
    Input("filter-dates", "start_date"),
    Input("filter-dates", "end_date"),
    Input("filter-season", "value"),
)
def update_dashboard(region, city, start_date, end_date, season):
    dff = filter_df(region, city, start_date, end_date, season)
    n   = len(dff)

    if n == 0:
        empty = go.Figure().update_layout(template=PLOTLY_TEMPLATE,
                                          title="Aucune donnée pour ces filtres")
        return (html.P("Aucune donnée"), empty, empty, empty, empty, empty, empty,
                html.P("Aucune donnée"))

    # ── KPIs ──
    blackout_rate = dff["blackout"].mean() * 100
    avg_risk      = dff["outage_risk"].mean()
    total_shed    = dff["load_shedding_mw"].sum()
    overload_rate = dff["overload"].mean() * 100
    avg_load      = dff["total_load_mw"].mean()

    kpi_row = dbc.Row([
        dbc.Col(kpi_card("Taux de Blackout",    f"{blackout_rate:.2f}%",  "fa-bolt",        "#E63946"), md=2),
        dbc.Col(kpi_card("Risque Moyen",         f"{avg_risk:.1f}%",       "fa-exclamation", "#F4A261"), md=2),
        dbc.Col(kpi_card("Délestage Total",      f"{total_shed:,.0f} MW",  "fa-plug",        "#E9C46A"), md=2),
        dbc.Col(kpi_card("Taux de Surcharge",    f"{overload_rate:.1f}%",  "fa-fire",        "#FF6B6B"), md=2),
        dbc.Col(kpi_card("Charge Moy.",          f"{avg_load:.0f} MW",     "fa-chart-bar",   "#457B9D"), md=2),
        dbc.Col(kpi_card("Observations",         f"{n:,}",                 "fa-database",    "#2A9D8F"), md=2),
    ])

    # ── Série temporelle charge ──
    ts = dff.set_index("datetime")["total_load_mw"].resample("D").mean().reset_index()
    fig_ts = px.line(ts, x="datetime", y="total_load_mw",
                     title="Charge Journalière (MW)",
                     labels={"total_load_mw": "Charge (MW)", "datetime": ""},
                     template=PLOTLY_TEMPLATE, color_discrete_sequence=["#457B9D"])
    fig_ts.update_layout(margin=dict(t=30, b=20, l=40, r=20))

    # ── Blackouts par région ──
    reg_bl = dff.groupby("region")["blackout"].agg(["sum", "count"]).reset_index()
    reg_bl["rate"] = reg_bl["sum"] / reg_bl["count"] * 100
    reg_bl = reg_bl.sort_values("rate", ascending=True)
    fig_region = px.bar(reg_bl, x="rate", y="region", orientation="h",
                        title="Taux de Blackout (%)",
                        labels={"rate": "%", "region": ""},
                        template=PLOTLY_TEMPLATE, color="rate",
                        color_continuous_scale="Reds")
    fig_region.update_layout(margin=dict(t=30, b=20, l=20, r=20), showlegend=False)

    # ── Heatmap heure × jour ──
    if "hour" in dff.columns and "day_of_week" in dff.columns:
        pivot = dff.pivot_table("total_load_mw", "hour", "day_of_week", aggfunc="mean")
        pivot.columns = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        fig_heat = px.imshow(pivot, title="Charge Moy. (MW)",
                             color_continuous_scale="YlOrRd",
                             labels={"x": "Jour", "y": "Heure"},
                             template=PLOTLY_TEMPLATE)
    else:
        fig_heat = go.Figure().update_layout(template=PLOTLY_TEMPLATE)
    fig_heat.update_layout(margin=dict(t=30, b=20, l=40, r=20))

    # ── Risque par heure ──
    hrisk = dff.groupby("hour").agg(
        risk=("outage_risk", "mean"),
        blackouts=("blackout", "sum")
    ).reset_index()
    fig_hrisk = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hrisk.add_trace(go.Bar(x=hrisk["hour"], y=hrisk["blackouts"], name="Blackouts",
                               marker_color="#E63946", opacity=0.6), secondary_y=False)
    fig_hrisk.add_trace(go.Scatter(x=hrisk["hour"], y=hrisk["risk"], name="Risque moyen",
                                   mode="lines+markers", line=dict(color="#F4A261", width=2)),
                        secondary_y=True)
    fig_hrisk.update_layout(template=PLOTLY_TEMPLATE, margin=dict(t=30, b=20, l=40, r=40))

    # ── Impact événements ──
    ev_stats = dff.groupby("event")["outage_risk"].mean().reset_index().sort_values("outage_risk")
    fig_events = px.bar(ev_stats, x="outage_risk", y="event", orientation="h",
                        title="Risque Moyen (%)", labels={"outage_risk": "Risque"},
                        template=PLOTLY_TEMPLATE, color="outage_risk",
                        color_continuous_scale="Oranges")
    fig_events.update_layout(margin=dict(t=30, b=20, l=20, r=20))

    # ── Scatter charge vs disponibilité ──
    sample = dff.sample(min(3000, n), random_state=42)
    fig_scatter = px.scatter(
        sample, x="available_power_mw", y="total_load_mw",
        color="blackout", size="outage_risk",
        color_discrete_map={0: "#2A9D8F", 1: "#E63946"},
        labels={"available_power_mw": "Puissance Disponible (MW)",
                "total_load_mw": "Charge (MW)"},
        template=PLOTLY_TEMPLATE, opacity=0.6
    )
    # Ligne y = x
    lim = max(sample["available_power_mw"].max(), sample["total_load_mw"].max())
    fig_scatter.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                     line=dict(dash="dash", color="white", width=1),
                                     name="Équilibre"))
    fig_scatter.update_layout(margin=dict(t=30, b=20, l=40, r=20))

    # ── Tableau des événements critiques ──
    critical = dff[dff["outage_risk"] > 90][[
        "datetime", "region", "city", "outage_risk", "total_load_mw",
        "available_power_mw", "load_shedding_mw", "event"
    ]].sort_values("outage_risk", ascending=False).head(20)
    critical["datetime"] = critical["datetime"].astype(str)

    table = dash_table.DataTable(
        data=critical.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in critical.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#E63946", "color": "white", "fontWeight": "bold"},
        style_data={"backgroundColor": "#1e1e3a", "color": "white"},
        style_data_conditional=[{
            "if": {"row_index": "odd"},
            "backgroundColor": "#252545",
        }],
        page_size=10,
    )

    return (kpi_row, fig_ts, fig_region, fig_heat, fig_hrisk, fig_events, fig_scatter, table)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
