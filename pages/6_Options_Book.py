"""
Page Streamlit — Options Book
Gestion d'un portefeuille d'options avec Greeks agrégés.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.options_book import (
    compute_book_greeks,
    compute_book_pnl,
    compute_greeks_profile,
)

# =========================
# CSS trading desk
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0a0f1c 0%, #111827 100%);
        border: 1px solid #1e293b;
        padding: 16px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }

    div[data-testid="stMetric"] label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #64748b;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    h1 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
        border-bottom: 2px solid #f59e0b;
        padding-bottom: 8px;
    }

    h2, h3 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.3px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1c 0%, #0f172a 100%);
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
    }

    .trading-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #f59e0b, transparent);
        margin: 1.5rem 0;
        border: none;
    }

    .tag-long {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .tag-short {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

plotly_layout = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
)

st.title("📒 Options Book")
st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Sidebar — Paramètres marché
# =========================
st.sidebar.header("PARAMÈTRES MARCHÉ")

S = st.sidebar.number_input("Spot (S)", min_value=0.01, value=100.0, step=1.0)
sigma_pct = st.sidebar.slider("Volatilité %", 1, 150, 20, 1)
sigma = sigma_pct / 100
r_pct = st.sidebar.slider("Taux sans risque %", 0, 15, 5, 1)
r = r_pct / 100

# =========================
# Stratégies prédéfinies
# =========================
st.sidebar.markdown("---")
st.sidebar.header("STRATÉGIES")

strategy = st.sidebar.selectbox(
    "Charger une stratégie",
    [
        "Personnalisé",
        "Long Straddle",
        "Short Straddle",
        "Bull Call Spread",
        "Bear Put Spread",
        "Long Butterfly",
        "Iron Condor",
        "Covered Call",
        "Protective Put",
    ],
)

# Stratégies prédéfinies — chaque stratégie est une liste de positions
preset_strategies = {
    "Long Straddle": [
        {"type": "call", "strike": 100.0, "maturity": 0.5, "quantity": 1},
        {"type": "put", "strike": 100.0, "maturity": 0.5, "quantity": 1},
    ],
    "Short Straddle": [
        {"type": "call", "strike": 100.0, "maturity": 0.5, "quantity": -1},
        {"type": "put", "strike": 100.0, "maturity": 0.5, "quantity": -1},
    ],
    "Bull Call Spread": [
        {"type": "call", "strike": 95.0, "maturity": 0.5, "quantity": 1},
        {"type": "call", "strike": 105.0, "maturity": 0.5, "quantity": -1},
    ],
    "Bear Put Spread": [
        {"type": "put", "strike": 105.0, "maturity": 0.5, "quantity": 1},
        {"type": "put", "strike": 95.0, "maturity": 0.5, "quantity": -1},
    ],
    "Long Butterfly": [
        {"type": "call", "strike": 90.0, "maturity": 0.5, "quantity": 1},
        {"type": "call", "strike": 100.0, "maturity": 0.5, "quantity": -2},
        {"type": "call", "strike": 110.0, "maturity": 0.5, "quantity": 1},
    ],
    "Iron Condor": [
        {"type": "put", "strike": 90.0, "maturity": 0.5, "quantity": -1},
        {"type": "put", "strike": 95.0, "maturity": 0.5, "quantity": 1},
        {"type": "call", "strike": 105.0, "maturity": 0.5, "quantity": 1},
        {"type": "call", "strike": 110.0, "maturity": 0.5, "quantity": -1},
    ],
    "Covered Call": [
        {"type": "call", "strike": 105.0, "maturity": 0.5, "quantity": -1},
    ],
    "Protective Put": [
        {"type": "put", "strike": 95.0, "maturity": 0.5, "quantity": 1},
    ],
}

# =========================
# Construction du book
# =========================
st.subheader("Construction du book")

if strategy != "Personnalisé":
    st.info(f"Stratégie chargée : **{strategy}**. Tu peux modifier les positions ci-dessous.")

# Initialiser les positions
if strategy != "Personnalisé" and strategy in preset_strategies:
    default_positions = preset_strategies[strategy]
else:
    default_positions = [
        {"type": "call", "strike": 100.0, "maturity": 1.0, "quantity": 1},
    ]

# Nombre de positions
n_positions = st.number_input(
    "Nombre de positions",
    min_value=1, max_value=10,
    value=len(default_positions),
    step=1,
)

# Saisie de chaque position
positions = []
for i in range(int(n_positions)):
    st.markdown(f"#### Position {i+1}")

    # Valeurs par défaut depuis la stratégie
    default = default_positions[i] if i < len(default_positions) else {
        "type": "call", "strike": 100.0, "maturity": 1.0, "quantity": 1
    }

    cols = st.columns(4)

    with cols[0]:
        opt_type = st.selectbox(
            f"Type #{i+1}", ["call", "put"],
            index=0 if default["type"] == "call" else 1,
            key=f"type_{i}",
        )

    with cols[1]:
        strike = st.number_input(
            f"Strike #{i+1}", min_value=0.01, value=default["strike"],
            step=1.0, key=f"strike_{i}",
        )

    with cols[2]:
        maturity = st.number_input(
            f"Maturité (ans) #{i+1}", min_value=0.01, value=default["maturity"],
            step=0.05, key=f"mat_{i}",
        )

    with cols[3]:
        quantity = st.number_input(
            f"Quantité #{i+1}", min_value=-100, max_value=100,
            value=default["quantity"], step=1, key=f"qty_{i}",
            help="Positif = Long, Négatif = Short",
        )

    if quantity != 0:
        positions.append({
            "type": opt_type,
            "strike": strike,
            "maturity": maturity,
            "quantity": int(quantity),
        })

if len(positions) == 0:
    st.warning("Aucune position active (toutes les quantités sont à 0).")
    st.stop()

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Greeks agrégés
# =========================
book = compute_book_greeks(positions, S, r, sigma)

st.subheader("Greeks agrégés du book")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Valeur du book", f"{book['total_value']:.4f}")
col2.metric("Δ Delta net", f"{book['total_delta']:.4f}")
col3.metric("Γ Gamma net", f"{book['total_gamma']:.4f}")
col4.metric("Θ Theta net", f"{book['total_theta']:.4f}")
col5.metric("ν Vega net", f"{book['total_vega']:.4f}")
col6.metric("ρ Rho net", f"{book['total_rho']:.4f}")

# Interprétation
if abs(book["total_delta"]) < 0.05:
    st.markdown('<span class="tag-long">DELTA NEUTRAL</span>', unsafe_allow_html=True)
elif book["total_delta"] > 0:
    st.markdown(f'<span class="tag-long">LONG DELTA ({book["total_delta"]:+.2f})</span>', unsafe_allow_html=True)
else:
    st.markdown(f'<span class="tag-short">SHORT DELTA ({book["total_delta"]:+.2f})</span>', unsafe_allow_html=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Détail par position
# =========================
st.subheader("Détail par position")

details_df = pd.DataFrame(book["details"])
display_df = details_df.copy()
display_df["direction"] = display_df["quantity"].apply(lambda x: "LONG" if x > 0 else "SHORT")
display_df["price"] = display_df["price"].apply(lambda x: f"{x:.4f}")
display_df["value"] = display_df["value"].apply(lambda x: f"{x:.4f}")
display_df["delta"] = display_df["delta"].apply(lambda x: f"{x:+.4f}")
display_df["gamma"] = display_df["gamma"].apply(lambda x: f"{x:+.4f}")
display_df["theta"] = display_df["theta"].apply(lambda x: f"{x:+.4f}")
display_df["vega"] = display_df["vega"].apply(lambda x: f"{x:+.4f}")

display_df = display_df[["direction", "type", "strike", "maturity", "quantity", "price", "value", "delta", "gamma", "theta", "vega"]]
display_df.columns = ["Dir", "Type", "Strike", "Maturité", "Qty", "Prix unit.", "Valeur", "Delta", "Gamma", "Theta", "Vega"]

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# P&L du book
# =========================
st.subheader("P&L du book en fonction du spot")

spot_range = np.linspace(S * 0.5, S * 1.5, 300)

pnl_result = compute_book_pnl(positions, S, r, sigma, spot_range)

fig_pnl = go.Figure()

# P&L actuel (avec valeur temps)
fig_pnl.add_trace(go.Scatter(
    x=spot_range, y=pnl_result["pnl_current"],
    name="P&L actuel",
    line=dict(color="#22d3ee", width=2.5),
))

# P&L à maturité (payoff pur)
fig_pnl.add_trace(go.Scatter(
    x=spot_range, y=pnl_result["pnl_at_expiry"],
    name="P&L à maturité",
    line=dict(color="#64748b", width=1.5, dash="dash"),
))

# Ligne de zéro
fig_pnl.add_hline(y=0, line_color="#64748b", line_dash="dot")

# Spot actuel
fig_pnl.add_vline(
    x=S, line_dash="dot", line_color="#f59e0b",
    annotation_text=f"Spot={S}",
)

# Strikes
for pos in positions:
    fig_pnl.add_vline(
        x=pos["strike"], line_dash="dot", line_color="rgba(167,139,250,0.3)",
    )

fig_pnl.update_layout(
    title="P&L du book — actuel vs à maturité",
    xaxis_title="Spot",
    yaxis_title="P&L",
    height=500,
    **plotly_layout,
)

st.plotly_chart(fig_pnl, use_container_width=True)

# Breakeven approximatif
breakevens = []
for i in range(1, len(spot_range)):
    if pnl_result["pnl_at_expiry"][i-1] * pnl_result["pnl_at_expiry"][i] < 0:
        breakevens.append(round(spot_range[i], 2))

if breakevens:
    be_cols = st.columns(len(breakevens))
    for i, be in enumerate(breakevens):
        be_cols[i].metric(f"Breakeven {i+1}", f"{be:.2f}")

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Profil des Greeks
# =========================
st.subheader("Profil des Greeks en fonction du spot")

greeks_profile = compute_greeks_profile(positions, r, sigma, spot_range)

fig_greeks = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Delta net (Δ)", "Gamma net (Γ)", "Theta net (Θ)", "Vega net (ν)"),
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

fig_greeks.add_trace(
    go.Scatter(x=spot_range, y=greeks_profile["deltas"], line=dict(color="#22d3ee", width=2)),
    row=1, col=1,
)
fig_greeks.add_trace(
    go.Scatter(x=spot_range, y=greeks_profile["gammas"], line=dict(color="#a78bfa", width=2)),
    row=1, col=2,
)
fig_greeks.add_trace(
    go.Scatter(x=spot_range, y=greeks_profile["thetas"], line=dict(color="#f87171", width=2)),
    row=2, col=1,
)
fig_greeks.add_trace(
    go.Scatter(x=spot_range, y=greeks_profile["vegas"], line=dict(color="#34d399", width=2)),
    row=2, col=2,
)

# Lignes de zéro et spot actuel sur chaque subplot
for row in [1, 2]:
    for col in [1, 2]:
        fig_greeks.add_hline(y=0, line_dash="dot", line_color="#64748b", row=row, col=col)
        fig_greeks.add_vline(x=S, line_dash="dot", line_color="#f59e0b", row=row, col=col)

fig_greeks.update_layout(
    showlegend=False,
    height=600,
    **plotly_layout,
)

st.plotly_chart(fig_greeks, use_container_width=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Scénario de choc
# =========================
st.subheader("Analyse de scénario")

st.markdown(
    "> Simule un choc sur le spot et/ou la volatilité "
    "pour voir l'impact sur le book."
)

sc1, sc2 = st.columns(2)

with sc1:
    spot_shock_pct = st.slider("Choc spot (%)", -30, 30, 0, 1)
with sc2:
    vol_shock_pts = st.slider("Choc vol (points)", -10, 10, 0, 1)

S_shocked = S * (1 + spot_shock_pct / 100)
sigma_shocked = max(sigma + vol_shock_pts / 100, 0.01)

book_shocked = compute_book_greeks(positions, S_shocked, r, sigma_shocked)

pnl_shock = book_shocked["total_value"] - book["total_value"]

col_sc1, col_sc2, col_sc3, col_sc4 = st.columns(4)
col_sc1.metric("Nouveau spot", f"{S_shocked:.2f}", delta=f"{spot_shock_pct:+d}%")
col_sc2.metric("Nouvelle vol", f"{sigma_shocked:.1%}", delta=f"{vol_shock_pts:+d}pts")
col_sc3.metric("P&L du choc", f"{pnl_shock:+.4f}")
col_sc4.metric("Nouveau delta", f"{book_shocked['total_delta']:+.4f}")

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Récapitulatif
# =========================
st.subheader("Récapitulatif")

recap = pd.DataFrame({
    "Paramètre": [
        "Spot", "Volatilité", "Taux", "Nb positions",
        "Valeur du book", "Delta net", "Gamma net",
        "Theta net (/jour)", "Vega net", "Rho net",
    ],
    "Valeur": [
        f"{S:.2f}", f"{sigma:.1%}", f"{r:.1%}", f"{len(positions)}",
        f"{book['total_value']:.4f}",
        f"{book['total_delta']:+.4f}",
        f"{book['total_gamma']:+.4f}",
        f"{book['total_theta']:+.4f}",
        f"{book['total_vega']:+.4f}",
        f"{book['total_rho']:+.4f}",
    ],
})

st.dataframe(recap, use_container_width=True, hide_index=True)