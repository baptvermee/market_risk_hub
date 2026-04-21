"""
Page Streamlit — Exotic Options Pricer
Même design que le Vanilla Option Pricer (dark trading desk style).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.exotic_option_pricer import (
    price_asian_option,
    price_vanilla_mc,
    black_scholes_price,
)

# =========================
# Page config & CSS (identique au vanilla pricer)
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
        border-bottom: 2px solid #22d3ee;
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
        background: linear-gradient(90deg, transparent, #22d3ee, transparent);
        margin: 1.5rem 0;
        border: none;
    }

    .tag-asian {
        background-color: #1e3a5f;
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .tag-discount {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Exotic Option Pricer — Asiatique")
st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Sidebar — Paramètres (même structure que vanilla)
# =========================
st.sidebar.header("PARAMÈTRES")

st.sidebar.markdown("#### Sous-jacent")
S = st.sidebar.number_input("Spot (S)", min_value=0.01, value=100.0, step=1.0)
sigma_pct = st.sidebar.slider("Volatilité %", 1, 150, 20, 1)
sigma = sigma_pct / 100

st.sidebar.markdown("#### Contrat")
option_type = st.sidebar.selectbox("Type", ["call", "put"])
K = st.sidebar.number_input("Strike (K)", min_value=0.01, value=100.0, step=1.0)
T = st.sidebar.slider("Maturité (années)", 0.01, 5.0, 1.0, 0.01)

st.sidebar.markdown("#### Marché")
r_pct = st.sidebar.slider("Taux sans risque %", 0, 15, 5, 1)
r = r_pct / 100

st.sidebar.markdown("#### Monte Carlo")
n_sims = st.sidebar.slider(
    "Simulations",
    min_value=10000,
    max_value=500000,
    value=100000,
    step=10000,
)
n_steps = st.sidebar.slider(
    "Pas de temps",
    min_value=50,
    max_value=504,
    value=252,
    step=1,
    help="252 = 1 point par jour de trading sur 1 an",
)

# =========================
# Cache Streamlit
# =========================

@st.cache_data
def cached_asian_price(S0, K, r, sigma, T, n_steps, n_sims, option_type):
    return price_asian_option(S0, K, r, sigma, T, n_steps, n_sims, option_type)

@st.cache_data
def cached_vanilla_mc(S0, K, r, sigma, T, n_sims, option_type):
    return price_vanilla_mc(S0, K, r, sigma, T, n_sims, option_type)

# =========================
# Calculs
# =========================

with st.spinner("Simulation Monte Carlo en cours..."):
    asian_result = cached_asian_price(S, K, r, sigma, T, n_steps, n_sims, option_type)
    vanilla_mc = cached_vanilla_mc(S, K, r, sigma, T, n_sims, option_type)
    bs_price = black_scholes_price(S, K, r, sigma, T, option_type)

discount_pct = (1 - asian_result["price"] / vanilla_mc["price"]) * 100 if vanilla_mc["price"] > 0 else 0

# =========================
# Ligne 1 — KPI principaux
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Prix Asiatique (MC)", f"{asian_result['price']:.4f}")
with col2:
    st.metric("Prix Vanille (MC)", f"{vanilla_mc['price']:.4f}")
with col3:
    st.metric("Prix Black-Scholes", f"{bs_price:.4f}")
with col4:
    st.metric("Discount Asiatique", f"{discount_pct:.1f}%")
    st.markdown(
        '<span class="tag-discount">MOINS CHÈRE</span>' if discount_pct > 0
        else '<span class="tag-asian">PLUS CHÈRE</span>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Ligne 2 — Statistiques MC
# =========================
st.subheader("Statistiques Monte Carlo")

col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)

pct_itm = np.mean(asian_result["payoffs"] > 0) * 100

col_s1.metric("Erreur standard", f"{asian_result['std_error']:.6f}")
col_s2.metric("IC 95% bas", f"{asian_result['ci_lower']:.4f}")
col_s3.metric("IC 95% haut", f"{asian_result['ci_upper']:.4f}")
col_s4.metric("% In-The-Money", f"{pct_itm:.1f}%")
col_s5.metric("Écart MC vs BS (vanille)", f"{abs(vanilla_mc['price'] - bs_price):.6f}")

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Graphique 1 — Trajectoires simulées
# =========================
st.subheader("Trajectoires simulées")

n_display = min(150, n_sims)
display_idx = np.random.choice(n_sims, n_display, replace=False)

time_axis = np.linspace(0, T * 252, n_steps + 1)
paths_sample = asian_result["paths"][:, display_idx]

fig_paths = go.Figure()

for i in range(n_display):
    fig_paths.add_trace(go.Scatter(
        x=time_axis,
        y=paths_sample[:, i],
        mode="lines",
        line=dict(width=0.5, color="rgba(34, 211, 238, 0.12)"),
        showlegend=False,
    ))

fig_paths.add_hline(
    y=K, line_dash="dash", line_color="#f59e0b",
    annotation_text=f"Strike K={K}", annotation_position="top right",
)
fig_paths.add_hline(
    y=S, line_dash="dot", line_color="#34d399",
    annotation_text=f"Spot S₀={S}", annotation_position="bottom right",
)

fig_paths.update_layout(
    title=f"{n_display} trajectoires affichées sur {n_sims:,} simulées",
    xaxis_title="Jours de trading",
    yaxis_title="Prix du sous-jacent",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=500,
)

st.plotly_chart(fig_paths, use_container_width=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Graphique 2 — Distribution : Moyenne vs Prix final
# =========================
st.subheader("Distribution — Moyenne des prix vs Prix final")

st.markdown(
    "> **Pourquoi l'asiatique est moins chère ?** La distribution des moyennes (bleu) "
    "est plus resserrée que celle des prix finaux (rouge). "
    "La moyenne lisse les extrêmes → les gros payoffs sont plus rares → l'option vaut moins."
)

fig_dist = go.Figure()

fig_dist.add_trace(go.Histogram(
    x=asian_result["avg_prices"],
    nbinsx=80,
    name="Moyenne des prix (asiatique)",
    opacity=0.6,
    marker_color="#22d3ee",
))

fig_dist.add_trace(go.Histogram(
    x=asian_result["paths"][-1, :],
    nbinsx=80,
    name="Prix final (vanille)",
    opacity=0.6,
    marker_color="#f87171",
))

fig_dist.add_vline(
    x=K, line_dash="dash", line_color="#f59e0b",
    annotation_text=f"Strike K={K}", annotation_position="top right",
)

fig_dist.update_layout(
    barmode="overlay",
    title="Dispersion réduite = option moins chère",
    xaxis_title="Prix",
    yaxis_title="Fréquence",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=450,
    legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95),
)

st.plotly_chart(fig_dist, use_container_width=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Graphique 3 — Distribution des payoffs
# =========================
st.subheader("Distribution des payoffs")

fig_payoffs = go.Figure()

fig_payoffs.add_trace(go.Histogram(
    x=asian_result["payoffs"],
    nbinsx=80,
    name="Payoffs asiatique",
    opacity=0.7,
    marker_color="#22d3ee",
))

fig_payoffs.update_layout(
    title=f"Payoffs — {pct_itm:.1f}% des simulations finissent In-The-Money",
    xaxis_title="Payoff",
    yaxis_title="Fréquence",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=400,
)

st.plotly_chart(fig_payoffs, use_container_width=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("% ITM", f"{pct_itm:.1f}%")
kpi2.metric("Payoff moyen", f"{np.mean(asian_result['payoffs']):.4f}")
kpi3.metric("Payoff max", f"{np.max(asian_result['payoffs']):.4f}")

itm_payoffs = asian_result["payoffs"][asian_result["payoffs"] > 0]
kpi4.metric(
    "Payoff médian (si ITM)",
    f"{np.median(itm_payoffs):.4f}" if len(itm_payoffs) > 0 else "N/A",
)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Graphique 4 — Convergence Monte Carlo
# =========================
st.subheader("Convergence Monte Carlo")

st.markdown(
    "> Ce graphique montre comment le prix estimé se stabilise quand on augmente "
    "le nombre de simulations. Si la courbe oscille encore à la fin, "
    "il faut plus de simulations."
)

discount_factor = np.exp(-r * T)
discounted_payoffs = discount_factor * asian_result["payoffs"]
cumulative_mean = np.cumsum(discounted_payoffs) / np.arange(1, n_sims + 1)

step = max(1, n_sims // 2000)
x_conv = np.arange(1, n_sims + 1)[::step]
y_conv = cumulative_mean[::step]

fig_conv = go.Figure()

fig_conv.add_trace(go.Scatter(
    x=x_conv,
    y=y_conv,
    mode="lines",
    name="Prix estimé",
    line=dict(color="#22d3ee", width=2),
))

fig_conv.add_hline(
    y=asian_result["price"],
    line_dash="dash",
    line_color="#f59e0b",
    annotation_text=f"Prix final : {asian_result['price']:.4f}",
)

fig_conv.update_layout(
    title="Le prix converge vers sa vraie valeur",
    xaxis_title="Nombre de simulations",
    yaxis_title="Prix estimé",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=400,
)

st.plotly_chart(fig_conv, use_container_width=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Graphique 5 — Prix en fonction du strike
# =========================
st.subheader("Profil de prix — Asiatique vs Vanille par strike")

st.markdown(
    "> L'asiatique est systématiquement moins chère que la vanille, "
    "quel que soit le strike. L'écart est maximal ATM."
)

strike_range = np.linspace(S * 0.7, S * 1.3, 30)

asian_prices_by_strike = []
vanilla_prices_by_strike = []
bs_prices_by_strike = []

for k_val in strike_range:
    a_res = price_asian_option(S, k_val, r, sigma, T, n_steps, 50000, option_type)
    asian_prices_by_strike.append(a_res["price"])

    v_res = price_vanilla_mc(S, k_val, r, sigma, T, 50000, option_type)
    vanilla_prices_by_strike.append(v_res["price"])

    bs_prices_by_strike.append(black_scholes_price(S, k_val, r, sigma, T, option_type))

fig_strike = go.Figure()

fig_strike.add_trace(go.Scatter(
    x=strike_range, y=asian_prices_by_strike,
    name="Asiatique (MC)",
    line=dict(color="#22d3ee", width=2.5),
))

fig_strike.add_trace(go.Scatter(
    x=strike_range, y=vanilla_prices_by_strike,
    name="Vanille (MC)",
    line=dict(color="#f87171", width=2, dash="dash"),
))

fig_strike.add_trace(go.Scatter(
    x=strike_range, y=bs_prices_by_strike,
    name="Black-Scholes",
    line=dict(color="#64748b", width=1.5, dash="dot"),
))

fig_strike.add_vline(
    x=S, line_dash="dot", line_color="#f59e0b",
    annotation_text=f"Spot={S}",
)

fig_strike.add_trace(go.Scatter(
    x=[K], y=[asian_result["price"]],
    mode="markers",
    name="Position actuelle",
    marker=dict(color="#f43f5e", size=10, symbol="diamond"),
))

fig_strike.update_layout(
    title=f"Prix du {option_type.upper()} en fonction du strike",
    xaxis_title="Strike (K)",
    yaxis_title="Prix de l'option",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=500,
)

st.plotly_chart(fig_strike, use_container_width=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Tableau récapitulatif
# =========================
st.subheader("Récapitulatif")

recap = pd.DataFrame({
    "Paramètre": [
        "Type d'exotique", "Type", "Spot (S)", "Strike (K)", "Maturité (T)",
        "Vol (σ)", "Taux (r)", "Simulations", "Pas de temps",
        "Prix Asiatique", "Prix Vanille (MC)", "Prix Black-Scholes",
        "Discount asiatique", "Erreur standard", "% ITM",
    ],
    "Valeur": [
        "Asiatique (moyenne arithmétique)",
        option_type.upper(),
        f"{S:.2f}",
        f"{K:.2f}",
        f"{T:.2f} ans",
        f"{sigma:.1%}",
        f"{r:.1%}",
        f"{n_sims:,}",
        f"{n_steps}",
        f"{asian_result['price']:.6f}",
        f"{vanilla_mc['price']:.6f}",
        f"{bs_price:.6f}",
        f"{discount_pct:.2f}%",
        f"{asian_result['std_error']:.6f}",
        f"{pct_itm:.1f}%",
    ],
})

st.dataframe(recap, use_container_width=True, hide_index=True)