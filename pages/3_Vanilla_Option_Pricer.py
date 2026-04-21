import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.vanilla_option_pricer import black_scholes_price, compute_greeks

# =========================
# Page config & CSS
# =========================

st.markdown("""
<style>
    /* --- Police & couleurs globales --- */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* --- KPI Cards style salle de marché --- */
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

    /* --- Headers --- */
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

    /* --- Sidebar --- */
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

    /* --- Divider custom --- */
    .trading-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #22d3ee, transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* --- Tag moneyness --- */
    .tag-itm {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .tag-atm {
        background-color: #1e3a5f;
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .tag-otm {
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

st.title("🎯 Option Pricer — Vanilles")
st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Sidebar — Paramètres
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

# =========================
# Calculs
# =========================
price = black_scholes_price(S, K, T, r, sigma, option_type)
greeks = compute_greeks(S, K, T, r, sigma, option_type)

# Valeur intrinsèque et valeur temps
if option_type == "call":
    intrinsic = max(S - K, 0)
else:
    intrinsic = max(K - S, 0)
time_value = price - intrinsic

# Moneyness
moneyness_ratio = S / K
if option_type == "call":
    if S > K:
        moneyness = "ITM"
    elif S == K:
        moneyness = "ATM"
    else:
        moneyness = "OTM"
else:
    if S < K:
        moneyness = "ITM"
    elif S == K:
        moneyness = "ATM"
    else:
        moneyness = "OTM"

moneyness_class = {"ITM": "tag-itm", "ATM": "tag-atm", "OTM": "tag-otm"}

# =========================
# Affichage principal
# =========================

# --- Ligne 1 : Prix + Moneyness ---
col_main1, col_main2, col_main3, col_main4 = st.columns(4)

with col_main1:
    st.metric("Prix de l'option", f"{price:.4f}")
with col_main2:
    st.metric("Valeur intrinsèque", f"{intrinsic:.4f}")
with col_main3:
    st.metric("Valeur temps", f"{time_value:.4f}")
with col_main4:
    st.metric("Moneyness (S/K)", f"{moneyness_ratio:.4f}")
    st.markdown(
        f'<span class="{moneyness_class[moneyness]}">{moneyness}</span>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# --- Ligne 2 : Greeks ---
st.subheader("Greeks")

col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)

col_g1.metric("Δ Delta", f"{greeks['delta']:.4f}")
col_g2.metric("Γ Gamma", f"{greeks['gamma']:.4f}")
col_g3.metric("Θ Theta", f"{greeks['theta']:.4f}")
col_g4.metric("ν Vega", f"{greeks['vega']:.4f}")
col_g5.metric("ρ Rho", f"{greeks['rho']:.4f}")

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Graphiques : Prix et Greeks en fonction de S
# =========================
st.subheader("Profil de l'option en fonction du spot")

# On génère une grille de prix spot autour du strike
spot_range = np.linspace(S * 0.5, S * 1.5, 200)

# On calcule le prix et les greeks pour chaque valeur de spot
prices_curve = []
deltas = []
gammas = []
thetas = []
vegas = []

for s in spot_range:
    prices_curve.append(black_scholes_price(s, K, T, r, sigma, option_type))
    g = compute_greeks(s, K, T, r, sigma, option_type)
    deltas.append(g["delta"])
    gammas.append(g["gamma"])
    thetas.append(g["theta"])
    vegas.append(g["vega"])

# --- Graphique prix ---
fig_price = go.Figure()

fig_price.add_trace(go.Scatter(
    x=spot_range, y=prices_curve,
    name="Prix BS",
    line=dict(color="#22d3ee", width=2.5),
))

# Payoff à maturité (valeur intrinsèque)
if option_type == "call":
    payoff = [max(s - K, 0) for s in spot_range]
else:
    payoff = [max(K - s, 0) for s in spot_range]

fig_price.add_trace(go.Scatter(
    x=spot_range, y=payoff,
    name="Payoff à maturité",
    line=dict(color="#64748b", width=1.5, dash="dash"),
))

# Ligne verticale au strike
fig_price.add_vline(x=K, line_dash="dot", line_color="#f59e0b",
                    annotation_text=f"K={K}", annotation_position="top right")

# Point actuel
fig_price.add_trace(go.Scatter(
    x=[S], y=[price],
    mode="markers",
    name="Position actuelle",
    marker=dict(color="#f43f5e", size=10, symbol="diamond"),
))

fig_price.update_layout(
    title=f"Prix du {option_type.upper()} (K={K}, T={T:.2f}y, σ={sigma:.0%})",
    xaxis_title="Spot (S)",
    yaxis_title="Prix de l'option",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=450,
)

st.plotly_chart(fig_price, use_container_width=True)

# --- Graphique Greeks ---
fig_greeks = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Delta (Δ)", "Gamma (Γ)", "Theta (Θ)", "Vega (ν)"),
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

fig_greeks.add_trace(
    go.Scatter(x=spot_range, y=deltas, line=dict(color="#22d3ee", width=2)),
    row=1, col=1,
)
fig_greeks.add_trace(
    go.Scatter(x=spot_range, y=gammas, line=dict(color="#a78bfa", width=2)),
    row=1, col=2,
)
fig_greeks.add_trace(
    go.Scatter(x=spot_range, y=thetas, line=dict(color="#f87171", width=2)),
    row=2, col=1,
)
fig_greeks.add_trace(
    go.Scatter(x=spot_range, y=vegas, line=dict(color="#34d399", width=2)),
    row=2, col=2,
)

# Ligne verticale au strike sur chaque subplot
for row in [1, 2]:
    for col in [1, 2]:
        fig_greeks.add_vline(
            x=K, line_dash="dot", line_color="#f59e0b",
            row=row, col=col,
        )

fig_greeks.update_layout(
    showlegend=False,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=600,
)

st.plotly_chart(fig_greeks, use_container_width=True)

# =========================
# Surface de volatilité : Prix en fonction de S et T
# =========================
st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)
st.subheader("Surface de prix — Spot × Maturité")

spot_grid = np.linspace(S * 0.6, S * 1.4, 60)
maturity_grid = np.linspace(0.05, T * 2, 60)

# Matrice de prix : chaque cellule = prix BS pour un couple (spot, maturité)
price_surface = np.zeros((len(maturity_grid), len(spot_grid)))

for i, t_val in enumerate(maturity_grid):
    for j, s_val in enumerate(spot_grid):
        price_surface[i, j] = black_scholes_price(s_val, K, t_val, r, sigma, option_type)

fig_surface = go.Figure(data=[go.Surface(
    x=spot_grid,
    y=maturity_grid,
    z=price_surface,
    colorscale="Viridis",
    colorbar=dict(title="Prix"),
)])

fig_surface.update_layout(
    title=f"Surface de prix — {option_type.upper()} (K={K}, σ={sigma:.0%})",
    scene=dict(
        xaxis_title="Spot (S)",
        yaxis_title="Maturité (T)",
        zaxis_title="Prix",
        bgcolor="rgba(10,15,28,0.8)",
    ),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono"),
    height=600,
)

st.plotly_chart(fig_surface, use_container_width=True)

# =========================
# PnL à maturité
# =========================
st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)
st.subheader("PnL à maturité")

position = st.radio("Position", ["Long", "Short"], horizontal=True)

pnl = []
for s in spot_range:
    if option_type == "call":
        payoff_val = max(s - K, 0)
    else:
        payoff_val = max(K - s, 0)

    if position == "Long":
        pnl.append(payoff_val - price)
    else:
        pnl.append(price - payoff_val)

fig_pnl = go.Figure()

# Zone de profit (vert) et zone de perte (rouge)
pnl_array = np.array(pnl)

fig_pnl.add_trace(go.Scatter(
    x=spot_range, y=pnl_array,
    fill="tozeroy",
    line=dict(color="#22d3ee", width=2),
    fillcolor="rgba(34, 211, 238, 0.15)",
    name="PnL",
))

fig_pnl.add_hline(y=0, line_color="#64748b", line_dash="dash")
fig_pnl.add_vline(x=K, line_dash="dot", line_color="#f59e0b",
                   annotation_text=f"K={K}")

fig_pnl.update_layout(
    title=f"PnL à maturité — {position} {option_type.upper()}",
    xaxis_title="Spot à maturité",
    yaxis_title="PnL",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=400,
)

st.plotly_chart(fig_pnl, use_container_width=True)

# =========================
# Tableau récapitulatif
# =========================
st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)
st.subheader("Récapitulatif")

recap = pd.DataFrame({
    "Paramètre": ["Type", "Spot (S)", "Strike (K)", "Maturité (T)", "Vol (σ)", "Taux (r)",
                   "Prix BS", "Δ Delta", "Γ Gamma", "Θ Theta/jour", "ν Vega", "ρ Rho"],
    "Valeur": [
        option_type.upper(), f"{S:.2f}", f"{K:.2f}", f"{T:.2f} ans",
        f"{sigma:.1%}", f"{r:.1%}",
        f"{price:.4f}", f"{greeks['delta']:.4f}", f"{greeks['gamma']:.6f}",
        f"{greeks['theta']:.4f}", f"{greeks['vega']:.4f}", f"{greeks['rho']:.4f}",
    ],
})

st.dataframe(recap, use_container_width=True, hide_index=True)

# =========================
# Implied Volatility
# =========================
st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)
st.subheader("📡 Implied Volatility — Reverse Engineering")

st.markdown(
    "Entre un prix de marché observé et le pricer retrouve la volatilité "
    "implicite par Newton-Raphson."
)

from src.vanilla_option_pricer import implied_volatility

iv_col1, iv_col2 = st.columns(2)

with iv_col1:
    market_price_input = st.number_input(
        "Prix de marché observé",
        min_value=0.01,
        value=round(price, 2),  # on pré-remplit avec le prix BS actuel
        step=0.01,
    )

with iv_col2:
    iv = implied_volatility(market_price_input, S, K, T, r, option_type)

    if np.isnan(iv):
        st.error("Pas de convergence — le prix est peut-être hors bornes d'arbitrage.")
    else:
        st.metric("Volatilité implicite", f"{iv:.2%}")

        # Comparaison avec la vol qu'on a mise dans le pricer
        diff = iv - sigma
        if abs(diff) < 0.001:
            st.success("La vol implicite correspond à la vol du pricer — cohérent.")
        elif diff > 0:
            st.warning(
                f"La vol implicite ({iv:.2%}) est supérieure à votre vol ({sigma:.2%}). "
                f"Le marché price plus de risque que votre hypothèse."
            )
        else:
            st.info(
                f"La vol implicite ({iv:.2%}) est inférieure à votre vol ({sigma:.2%}). "
                f"Le marché price moins de risque que votre hypothèse."
            )

# =========================
# Smile de volatilité
# =========================
st.markdown("### Volatility Smile")
st.markdown(
    "On calcule la vol implicite pour différents strikes, à prix fixé par BS. "
    "En réalité, le smile vient du fait que le marché ne suit pas parfaitement Black-Scholes."
)

# Grille de strikes autour du spot
strikes_smile = np.linspace(S * 0.7, S * 1.3, 30)

iv_smile = []
for k in strikes_smile:
    # On génère un "prix de marché" via BS avec notre vol
    mkt_price = black_scholes_price(S, k, T, r, sigma, option_type)
    # On retrouve la vol implicite
    iv_val = implied_volatility(mkt_price, S, k, T, r, option_type)
    iv_smile.append(iv_val if not np.isnan(iv_val) else None)

fig_smile = go.Figure()

fig_smile.add_trace(go.Scatter(
    x=strikes_smile,
    y=iv_smile,
    line=dict(color="#22d3ee", width=2.5),
    name="Vol implicite",
))

fig_smile.add_vline(
    x=S, line_dash="dot", line_color="#f59e0b",
    annotation_text=f"Spot={S}",
)

fig_smile.update_layout(
    title="Volatility Smile (vol implicite par strike)",
    xaxis_title="Strike (K)",
    yaxis_title="Vol implicite",
    yaxis_tickformat=".1%",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=450,
)

st.plotly_chart(fig_smile, use_container_width=True)

st.info(
    "Ici le smile est plat car on utilise BS pour générer les prix puis BS pour retrouver la vol — "
    "c'est cohérent. Avec des prix de marché réels, le smile serait courbé : "
    "les options OTM (loin du strike) ont une vol implicite plus élevée, "
    "ce qui reflète le fait que le marché anticipe des queues de distribution plus épaisses "
    "que ce que prédit la loi normale de Black-Scholes."
)

