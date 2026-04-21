import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from scipy.stats import norm

from src.risk_engine import (
    var_historical,
    var_parametric,
    expected_shortfall,
    compute_drawdown,
    monte_carlo_multivariate,
    kupiec_test,
    christoffersen_test,
)

from src.data_loader import load_prices, compute_returns

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
        border-bottom: 2px solid #f87171;
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
        background: linear-gradient(90deg, transparent, #f87171, transparent);
        margin: 1.5rem 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

plotly_layout = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
)

st.title("⚠️ Risk Analytics")
st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
st.sidebar.header("PARAMÈTRES")

st.sidebar.markdown("#### Portefeuille")
tickers_input = st.sidebar.text_input(
    "Tickers (séparés par des virgules)",
    value="SPY"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

st.sidebar.markdown("#### Période")
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("today"))

st.sidebar.markdown("#### Risque")
confidence_level = st.sidebar.slider(
    "Niveau de confiance VaR / ES",
    min_value=0.90, max_value=0.99, value=0.95, step=0.01,
)

portfolio_value = st.sidebar.number_input(
    "Valeur du portefeuille",
    min_value=1000.0, value=1_000_000.0, step=10000.0,
)

var_method = st.sidebar.selectbox("Méthode de VaR", ["Historique", "Paramétrique"])

rolling_window = st.sidebar.slider(
    "Fenêtre rolling (jours)",
    min_value=20, max_value=252, value=60, step=5,
)

# =========================
# Chargement données
# =========================
@st.cache_data
def get_prices(tickers, start, end):
    return load_prices(tickers, start, end)

if len(tickers) == 0:
    st.error("Veuillez saisir au moins un ticker.")
    st.stop()

data = get_prices(tickers, start_date, end_date)

if data.empty:
    st.error("Aucune donnée récupérée. Vérifie les tickers saisis.")
    st.stop()

data = data.dropna(how="all")
returns = compute_returns(data)

if returns.empty:
    st.error("Impossible de calculer les rendements.")
    st.stop()

valid_tickers = list(returns.columns)

# =========================
# Pondérations du portefeuille
# =========================
if len(valid_tickers) == 1:
    single_ticker = valid_tickers[0]
    weights_series = pd.Series({single_ticker: 1.0}, dtype=float)
else:
    st.subheader("Pondérations du portefeuille")

    weights = {}
    cols = st.columns(min(4, len(valid_tickers)))

    for i, ticker in enumerate(valid_tickers):
        col = cols[i % len(cols)]
        with col:
            weights[ticker] = st.number_input(
                f"Poids {ticker}",
                min_value=0.0, max_value=1.0,
                value=float(round(1.0 / len(valid_tickers), 4)),
                step=0.01, key=f"weight_{ticker}"
            )

    weights_series = pd.Series(weights, dtype=float)

    if weights_series.sum() == 0:
        st.error("La somme des pondérations ne peut pas être nulle.")
        st.stop()

    weights_series = weights_series / weights_series.sum()

    fig_weights = go.Figure(data=[go.Pie(
        labels=weights_series.index,
        values=weights_series.values,
        hole=0.4,
        marker=dict(colors=["#22d3ee", "#a78bfa", "#f87171", "#34d399", "#f59e0b"]),
        textfont=dict(family="JetBrains Mono"),
    )])
    fig_weights.update_layout(title="Répartition du portefeuille", height=350, **plotly_layout)
    st.plotly_chart(fig_weights, use_container_width=True)

    st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Rendement portefeuille
# =========================
portfolio_returns = returns.mul(weights_series, axis=1).sum(axis=1)

# =========================
# VaR / ES
# =========================
st.subheader("VaR & Expected Shortfall")

if var_method == "Historique":
    var_return = var_historical(portfolio_returns, confidence_level)
    method_label = "VaR historique"
else:
    var_return = var_parametric(portfolio_returns, confidence_level)
    method_label = "VaR paramétrique"

es_return = expected_shortfall(portfolio_returns, confidence_level)

var_amount = -var_return * portfolio_value
es_amount = -es_return * portfolio_value if pd.notna(es_return) else np.nan

col1, col2, col3, col4 = st.columns(4)
col1.metric("VaR (%)", f"{var_return:.2%}")
col2.metric("VaR (montant)", f"{var_amount:,.0f}")
col3.metric("Expected Shortfall", f"{es_amount:,.0f}" if pd.notna(es_amount) else "N/A")
col4.metric("Vol journalière", f"{portfolio_returns.std():.2%}")

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# Distribution des rendements
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=portfolio_returns,
    nbinsx=60,
    marker_color="#22d3ee",
    opacity=0.7,
))
fig_hist.add_vline(
    x=var_return, line_dash="dash", line_color="#f87171",
    annotation_text=f"VaR {confidence_level:.0%}",
)
fig_hist.update_layout(
    title="Distribution des rendements du portefeuille",
    xaxis_title="Rendement journalier",
    yaxis_title="Fréquence",
    height=400,
    **plotly_layout,
)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Rolling VaR
# =========================
st.subheader("Rolling VaR")

alpha = 1 - confidence_level

if var_method == "Historique":
    rolling_var = portfolio_returns.rolling(rolling_window).quantile(alpha)
else:
    rolling_mean = portfolio_returns.rolling(rolling_window).mean()
    rolling_std = portfolio_returns.rolling(rolling_window).std()
    z_score = norm.ppf(confidence_level)
    rolling_var = rolling_mean - z_score * rolling_std

rolling_var_amount = -rolling_var * portfolio_value

fig_rvar = go.Figure()
fig_rvar.add_trace(go.Scatter(
    x=rolling_var_amount.index, y=rolling_var_amount,
    line=dict(color="#f87171", width=2),
    fill="tozeroy",
    fillcolor="rgba(248, 113, 113, 0.1)",
))
fig_rvar.update_layout(
    title=f"Rolling {method_label} ({rolling_window} jours)",
    xaxis_title="Date",
    yaxis_title="VaR (montant)",
    height=400,
    **plotly_layout,
)
st.plotly_chart(fig_rvar, use_container_width=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Backtesting VaR
# =========================
st.subheader("Backtesting VaR")

backtest_df = pd.DataFrame({
    "Return": portfolio_returns,
    "VaR": rolling_var
}).dropna()

backtest_df["Breach"] = backtest_df["Return"] < backtest_df["VaR"]

breach_count = int(backtest_df["Breach"].sum())
obs_count = len(backtest_df)
breach_rate = breach_count / obs_count if obs_count > 0 else np.nan
expected_breach_rate = 1 - confidence_level

col5, col6, col7 = st.columns(3)
col5.metric("Dépassements", f"{breach_count}")
col6.metric("Taux observé", f"{breach_rate:.2%}" if pd.notna(breach_rate) else "N/A")
col7.metric("Taux théorique", f"{expected_breach_rate:.2%}")

fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(
    x=backtest_df.index, y=backtest_df["Return"],
    name="Rendement", line=dict(color="#22d3ee", width=1.5),
))
fig_bt.add_trace(go.Scatter(
    x=backtest_df.index, y=backtest_df["VaR"],
    name="VaR", line=dict(color="#f87171", width=2, dash="dash"),
))
fig_bt.update_layout(
    title="Backtesting : rendement vs VaR",
    xaxis_title="Date",
    yaxis_title="Rendement",
    height=450,
    **plotly_layout,
)
st.plotly_chart(fig_bt, use_container_width=True)

# Tests statistiques
st.markdown("#### Tests statistiques")

kupiec = kupiec_test(portfolio_returns, rolling_var, confidence_level)
christoff = christoffersen_test(portfolio_returns, rolling_var, confidence_level)

col_k1, col_k2, col_k3 = st.columns(3)
col_k1.metric("Kupiec p-value", f"{kupiec['p_value']:.4f}" if pd.notna(kupiec['p_value']) else "N/A")
col_k2.metric("Christoffersen p-value", f"{christoff['p_value']:.4f}" if pd.notna(christoff['p_value']) else "N/A")
col_k3.metric("Dépassements", f"{kupiec['n_breaches']} / {kupiec['n_obs']}")

if pd.notna(kupiec["p_value"]):
    if kupiec["reject"]:
        st.error("❌ Kupiec rejeté : le nombre de dépassements est incohérent avec le modèle.")
    else:
        st.success("✅ Kupiec validé : dépassements cohérents.")

if pd.notna(christoff["p_value"]):
    if christoff["reject"]:
        st.error("❌ Christoffersen rejeté : les dépassements arrivent en cluster.")
    else:
        st.success("✅ Christoffersen validé : dépassements indépendants.")

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Drawdown
# =========================
st.subheader("Drawdown")

drawdown = compute_drawdown(portfolio_returns)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown.index, y=drawdown,
    line=dict(color="#f87171", width=2),
    fill="tozeroy",
    fillcolor="rgba(248, 113, 113, 0.15)",
))
fig_dd.update_layout(
    title="Drawdown du portefeuille",
    xaxis_title="Date",
    yaxis_title="Drawdown",
    yaxis_tickformat=".0%",
    height=400,
    **plotly_layout,
)
st.plotly_chart(fig_dd, use_container_width=True)

max_dd = drawdown.min()
st.metric("Max Drawdown", f"{max_dd:.2%}")

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Stress tests
# =========================
st.subheader("Stress Testing")

scenario = st.selectbox(
    "Scénario prédéfini",
    ["Scénario personnalisé", "Correction actions modérée",
     "Choc inflation / taux", "Stress volatilité", "Crise sévère multi-facteurs"],
)

default_equity, default_rate, default_vol = -10, 100, 20

if scenario == "Correction actions modérée":
    default_equity, default_rate, default_vol = -10, 50, 15
elif scenario == "Choc inflation / taux":
    default_equity, default_rate, default_vol = -8, 150, 10
elif scenario == "Stress volatilité":
    default_equity, default_rate, default_vol = -5, 25, 40
elif scenario == "Crise sévère multi-facteurs":
    default_equity, default_rate, default_vol = -25, 200, 60

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    equity_shock = st.slider("Choc actions (%)", -50, 20, default_equity, 1)
with col_s2:
    rate_shock = st.slider("Choc taux (bps)", -300, 300, default_rate, 10)
with col_s3:
    vol_shock = st.slider("Choc vol implicite (%)", -50, 100, default_vol, 5)

st.markdown("#### Stress par actif")

asset_stress = {}
stress_cols = st.columns(min(4, max(1, len(valid_tickers))))

for i, ticker in enumerate(valid_tickers):
    col = stress_cols[i % len(stress_cols)]
    with col:
        asset_stress[ticker] = st.slider(
            f"{ticker} (%)", -50, 20, equity_shock, 1, key=f"stress_{ticker}"
        )

asset_stress_series = pd.Series(asset_stress, dtype=float) / 100
asset_pnl = portfolio_value * weights_series * asset_stress_series
equity_loss = asset_pnl.sum()

rate_loss = portfolio_value * (rate_shock * 0.00005)
vol_loss = portfolio_value * (vol_shock * 0.001)
total_stress_pnl = equity_loss - rate_loss - vol_loss

stress_df = pd.DataFrame({
    "Facteur": list(asset_pnl.index) + ["Choc taux", "Choc volatilité", "Impact total"],
    "PnL estimé": list(asset_pnl.values) + [-rate_loss, -vol_loss, total_stress_pnl]
})

fig_stress = go.Figure()
colors = ["#f87171" if v < 0 else "#34d399" for v in stress_df["PnL estimé"]]
fig_stress.add_trace(go.Bar(
    x=stress_df["Facteur"], y=stress_df["PnL estimé"],
    marker_color=colors,
))
fig_stress.update_layout(
    title="Impact du stress test",
    xaxis_title="Facteur",
    yaxis_title="PnL estimé",
    height=400,
    **plotly_layout,
)
st.plotly_chart(fig_stress, use_container_width=True)

col8, col9, col10 = st.columns(3)
col8.metric("PnL stressé", f"{total_stress_pnl:,.0f}")
col9.metric("Impact (%)", f"{total_stress_pnl / portfolio_value:.2%}")
col10.metric("Scénario", scenario)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Contribution au risque
# =========================
st.subheader("Contribution au risque")

cov_matrix = returns.cov()
w = weights_series.values

portfolio_var_daily = float(w.T @ cov_matrix.values @ w)
portfolio_vol_daily = np.sqrt(portfolio_var_daily)

if portfolio_vol_daily == 0:
    st.warning("Volatilité nulle — impossible de calculer les contributions.")
else:
    marginal_component = cov_matrix.values @ w
    risk_contributions = w * marginal_component / portfolio_vol_daily
    relative_risk_contributions = risk_contributions / portfolio_vol_daily

    z_score = norm.ppf(confidence_level)
    portfolio_var_param_amount = z_score * portfolio_vol_daily * portfolio_value
    var_contributions_amount = relative_risk_contributions * portfolio_var_param_amount

    risk_contrib_df = pd.DataFrame({
        "Ticker": weights_series.index,
        "Weight": weights_series.values,
        "Risk Contribution %": relative_risk_contributions,
        "VaR Contribution": var_contributions_amount,
    }).sort_values("Risk Contribution %", ascending=False)

    top_ticker = risk_contrib_df.iloc[0]["Ticker"]
    top_rc_pct = risk_contrib_df.iloc[0]["Risk Contribution %"]

    col_rc1, col_rc2, col_rc3 = st.columns(3)
    col_rc1.metric("Actif dominant", f"{top_ticker}")
    col_rc2.metric("Part du risque", f"{top_rc_pct:.2%}")
    col_rc3.metric("Contribution VaR", f"{risk_contrib_df.iloc[0]['VaR Contribution']:,.0f}")

    fig_rc = go.Figure()
    fig_rc.add_trace(go.Bar(
        x=risk_contrib_df["Ticker"],
        y=risk_contrib_df["Risk Contribution %"],
        marker_color="#a78bfa",
    ))
    fig_rc.update_layout(
        title="Contribution relative au risque",
        yaxis_tickformat=".1%",
        height=400,
        **plotly_layout,
    )
    st.plotly_chart(fig_rc, use_container_width=True)

    st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Monte Carlo VaR
# =========================
@st.cache_data
def cached_monte_carlo(returns_values, returns_columns, weights,
                        portfolio_value, n_sims, horizon, seed=42):
    return monte_carlo_multivariate(
        returns_values, returns_columns, weights,
        portfolio_value, n_sims, horizon, seed
    )

st.subheader("Monte Carlo VaR")

mc_col1, mc_col2 = st.columns(2)
with mc_col1:
    n_sims = st.slider("Simulations", 1000, 1000000, 100000, 5000)
with mc_col2:
    mc_horizon = st.slider("Horizon (jours)", 1, 60, 20, 1)

mc_results = cached_monte_carlo(
    returns_values=tuple(map(tuple, returns[valid_tickers].values)),
    returns_columns=tuple(returns[valid_tickers].columns),
    weights=tuple(weights_series.values),
    portfolio_value=portfolio_value,
    n_sims=n_sims,
    horizon=mc_horizon,
)

# Trajectoires
n_display = min(200, n_sims)
display_indices = np.random.choice(n_sims, n_display, replace=False)

fig_paths = go.Figure()
paths_sample = mc_results["paths"][:, display_indices]
for i in range(n_display):
    fig_paths.add_trace(go.Scatter(
        y=paths_sample[:, i],
        mode="lines",
        line=dict(width=0.5, color="rgba(34, 211, 238, 0.1)"),
        showlegend=False,
    ))
fig_paths.update_layout(
    title=f"Monte Carlo — {n_display} trajectoires sur {n_sims:,} simulées",
    xaxis_title="Jours",
    yaxis_title="Valeur du portefeuille",
    height=450,
    **plotly_layout,
)
st.plotly_chart(fig_paths, use_container_width=True)

# Distribution PnL
final_pnl = mc_results["final_pnl"]
final_values = mc_results["final_values"]

mc_var_amount = -np.quantile(final_pnl, 1 - confidence_level)
mc_es_losses = final_pnl[final_pnl <= np.quantile(final_pnl, 1 - confidence_level)]
mc_es_amount = -mc_es_losses.mean()

fig_pnl = go.Figure()
fig_pnl.add_trace(go.Histogram(
    x=final_pnl, nbinsx=60,
    marker_color="#22d3ee", opacity=0.7,
))
fig_pnl.add_vline(
    x=-mc_var_amount, line_dash="dash", line_color="#f87171",
    annotation_text=f"MC VaR {confidence_level:.0%}",
)
fig_pnl.update_layout(
    title="Distribution des PnL simulés",
    xaxis_title="PnL final",
    yaxis_title="Fréquence",
    height=400,
    **plotly_layout,
)
st.plotly_chart(fig_pnl, use_container_width=True)

mc_k1, mc_k2, mc_k3 = st.columns(3)
mc_k1.metric("Monte Carlo VaR", f"{mc_var_amount:,.0f}")
mc_k2.metric("Monte Carlo ES", f"{mc_es_amount:,.0f}")
mc_k3.metric("Valeur moyenne finale", f"{final_values.mean():,.0f}")

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Synthèse
# =========================
st.subheader("Synthèse exécutive")

summary_df = pd.DataFrame({
    "Metric": [
        method_label, "Expected Shortfall", "Max Drawdown",
        "Dépassements VaR", "Stress PnL", "Stress Impact %",
    ],
    "Value": [
        f"{var_amount:,.0f}",
        f"{es_amount:,.0f}" if pd.notna(es_amount) else "N/A",
        f"{max_dd:.2%}",
        f"{breach_count}",
        f"{total_stress_pnl:,.0f}",
        f"{total_stress_pnl / portfolio_value:.2%}",
    ]
})

st.dataframe(summary_df, use_container_width=True, hide_index=True)