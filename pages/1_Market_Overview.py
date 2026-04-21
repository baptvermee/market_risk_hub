import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
</style>
""", unsafe_allow_html=True)

# Plotly template réutilisable
plotly_layout = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
)

st.title("📊 Market Overview")
st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
st.sidebar.header("PARAMÈTRES")

st.sidebar.markdown("#### Univers")
tickers_input = st.sidebar.text_input(
    "Tickers (séparés par des virgules)",
    value="SPY"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

st.sidebar.markdown("#### Période")
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("today"))

st.sidebar.markdown("#### Analyse")
window = st.sidebar.slider("Fenêtre rolling (jours)", 20, 252, 60)

# =========================
# Chargement données
# =========================
@st.cache_data
def get_prices(tickers, start, end):
    return load_prices(tickers, start, end)

data = get_prices(tickers, start_date, end_date)

if data.empty:
    st.error("Aucune donnée récupérée.")
    st.stop()

data = data.dropna(how="all")
returns = compute_returns(data)

# =========================
# KPIs principaux
# =========================
total_return = (data.iloc[-1] / data.iloc[0] - 1)
volatility = returns.std() * (252 ** 0.5)
sharpe = (returns.mean() * 252) / (returns.std() * (252 ** 0.5))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Return moyen", f"{total_return.mean():.2%}")
col2.metric("Volatilité moyenne", f"{volatility.mean():.2%}")
col3.metric("Sharpe moyen", f"{sharpe.mean():.2f}")
col4.metric("Nb actifs", f"{len(data.columns)}")

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# KPIs par ticker
# =========================
if len(data.columns) > 1:
    st.subheader("Métriques par actif")

    cols = st.columns(min(5, len(data.columns)))
    for i, ticker in enumerate(data.columns):
        with cols[i % len(cols)]:
            st.metric(
                f"{ticker} Return",
                f"{total_return[ticker]:.2%}",
            )
            st.metric(
                f"{ticker} Vol",
                f"{volatility[ticker]:.2%}",
            )
            st.metric(
                f"{ticker} Sharpe",
                f"{sharpe[ticker]:.2f}",
            )

    st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Graph prix
# =========================
st.subheader("Prix des actifs")

fig_prices = go.Figure()
for ticker in data.columns:
    fig_prices.add_trace(go.Scatter(
        x=data.index, y=data[ticker],
        name=ticker,
        line=dict(width=2),
    ))

fig_prices.update_layout(
    title="Évolution des prix",
    xaxis_title="Date",
    yaxis_title="Prix",
    height=450,
    **plotly_layout,
)
st.plotly_chart(fig_prices, use_container_width=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Rendements cumulés
# =========================
st.subheader("Rendements cumulés")

cumulative_returns = (1 + returns).cumprod()

fig_cum = go.Figure()
for ticker in cumulative_returns.columns:
    fig_cum.add_trace(go.Scatter(
        x=cumulative_returns.index, y=cumulative_returns[ticker],
        name=ticker,
        line=dict(width=2),
    ))

fig_cum.add_hline(y=1.0, line_dash="dot", line_color="#64748b")

fig_cum.update_layout(
    title="Performance cumulée (base 1)",
    xaxis_title="Date",
    yaxis_title="Rendement cumulé",
    height=450,
    **plotly_layout,
)
st.plotly_chart(fig_cum, use_container_width=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Corrélation
# =========================
st.subheader("Matrice de corrélation")

corr_matrix = returns.corr()

fig_corr = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale="RdBu",
    zmin=-1, zmax=1,
    text=np.round(corr_matrix.values, 2),
    texttemplate="%{text}",
    textfont=dict(family="JetBrains Mono", size=12),
))

fig_corr.update_layout(
    title="Corrélation des rendements",
    height=450,
    **plotly_layout,
)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Benchmark & portefeuille
# =========================
benchmark = "SPY"

if benchmark not in data.columns:
    st.warning("Ajoute SPY dans les tickers pour l'analyse benchmark.")
else:
    portfolio_tickers = [t for t in data.columns if t != benchmark]

    if len(portfolio_tickers) > 0:
        portfolio_returns = returns[portfolio_tickers].mean(axis=1)
        benchmark_returns = returns[benchmark]

        portfolio_cum = (1 + portfolio_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()

        st.subheader("Portefeuille vs SPY")

        # KPIs
        outperformance = portfolio_cum.iloc[-1] - benchmark_cum.iloc[-1]
        beta = portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()

        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Surperformance vs SPY", f"{outperformance:.2%}")
        col_b2.metric("Beta", f"{beta:.2f}")
        col_b3.metric("Corrélation", f"{portfolio_returns.corr(benchmark_returns):.2f}")

        # Graph comparaison
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=portfolio_cum.index, y=portfolio_cum,
            name="Portfolio",
            line=dict(color="#22d3ee", width=2.5),
        ))
        fig_compare.add_trace(go.Scatter(
            x=benchmark_cum.index, y=benchmark_cum,
            name="SPY",
            line=dict(color="#f87171", width=2, dash="dash"),
        ))
        fig_compare.update_layout(
            title="Performance cumulée — Portfolio vs SPY",
            xaxis_title="Date",
            yaxis_title="Rendement cumulé",
            height=450,
            **plotly_layout,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

        # =========================
        # Rolling volatility
        # =========================
        st.subheader("Volatilité roulante")

        rolling_vol = portfolio_returns.rolling(window).std() * (252 ** 0.5)

        fig_rvol = go.Figure()
        fig_rvol.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol,
            line=dict(color="#a78bfa", width=2),
            fill="tozeroy",
            fillcolor="rgba(167, 139, 250, 0.1)",
        ))
        fig_rvol.update_layout(
            title=f"Volatilité roulante ({window} jours, annualisée)",
            xaxis_title="Date",
            yaxis_title="Volatilité",
            yaxis_tickformat=".0%",
            height=400,
            **plotly_layout,
        )
        st.plotly_chart(fig_rvol, use_container_width=True)

        st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

        # =========================
        # Rolling beta
        # =========================
        st.subheader("Beta roulant")

        rolling_beta = (
            portfolio_returns.rolling(window).cov(benchmark_returns)
            / benchmark_returns.rolling(window).var()
        )

        fig_rbeta = go.Figure()
        fig_rbeta.add_trace(go.Scatter(
            x=rolling_beta.index, y=rolling_beta,
            line=dict(color="#22d3ee", width=2),
        ))
        fig_rbeta.add_hline(y=1.0, line_dash="dot", line_color="#f59e0b", annotation_text="β=1")
        fig_rbeta.update_layout(
            title=f"Beta roulant vs SPY ({window} jours)",
            xaxis_title="Date",
            yaxis_title="Beta",
            height=400,
            **plotly_layout,
        )
        st.plotly_chart(fig_rbeta, use_container_width=True)

        st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

        # =========================
        # Drawdown
        # =========================
        st.subheader("Drawdown")

        rolling_max = portfolio_cum.cummax()
        drawdown = portfolio_cum / rolling_max - 1

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

        st.metric("Max Drawdown", f"{drawdown.min():.2%}")