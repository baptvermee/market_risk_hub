import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_prices, compute_returns

@st.cache_data
def get_prices(tickers, start, end):
    """Couche de cache Streamlit autour de notre data loader."""
    return load_prices(tickers, start, end)

st.title("📊 Market Overview")

# =========================
# Sidebar
# =========================
st.sidebar.header("Paramètres")

tickers_input = st.sidebar.text_input(
    "Entrer les tickers (séparés par des virgules)",
    value="SPY"
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("today"))

window = st.sidebar.slider("Fenêtre rolling (jours)", 20, 252, 60)


data = get_prices(tickers, start_date, end_date)

if data.empty:
    st.error("Aucune donnée récupérée.")
    st.stop()

data = data.dropna(how="all")
returns = compute_returns(data)

st.write("Tickers valides :", list(data.columns))

# =========================
# KPIs
# =========================
total_return = (data.iloc[-1] / data.iloc[0] - 1).mean()
volatility = returns.std().mean() * (252 ** 0.5)

col1, col2 = st.columns(2)
col1.metric("Return moyen", f"{total_return:.2%}")
col2.metric("Volatilité moyenne", f"{volatility:.2%}")

# =========================
# Graph prix
# =========================
st.subheader("Prix des actifs")

prices_long = data.reset_index().melt(
    id_vars=data.index.name or "Date",
    var_name="Ticker",
    value_name="Price"
)

prices_long = prices_long.rename(columns={prices_long.columns[0]: "Date"})

fig_prices = px.line(prices_long, x="Date", y="Price", color="Ticker")
st.plotly_chart(fig_prices, use_container_width=True)

# =========================
# Rendements cumulés
# =========================
st.subheader("Rendements cumulés")

cumulative_returns = (1 + returns).cumprod()

cum_long = cumulative_returns.reset_index().melt(
    id_vars=cumulative_returns.index.name or "Date",
    var_name="Ticker",
    value_name="Cumulative Return"
)

cum_long = cum_long.rename(columns={cum_long.columns[0]: "Date"})

fig_cum = px.line(cum_long, x="Date", y="Cumulative Return", color="Ticker")
st.plotly_chart(fig_cum, use_container_width=True)

# =========================
# Corrélation
# =========================
st.subheader("Matrice de corrélation")

corr_matrix = returns.corr()

fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1
)

st.plotly_chart(fig_corr, use_container_width=True)

# =========================
# Benchmark & portefeuille
# =========================
benchmark = "SPY"

if benchmark not in data.columns:
    st.warning("Ajoute SPY pour l'analyse benchmark.")
else:
    portfolio_tickers = [t for t in data.columns if t != benchmark]

    if len(portfolio_tickers) > 0:
        portfolio_returns = returns[portfolio_tickers].mean(axis=1)
        benchmark_returns = returns[benchmark]

        portfolio_cum = (1 + portfolio_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()

        st.subheader("Portefeuille vs SPY")

        df_compare = pd.DataFrame({
            "Portfolio": portfolio_cum,
            "SPY": benchmark_cum
        })

        fig_compare = px.line(df_compare)
        st.plotly_chart(fig_compare, use_container_width=True)

        outperformance = portfolio_cum.iloc[-1] - benchmark_cum.iloc[-1]
        st.metric("Surperformance vs SPY", f"{outperformance:.2%}")

        beta = portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()
        st.metric("Beta", f"{beta:.2f}")

        # =========================
        # Rolling volatility
        # =========================
        st.subheader("Volatilité roulante")

        rolling_vol = portfolio_returns.rolling(window).std() * (252 ** 0.5)
        st.plotly_chart(px.line(rolling_vol), use_container_width=True)

        # =========================
        # Rolling beta
        # =========================
        st.subheader("Beta roulant")

        rolling_beta = (
            portfolio_returns.rolling(window).cov(benchmark_returns)
            / benchmark_returns.rolling(window).var()
        )

        st.plotly_chart(px.line(rolling_beta), use_container_width=True)

        # =========================
        # Drawdown
        # =========================
        st.subheader("Drawdown")

        rolling_max = portfolio_cum.cummax()
        drawdown = portfolio_cum / rolling_max - 1

        st.plotly_chart(px.line(drawdown), use_container_width=True)

        st.metric("Max Drawdown", f"{drawdown.min():.2%}")