import streamlit as st
import pandas as pd
import numpy as np
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

@st.cache_data
def get_prices(tickers, start, end):
    """Couche de cache Streamlit autour de notre data loader."""
    return load_prices(tickers, start, end)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 14px;
        border-radius: 12px;
    }
    h1, h2, h3 {
        letter-spacing: 0.2px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚠️ Risk Analytics")
st.markdown("### Market Risk Monitoring, VaR, Expected Shortfall, Backtesting and Stress Testing")

# =========================
# Sidebar
# =========================
st.sidebar.header("Paramètres Risk")

tickers_input = st.sidebar.text_input(
    "Tickers portefeuille (séparés par des virgules)",
    value="SPY"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("today"))

confidence_level = st.sidebar.slider(
    "Niveau de confiance VaR / ES",
    min_value=0.90,
    max_value=0.99,
    value=0.95,
    step=0.01,
)

portfolio_value = st.sidebar.number_input(
    "Valeur du portefeuille (€ ou $)",
    min_value=1000.0,
    value=1_000_000.0,
    step=10000.0,
)

var_method = st.sidebar.selectbox(
    "Méthode de VaR",
    ["Historique", "Paramétrique"]
)

rolling_window = st.sidebar.slider(
    "Fenêtre rolling VaR / backtesting",
    min_value=20,
    max_value=252,
    value=60,
    step=5
)



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

st.markdown("## Données chargées")
st.write("Tickers valides :", valid_tickers)

# =========================
# Pondérations du portefeuille
# =========================
if len(valid_tickers) == 1:
    single_ticker = valid_tickers[0]
    weights_series = pd.Series({single_ticker: 1.0}, dtype=float)

    st.markdown("## Pondération du portefeuille")
    st.info(f"Portefeuille mono-actif : 100 % investi sur {single_ticker}")

else:
    st.markdown("## Pondérations du portefeuille")
    st.markdown("Définis les pondérations de chaque actif. La somme sera normalisée automatiquement.")

    weights = {}
    cols = st.columns(min(4, len(valid_tickers)))

    for i, ticker in enumerate(valid_tickers):
        col = cols[i % len(cols)]
        with col:
            weights[ticker] = st.number_input(
                f"Poids {ticker}",
                min_value=0.0,
                max_value=1.0,
                value=float(round(1.0 / len(valid_tickers), 4)),
                step=0.01,
                key=f"weight_{ticker}"
            )

    weights_series = pd.Series(weights, dtype=float)

    if weights_series.sum() == 0:
        st.error("La somme des pondérations ne peut pas être nulle.")
        st.stop()

    weights_series = weights_series / weights_series.sum()

    weights_df = pd.DataFrame({
        "Ticker": weights_series.index,
        "Weight": weights_series.values
    })

    fig_weights = px.pie(
        weights_df,
        names="Ticker",
        values="Weight",
        title="Répartition du portefeuille"
    )
    st.plotly_chart(fig_weights, use_container_width=True)

# =========================
# Rendement portefeuille
# =========================
portfolio_returns = returns.mul(weights_series, axis=1).sum(axis=1)

# =========================
# VaR / ES
# =========================
st.markdown("## 📊 VaR & Expected Shortfall")

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

with col1:
    st.metric("VaR (%)", f"{var_return:.2%}")

with col2:
    st.metric("VaR (montant)", f"{var_amount:,.0f}")

with col3:
    st.metric("Expected Shortfall", f"{es_amount:,.0f}" if pd.notna(es_amount) else "N/A")

with col4:
    st.metric("Volatilité journalière", f"{portfolio_returns.std():.2%}")

st.info(
    f"La VaR estimée correspond à une perte potentielle journalière d’environ "
    f"{var_amount:,.0f} avec un niveau de confiance de {confidence_level:.0%}."
)

hist_df = pd.DataFrame({"Portfolio Returns": portfolio_returns})

fig_hist = px.histogram(
    hist_df,
    x="Portfolio Returns",
    nbins=60,
    title="Distribution des rendements du portefeuille"
)

fig_hist.add_vline(
    x=var_return,
    line_dash="dash",
    annotation_text=f"VaR {confidence_level:.0%}",
    annotation_position="top right"
)

st.plotly_chart(fig_hist, use_container_width=True)

# =========================
# Rolling VaR
# =========================
st.markdown("## 📈 Rolling VaR")

alpha = 1 - confidence_level

if var_method == "Historique":
    rolling_var = portfolio_returns.rolling(rolling_window).quantile(alpha)
else:
    rolling_mean = portfolio_returns.rolling(rolling_window).mean()
    rolling_std = portfolio_returns.rolling(rolling_window).std()
    z_score = norm.ppf(confidence_level)
    rolling_var = rolling_mean - z_score * rolling_std

rolling_var_amount = -rolling_var * portfolio_value

fig_rvar = px.line(
    rolling_var_amount,
    title=f"Rolling {method_label} ({rolling_window} jours)"
)
st.plotly_chart(fig_rvar, use_container_width=True)

# =========================
# Backtesting VaR
# =========================
st.markdown("## 🔍 Backtesting VaR")

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
col5.metric("Nombre de dépassements", f"{breach_count}")
col6.metric("Taux observé", f"{breach_rate:.2%}" if pd.notna(breach_rate) else "N/A")
col7.metric("Taux théorique", f"{expected_breach_rate:.2%}")

fig_backtest = px.line(
    backtest_df,
    y=["Return", "VaR"],
    title="Backtesting : rendement portefeuille vs VaR"
)
st.plotly_chart(fig_backtest, use_container_width=True)

breaches_only = backtest_df[backtest_df["Breach"]].copy()
if not breaches_only.empty:
    breaches_plot = breaches_only.reset_index()
    x_col = breaches_plot.columns[0]

    fig_breaches = px.scatter(
        breaches_plot,
        x=x_col,
        y="Return",
        title="Jours de dépassement de VaR"
    )
    st.plotly_chart(fig_breaches, use_container_width=True)

# --- Tests statistiques formels ---
st.markdown("### Tests statistiques")

kupiec = kupiec_test(portfolio_returns, rolling_var, confidence_level)
christoff = christoffersen_test(portfolio_returns, rolling_var, confidence_level)

col_k1, col_k2, col_k3 = st.columns(3)
col_k1.metric("Kupiec p-value", f"{kupiec['p_value']:.4f}" if pd.notna(kupiec['p_value']) else "N/A")
col_k2.metric("Christoffersen p-value", f"{christoff['p_value']:.4f}" if pd.notna(christoff['p_value']) else "N/A")
col_k3.metric("Dépassements observés", f"{kupiec['n_breaches']} / {kupiec['n_obs']}")

if pd.notna(kupiec["p_value"]):
    if kupiec["reject"]:
        st.error("❌ Test de Kupiec rejeté : le nombre de dépassements est statistiquement incohérent avec le modèle.")
    else:
        st.success("✅ Test de Kupiec validé : le nombre de dépassements est cohérent.")

if pd.notna(christoff["p_value"]):
    if christoff["reject"]:
        st.error("❌ Test de Christoffersen rejeté : les dépassements arrivent en cluster, le modèle ne réagit pas assez vite.")
    else:
        st.success("✅ Test de Christoffersen validé : les dépassements sont indépendants.")

# =========================
# Drawdown
# =========================
st.markdown("## 📉 Drawdown Analysis")

drawdown = compute_drawdown(portfolio_returns)

fig_dd = px.line(
    drawdown,
    title="Drawdown du portefeuille"
)
st.plotly_chart(fig_dd, use_container_width=True)

max_dd = drawdown.min()
st.metric("Max Drawdown", f"{max_dd:.2%}")

if max_dd < -0.30:
    st.error("Drawdown sévère : le portefeuille présente une forte vulnérabilité historique.")
elif max_dd < -0.15:
    st.warning("Drawdown significatif : le portefeuille reste sensible aux phases de stress.")
else:
    st.success("Drawdown globalement maîtrisé sur la période observée.")

# =========================
# Stress tests
# =========================
st.markdown("## ⚡ Stress Testing")

scenario = st.selectbox(
    "Choisir un scénario prédéfini",
    [
        "Scénario personnalisé",
        "Correction actions modérée",
        "Choc inflation / taux",
        "Stress volatilité",
        "Crise sévère multi-facteurs",
    ]
)

default_equity = -10
default_rate = 100
default_vol = 20

if scenario == "Correction actions modérée":
    default_equity = -10
    default_rate = 50
    default_vol = 15
elif scenario == "Choc inflation / taux":
    default_equity = -8
    default_rate = 150
    default_vol = 10
elif scenario == "Stress volatilité":
    default_equity = -5
    default_rate = 25
    default_vol = 40
elif scenario == "Crise sévère multi-facteurs":
    default_equity = -25
    default_rate = 200
    default_vol = 60

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    equity_shock = st.slider(
        "Choc actions (%)",
        min_value=-50,
        max_value=20,
        value=default_equity,
        step=1
    )

with col_s2:
    rate_shock = st.slider(
        "Choc taux (bps)",
        min_value=-300,
        max_value=300,
        value=default_rate,
        step=10
    )

with col_s3:
    vol_shock = st.slider(
        "Choc volatilité implicite (%)",
        min_value=-50,
        max_value=100,
        value=default_vol,
        step=5
    )

st.markdown("### Stress par actif")
st.markdown("Définis un choc spécifique par actif si nécessaire.")

asset_stress = {}
stress_cols = st.columns(min(4, max(1, len(valid_tickers))))

for i, ticker in enumerate(valid_tickers):
    col = stress_cols[i % len(stress_cols)]
    with col:
        asset_stress[ticker] = st.slider(
            f"{ticker} (%)",
            min_value=-50,
            max_value=20,
            value=equity_shock,
            step=1,
            key=f"stress_{ticker}"
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

st.markdown("### Impact détaillé du stress")
st.dataframe(stress_df, use_container_width=True)

fig_stress = px.bar(
    stress_df,
    x="Facteur",
    y="PnL estimé",
    title="Impact estimé du stress test"
)
st.plotly_chart(fig_stress, use_container_width=True)

col8, col9, col10 = st.columns(3)
col8.metric("PnL stressé estimé", f"{total_stress_pnl:,.0f}")
col9.metric("Impact stressé (%)", f"{total_stress_pnl / portfolio_value:.2%}")
col10.metric("Scénario", scenario)

# =========================
# Contribution au risque
# =========================
st.markdown("## 🧩 Contribution au risque")

cov_matrix = returns.cov()
w = weights_series.values

portfolio_var_daily = float(w.T @ cov_matrix.values @ w)
portfolio_vol_daily = np.sqrt(portfolio_var_daily)

if portfolio_vol_daily == 0:
    st.warning("La volatilité du portefeuille est nulle ; impossible de calculer les contributions au risque.")
else:
    marginal_component = cov_matrix.values @ w

    # Contribution absolue à la volatilité
    risk_contributions = w * marginal_component / portfolio_vol_daily

    # Contribution relative au risque total
    relative_risk_contributions = risk_contributions / portfolio_vol_daily

    # VaR paramétrique portefeuille
    z_score = norm.ppf(confidence_level)
    portfolio_var_param_return = z_score * portfolio_vol_daily
    portfolio_var_param_amount = portfolio_var_param_return * portfolio_value

    # Contribution à la VaR
    var_contributions_amount = relative_risk_contributions * portfolio_var_param_amount

    risk_contrib_df = pd.DataFrame({
        "Ticker": weights_series.index,
        "Weight": weights_series.values,
        "Risk Contribution": risk_contributions,
        "Risk Contribution %": relative_risk_contributions,
        "VaR Contribution Amount": var_contributions_amount
    }).sort_values("Risk Contribution %", ascending=False)

    # KPI principaux
    top_ticker = risk_contrib_df.iloc[0]["Ticker"]
    top_rc_pct = risk_contrib_df.iloc[0]["Risk Contribution %"]
    top_var_amount = risk_contrib_df.iloc[0]["VaR Contribution Amount"]

    col_rc1, col_rc2, col_rc3 = st.columns(3)
    col_rc1.metric("Actif dominant dans le risque", f"{top_ticker}")
    col_rc2.metric("Part du risque portée", f"{top_rc_pct:.2%}")
    col_rc3.metric("Contribution VaR", f"{top_var_amount:,.0f}")

    # Graphique 1 : contribution relative
    fig_rc_pct = px.bar(
        risk_contrib_df,
        x="Ticker",
        y="Risk Contribution %",
        title="Contribution relative au risque total"
    )
    fig_rc_pct.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig_rc_pct, use_container_width=True)

    # Graphique 2 : contribution VaR
    fig_var_contrib = px.bar(
        risk_contrib_df,
        x="Ticker",
        y="VaR Contribution Amount",
        title="Contribution à la VaR paramétrique (montant)"
    )
    st.plotly_chart(fig_var_contrib, use_container_width=True)

    # Tableau propre
    st.markdown("### Détail par actif")

    display_df = risk_contrib_df.copy()
    display_df["Weight"] = display_df["Weight"].map(lambda x: f"{x:.2%}")
    display_df["Risk Contribution %"] = display_df["Risk Contribution %"].map(lambda x: f"{x:.2%}")
    display_df["VaR Contribution Amount"] = display_df["VaR Contribution Amount"].map(lambda x: f"{x:,.0f}")

    display_df = display_df[["Ticker", "Weight", "Risk Contribution %", "VaR Contribution Amount"]]

    st.dataframe(display_df, use_container_width=True)

    # Interprétation automatique
    if top_rc_pct >= 0.50:
        st.warning(
            f"Le risque du portefeuille est fortement concentré sur {top_ticker}, "
            f"qui représente à lui seul {top_rc_pct:.2%} du risque total."
        )
    elif top_rc_pct >= 0.35:
        st.info(
            f"{top_ticker} est le principal contributeur au risque avec {top_rc_pct:.2%} du risque total."
        )
    else:
        st.success(
            "Le risque apparaît relativement réparti entre les actifs du portefeuille."
        )


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

st.markdown("## 🎲 Monte Carlo VaR")

mc_col1, mc_col2 = st.columns(2)

with mc_col1:
    n_sims = st.slider(
        "Nombre de simulations",
        min_value=1000,
        max_value=1000000,  
        value=100000,        
        step=5000
    )

with mc_col2:
    mc_horizon = st.slider(
        "Horizon de simulation (jours)",
        min_value=1,
        max_value=60,
        value=20,
        step=1
    )

# --- APPEL avec conversion en tuples pour le cache ---
# Pourquoi cette conversion ?
# st.cache_data compare les arguments pour savoir si le résultat est déjà en cache
# Il a besoin d'objets "hashables" (immutables) pour faire cette comparaison
# - DataFrame et ndarray ne sont PAS hashables → on les convertit en tuples
# - float, int, str SONT hashables → on les passe directement
mc_results = cached_monte_carlo(
    returns_values=tuple(map(tuple, returns[valid_tickers].values)),
    returns_columns=tuple(returns[valid_tickers].columns),
    weights=tuple(weights_series.values),
    portfolio_value=portfolio_value,
    n_sims=n_sims,
    horizon=mc_horizon,
)



# Trajectoires simulées
st.markdown("### Trajectoires simulées")

# On a calculé 50 000 (ou plus) trajectoires pour la précision statistique
# Mais on n'en AFFICHE que 200 — c'est largement suffisant visuellement
# et ça évite de faire planter Plotly/le navigateur
n_display = min(200, n_sims)  # au cas où n_sims < 200

# On sélectionne 200 colonnes au hasard parmi toutes les simulations
# np.random.choice(n_sims, n_display, replace=False) tire 200 indices uniques
display_indices = np.random.choice(n_sims, n_display, replace=False)

# mc_results["paths"] a pour shape (horizon+1, n_sims)
# [:, display_indices] sélectionne seulement les 200 colonnes choisies
paths_df = pd.DataFrame(mc_results["paths"][:, display_indices])
paths_df.index.name = "Step"

fig_paths = px.line(
    paths_df,
    title=f"Monte Carlo multivariée — {n_display} trajectoires affichées sur {n_sims} simulées"
)
fig_paths.update_layout(showlegend=False)
st.plotly_chart(fig_paths, use_container_width=True)

# Distribution des PnL finaux
final_pnl = mc_results["final_pnl"]
final_values = mc_results["final_values"]

mc_var_amount = -np.quantile(final_pnl, 1 - confidence_level)
mc_es_losses = final_pnl[final_pnl <= np.quantile(final_pnl, 1 - confidence_level)]
mc_es_amount = -mc_es_losses.mean()

st.markdown("### Distribution des PnL simulés")

pnl_df = pd.DataFrame({"PnL final": final_pnl})

fig_pnl = px.histogram(
    pnl_df,
    x="PnL final",
    nbins=60,
    title="Distribution simulée des PnL finaux"
)
fig_pnl.add_vline(
    x=-mc_var_amount,
    line_dash="dash",
    annotation_text=f"MC VaR {confidence_level:.0%}",
    annotation_position="top right"
)
st.plotly_chart(fig_pnl, use_container_width=True)

mc_kpi1, mc_kpi2, mc_kpi3 = st.columns(3)
mc_kpi1.metric("Monte Carlo VaR", f"{mc_var_amount:,.0f}")
mc_kpi2.metric("Monte Carlo ES", f"{mc_es_amount:,.0f}")
mc_kpi3.metric("Valeur moyenne finale", f"{final_values.mean():,.0f}")
# =========================
# Synthèse
# =========================
st.markdown("## 🧾 Synthèse exécutive")

summary_df = pd.DataFrame({
    "Metric": [
        method_label,
        "Expected Shortfall",
        "Max Drawdown",
        "Nombre de dépassements VaR",
        "Stress PnL",
        "Stress Impact %"
    ],
    "Value": [
        f"{var_amount:,.0f}",
        f"{es_amount:,.0f}" if pd.notna(es_amount) else "N/A",
        f"{max_dd:.2%}",
        f"{breach_count}",
        f"{total_stress_pnl:,.0f}",
        f"{total_stress_pnl / portfolio_value:.2%}"
    ]
})

st.dataframe(summary_df, use_container_width=True)

if total_stress_pnl < -0.20 * portfolio_value:
    st.error("Le portefeuille est fortement exposé à un scénario de crise.")
elif total_stress_pnl < -0.10 * portfolio_value:
    st.warning("Le portefeuille présente une sensibilité notable aux chocs simulés.")
else:
    st.success("Le portefeuille paraît relativement résilient aux scénarios testés.")

st.markdown(f"""
- Méthode de VaR utilisée : **{method_label}**
- Niveau de confiance : **{confidence_level:.0%}**
- Valeur de portefeuille retenue : **{portfolio_value:,.0f}**
- Fenêtre de backtesting / rolling : **{rolling_window} jours**
- Taux de dépassement observé : **{breach_rate:.2%}**
- Max drawdown observé : **{max_dd:.2%}**
- Impact du scénario de stress : **{total_stress_pnl:,.0f}**
""") 