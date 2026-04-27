"""
risk_engine.py
--------------
Moteur de calcul du risque de marché.

Contient les fonctions de calcul de VaR, Expected Shortfall,
volatilité, drawdown, et les métriques associées.

Aucune dépendance à Streamlit — ce module est purement analytique.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# =============================================
# VaR (Value at Risk)
# =============================================

def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    alpha = 1 - confidence
    return float(returns.quantile(alpha))


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(1 - confidence)  # z est négatif (queue gauche)
    return float(mu + z * sigma)

# =============================================
# Expected Shortfall (CVaR)
# =============================================

def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    alpha = 1 - confidence
    var_threshold = returns.quantile(alpha)
    tail_losses = returns[returns <= var_threshold]

    if tail_losses.empty:
        return np.nan

    return float(tail_losses.mean())


# =============================================
# Drawdown
# =============================================

def compute_drawdown(returns: pd.Series) -> pd.Series:
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return drawdown

def monte_carlo_multivariate(
    returns_values: tuple,
    returns_columns: tuple,
    weights: tuple,
    portfolio_value: float,
    n_sims: int = 50000,
    horizon: int = 20,
    seed: int = 42,
) -> dict:

    np.random.seed(seed)

    returns_array = np.array(returns_values)
    w = np.array(weights)
    n_assets = returns_array.shape[1]

    mu = returns_array.mean(axis=0)
    cov = np.cov(returns_array, rowvar=False)

    # --- CAS LIMITE : un seul actif ---
    # np.cov renvoie un scalaire quand il n'y a qu'une colonne
    # On le force en matrice 2D pour que Cholesky fonctionne
    # np.atleast_2d transforme :
    #   - un scalaire 0.0004 → une matrice [[0.0004]]
    #   - une matrice déjà 2D → reste inchangée
    cov = np.atleast_2d(cov)

    L = np.linalg.cholesky(cov)

    all_Z = np.random.normal(size=(horizon, n_assets, n_sims))
    all_correlated = np.array([
        mu.reshape(-1, 1) + L @ all_Z[t] for t in range(horizon)
    ])

    paths = np.zeros((horizon + 1, n_sims))
    paths[0, :] = portfolio_value
    asset_values = np.outer(np.ones(n_sims), w * portfolio_value)

    for t in range(horizon):
        asset_values = asset_values * (1 + all_correlated[t].T)
        paths[t + 1, :] = asset_values.sum(axis=1)

    final_values = paths[-1, :]
    final_pnl = final_values - portfolio_value

    return {
        "paths": paths,
        "final_values": final_values,
        "final_pnl": final_pnl,
    }

def kupiec_test(returns: pd.Series, var_series: pd.Series, confidence: float = 0.95) -> dict:
    from scipy.stats import chi2

    # Aligner les deux séries sur les mêmes dates
    aligned = pd.concat([returns, var_series], axis=1).dropna()
    aligned.columns = ["return", "var"]

    n = len(aligned)
    # Un dépassement = le rendement est INFÉRIEUR à la VaR
    x = int((aligned["return"] < aligned["var"]).sum())

    p = 1 - confidence       # taux théorique (ex: 0.05)
    p_hat = x / n if n > 0 else 0  # taux observé

    # --- Calcul du likelihood ratio ---
    # LR = -2 * ln(L0 / L1)
    # L0 = vraisemblance sous H0 (taux = p)
    # L1 = vraisemblance sous H1 (taux = p_hat)
    #
    # Formule développée :
    # LR = -2 * [x*ln(p) + (n-x)*ln(1-p) - x*ln(p_hat) - (n-x)*ln(1-p_hat)]

    if x == 0 or x == n:
        # Cas extrêmes : aucun dépassement ou que des dépassements
        # Le test n'est pas applicable proprement
        return {
            "n_obs": n,
            "n_breaches": x,
            "breach_rate": p_hat,
            "expected_rate": p,
            "lr_statistic": np.nan,
            "p_value": np.nan,
            "reject": False,
        }

    lr = -2 * (
        x * np.log(p) + (n - x) * np.log(1 - p)
        - x * np.log(p_hat) - (n - x) * np.log(1 - p_hat)
    )

    # La statistique LR suit un chi²(1)
    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        "n_obs": n,
        "n_breaches": x,
        "breach_rate": p_hat,
        "expected_rate": p,
        "lr_statistic": float(lr),
        "p_value": float(p_value),
        "reject": p_value < 0.05,
    }


def christoffersen_test(returns: pd.Series, var_series: pd.Series, confidence: float = 0.95) -> dict:
    from scipy.stats import chi2

    aligned = pd.concat([returns, var_series], axis=1).dropna()
    aligned.columns = ["return", "var"]

    # Série binaire : 1 = dépassement, 0 = pas de dépassement
    breaches = (aligned["return"] < aligned["var"]).astype(int).values

    # Construction de la matrice de transition
    n00 = n01 = n10 = n11 = 0

    for i in range(1, len(breaches)):
        yesterday = breaches[i - 1]
        today = breaches[i]

        if yesterday == 0 and today == 0:
            n00 += 1
        elif yesterday == 0 and today == 1:
            n01 += 1
        elif yesterday == 1 and today == 0:
            n10 += 1
        elif yesterday == 1 and today == 1:
            n11 += 1

    # Probabilités de transition
    # p01 = P(dépassement aujourd'hui | pas de dépassement hier)
    # p11 = P(dépassement aujourd'hui | dépassement hier)
    # Si indépendance : p01 ≈ p11

    total_0 = n00 + n01  # jours précédés d'un non-dépassement
    total_1 = n10 + n11  # jours précédés d'un dépassement

    if total_0 == 0 or total_1 == 0 or n01 == 0 or n11 == 0:
        return {
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "lr_independence": np.nan,
            "p_value": np.nan,
            "reject": False,
        }

    p01 = n01 / total_0
    p11 = n11 / total_1

    # Probabilité globale de dépassement (sous H0 : indépendance)
    p_global = (n01 + n11) / (total_0 + total_1)

    # Likelihood ratio
    lr = -2 * (
        n00 * np.log(1 - p_global) + n01 * np.log(p_global)
        + n10 * np.log(1 - p_global) + n11 * np.log(p_global)
        - n00 * np.log(1 - p01) - n01 * np.log(p01)
        - n10 * np.log(1 - p11) - n11 * np.log(p11)
    )

    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "lr_independence": float(lr),
        "p_value": float(p_value),
        "reject": p_value < 0.05,
    }