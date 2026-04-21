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
    """
    Calcule la VaR historique.

    Principe : on prend l'historique réel des rendements et on lit
    directement le quantile correspondant au niveau de confiance.

    Exemple concret :
    Si tu as 1000 jours de rendements et confidence = 0.95,
    on trie les rendements du pire au meilleur et on prend
    le 50ème pire (car 1000 × 5% = 50). Ce rendement-là,
    c'est ta VaR : "dans 95% des cas, tu ne perdras pas plus que ça."

    Paramètres
    ----------
    returns : pd.Series
        Série des rendements journaliers du portefeuille
    confidence : float
        Niveau de confiance, entre 0 et 1 (ex: 0.95 pour 95%)

    Retourne
    --------
    float
        Le rendement correspondant au seuil de perte (valeur négative)
    """

    alpha = 1 - confidence
    return float(returns.quantile(alpha))


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calcule la VaR paramétrique (méthode variance-covariance).

    Hypothèse : les rendements suivent une loi normale.
    On utilise la moyenne et l'écart-type des rendements,
    puis on applique le z-score correspondant au niveau de confiance.

    Formule : VaR = μ - z × σ

    où :
    - μ  = rendement moyen journalier
    - σ  = écart-type journalier des rendements
    - z  = quantile de la loi normale (ex: 1.6449 pour 95%)

    Note : ici on utilise scipy.stats.norm.ppf() au lieu d'un
    dictionnaire de z-scores codé en dur. ppf = "percent point function",
    c'est l'inverse de la fonction de répartition de la loi normale.
    norm.ppf(0.95) te donne 1.6449 automatiquement.

    Paramètres
    ----------
    returns : pd.Series
        Série des rendements journaliers
    confidence : float
        Niveau de confiance

    Retourne
    --------
    float
        Le seuil de perte estimé (valeur négative)
    """

    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(1 - confidence)  # z est négatif (queue gauche)
    return float(mu + z * sigma)


# =============================================
# Expected Shortfall (CVaR)
# =============================================

def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calcule l'Expected Shortfall (ES), aussi appelé CVaR.

    La VaR te dit : "tu ne perdras pas plus que X dans 95% des cas."
    Mais elle ne dit rien sur ce qui se passe dans les 5% restants.

    L'ES répond à cette question : "quand ça va mal (les 5% pires jours),
    quelle est la perte MOYENNE ?"

    C'est donc une mesure plus conservatrice que la VaR.

    Exemple : si ta VaR 95% est -2%, et ton ES est -3.5%,
    ça veut dire que les jours où tu dépasses la VaR,
    tu perds en moyenne 3.5%.

    Paramètres
    ----------
    returns : pd.Series
        Série des rendements journaliers
    confidence : float
        Niveau de confiance

    Retourne
    --------
    float
        La perte moyenne conditionnelle (valeur négative)
    """

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
    """
    Calcule le drawdown à partir d'une série de rendements.

    Le drawdown mesure la chute depuis le dernier plus haut.
    C'est LA métrique que tout investisseur regarde en premier.

    Calcul :
    1. On construit la courbe de richesse cumulée
    2. À chaque instant, on note le maximum atteint jusque-là
    3. Le drawdown = (valeur actuelle / maximum) - 1

    Un drawdown de -15% signifie que depuis ton pic de richesse,
    tu as perdu 15%.

    Paramètres
    ----------
    returns : pd.Series
        Série des rendements journaliers

    Retourne
    --------
    pd.Series
        Série du drawdown (valeurs négatives ou nulles)
    """

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
    """
    Test de Kupiec (Proportion of Failures).

    On compare le taux de dépassements observé au taux théorique
    via un test du rapport de vraisemblance (likelihood ratio).

    Intuition :
    - On a N jours d'observation
    - On compte x = nombre de jours où le rendement < VaR
    - Le taux observé est p_hat = x / N
    - Le taux théorique est p = 1 - confidence (ex: 5% pour VaR 95%)
    - On teste H0 : p_hat = p  vs  H1 : p_hat ≠ p

    La statistique LR suit une loi du chi² à 1 degré de liberté.
    Si la p-value est < 0.05, on rejette H0 : le modèle est inadéquat.

    Paramètres
    ----------
    returns : pd.Series
        Rendements journaliers du portefeuille
    var_series : pd.Series
        Série de VaR (même index que returns)
    confidence : float
        Niveau de confiance de la VaR

    Retourne
    --------
    dict avec :
        - "n_obs"          : nombre d'observations
        - "n_breaches"     : nombre de dépassements
        - "breach_rate"    : taux observé
        - "expected_rate"  : taux théorique
        - "lr_statistic"   : statistique du test
        - "p_value"        : p-value du test
        - "reject"         : True si on rejette H0 au seuil 5%
    """
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
    """
    Test de Christoffersen (indépendance des dépassements).

    Kupiec vérifie le NOMBRE de dépassements.
    Christoffersen vérifie leur RÉPARTITION dans le temps.

    On construit une matrice de transition 2×2 :
        - n00 : pas de dépassement hier, pas aujourd'hui
        - n01 : pas de dépassement hier, dépassement aujourd'hui
        - n10 : dépassement hier, pas aujourd'hui
        - n11 : dépassement hier, dépassement aujourd'hui

    Si les dépassements sont indépendants, la probabilité
    d'un dépassement aujourd'hui ne devrait pas dépendre
    de ce qui s'est passé hier.

    Paramètres
    ----------
    returns : pd.Series
        Rendements journaliers
    var_series : pd.Series
        Série de VaR
    confidence : float
        Niveau de confiance

    Retourne
    --------
    dict avec :
        - "n00", "n01", "n10", "n11" : matrice de transition
        - "lr_independence"  : statistique du test d'indépendance
        - "p_value"          : p-value
        - "reject"           : True si on rejette l'indépendance
    """
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