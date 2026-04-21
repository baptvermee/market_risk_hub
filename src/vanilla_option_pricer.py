"""
option_pricer.py
----------------
Moteur de pricing d'options vanilles (Black-Scholes)
et calcul des Greeks.

Aucune dépendance à Streamlit.
"""

import numpy as np
from scipy.stats import norm


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Prix d'une option européenne via Black-Scholes.

    Paramètres
    ----------
    S : float
        Prix spot du sous-jacent
    K : float
        Strike (prix d'exercice)
    T : float
        Temps jusqu'à maturité (en années). Ex: 0.25 = 3 mois
    r : float
        Taux sans risque annualisé (ex: 0.05 = 5%)
    sigma : float
        Volatilité annualisée (ex: 0.20 = 20%)
    option_type : str
        "call" ou "put"

    Retourne
    --------
    float
        Prix théorique de l'option
    """

    # Protection : si T = 0, l'option est expirée → payoff intrinsèque
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    # Protection : si sigma = 0, pas d'incertitude → valeur actualisée du payoff
    if sigma <= 0:
        if option_type == "call":
            return max(S - K * np.exp(-r * T), 0.0)
        else:
            return max(K * np.exp(-r * T) - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(price)


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """
    Calcule les Greeks d'une option européenne.

    Les Greeks mesurent la sensibilité du prix de l'option
    à chaque paramètre. En salle de marché, ce sont les
    indicateurs de risque principaux sur un book d'options.

    Delta : dC/dS — de combien bouge le prix si S bouge de 1$
        Call : entre 0 et 1  (un call deep ITM a un delta ~1)
        Put  : entre -1 et 0 (un put deep ITM a un delta ~-1)

    Gamma : d²C/dS² — de combien bouge le delta si S bouge de 1$
        Toujours positif. Maximum quand l'option est ATM.
        Un gamma élevé = le delta change vite = risque de hedging.

    Theta : dC/dT — combien l'option perd par jour qui passe
        Presque toujours négatif (l'option perd de la valeur avec le temps).
        On le divise par 365 pour avoir la perte PAR JOUR.

    Vega : dC/dσ — de combien bouge le prix si la vol bouge de 1%
        Toujours positif. On le divise par 100 pour avoir
        la sensibilité à 1 point de volatilité.

    Rho : dC/dr — de combien bouge le prix si le taux bouge de 1%
        Call : positif (un taux plus haut augmente la valeur du call)
        Put  : négatif

    Paramètres
    ----------
    S, K, T, r, sigma, option_type : mêmes que black_scholes_price

    Retourne
    --------
    dict avec : delta, gamma, theta, vega, rho
    """

    if T <= 0 or sigma <= 0:
        return {
            "delta": 1.0 if option_type == "call" and S > K else (
                -1.0 if option_type == "put" and S < K else 0.0
            ),
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # --- Delta ---
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1  # = -N(-d1)

    # --- Gamma (identique pour call et put) ---
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # --- Theta ---
    # Composante commune liée au time decay de la volatilité
    theta_common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        theta = theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)

    # On convertit en theta PAR JOUR (le trader veut savoir combien il perd par jour)
    theta_daily = theta / 365

    # --- Vega ---
    # Sensibilité à 1 point de vol (pas 100%)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # --- Rho ---
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta_daily),
        "vega": float(vega),
        "rho": float(rho),
    }

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-10,
    max_iter: int = 500,
) -> float:
    """
    Calcule la volatilité implicite par Newton-Raphson amélioré.
    """

    # Protection de base
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan

    # --- Estimation initiale de σ ---
    # Brenner-Subrahmanyam pour ATM
    sigma_est = np.sqrt(2 * np.pi / T) * market_price / S
    sigma_est = np.clip(sigma_est, 0.01, 5.0)

    # --- Boucle Newton-Raphson ---
    for i in range(max_iter):
        bs_price = black_scholes_price(S, K, T, r, sigma_est, option_type)
        error = bs_price - market_price

        if abs(error) < tol:
            return float(sigma_est)

        d1 = (np.log(S / K) + (r + 0.5 * sigma_est ** 2) * T) / (
            sigma_est * np.sqrt(T)
        )
        vega = S * norm.pdf(d1) * np.sqrt(T)

        if vega < 1e-12:
            # Vega trop petit → on passe en bissection
            break

        # Mise à jour Newton-Raphson avec amortissement
        # Le facteur 0.5-1.0 empêche les sauts trop grands
        step = error / vega
        # Si le pas est trop grand par rapport à sigma, on le réduit
        if abs(step) > 0.5 * sigma_est:
            step = 0.5 * sigma_est * np.sign(step)

        sigma_est = sigma_est - step
        sigma_est = np.clip(sigma_est, 0.001, 5.0)

    # --- Fallback : bissection si Newton-Raphson n'a pas convergé ---
    # La bissection est plus lente mais GARANTIT la convergence
    sigma_low = 0.001
    sigma_high = 5.0

    for i in range(200):
        sigma_mid = (sigma_low + sigma_high) / 2
        bs_mid = black_scholes_price(S, K, T, r, sigma_mid, option_type)

        if abs(bs_mid - market_price) < tol:
            return float(sigma_mid)

        if bs_mid > market_price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid

    return float((sigma_low + sigma_high) / 2)