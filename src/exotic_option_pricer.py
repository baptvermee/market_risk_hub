import numpy as np
from scipy.stats import norm


# =============================================================================
# 1. SIMULATEUR DE CHEMINS — Mouvement Brownien Géométrique (GBM)
# =============================================================================

def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_sims: int,
    seed: int = 42,
) -> np.ndarray:
    np.random.seed(seed)

    dt = T / n_steps
    Z = np.random.normal(size=(n_steps, n_sims))
    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    cumulative_log_returns = np.vstack([
        np.zeros((1, n_sims)),    # t=0 : log-rendement cumulé = 0
        np.cumsum(log_returns, axis=0)  # t=1 à t=n_steps
    ])

    paths = S0 * np.exp(cumulative_log_returns)

    return paths


# =============================================================================
# 2. OPTION ASIATIQUE — Moyenne Arithmétique
# =============================================================================

def price_asian_option(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    n_sims: int = 100000,
    option_type: str = "call",
    seed: int = 42,
) -> dict:
    # 1. Simuler les chemins
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_sims, seed)
    avg_prices = np.mean(paths, axis=0) 

    # 3. Calculer les payoffs
    if option_type == "call":
        payoffs = np.maximum(avg_prices - K, 0)
    else:
        payoffs = np.maximum(K - avg_prices, 0)

    # 4. Actualiser les payoffs
    #    Le prix de l'option = espérance des payoffs × facteur d'actualisation
    #    exp(-rT) ramène les flux futurs en valeur présente
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    # 5. Calculer le prix et les statistiques d'erreur
    price = np.mean(discounted_payoffs)

    # L'erreur standard mesure la précision de notre estimation
    # Elle décroît en 1/sqrt(n_sims) : 4x plus de sims = 2x plus précis
    std_error = np.std(discounted_payoffs) / np.sqrt(n_sims)

    # Intervalle de confiance à 95% : prix ± 1.96 × erreur standard
    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error

    return {
        "price": price,
        "std_error": std_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "paths": paths,
        "payoffs": payoffs,
        "avg_prices": avg_prices,
    }


# =============================================================================
# 3. COMPARAISON VANILLE — Pour voir la différence de prix
# =============================================================================

def price_vanilla_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_sims: int = 100000,
    option_type: str = "call",
    seed: int = 42,
) -> dict:
    paths = simulate_gbm_paths(S0, r, sigma, T, 252, n_sims, seed)

    # Pour la vanille, seul le prix FINAL compte
    final_prices = paths[-1, :]  # dernière ligne = prix à maturité

    if option_type == "call":
        payoffs = np.maximum(final_prices - K, 0)
    else:
        payoffs = np.maximum(K - final_prices, 0)

    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_sims)

    return {
        "price": price,
        "std_error": std_error,
        "ci_lower": price - 1.96 * std_error,
        "ci_upper": price + 1.96 * std_error,
    }


# =============================================================================
# 4. BLACK-SCHOLES ANALYTIQUE — Référence exacte
# =============================================================================

def black_scholes_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
) -> float:
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)