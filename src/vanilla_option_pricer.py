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
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

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

    # --- Gamma ---
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # --- Theta ---
    theta_common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        theta = theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)

    theta_daily = theta / 365

    # --- Vega ---
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
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan

    # Bornes d'arbitrage
    if option_type == "call":
        lower_bound = max(S - K * np.exp(-r * T), 0)
        upper_bound = S
    else:
        lower_bound = max(K * np.exp(-r * T) - S, 0)
        upper_bound = K * np.exp(-r * T)

    if market_price <= lower_bound or market_price >= upper_bound:
        return np.nan

    # Estimation initiale (Brenner-Subrahmanyam)
    sigma_est = np.sqrt(2 * np.pi / T) * market_price / S
    sigma_est = np.clip(sigma_est, 0.01, 5.0)

    # Newton-Raphson avec amortissement
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
            break

        step = error / vega
        if abs(step) > 0.5 * sigma_est:
            step = 0.5 * sigma_est * np.sign(step)

        sigma_est = sigma_est - step
        sigma_est = np.clip(sigma_est, 0.001, 5.0)

    # Fallback bissection
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