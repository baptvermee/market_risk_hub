import numpy as np
from src.vanilla_option_pricer import black_scholes_price, compute_greeks


def compute_book_greeks(positions: list, S: float, r: float, sigma: float) -> dict:
    total_delta = 0
    total_gamma = 0
    total_theta = 0
    total_vega = 0
    total_rho = 0
    total_value = 0

    details = []

    for pos in positions:
        opt_type = pos["type"]
        K = pos["strike"]
        T = pos["maturity"]
        qty = pos["quantity"]

        # Prix et Greeks de cette option individuelle
        price = black_scholes_price(S, K, T, r, sigma, opt_type)
        greeks = compute_greeks(S, K, T, r, sigma, opt_type)

        # Contribution de cette position = Greeks × quantité
        # Si qty > 0 (long), on ajoute
        # Si qty < 0 (short), on soustrait
        pos_delta = greeks["delta"] * qty
        pos_gamma = greeks["gamma"] * qty
        pos_theta = greeks["theta"] * qty
        pos_vega = greeks["vega"] * qty
        pos_rho = greeks["rho"] * qty
        pos_value = price * qty

        total_delta += pos_delta
        total_gamma += pos_gamma
        total_theta += pos_theta
        total_vega += pos_vega
        total_rho += pos_rho
        total_value += pos_value

        details.append({
            "type": opt_type,
            "strike": K,
            "maturity": T,
            "quantity": qty,
            "price": price,
            "value": pos_value,
            "delta": pos_delta,
            "gamma": pos_gamma,
            "theta": pos_theta,
            "vega": pos_vega,
            "rho": pos_rho,
        })

    return {
        "total_delta": total_delta,
        "total_gamma": total_gamma,
        "total_theta": total_theta,
        "total_vega": total_vega,
        "total_rho": total_rho,
        "total_value": total_value,
        "details": details,
    }


def compute_book_pnl(
    positions: list,
    S_current: float,
    r: float,
    sigma: float,
    spot_range: np.ndarray,
) -> dict:
    # Valeur actuelle du book (au spot actuel)
    current_book = compute_book_greeks(positions, S_current, r, sigma)
    current_value = current_book["total_value"]

    pnl_current = np.zeros(len(spot_range))
    pnl_at_expiry = np.zeros(len(spot_range))

    for i, s in enumerate(spot_range):
        # P&L mark-to-market : on recalcule la valeur du book à ce spot
        new_book = compute_book_greeks(positions, s, r, sigma)
        pnl_current[i] = new_book["total_value"] - current_value

        # P&L à maturité : seulement la valeur intrinsèque
        expiry_value = 0
        for pos in positions:
            if pos["type"] == "call":
                payoff = max(s - pos["strike"], 0)
            else:
                payoff = max(pos["strike"] - s, 0)
            expiry_value += payoff * pos["quantity"]

        # On soustrait la prime payée/reçue
        pnl_at_expiry[i] = expiry_value - current_value

    return {
        "spot_range": spot_range,
        "pnl_current": pnl_current,
        "pnl_at_expiry": pnl_at_expiry,
        "current_value": current_value,
    }


def compute_greeks_profile(
    positions: list,
    r: float,
    sigma: float,
    spot_range: np.ndarray,
) -> dict:
    deltas = np.zeros(len(spot_range))
    gammas = np.zeros(len(spot_range))
    thetas = np.zeros(len(spot_range))
    vegas = np.zeros(len(spot_range))

    for i, s in enumerate(spot_range):
        book = compute_book_greeks(positions, s, r, sigma)
        deltas[i] = book["total_delta"]
        gammas[i] = book["total_gamma"]
        thetas[i] = book["total_theta"]
        vegas[i] = book["total_vega"]

    return {
        "spot_range": spot_range,
        "deltas": deltas,
        "gammas": gammas,
        "thetas": thetas,
        "vegas": vegas,
    }