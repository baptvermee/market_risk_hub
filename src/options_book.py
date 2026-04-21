"""
Moteur de gestion d'un book d'options.

==========================================================================
C'EST QUOI UN BOOK D'OPTIONS ?
==========================================================================

Un trader d'options ne détient jamais une seule option. Il a un "book" :
des dizaines ou centaines de positions (calls, puts, différents strikes,
différentes maturités, long et short).

Le book se gère via les GREEKS AGRÉGÉS :
- Delta net : sensibilité totale du book au prix du sous-jacent
- Gamma net : comment le delta net change quand le spot bouge
- Theta net : combien le book gagne/perd par jour qui passe
- Vega net : sensibilité totale du book à la volatilité

Un trader delta-neutral a un delta net ≈ 0 : son P&L ne dépend pas
de la direction du marché, seulement de la volatilité et du gamma.

==========================================================================
ARCHITECTURE
==========================================================================

- compute_book_greeks()    : agrège les Greeks de toutes les positions
- compute_book_pnl()       : calcule le P&L du book en fonction du spot
- compute_pnl_surface()    : P&L en fonction du spot ET du temps
"""

import numpy as np
from src.vanilla_option_pricer import black_scholes_price, compute_greeks


def compute_book_greeks(positions: list, S: float, r: float, sigma: float) -> dict:
    """
    Calcule les Greeks agrégés d'un book d'options.

    Chaque position est un dict :
    {
        "type": "call" ou "put",
        "strike": float,
        "maturity": float (en années),
        "quantity": int (positif = long, négatif = short),
    }

    Les Greeks sont ADDITIFS : le delta du book = somme des deltas
    de chaque position × quantité. Idem pour gamma, theta, vega, rho.

    C'est une propriété fondamentale qui vient de la linéarité
    de la dérivée : d(A+B)/dS = dA/dS + dB/dS

    Paramètres
    ----------
    positions : list de dicts décrivant chaque position
    S         : prix spot actuel
    r         : taux sans risque
    sigma     : volatilité (commune à toutes les positions)

    Retourne
    --------
    dict avec :
        - Greeks agrégés (delta, gamma, theta, vega, rho)
        - Valeur totale du book
        - Détail par position
    """

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
    """
    Calcule le P&L du book pour une gamme de prix spot.

    Pour chaque spot dans spot_range, on recalcule la valeur du book
    et on compare à la valeur actuelle → P&L = valeur_nouvelle - valeur_actuelle

    On calcule aussi le P&L à maturité (valeur intrinsèque seulement,
    sans valeur temps) pour voir le profil de payoff pur.

    Paramètres
    ----------
    positions   : list de dicts décrivant chaque position
    S_current   : spot actuel (pour calculer le P&L relatif)
    r           : taux sans risque
    sigma       : volatilité
    spot_range  : array de spots pour lesquels calculer le P&L

    Retourne
    --------
    dict avec :
        - "spot_range"     : array des spots
        - "pnl_current"    : P&L si on clôture maintenant à chaque spot
        - "pnl_at_expiry"  : P&L à maturité (payoff intrinsèque)
        - "current_value"  : valeur actuelle du book
    """

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
    """
    Calcule les Greeks agrégés du book pour chaque spot dans spot_range.

    Ça donne le profil de chaque Greek en fonction du spot :
    - Comment le delta net évolue quand le spot bouge
    - Où le gamma est maximal (zones de risque de hedging)
    - Etc.

    Retourne
    --------
    dict avec des arrays : deltas, gammas, thetas, vegas
    """

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