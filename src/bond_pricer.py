import numpy as np
from scipy.optimize import brentq


# =============================================================================
# 1. GÉNÉRATION DES FLUX (CASH FLOWS)
# =============================================================================

def generate_cash_flows(
    face_value: float,
    coupon_rate: float,
    maturity: int,
    frequency: int = 2,
    bond_type: str = "fixed",
) -> dict:
    # Nombre total de périodes
    n_periods = maturity * frequency

    # Coupon par période
    coupon_per_period = face_value * coupon_rate / frequency

    # Dates de flux (en années)
    times = np.array([(i + 1) / frequency for i in range(n_periods)])

    if bond_type == "zero":
        # --- Zéro coupon ---
        coupons = np.zeros(n_periods)
        principals = np.zeros(n_periods)
        principals[-1] = face_value  # remboursement total à la fin

    elif bond_type == "amortizing":
        # --- Obligation amortissable ---
        principal_per_period = face_value / n_periods
        principals = np.full(n_periods, principal_per_period)

        remaining = face_value
        coupons = np.zeros(n_periods)
        for i in range(n_periods):
            coupons[i] = remaining * coupon_rate / frequency
            remaining -= principal_per_period

    else:
        # --- Taux fixe classique (in fine) ---
        # Coupons identiques à chaque période + remboursement total à la fin
        coupons = np.full(n_periods, coupon_per_period)
        principals = np.zeros(n_periods)
        principals[-1] = face_value  # bullet repayment

    total_flows = coupons + principals

    # Principal restant après chaque date
    remaining_principal = face_value - np.cumsum(principals)

    return {
        "times": times,
        "coupons": coupons,
        "principals": principals,
        "total_flows": total_flows,
        "remaining_principal": remaining_principal,
        "n_periods": n_periods,
        "frequency": frequency,
    }


# =============================================================================
# 2. PRIX DE L'OBLIGATION
# =============================================================================

def bond_price(
    face_value: float,
    coupon_rate: float,
    maturity: int,
    ytm: float,
    frequency: int = 2,
    bond_type: str = "fixed",
) -> dict:
    cf = generate_cash_flows(face_value, coupon_rate, maturity, frequency, bond_type)

    # Taux par période
    y_per_period = ytm / frequency

    # Facteurs d'actualisation pour chaque période
    periods = np.arange(1, cf["n_periods"] + 1)
    discount_factors = 1 / (1 + y_per_period) ** periods

    # Valeur présente de chaque flux
    pv_flows = cf["total_flows"] * discount_factors

    # Prix = somme des valeurs présentes
    dirty_price = np.sum(pv_flows)

    return {
        "dirty_price": dirty_price,
        "cash_flows": cf,
        "pv_flows": pv_flows,
        "discount_factors": discount_factors,
    }


# =============================================================================
# 3. ACCRUED INTEREST & CLEAN PRICE
# =============================================================================

def clean_dirty_price(
    face_value: float,
    coupon_rate: float,
    maturity: int,
    ytm: float,
    frequency: int = 2,
    bond_type: str = "fixed",
    days_since_last_coupon: int = 0,
    days_in_coupon_period: int = 182,
) -> dict:
    result = bond_price(face_value, coupon_rate, maturity, ytm, frequency, bond_type)

    if bond_type == "zero":
        accrued = 0.0
    else:
        coupon_per_period = face_value * coupon_rate / frequency
        # Proportion de la période écoulée depuis le dernier coupon
        accrual_fraction = days_since_last_coupon / days_in_coupon_period
        accrued = coupon_per_period * accrual_fraction

    clean_price = result["dirty_price"] - accrued

    return {
        "dirty_price": result["dirty_price"],
        "clean_price": clean_price,
        "accrued_interest": accrued,
        "cash_flows": result["cash_flows"],
        "pv_flows": result["pv_flows"],
        "discount_factors": result["discount_factors"],
    }


# =============================================================================
# 4. YIELD TO MATURITY (YTM) — Recherche du taux implicite
# =============================================================================

def yield_to_maturity(
    market_price: float,
    face_value: float,
    coupon_rate: float,
    maturity: int,
    frequency: int = 2,
    bond_type: str = "fixed",
) -> float:
    def objective(ytm_guess):
        result = bond_price(face_value, coupon_rate, maturity, ytm_guess, frequency, bond_type)
        return result["dirty_price"] - market_price

    try:
        ytm = brentq(objective, -0.05, 1.0, xtol=1e-10)
        return ytm
    except ValueError:
        return np.nan


# =============================================================================
# 5. DURATION DE MACAULAY
# =============================================================================

def macaulay_duration(
    face_value: float,
    coupon_rate: float,
    maturity: int,
    ytm: float,
    frequency: int = 2,
    bond_type: str = "fixed",
) -> float:
    result = bond_price(face_value, coupon_rate, maturity, ytm, frequency, bond_type)
    cf = result["cash_flows"]
    price = result["dirty_price"]

    if price <= 0:
        return np.nan
    weighted_times = cf["times"] * result["pv_flows"]
    duration = np.sum(weighted_times) / price

    return duration


# =============================================================================
# 6. DURATION MODIFIÉE
# =============================================================================

def modified_duration(
    face_value: float,
    coupon_rate: float,
    maturity: int,
    ytm: float,
    frequency: int = 2,
    bond_type: str = "fixed",
) -> float:
    d_mac = macaulay_duration(face_value, coupon_rate, maturity, ytm, frequency, bond_type)

    if np.isnan(d_mac):
        return np.nan

    return d_mac / (1 + ytm / frequency)


# =============================================================================
# 7. CONVEXITÉ
# =============================================================================

def convexity(
    face_value: float,
    coupon_rate: float,
    maturity: int,
    ytm: float,
    frequency: int = 2,
    bond_type: str = "fixed",
) -> float:
    result = bond_price(face_value, coupon_rate, maturity, ytm, frequency, bond_type)
    cf = result["cash_flows"]
    price = result["dirty_price"]

    if price <= 0:
        return np.nan

    y_per = ytm / frequency
    periods = np.arange(1, cf["n_periods"] + 1)

    # Formule de la convexité
    weighted = periods * (periods + 1) * result["pv_flows"]
    conv = np.sum(weighted) / (price * frequency ** 2 * (1 + y_per) ** 2)

    return conv


# =============================================================================
# 8. ANALYSE DE SENSIBILITÉ AUX TAUX
# =============================================================================

def rate_sensitivity_analysis(
    face_value: float,
    coupon_rate: float,
    maturity: int,
    ytm: float,
    frequency: int = 2,
    bond_type: str = "fixed",
    shocks_bps: list = None,
) -> dict:
    if shocks_bps is None:
        shocks_bps = [-200, -100, -50, -25, 0, 25, 50, 100, 200]

    # Prix, duration et convexité au taux actuel
    base = bond_price(face_value, coupon_rate, maturity, ytm, frequency, bond_type)
    base_price = base["dirty_price"]
    d_mod = modified_duration(face_value, coupon_rate, maturity, ytm, frequency, bond_type)
    conv = convexity(face_value, coupon_rate, maturity, ytm, frequency, bond_type)

    results = []
    for shock_bp in shocks_bps:
        shock = shock_bp / 10000  # conversion bps
        new_ytm = ytm + shock

        # Prix exact (recalcul complet avec le nouveau taux)
        new_result = bond_price(face_value, coupon_rate, maturity, new_ytm, frequency, bond_type)
        exact_price = new_result["dirty_price"]

        # Approximation duration seule : ΔP/P ≈ -D_mod × Δy
        duration_approx = base_price * (1 - d_mod * shock)

        # Approximation duration + convexité : ΔP/P ≈ -D_mod × Δy + 0.5 × C × (Δy)²
        full_approx = base_price * (1 - d_mod * shock + 0.5 * conv * shock ** 2)

        # Erreurs d'approximation
        duration_error = duration_approx - exact_price
        full_error = full_approx - exact_price

        results.append({
            "shock_bps": shock_bp,
            "new_ytm": new_ytm,
            "exact_price": exact_price,
            "duration_approx": duration_approx,
            "full_approx": full_approx,
            "exact_change_pct": (exact_price / base_price - 1) * 100,
            "duration_error": duration_error,
            "full_error": full_error,
        })

    return {
        "base_price": base_price,
        "modified_duration": d_mod,
        "convexity": conv,
        "shocks": results,
    }


# =============================================================================
# 9. COURBE PRIX-TAUX
# =============================================================================

def price_yield_curve(
    face_value: float,
    coupon_rate: float,
    maturity: int,
    frequency: int = 2,
    bond_type: str = "fixed",
    ytm_range: tuple = (0.001, 0.15),
    n_points: int = 200,
) -> dict:
    ytm_values = np.linspace(ytm_range[0], ytm_range[1], n_points)
    prices = []

    for y in ytm_values:
        result = bond_price(face_value, coupon_rate, maturity, y, frequency, bond_type)
        prices.append(result["dirty_price"])

    return {
        "ytm_values": ytm_values,
        "prices": np.array(prices),
    }