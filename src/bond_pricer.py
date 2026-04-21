"""
Moteur de pricing d'obligations (Fixed Income).

==========================================================================
LES BASES DU PRICING OBLIGATAIRE
==========================================================================

Une obligation est un contrat où :
- L'émetteur (entreprise, État) emprunte de l'argent
- L'investisseur prête son argent et reçoit en échange :
  → Des coupons périodiques (intérêts)
  → Le remboursement du principal (nominal) à maturité

Le PRIX d'une obligation = somme des flux futurs actualisés :
    P = Σ (Coupon_i / (1+y)^i) + Nominal / (1+y)^N

où y = taux de rendement (yield) et N = nombre de périodes.

TYPES D'OBLIGATIONS :
- Taux fixe classique : coupons réguliers + remboursement in fine
- Zéro coupon : pas de coupon, achetée à discount, remboursée au pair
- Amortissable : le principal est remboursé progressivement à chaque période

MESURES DE RISQUE :
- Duration de Macaulay : durée de vie moyenne pondérée des flux
- Duration modifiée : sensibilité du prix à un changement de taux
  ΔP/P ≈ -D_mod × Δy
- Convexité : correction du second ordre
  ΔP/P ≈ -D_mod × Δy + 0.5 × Convexité × (Δy)²
"""

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
    """
    Génère l'échéancier des flux d'une obligation.

    Paramètres
    ----------
    face_value  : valeur nominale (ex: 1000€)
    coupon_rate : taux de coupon annuel (ex: 0.05 = 5%)
    maturity    : maturité en années (ex: 10)
    frequency   : nombre de coupons par an (1=annuel, 2=semestriel, 4=trimestriel)
    bond_type   : "fixed" (taux fixe), "zero" (zéro coupon), "amortizing" (amortissable)

    Retourne
    --------
    dict avec :
        - "times"       : array des dates de flux (en années)
        - "coupons"     : array des montants de coupon à chaque date
        - "principals"  : array des remboursements de principal à chaque date
        - "total_flows" : coupons + principals (ce qu'on reçoit vraiment)
        - "remaining_principal" : principal restant après chaque date
    """

    # Nombre total de périodes
    n_periods = maturity * frequency

    # Coupon par période
    # Ex: coupon annuel 5% payé semestriellement → 2.5% par semestre
    coupon_per_period = face_value * coupon_rate / frequency

    # Dates de flux (en années)
    # Ex: semestriel sur 5 ans → [0.5, 1.0, 1.5, 2.0, ..., 5.0]
    times = np.array([(i + 1) / frequency for i in range(n_periods)])

    if bond_type == "zero":
        # --- Zéro coupon ---
        # Pas de coupon du tout, juste le remboursement à maturité
        # L'investisseur achète à discount (ex: 850€) et reçoit 1000€ à maturité
        coupons = np.zeros(n_periods)
        principals = np.zeros(n_periods)
        principals[-1] = face_value  # remboursement total à la fin

    elif bond_type == "amortizing":
        # --- Obligation amortissable ---
        # Le principal est remboursé en parts égales à chaque période
        # Le coupon est calculé sur le principal RESTANT (pas le nominal initial)
        #
        # Ex: 1000€, 10 périodes → 100€ de principal remboursé à chaque période
        # Le coupon de la période 1 = taux × 1000€
        # Le coupon de la période 2 = taux × 900€ (il reste 900€)
        # etc.
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
    """
    Calcule le prix d'une obligation (dirty price).

    Prix = Σ (CF_i / (1 + y/freq)^i)

    où :
    - CF_i = flux à la période i (coupon + éventuel remboursement de principal)
    - y    = yield to maturity (taux de rendement)
    - freq = nombre de coupons par an

    On actualise chaque flux par le facteur (1 + y/freq)^i
    Le y est divisé par freq car c'est un taux annuel appliqué à des périodes infra-annuelles.

    Paramètres
    ----------
    face_value  : nominal
    coupon_rate : taux de coupon annuel
    maturity    : maturité en années
    ytm         : yield to maturity (taux de rendement annuel)
    frequency   : coupons par an
    bond_type   : "fixed", "zero", "amortizing"

    Retourne
    --------
    dict avec :
        - "dirty_price"   : prix total (incluant les intérêts courus)
        - "cash_flows"    : détail de l'échéancier
        - "pv_flows"      : valeur présente de chaque flux
        - "discount_factors" : facteurs d'actualisation
    """

    cf = generate_cash_flows(face_value, coupon_rate, maturity, frequency, bond_type)

    # Taux par période
    y_per_period = ytm / frequency

    # Facteurs d'actualisation pour chaque période
    # df[i] = 1 / (1 + y/freq)^(i+1)
    # Le flux à la période i est multiplié par df[i] pour obtenir sa valeur présente
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
    """
    Calcule le prix clean, dirty et les intérêts courus (accrued interest).

    DIRTY PRICE = ce que l'acheteur paye réellement
    ACCRUED INTEREST = la part du prochain coupon qui "appartient" au vendeur
    CLEAN PRICE = dirty - accrued = le prix coté sur les écrans

    Pourquoi cette distinction ?
    Si tu achètes une obligation 1 jour avant le coupon, tu vas recevoir
    le coupon entier. Mais tu ne le "mérites" pas — le vendeur détenait
    l'obligation pendant presque toute la période. Donc tu lui payes
    les intérêts courus en plus du prix clean.

    Accrued = Coupon_annuel/freq × (jours depuis dernier coupon / jours dans la période)
    """

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
    """
    Trouve le YTM à partir du prix de marché par résolution numérique.

    Le YTM est le taux y tel que :
        Prix_marché = Σ (CF_i / (1 + y/freq)^i)

    C'est l'inverse du pricing : on connaît le prix, on cherche le taux.
    C'est comme la vol implicite pour les options — même idée, domaine différent.

    On utilise brentq (méthode de Brent) pour trouver la racine de :
        f(y) = prix_théorique(y) - prix_marché = 0
    """

    def objective(ytm_guess):
        result = bond_price(face_value, coupon_rate, maturity, ytm_guess, frequency, bond_type)
        return result["dirty_price"] - market_price

    try:
        # brentq cherche la racine dans l'intervalle [-0.05, 1.0]
        # -5% (taux négatifs existent !) à 100% (obligation en détresse)
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
    """
    Duration de Macaulay (en années).

    C'est la DURÉE DE VIE MOYENNE PONDÉRÉE des flux de l'obligation,
    où les poids sont les valeurs présentes de chaque flux.

    D_mac = (1/P) × Σ (t_i × PV(CF_i))

    Intuition : si tous les flux arrivent dans 5 ans (zéro coupon),
    la duration = 5 ans. Si une partie arrive avant (coupons),
    la duration < maturité.

    La duration mesure aussi la sensibilité "temporelle" :
    c'est le temps qu'il faut pour récupérer son investissement
    en moyenne pondérée.
    """

    result = bond_price(face_value, coupon_rate, maturity, ytm, frequency, bond_type)
    cf = result["cash_flows"]
    price = result["dirty_price"]

    if price <= 0:
        return np.nan

    # Somme pondérée : chaque flux est multiplié par son temps (en années)
    # puis on divise par le prix
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
    """
    Duration modifiée = D_mac / (1 + y/freq)

    C'est LA mesure de risque de taux la plus utilisée en fixed income.
    Elle donne directement la sensibilité du prix à un changement de taux :

        ΔP/P ≈ -D_mod × Δy

    Ex: D_mod = 7, Δy = +1% → ΔP/P ≈ -7% (le prix baisse de 7%)

    La division par (1 + y/freq) convertit la duration de Macaulay
    (qui est en "temps") en une sensibilité (qui est en "% par % de taux").
    """

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
    """
    Convexité de l'obligation.

    La duration est une approximation LINÉAIRE de la relation prix-taux.
    Mais cette relation est en réalité CONVEXE (courbée).
    La convexité mesure cette courbure.

    C = (1/P) × Σ [t_i × (t_i + 1/freq) × PV(CF_i)] / (1 + y/freq)²

    Avec la convexité, l'approximation devient :
        ΔP/P ≈ -D_mod × Δy + 0.5 × C × (Δy)²

    Le terme de convexité est toujours POSITIF (pour une obligation classique),
    ce qui signifie que :
    - Quand les taux baissent, le prix monte PLUS que ce que la duration prédit
    - Quand les taux montent, le prix baisse MOINS que ce que la duration prédit
    → La convexité est une bonne chose pour l'investisseur !
    """

    result = bond_price(face_value, coupon_rate, maturity, ytm, frequency, bond_type)
    cf = result["cash_flows"]
    price = result["dirty_price"]

    if price <= 0:
        return np.nan

    y_per = ytm / frequency
    periods = np.arange(1, cf["n_periods"] + 1)

    # Formule de la convexité
    # On utilise t × (t+1) au lieu de t² pour la correction discrète
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
    """
    Analyse l'impact de chocs de taux sur le prix de l'obligation.

    Pour chaque choc (en basis points), on calcule :
    - Le prix exact (recalcul complet)
    - L'approximation par la duration seule
    - L'approximation par duration + convexité
    → On voit que la convexité améliore l'approximation pour les gros chocs.

    1 basis point (bp) = 0.01% = 0.0001
    """

    if shocks_bps is None:
        shocks_bps = [-200, -100, -50, -25, 0, 25, 50, 100, 200]

    # Prix, duration et convexité au taux actuel
    base = bond_price(face_value, coupon_rate, maturity, ytm, frequency, bond_type)
    base_price = base["dirty_price"]
    d_mod = modified_duration(face_value, coupon_rate, maturity, ytm, frequency, bond_type)
    conv = convexity(face_value, coupon_rate, maturity, ytm, frequency, bond_type)

    results = []
    for shock_bp in shocks_bps:
        shock = shock_bp / 10000  # conversion bps → décimal
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
    """
    Calcule le prix de l'obligation pour une gamme de taux.

    C'est la courbe fondamentale du fixed income :
    - Elle est DÉCROISSANTE (taux ↑ → prix ↓)
    - Elle est CONVEXE (courbée vers le haut)
    - La pente à un point donné = -duration modifiée × prix
    """

    ytm_values = np.linspace(ytm_range[0], ytm_range[1], n_points)
    prices = []

    for y in ytm_values:
        result = bond_price(face_value, coupon_rate, maturity, y, frequency, bond_type)
        prices.append(result["dirty_price"])

    return {
        "ytm_values": ytm_values,
        "prices": np.array(prices),
    }