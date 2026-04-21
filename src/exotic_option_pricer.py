"""
Moteur de pricing d'options exotiques par Monte Carlo.

Pourquoi Monte Carlo ?
Les options exotiques ont des payoffs qui dépendent du CHEMIN du prix
(la moyenne, le max, le min, le franchissement d'une barrière...).
Il n'existe pas de formule fermée comme Black-Scholes pour les pricer.
On doit donc simuler des milliers de chemins de prix possibles
et calculer le payoff sur chacun.

Architecture :
- simulate_gbm_paths()   : génère les chemins de prix (réutilisable pour tout type d'exotique)
- price_asian_option()    : price une option asiatique (moyenne arithmétique)
- Les fonctions pour barrières, lookback, digitales seront ajoutées ici plus tard
"""

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
    """
    Simule des chemins de prix selon le Mouvement Brownien Géométrique (GBM).

    Le GBM est le modèle standard en finance. Il suppose que le prix suit :
        dS = r * S * dt + sigma * S * dW

    où :
        - r     = taux sans risque (le "drift" en risque neutre)
        - sigma = volatilité
        - dW    = incrément d'un mouvement brownien (= choc aléatoire)

    On discrétise cette équation en N pas de temps :
        S(t+dt) = S(t) * exp((r - 0.5*sigma²)*dt + sigma*sqrt(dt)*Z)

    où Z ~ N(0,1). Le terme -0.5*sigma² est la "correction d'Itô" qui garantit
    que E[S(T)] = S(0) * exp(r*T) (pas de biais systématique).

    Paramètres
    ----------
    S0      : prix initial du sous-jacent
    r       : taux sans risque annualisé (ex: 0.05 = 5%)
    sigma   : volatilité annualisée (ex: 0.20 = 20%)
    T       : maturité en années (ex: 0.5 = 6 mois)
    n_steps : nombre de pas de temps dans chaque chemin
              Plus c'est élevé, plus la simulation est précise
              (surtout important pour les barrières et lookback)
    n_sims  : nombre de chemins simulés
    seed    : graine aléatoire pour la reproductibilité

    Retourne
    --------
    paths : np.ndarray de shape (n_steps + 1, n_sims)
            Chaque colonne est un chemin de prix.
            paths[0, :] = S0 pour tous les chemins
            paths[-1, :] = prix final de chaque chemin
    """

    np.random.seed(seed)

    # --- Pas de temps ---
    # dt = fraction d'année entre deux pas
    # Ex: T=1 an, n_steps=252 → dt ≈ 1 jour de trading
    dt = T / n_steps

    # --- Génération vectorisée de TOUS les chocs d'un coup ---
    # Au lieu de boucler pas par pas, on génère une matrice complète
    # de nombres aléatoires : (n_steps lignes) × (n_sims colonnes)
    # Chaque élément Z[i,j] est le choc aléatoire au pas i, simulation j
    Z = np.random.normal(size=(n_steps, n_sims))

    # --- Calcul des rendements log à chaque pas ---
    # La formule discrétisée du GBM donne le rendement logarithmique :
    #   log(S(t+dt)/S(t)) = (r - 0.5*sigma²)*dt + sigma*sqrt(dt)*Z
    #
    # (r - 0.5*sigma²)*dt  = partie déterministe (drift)
    # sigma*sqrt(dt)*Z     = partie aléatoire (diffusion)
    #
    # On calcule ça pour TOUS les pas et TOUTES les simulations d'un coup
    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z

    # --- Construction des chemins de prix ---
    # On veut S(t) = S0 * exp(somme cumulative des log-rendements)
    #
    # np.cumsum(log_returns, axis=0) fait la somme cumulative le long de l'axe temps
    # Pour chaque simulation j : cumsum[i, j] = sum(log_returns[0:i+1, j])
    #
    # np.vstack ajoute une ligne de zéros en haut (le temps 0, avant tout mouvement)
    # Puis exp() transforme les log-prix en prix
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
    """
    Price une option asiatique à moyenne arithmétique par Monte Carlo.

    Payoff call asiatique : max(moyenne(S) - K, 0)
    Payoff put asiatique  : max(K - moyenne(S), 0)

    La "moyenne(S)" est calculée sur TOUT le chemin du prix,
    pas juste sur le prix final. C'est ça qui rend l'option asiatique
    différente d'une vanille, et c'est pour ça qu'on a besoin de Monte Carlo.

    Paramètres
    ----------
    S0          : prix spot du sous-jacent
    K           : strike (prix d'exercice)
    r           : taux sans risque annualisé
    sigma       : volatilité annualisée
    T           : maturité en années
    n_steps     : nombre de points pour calculer la moyenne
                  252 = un point par jour de trading sur 1 an
    n_sims      : nombre de simulations Monte Carlo
    option_type : "call" ou "put"
    seed        : graine aléatoire

    Retourne
    --------
    dict avec :
        - "price"        : prix de l'option (espérance actualisée des payoffs)
        - "std_error"    : erreur standard de l'estimation
                           (mesure la précision : plus n_sims est grand, plus c'est petit)
        - "ci_lower/upper" : intervalle de confiance à 95%
        - "paths"        : les chemins simulés (pour l'affichage)
        - "payoffs"      : les payoffs de chaque simulation
        - "avg_prices"   : la moyenne du prix sur chaque chemin
    """

    # 1. Simuler les chemins
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_sims, seed)

    # 2. Calculer la moyenne du prix sur chaque chemin
    #    np.mean(paths, axis=0) calcule la moyenne le long de l'axe temps (axis=0)
    #    Pour chaque simulation j : avg_prices[j] = moyenne de paths[:, j]
    #    C'est la valeur clé de l'option asiatique !
    avg_prices = np.mean(paths, axis=0)  # shape: (n_sims,)

    # 3. Calculer les payoffs
    #    Call : on gagne si la moyenne est AU-DESSUS du strike
    #    Put  : on gagne si la moyenne est EN-DESSOUS du strike
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
    """
    Price une option vanille par Monte Carlo (pour comparaison avec l'asiatique).

    On utilise Monte Carlo au lieu de Black-Scholes pour que la comparaison
    soit fair : même méthode, même chemins aléatoires, seul le payoff change.

    Payoff call vanille : max(S_T - K, 0)  ← dépend UNIQUEMENT du prix final
    Payoff call asiatique : max(avg(S) - K, 0)  ← dépend de la MOYENNE
    """

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
    """
    Prix Black-Scholes analytique (formule fermée).

    Sert de référence pour valider notre Monte Carlo vanille :
    si le MC vanille donne un prix proche du BS, notre simulateur est correct.
    """

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)