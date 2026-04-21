"""
data_loader.py
--------------
Module responsable du téléchargement et du nettoyage
des données de marché via Yahoo Finance.

Toutes les pages Streamlit importeront ce module.
"""

import yfinance as yf
import pandas as pd


def load_prices(tickers: list[str], start, end) -> pd.DataFrame:
    """
    Télécharge les prix de clôture pour une liste de tickers.

    Paramètres
    ----------
    tickers : list[str]
        Liste des codes Yahoo Finance, ex: ["AAPL", "MSFT"]
    start : date
        Date de début de l'historique
    end : date
        Date de fin de l'historique

    Retourne
    --------
    pd.DataFrame
        Un DataFrame avec une colonne par ticker, indexé par date.
        Les colonnes sans aucune donnée sont supprimées.
    """

    # --- Téléchargement brut via yfinance ---
    raw = yf.download(tickers, start=start, end=end, progress=False)

    # Si rien n'est revenu, on retourne un DataFrame vide
    if raw.empty:
        return pd.DataFrame()

    # --- Extraction de la colonne "Close" ---
    # yfinance retourne un MultiIndex quand on demande plusieurs tickers
    # (les colonnes sont organisées en deux niveaux : type de prix × ticker)
    # Quand on demande un seul ticker, c'est un index simple.

    if isinstance(raw.columns, pd.MultiIndex):
        # Cas multi-tickers : on récupère le niveau "Close"
        if "Close" in raw.columns.get_level_values(0):
            data = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            data = raw["Adj Close"].copy()
        else:
            return pd.DataFrame()
    else:
        # Cas mono-ticker : colonnes simples
        if "Close" in raw.columns:
            data = raw[["Close"]].copy()
        elif "Adj Close" in raw.columns:
            data = raw[["Adj Close"]].copy()
        else:
            return pd.DataFrame()

        # On renomme la colonne pour qu'elle porte le nom du ticker
        data.columns = [tickers[0]]

    # Supprime les colonnes entièrement vides (ticker invalide par ex.)
    data = data.dropna(axis=1, how="all")

    return data


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements journaliers simples à partir des prix.

    Le rendement simple, c'est :  r_t = (P_t / P_{t-1}) - 1

    C'est la variation en pourcentage du prix d'un jour à l'autre.
    Par exemple, si AAPL passe de 100$ à 103$, le rendement est +3%.

    Paramètres
    ----------
    prices : pd.DataFrame
        DataFrame de prix (sortie de load_prices)

    Retourne
    --------
    pd.DataFrame
        DataFrame des rendements journaliers (une ligne de moins que prices
        car le premier jour n'a pas de rendement précédent)
    """

    returns = prices.pct_change().dropna(how="all")
    return returns

import yfinance as yf  # déjà importé en haut, pas besoin de re-importer


def load_option_chain(ticker: str, expiry: str = None) -> dict:
    """
    Télécharge la chaîne d'options depuis Yahoo Finance.

    Une chaîne d'options = la liste de tous les calls et puts
    disponibles pour un ticker, à une date d'expiration donnée.

    Chaque ligne contient :
    - strike : prix d'exercice
    - lastPrice : dernier prix échangé
    - bid / ask : meilleure offre d'achat / de vente
    - volume : nombre de contrats échangés dans la journée
    - openInterest : nombre total de contrats ouverts
    - impliedVolatility : vol implicite estimée par Yahoo

    Paramètres
    ----------
    ticker : str
        Code Yahoo Finance (ex: "AAPL")
    expiry : str or None
        Date d'expiration au format "YYYY-MM-DD".
        Si None, on prend la première expiration disponible.

    Retourne
    --------
    dict avec :
        - "calls" : pd.DataFrame de la chaîne de calls
        - "puts"  : pd.DataFrame de la chaîne de puts
        - "expiry": str, la date d'expiration utilisée
        - "spot"  : float, le prix spot actuel
        - "available_expiries" : list[str], toutes les dates dispo
    """

    stock = yf.Ticker(ticker)

    # Récupérer le prix spot actuel
    spot = stock.history(period="1d")["Close"].iloc[-1]

    # Liste de toutes les expirations disponibles
    available_expiries = list(stock.options)

    if len(available_expiries) == 0:
        return {
            "calls": pd.DataFrame(),
            "puts": pd.DataFrame(),
            "expiry": None,
            "spot": spot,
            "available_expiries": [],
        }

    # Choisir l'expiration
    if expiry is not None and expiry in available_expiries:
        selected_expiry = expiry
    else:
        selected_expiry = available_expiries[0]

    # Télécharger la chaîne
    chain = stock.option_chain(selected_expiry)

    return {
        "calls": chain.calls,
        "puts": chain.puts,
        "expiry": selected_expiry,
        "spot": float(spot),
        "available_expiries": available_expiries,
    }