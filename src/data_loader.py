import yfinance as yf
import pandas as pd


def load_prices(tickers: list[str], start, end) -> pd.DataFrame:

    raw = yf.download(tickers, start=start, end=end, progress=False)

    if raw.empty:
        return pd.DataFrame()

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

        data.columns = [tickers[0]]

    data = data.dropna(axis=1, how="all")

    return data


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="all")
    return returns 