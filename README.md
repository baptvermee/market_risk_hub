# 📊 Market Risk Hub

A professional-grade market risk and derivatives pricing platform built with Python and Streamlit.

**[🚀 Launch the App](https://marketriskapp-wkmwdjrjkfw3uazg4cvft7.streamlit.app/)**

---

## What is this?

Market Risk Hub is an interactive dashboard that combines market data analysis, portfolio risk management, and derivatives pricing into a single platform. It connects to real market data via Yahoo Finance and implements quantitative finance models from scratch.

---

## Features

### 📊 Market Overview
Real-time portfolio monitoring with price tracking, cumulative returns, correlation matrix, rolling beta and volatility, and drawdown analysis. Supports any ticker available on Yahoo Finance.

### 📈 Market Data
Market data exploration and visualization.

### ⚠️ Risk Analytics
Portfolio risk measurement including Monte Carlo VaR (multivariate with Cholesky decomposition for asset correlations), Expected Shortfall (CVaR), and simulated portfolio trajectories. Vectorized Monte Carlo engine supporting 200k+ simulations.

### 🎯 Vanilla Option Pricer
Black-Scholes pricing with full Greeks (Delta, Gamma, Theta, Vega, Rho), price and Greeks profiles as a function of spot, 3D price surface (Spot × Maturity), PnL at expiry, implied volatility via Newton-Raphson, and real market volatility smile from Yahoo Finance option chains.

### 🎲 Exotic Option Pricer
Monte Carlo pricing for Asian options (arithmetic average), with comparison against vanilla pricing, convergence analysis, payoff distributions, and price profiles by strike demonstrating why Asian options are always cheaper than vanilla.

### 🏛️ Bond Pricer
Fixed income pricing for fixed-rate, zero-coupon, and amortizing bonds. Includes Macaulay and modified duration, convexity, DV01, price-yield curve with duration tangent, rate sensitivity analysis comparing duration approximation vs duration + convexity, yield to maturity solver, and side-by-side comparison of all three bond types.

---

## Tech Stack

- **Python** — NumPy, SciPy, Pandas
- **Streamlit** — interactive web interface
- **Plotly** — professional dark-theme visualizations
- **Yahoo Finance** — real-time market data

---

## Project Structure

```
market_risk_hub/
├── app.py                  # Main entry point
├── pages/
│   ├── 1_Overview.py       # Market overview dashboard
│   ├── 2_Market_Data.py    # Market data explorer
│   ├── 3_Risk_Analystics.py # VaR, ES, Monte Carlo
│   ├── 4_Vanilla_Option_Pricer.py # Black-Scholes + Greeks
│   ├── 5_Exotic_Option_Pricer.py  # Asian options (MC)
│   └── 6_Bond_Pricer.py    # Fixed income pricing
├── src/
│   ├── data_loader.py      # Yahoo Finance data pipeline
│   ├── risk_engine.py      # Monte Carlo VaR engine
│   ├── vanilla_option_pricer.py # BS pricing + implied vol
│   ├── exotic_option_pricer.py  # GBM simulation + Asian
│   └── bond_pricer.py      # Bond pricing + duration + convexity
└── requirements.txt
```

---

## Run Locally

```bash
git clone https://github.com/baptvermee/market_risk_hub.git
cd market_risk_hub
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Baptiste Vermee** — [GitHub](https://github.com/baptvermee)