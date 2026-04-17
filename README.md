# stocker — Regime-Aware Adaptive Stock Prediction

A machine learning system for NSE equity price prediction that dynamically selects its prediction strategy (regression vs classification) based on the current market regime.

## Setup

```bash
pip install -r requirements.txt
```

Create local environment config from template:

```bash
cp .env.example .env
```

Fill `.env` with your local credentials. Do not commit `.env`.

## Run the app

```bash
uvicorn ui.app:app --reload
```

## Run backtest only

```bash
python -m backtest.engine --ticker RELIANCE.NS
```

## Run tests

```bash
python -m pytest tests/
```

## Project structure

```
stocker/
├── data/           # Data ingestion and storage
├── features/       # Feature engineering
├── regime/         # Regime detection (heuristic + HMM)
├── models/         # Per-regime model training and registry
├── router/         # Adaptive paradigm router
├── backtest/       # Walk-forward backtester and metrics
├── ui/             # FastAPI + HTML/Tailwind/JS frontend
├── utils/          # Shared helpers
└── tests/          # Unit tests
```

## How it works

1. Fetches 5yr daily OHLCV for any NSE ticker via yfinance
2. Engineers a technical feature matrix (RSI, MACD, Bollinger, lags, rolling stats)
3. Detects current market regime: Bull / Bear / High-Volatility using HMM
4. Routes prediction to the specialist model validated for that regime
5. Outputs predicted price (regression) or direction + confidence (classification)
