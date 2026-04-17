import numpy as np
import pandas as pd
from utils.config import TRADING_DAYS_YEAR


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS_YEAR
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS_YEAR) * excess.mean() / excess.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    ann_return = returns.mean() * TRADING_DAYS_YEAR
    mdd = abs(max_drawdown((1 + returns).cumprod()))
    if mdd == 0:
        if ann_return > 0:
            return float("inf")
        return 0.0
    return float(ann_return / mdd)


def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def hit_rate(predicted_returns: np.ndarray, actual_returns: np.ndarray) -> float:
    """Percentage of times the predicted direction was correct."""
    pred_dir = np.sign(predicted_returns)
    actual_dir = np.sign(actual_returns)
    if len(pred_dir) == 0:
        return 0.0
    return float(np.mean(pred_dir == actual_dir) * 100)


def profit_factor(strategy_returns: np.ndarray) -> float:
    """Gross gains / gross losses."""
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = abs(strategy_returns[strategy_returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression-specific metrics: MAE, RMSE, R², directional accuracy."""
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    dir_acc = hit_rate(y_pred, y_true)

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 4),
        "directional_accuracy": round(dir_acc, 2),
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification-specific metrics: accuracy, F1, precision, recall."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def compute_trading_metrics(strategy_returns: np.ndarray) -> dict:
    """Compute trading-specific metrics from strategy returns."""
    rets = pd.Series(strategy_returns)
    equity = (1 + rets).cumprod()
    n_trades = int((rets != 0).sum())
    win_trades = int((rets > 0).sum())
    loss_trades = int((rets < 0).sum())

    return {
        "sharpe": round(sharpe_ratio(rets), 4),
        "max_drawdown": round(max_drawdown(equity), 4),
        "total_return": round(total_return(rets), 4),
        "calmar": round(calmar_ratio(rets), 4),
        "hit_rate": round(win_trades / n_trades * 100, 2) if n_trades > 0 else 0.0,
        "profit_factor": round(profit_factor(strategy_returns), 4),
        "n_trades": n_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
    }


def compute_all_metrics(returns: pd.Series) -> dict:
    equity = (1 + returns).cumprod()
    return {
        "sharpe": round(sharpe_ratio(returns), 4),
        "max_drawdown": round(max_drawdown(equity), 4),
        "total_return": round(total_return(returns), 4),
        "calmar": round(calmar_ratio(returns), 4),
        "n_trades": int((returns != 0).sum()),
    }

