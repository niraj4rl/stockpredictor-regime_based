import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from utils.metrics import sharpe_ratio, max_drawdown, calmar_ratio, total_return, compute_all_metrics


def test_sharpe_positive_returns():
    returns = pd.Series([0.01] * 100)
    s = sharpe_ratio(returns)
    assert s > 0


def test_sharpe_zero_variance():
    returns = pd.Series([0.0] * 100)
    s = sharpe_ratio(returns)
    assert s == 0.0


def test_max_drawdown_is_negative():
    equity = pd.Series([1.0, 1.1, 0.9, 0.95, 1.05])
    mdd = max_drawdown(equity)
    assert mdd < 0


def test_max_drawdown_no_drawdown():
    equity = pd.Series([1.0, 1.1, 1.2, 1.3])
    mdd = max_drawdown(equity)
    assert mdd == 0.0


def test_total_return():
    returns = pd.Series([0.1, -0.05, 0.08])
    tr = total_return(returns)
    expected = (1.1 * 0.95 * 1.08) - 1
    assert abs(tr - expected) < 1e-10


def test_compute_all_metrics_keys():
    returns = pd.Series(np.random.randn(100) * 0.01)
    metrics = compute_all_metrics(returns)
    for key in ["sharpe", "max_drawdown", "total_return", "calmar", "n_trades"]:
        assert key in metrics


def test_calmar_zero_drawdown():
    returns = pd.Series([0.001] * 252)
    c = calmar_ratio(returns)
    assert c > 0
