import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from router.adaptive import AdaptiveRouter
from utils.config import (
    REGIME_BULL, REGIME_BEAR, REGIME_HIGHVOL,
    PARADIGM_REGRESSION, PARADIGM_CLASSIFICATION,
)


def make_mock_registry():
    return {
        REGIME_BULL: {
            PARADIGM_REGRESSION: {"model": object(), "sharpe": 0.8, "feature_cols": ["f1"]},
            PARADIGM_CLASSIFICATION: {"model": object(), "sharpe": 0.5, "feature_cols": ["f1"]},
        },
        REGIME_BEAR: {
            PARADIGM_REGRESSION: {"model": object(), "sharpe": 0.2, "feature_cols": ["f1"]},
            PARADIGM_CLASSIFICATION: {"model": object(), "sharpe": 0.7, "feature_cols": ["f1"]},
        },
        REGIME_HIGHVOL: {
            PARADIGM_REGRESSION: {"model": object(), "sharpe": 0.1, "feature_cols": ["f1"]},
            PARADIGM_CLASSIFICATION: {"model": object(), "sharpe": 0.9, "feature_cols": ["f1"]},
        },
    }


def test_router_selects_correct_paradigm():
    registry = make_mock_registry()
    router = AdaptiveRouter()
    router.build_from_registry(registry)
    assert router.lookup[REGIME_BULL] == PARADIGM_REGRESSION
    assert router.lookup[REGIME_BEAR] == PARADIGM_CLASSIFICATION
    assert router.lookup[REGIME_HIGHVOL] == PARADIGM_CLASSIFICATION


def test_router_logs_decisions():
    registry = make_mock_registry()
    router = AdaptiveRouter()
    router.build_from_registry(registry)
    router.route(REGIME_BULL, date="2024-01-01")
    router.route(REGIME_BEAR, date="2024-01-02")
    assert len(router.audit_log) == 2
    assert router.audit_log[0]["regime"] == REGIME_BULL
    assert router.audit_log[1]["paradigm"] == PARADIGM_CLASSIFICATION


def test_get_model_returns_correct_entry():
    registry = make_mock_registry()
    router = AdaptiveRouter()
    router.build_from_registry(registry)
    model, paradigm, cols = router.get_model(registry, REGIME_BULL)
    assert model is not None
    assert paradigm == PARADIGM_REGRESSION
    assert cols == ["f1"]


def test_router_handles_missing_regime():
    router = AdaptiveRouter()
    router.build_from_registry({})
    paradigm = router.route("UnknownRegime")
    assert paradigm in [PARADIGM_REGRESSION, PARADIGM_CLASSIFICATION]


def test_router_summary():
    registry = make_mock_registry()
    router = AdaptiveRouter()
    router.build_from_registry(registry)
    router.route(REGIME_BULL)
    summary = router.summary()
    assert "routing_table" in summary
    assert summary["audit_entries"] == 1
