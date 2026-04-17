from utils.config import (
    PARADIGM_REGRESSION, PARADIGM_CLASSIFICATION,
    REGIME_BULL, REGIME_BEAR, REGIME_HIGHVOL, REGIME_LABELS,
)


class AdaptiveRouter:
    def __init__(self):
        self.lookup: dict[str, str] = {}
        self.audit_log: list[dict] = []

    def build_from_registry(self, registry: dict) -> "AdaptiveRouter":
        self.lookup = {}
        for regime in REGIME_LABELS:
            if regime not in registry:
                self.lookup[regime] = PARADIGM_CLASSIFICATION
                continue
            reg_sharpe = registry[regime].get(PARADIGM_REGRESSION, {}).get("sharpe", -999)
            cls_sharpe = registry[regime].get(PARADIGM_CLASSIFICATION, {}).get("sharpe", -999)
            winner = PARADIGM_REGRESSION if reg_sharpe >= cls_sharpe else PARADIGM_CLASSIFICATION
            self.lookup[regime] = winner
            print(
                f"[router] {regime}: {winner} wins "
                f"(reg={reg_sharpe:.3f}, cls={cls_sharpe:.3f})"
            )
        return self

    def route(self, regime: str, date=None) -> str:
        paradigm = self.lookup.get(regime, PARADIGM_CLASSIFICATION)
        self.audit_log.append({"date": str(date), "regime": regime, "paradigm": paradigm})
        return paradigm

    def get_model(self, registry: dict, regime: str) -> tuple:
        paradigm = self.route(regime)
        # Try current regime first
        if regime in registry and paradigm in registry[regime]:
            entry = registry[regime][paradigm]
            return entry["model"], paradigm, entry["feature_cols"]

        # Fallback: try the other paradigm for this regime
        other = PARADIGM_CLASSIFICATION if paradigm == PARADIGM_REGRESSION else PARADIGM_REGRESSION
        if regime in registry and other in registry[regime]:
            entry = registry[regime][other]
            return entry["model"], other, entry["feature_cols"]

        # Fallback: try any other regime that has a model (prefer Bull)
        for fallback_regime in [REGIME_BULL, REGIME_BEAR, REGIME_HIGHVOL]:
            if fallback_regime in registry:
                for p in [PARADIGM_REGRESSION, PARADIGM_CLASSIFICATION]:
                    if p in registry[fallback_regime] and registry[fallback_regime][p].get("model"):
                        entry = registry[fallback_regime][p]
                        print(f"[router] Fallback: using {fallback_regime}/{p} model for regime {regime}")
                        return entry["model"], p, entry["feature_cols"]

        return None, paradigm, None

    def summary(self) -> dict:
        return {
            "routing_table": self.lookup.copy(),
            "audit_entries": len(self.audit_log),
        }
