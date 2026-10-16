"""Scientific thresholds are separate from successful process completion."""
import math


def evaluate_mechanism_gate(metrics, thresholds):
    specifications = {
        "min_validation_gap": ("matched_minus_shuffled_logp", "min"),
        "min_fraction_gap_gt_0": ("fraction_gap_gt_0", "min"),
        "min_nll_improvement": ("nll_improvement_over_no_memory", "min"),
        "max_relative_update_norm": ("mean_relative_update_norm", "max"),
    }
    checks = {}
    for name, limit in thresholds.items():
        if name not in specifications:
            checks[name] = {"passed": None, "reason": "unsupported_threshold"}
            continue
        metric, direction = specifications[name]
        value = metrics.get(metric)
        if value is None or not math.isfinite(float(value)):
            checks[name] = {"passed": None, "reason": "missing_or_nonfinite_metric", "metric": metric}
        else:
            checks[name] = {"passed": value >= limit if direction == "min" else value <= limit,
                            "value": value, "threshold": limit, "metric": metric}
    decisions = [c["passed"] for c in checks.values()]
    status = "failed" if False in decisions else "not_evaluated" if not decisions or None in decisions else "passed"
    return {"status": status, "checks": checks, "scope": "mechanism_thresholds_not_humor_quality"}
