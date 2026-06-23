from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"


def ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, encoding="utf-8-sig")


required = [
    ROOT / "summary_phase4F_zh.md",
    LOGS / "phase4F_preflight_check_log.md",
    LOGS / "phase4F_fold_failure_analysis_log.md",
    LOGS / "phase4F_threshold_robustness_log.md",
    LOGS / "phase4F_phase5_threshold_decision_log.md",
    TABLES / "phase4F_input_integrity_check.csv",
    TABLES / "phase4F_fold_failure_analysis.csv",
    TABLES / "phase4F_fold_class_distribution.csv",
    TABLES / "phase4F_fold_treatment_distribution.csv",
    TABLES / "phase4F_threshold_robustness_primary.csv",
    TABLES / "phase4F_threshold_robustness_repeated_folds.csv",
    TABLES / "phase4F_threshold_robustness_GSE72056.csv",
    TABLES / "phase4F_recommended_phase5_threshold.csv",
    FIGURES / "phase4F_threshold_tradeoff_primary.png",
    FIGURES / "phase4F_threshold_tradeoff_GSE72056.png",
]

missing = [str(path.relative_to(ROOT)) for path in required if not ok(path)]
preflight = read_csv("phase4F_input_integrity_check.csv")
folds = read_csv("phase4F_fold_failure_analysis.csv")
primary = read_csv("phase4F_threshold_robustness_primary.csv")
repeated = read_csv("phase4F_threshold_robustness_repeated_folds.csv")
gse = read_csv("phase4F_threshold_robustness_GSE72056.csv")
recommended = read_csv("phase4F_recommended_phase5_threshold.csv")
summary = (ROOT / "summary_phase4F_zh.md").read_text(encoding="utf-8-sig")
ready_line = next((line.strip() for line in summary.splitlines() if line.strip().startswith("READY_FOR_PHASE5")), "")

report = {
    "missing_or_empty_files": missing,
    "preflight_status_counts": preflight["status"].astype(str).value_counts().to_dict() if "status" in preflight.columns else {},
    "fold_count": int(folds["fold_id"].nunique()) if "fold_id" in folds.columns else 0,
    "fold_failures": folds[["fold_id", "failure_reason", "macro_f1", "balanced_accuracy", "melanocytic_like_recall", "adverse_like_recall"]].to_dict(orient="records"),
    "primary_thresholds": primary[["threshold_adverse_probability", "macro_f1", "balanced_accuracy", "melanocytic_like_recall", "adverse_like_recall"]].to_dict(orient="records"),
    "repeated_statuses": repeated["status"].astype(str).unique().tolist() if "status" in repeated.columns else [],
    "GSE72056_thresholds": gse[["threshold_adverse_probability", "macro_f1", "balanced_accuracy", "melanocytic_like_recall", "adverse_like_recall"]].to_dict(orient="records"),
    "recommended_rows": recommended.loc[recommended["recommended_for_phase5_pilot"].astype(bool)].to_dict(orient="records") if "recommended_for_phase5_pilot" in recommended.columns else [],
    "summary_ready_line": ready_line,
}

errors = []
if missing:
    errors.append("missing_or_empty_files")
if (preflight["status"].astype(str).isin(["failed", "missing_or_empty"])).any():
    errors.append("preflight_failed_status")
if report["fold_count"] != 5:
    errors.append("fold_count_not_5")
if len(primary) != 5 or len(gse) != 5 or len(repeated) != 5:
    errors.append("threshold_row_count_not_5")
if set(repeated["status"].astype(str)) != {"unable_to_recompute_fold_thresholds_without_probabilities"}:
    errors.append("repeated_threshold_status_not_marked_unable")
if len(report["recommended_rows"]) != 1:
    errors.append("recommended_threshold_not_unique")
if ready_line not in {"READY_FOR_PHASE5 = NO", "READY_FOR_PHASE5 = CONDITIONAL_PILOT", "READY_FOR_PHASE5 = YES_FORMAL"}:
    errors.append("invalid_ready_line")

print(json.dumps(report, ensure_ascii=False, indent=2))
if errors:
    raise SystemExit("PHASE4F_VERIFICATION_FAILED: " + "; ".join(errors))
print("PHASE4F_VERIFICATION: PASS")
