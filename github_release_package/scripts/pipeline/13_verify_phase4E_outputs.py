from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"


def ok(path: Path) -> bool:
    return path.exists() and (path.is_dir() or path.stat().st_size > 0)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, encoding="utf-8-sig")


required = [
    ROOT / "summary_phase4E_zh.md",
    LOGS / "phase4E_preflight_check_log.md",
    LOGS / "phase4E_error_analysis_log.md",
    LOGS / "phase4E_threshold_calibration_log.md",
    LOGS / "phase4E_class_weighted_training_log.md",
    LOGS / "phase4E_focal_loss_training_log.md",
    LOGS / "phase4E_repeated_grouped_retraining_log.md",
    LOGS / "phase4E_GSE72056_sensitivity_log.md",
    TABLES / "phase4E_input_integrity_check.csv",
    TABLES / "phase4E_binary_A_error_analysis.csv",
    TABLES / "phase4E_binary_A_error_by_sample_id.csv",
    TABLES / "phase4E_binary_A_error_by_treatment_group.csv",
    TABLES / "phase4E_threshold_calibration_metrics.csv",
    TABLES / "phase4E_class_weighted_primary_test_metrics_threshold_050.csv",
    TABLES / "phase4E_class_weighted_primary_test_metrics_calibrated.csv",
    TABLES / "phase4E_class_weighted_per_class_metrics.csv",
    TABLES / "phase4E_gpu_memory_log.csv",
    TABLES / "phase4E_focal_loss_primary_test_metrics_threshold_050.csv",
    TABLES / "phase4E_focal_loss_primary_test_metrics_calibrated.csv",
    TABLES / "phase4E_repeated_grouped_retraining_metrics.csv",
    TABLES / "phase4E_repeated_grouped_fold_details.csv",
    TABLES / "phase4E_GSE72056_binary_A_sensitivity_threshold_050.csv",
    TABLES / "phase4E_GSE72056_binary_A_sensitivity_calibrated.csv",
    FIGURES / "phase4E_binary_A_probability_distribution.png",
    FIGURES / "phase4E_threshold_tradeoff_curve.png",
    FIGURES / "phase4E_class_weighted_confusion_matrix_calibrated.png",
    FIGURES / "phase4E_GSE72056_binary_A_confusion_matrix_calibrated.png",
    MODELS / "phase4E_geneformer_v2_binary_A_class_weighted" / "best_model.pt",
]

missing = [str(path.relative_to(ROOT)) for path in required if not ok(path)]

cw_050 = read_csv("phase4E_class_weighted_primary_test_metrics_threshold_050.csv")
cw_cal = read_csv("phase4E_class_weighted_primary_test_metrics_calibrated.csv")
repeated = read_csv("phase4E_repeated_grouped_retraining_metrics.csv")
folds = read_csv("phase4E_repeated_grouped_fold_details.csv")
sens_050 = read_csv("phase4E_GSE72056_binary_A_sensitivity_threshold_050.csv")
sens_cal = read_csv("phase4E_GSE72056_binary_A_sensitivity_calibrated.csv")
error_cells = read_csv("phase4E_binary_A_error_analysis.csv")
thresholds = read_csv("phase4E_threshold_calibration_metrics.csv")

summary_text = (ROOT / "summary_phase4E_zh.md").read_text(encoding="utf-8-sig")
ready_line = next((line.strip() for line in summary_text.splitlines() if line.strip().startswith("READY_FOR_PHASE5")), "")


def leakage(df: pd.DataFrame) -> list[str]:
    if "split_unit" not in df.columns or "split" not in df.columns:
        return ["missing_columns"]
    grouped = df.groupby("split_unit")["split"].nunique()
    return grouped[grouped > 1].index.astype(str).tolist()


supervised_errors = error_cells.loc[error_cells["error_type"].astype(str) != "correct"] if "error_type" in error_cells.columns else pd.DataFrame()

report = {
    "missing_or_empty_files": missing,
    "class_weighted_threshold_050": cw_050.to_dict(orient="records"),
    "class_weighted_calibrated": cw_cal.to_dict(orient="records"),
    "repeated_status_counts": repeated["retraining_status"].astype(str).value_counts().to_dict() if "retraining_status" in repeated.columns else {},
    "repeated_fold_count": int(repeated["fold"].nunique()) if "fold" in repeated.columns else 0,
    "repeated_metrics": repeated[
        [col for col in ["fold", "macro_f1", "balanced_accuracy", "adverse_like_recall", "melanocytic_like_recall", "retraining_status"] if col in repeated.columns]
    ].to_dict(orient="records"),
    "fold_details_count": int(len(folds)),
    "GSE72056_threshold_050": sens_050.to_dict(orient="records"),
    "GSE72056_calibrated": sens_cal.to_dict(orient="records"),
    "error_cell_count": int(len(supervised_errors)),
    "selected_threshold_rows": thresholds.loc[thresholds.get("selected_threshold", False).astype(bool)].to_dict(orient="records") if "selected_threshold" in thresholds.columns else [],
    "summary_ready_line": ready_line,
}

errors = []
if missing:
    errors.append("missing_or_empty_files")
if cw_cal.empty or str(cw_cal.iloc[0].get("status", "")) != "success":
    errors.append("class_weighted_calibrated_not_success")
if "retraining_status" not in repeated.columns or set(repeated["retraining_status"].astype(str)) != {"success"}:
    errors.append("repeated_not_all_success")
if report["repeated_fold_count"] != 5:
    errors.append("repeated_fold_count_not_5")
if "READY_FOR_PHASE5" not in ready_line:
    errors.append("missing_READY_FOR_PHASE5")
if sens_cal.empty or str(sens_cal.iloc[0].get("status", "")) != "success":
    errors.append("GSE72056_calibrated_not_success")
if not report["selected_threshold_rows"]:
    errors.append("missing_selected_threshold")

print(json.dumps(report, ensure_ascii=False, indent=2))
if errors:
    raise SystemExit("PHASE4E_VERIFICATION_FAILED: " + "; ".join(errors))
print("PHASE4E_VERIFICATION: PASS")
