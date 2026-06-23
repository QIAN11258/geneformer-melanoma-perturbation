from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, encoding="utf-8-sig")


def file_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


required_files = [
    ROOT / "summary_phase4D_zh.md",
    LOGS / "phase4D_preflight_check_log.md",
    LOGS / "phase4D_label_construction_log.md",
    LOGS / "phase4D_grouped_split_log.md",
    LOGS / "phase4D_binary_A_training_log.md",
    LOGS / "phase4D_three_class_training_log.md",
    LOGS / "phase4D_baseline_comparison_log.md",
    LOGS / "phase4D_GSE72056_sensitivity_log.md",
    TABLES / "phase4D_input_integrity_check.csv",
    TABLES / "phase4D_binary_A_label_distribution.csv",
    TABLES / "phase4D_three_class_label_distribution.csv",
    TABLES / "phase4D_label_by_sample_id_distribution.csv",
    TABLES / "phase4D_label_by_treatment_group_distribution.csv",
    TABLES / "phase4D_binary_A_grouped_split_plan.csv",
    TABLES / "phase4D_three_class_grouped_split_plan.csv",
    TABLES / "phase4D_binary_A_metadata_with_split.csv",
    TABLES / "phase4D_three_class_metadata_with_split.csv",
    TABLES / "phase4D_binary_A_repeated_grouped_metrics.csv",
    TABLES / "phase4D_three_class_repeated_grouped_metrics.csv",
    TABLES / "phase4D_binary_A_primary_test_metrics.csv",
    TABLES / "phase4D_binary_A_per_class_metrics.csv",
    TABLES / "phase4D_binary_A_gpu_memory_log.csv",
    TABLES / "phase4D_three_class_primary_test_metrics.csv",
    TABLES / "phase4D_three_class_per_class_metrics.csv",
    TABLES / "phase4D_binary_A_baseline_metrics.csv",
    TABLES / "phase4D_three_class_baseline_metrics.csv",
    TABLES / "phase4D_GSE72056_binary_A_sensitivity_metrics.csv",
    TABLES / "phase4D_GSE72056_three_class_sensitivity_metrics.csv",
    TABLES / "phase4D_GSE72056_prediction_distribution.csv",
    FIGURES / "phase4D_binary_A_confusion_matrix.png",
    FIGURES / "phase4D_three_class_confusion_matrix.png",
    FIGURES / "phase4D_geneformer_vs_baseline_binary_A.png",
    FIGURES / "phase4D_geneformer_vs_baseline_three_class.png",
    FIGURES / "phase4D_GSE72056_binary_A_confusion_matrix.png",
    FIGURES / "phase4D_GSE72056_three_class_confusion_matrix.png",
    MODELS / "phase4D_geneformer_v2_binary_A" / "best_model.pt",
    MODELS / "phase4D_geneformer_v2_three_class" / "best_model.pt",
]

missing = [str(path.relative_to(ROOT)) for path in required_files if not file_ok(path)]

binary_metrics = read_csv("phase4D_binary_A_primary_test_metrics.csv")
binary_per_class = read_csv("phase4D_binary_A_per_class_metrics.csv")
three_metrics = read_csv("phase4D_three_class_primary_test_metrics.csv")
three_per_class = read_csv("phase4D_three_class_per_class_metrics.csv")
binary_meta = read_csv("phase4D_binary_A_metadata_with_split.csv")
three_meta = read_csv("phase4D_three_class_metadata_with_split.csv")
binary_repeated = read_csv("phase4D_binary_A_repeated_grouped_metrics.csv")
three_repeated = read_csv("phase4D_three_class_repeated_grouped_metrics.csv")
binary_sens = read_csv("phase4D_GSE72056_binary_A_sensitivity_metrics.csv")
three_sens = read_csv("phase4D_GSE72056_three_class_sensitivity_metrics.csv")


def split_leakage(df: pd.DataFrame) -> dict[str, list[str]]:
    candidates = [col for col in ["sample_id", "split_unit", "tumor_id"] if col in df.columns]
    result: dict[str, list[str]] = {}
    for col in candidates:
        checked = df.loc[df["phase4D_supervised_use"].astype(bool)].copy()
        checked = checked.loc[~checked[col].astype(str).isin(["", "nan", "None", "not_available_in_source"])]
        if checked.empty:
            result[col] = []
            continue
        bad = checked.groupby(col)["split"].nunique()
        result[col] = bad[bad > 1].index.astype(str).tolist()
    return result


def repeated_statuses(df: pd.DataFrame) -> list[str]:
    status_col = "retraining_status" if "retraining_status" in df.columns else "status"
    if status_col not in df.columns:
        return []
    return sorted(df[status_col].dropna().astype(str).unique().tolist())


summary_text = (ROOT / "summary_phase4D_zh.md").read_text(encoding="utf-8-sig")
ready_line = next((line.strip() for line in summary_text.splitlines() if line.strip().startswith("READY_FOR_PHASE5")), "")

report = {
    "missing_or_empty_files": missing,
    "binary_A_primary": binary_metrics.to_dict(orient="records"),
    "binary_A_per_class": binary_per_class.to_dict(orient="records"),
    "three_class_primary": three_metrics.to_dict(orient="records"),
    "three_class_per_class": three_per_class.to_dict(orient="records"),
    "binary_A_repeated_statuses": repeated_statuses(binary_repeated),
    "three_class_repeated_statuses": repeated_statuses(three_repeated),
    "binary_A_split_leakage": split_leakage(binary_meta),
    "three_class_split_leakage": split_leakage(three_meta),
    "GSE72056_binary_A_sensitivity": binary_sens.to_dict(orient="records"),
    "GSE72056_three_class_sensitivity": three_sens.to_dict(orient="records"),
    "summary_ready_line": ready_line,
}

errors = []
if missing:
    errors.append("missing_or_empty_files")
for key in ["binary_A_split_leakage", "three_class_split_leakage"]:
    for field, bad_units in report[key].items():
        if bad_units:
            errors.append(f"{key}:{field}")
if "status" in binary_metrics.columns and not (binary_metrics["status"].astype(str) == "success").all():
    errors.append("binary_A_primary_not_success")
if "status" in three_metrics.columns and not (three_metrics["status"].astype(str) == "success").all():
    errors.append("three_class_primary_not_success")
for key in ["binary_A_repeated_statuses", "three_class_repeated_statuses"]:
    statuses = report[key]
    if statuses != ["not_run_resource_deferred"]:
        errors.append(f"{key}_unexpected")
if not ready_line:
    errors.append("missing_READY_FOR_PHASE5")

print(json.dumps(report, ensure_ascii=False, indent=2))
if errors:
    raise SystemExit("PHASE4D_VERIFICATION_FAILED: " + "; ".join(errors))
print("PHASE4D_VERIFICATION: PASS")
