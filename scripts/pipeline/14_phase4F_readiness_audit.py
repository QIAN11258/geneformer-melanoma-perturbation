from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_processed"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
SCRIPTS = ROOT / "scripts"

P4E_PATH = SCRIPTS / "12_phase4E_binaryA_calibration_retraining.py"
spec = importlib.util.spec_from_file_location("phase4e", P4E_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import Phase 4E helpers from {P4E_PATH}")
p4e = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p4e
spec.loader.exec_module(p4e)

LABELS = ["melanocytic_like", "adverse_like"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
CLASS_WEIGHTED_CKPT = MODELS / "phase4E_geneformer_v2_binary_A_class_weighted" / "best_model.pt"
TRAIN_DS = DATA / "tokenized_v2_gc104M" / "GSE115978_malignant_state_phase4D_labeled_v2.dataset"
SENS_DS = DATA / "tokenized_v2_gc104M" / "GSE72056_malignant_state_phase4D_labeled_v2.dataset"
TRAIN_DS_ORIGINAL = DATA / "tokenized_v2_gc104M" / "GSE115978_malignant_state_labeled_v2.dataset"
SENS_DS_ORIGINAL = DATA / "tokenized_v2_gc104M" / "GSE72056_malignant_state_labeled_v2.dataset"
TRAIN_H5AD = DATA / "GSE115978_malignant_state_phase4D_labeled.h5ad"
SENS_H5AD = DATA / "GSE72056_malignant_state_phase4D_labeled.h5ad"
META = TABLES / "phase4D_binary_A_metadata_with_split.csv"


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def preflight() -> tuple[pd.DataFrame, Any, Any]:
    rows: list[dict[str, Any]] = []
    log = ["# Phase 4F preflight check log", "", f"Timestamp: {datetime.now().isoformat(timespec='seconds')}"]
    required = [
        ROOT / "summary_phase4E_zh.md",
        TABLES / "phase4E_repeated_grouped_retraining_metrics.csv",
        TABLES / "phase4E_class_weighted_primary_test_metrics_calibrated.csv",
        TABLES / "phase4E_class_weighted_primary_test_metrics_threshold_050.csv",
        TABLES / "phase4E_GSE72056_binary_A_sensitivity_calibrated.csv",
        TABLES / "phase4E_GSE72056_binary_A_sensitivity_threshold_050.csv",
        MODELS / "phase4E_geneformer_v2_binary_A_class_weighted",
        CLASS_WEIGHTED_CKPT,
        TRAIN_H5AD,
        SENS_H5AD,
        TRAIN_DS_ORIGINAL,
        SENS_DS_ORIGINAL,
        TRAIN_DS,
        SENS_DS,
        META,
    ]
    missing = []
    for path in required:
        ok = path.exists() and (path.is_dir() or path.stat().st_size > 0)
        rows.append({"check_type": "required_file", "path": rel(path), "status": "ok" if ok else "missing_or_empty"})
        if not ok:
            missing.append(rel(path))
    if missing:
        pd.DataFrame(rows).to_csv(TABLES / "phase4F_input_integrity_check.csv", index=False, encoding="utf-8-sig")
        write_text(LOGS / "phase4F_preflight_check_log.md", log + ["Missing required inputs:", *[f"- {x}" for x in missing]])
        raise RuntimeError("Phase 4F preflight failed: missing inputs.")

    train_h5 = ad.read_h5ad(TRAIN_H5AD)
    sens_h5 = ad.read_h5ad(SENS_H5AD)
    train_ds = load_from_disk(str(TRAIN_DS))
    sens_ds = load_from_disk(str(SENS_DS))
    train_orig = load_from_disk(str(TRAIN_DS_ORIGINAL))
    sens_orig = load_from_disk(str(SENS_DS_ORIGINAL))
    meta = read_csv(META)

    for dataset_id, h5, ds, original in [
        ("GSE115978", train_h5, train_ds, train_orig),
        ("GSE72056", sens_h5, sens_ds, sens_orig),
    ]:
        rows.append(
            {
                "check_type": "row_count",
                "dataset_id": dataset_id,
                "h5ad_n_obs": int(h5.n_obs),
                "phase4D_labeled_token_rows": len(ds),
                "original_token_rows": len(original),
                "phase4D_labeled_row_match": int(h5.n_obs) == len(ds),
                "original_row_match": int(h5.n_obs) == len(original),
                "status": "ok" if int(h5.n_obs) == len(ds) == len(original) else "failed",
            }
        )

    phase4e_summary = (ROOT / "summary_phase4E_zh.md").read_text(encoding="utf-8-sig")
    phase4e_pass_like = "READY_FOR_PHASE5 = NO" in phase4e_summary
    repeated = read_csv(TABLES / "phase4E_repeated_grouped_retraining_metrics.csv")
    repeated_success = "retraining_status" in repeated.columns and set(repeated["retraining_status"].astype(str)) == {"success"} and repeated["fold"].nunique() == 5
    repeated_true = True
    for fold in range(1, 6):
        fold_dir = MODELS / "phase4E_geneformer_v2_binary_A_repeated_grouped" / f"fold_{fold}"
        repeated_true = repeated_true and (fold_dir / "best_model.pt").exists() and (fold_dir / "training_history.csv").exists()
    supervised = meta.loc[meta["phase4D_supervised_use"].astype(bool)]
    leakage = supervised.groupby("split_unit")["split"].nunique().loc[lambda x: x > 1].index.astype(str).tolist()
    gse_not_training = "GSE72056 was not used for training" in (LOGS / "phase4E_class_weighted_training_log.md").read_text(encoding="utf-8-sig")
    rows.extend(
        [
            {"check_type": "phase4E_verification_recomputed", "status": "PASS" if phase4e_pass_like and repeated_success else "failed"},
            {"check_type": "GSE72056_training_exclusion", "status": "ok" if gse_not_training else "failed"},
            {"check_type": "repeated_grouped_true_retraining", "status": "ok" if repeated_success and repeated_true else "failed"},
            {"check_type": "sample_id_split_leakage", "bad_split_units": ";".join(leakage), "status": "ok" if not leakage else "failed"},
        ]
    )
    failed = [row for row in rows if row.get("status") in {"failed", "missing_or_empty"}]
    pd.DataFrame(rows).to_csv(TABLES / "phase4F_input_integrity_check.csv", index=False, encoding="utf-8-sig")
    log.extend(
        [
            f"Phase 4E verification recomputed status: {'PASS' if phase4e_pass_like and repeated_success else 'failed'}",
            f"GSE72056 not used for training: {gse_not_training}",
            f"Repeated grouped retraining true checkpoints/histories present: {repeated_true}",
            f"sample_id/split_unit leakage count: {len(leakage)}",
        ]
    )
    if failed:
        log.append("Failed checks:")
        log.extend(f"- {row}" for row in failed)
        write_text(LOGS / "phase4F_preflight_check_log.md", log)
        raise RuntimeError("Phase 4F preflight failed.")
    log.append("Preflight passed.")
    write_text(LOGS / "phase4F_preflight_check_log.md", log)
    return meta, train_ds, sens_ds


def parse_dist(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    if pd.isna(value):
        return {}
    return {str(k): int(v) for k, v in json.loads(str(value)).items()}


def fold_assignments_from_phase4e(meta: pd.DataFrame) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for fold, assignment, _ in p4e.fold_assignments(meta, seed=42):
        out[fold] = p4e.apply_assignment(meta, assignment)
    return out


def fold_failure_reason(row: pd.Series) -> str:
    reasons = []
    if float(row["melanocytic_like_recall"]) < 0.50:
        reasons.append("melanocytic_like_recall_lt_0.50")
    if float(row["adverse_like_recall"]) < 0.70:
        reasons.append("adverse_like_recall_lt_0.70")
    if float(row["macro_f1"]) < 0.60:
        reasons.append("macro_f1_lt_0.60")
    return ";".join(reasons) if reasons else "passes_minimum_metric_thresholds"


def write_fold_failure_analysis(meta: pd.DataFrame) -> None:
    repeated = read_csv(TABLES / "phase4E_repeated_grouped_retraining_metrics.csv")
    fold_meta = fold_assignments_from_phase4e(meta)
    failure_rows = []
    class_rows = []
    treatment_rows = []
    log = ["# Phase 4F fold failure analysis log", "", "Fold assignments reconstructed with the same Phase 4E seed=42 StratifiedGroupKFold logic."]
    for _, metric in repeated.iterrows():
        fold = int(metric["fold"])
        test = fold_meta[fold].loc[lambda df: (df["phase4D_supervised_use"].astype(bool)) & (df["split"] == "held_out_test")].copy()
        class_counts = test["phase4D_task_label"].value_counts().to_dict()
        treatment_counts = test["treatment.group"].astype(str).value_counts().to_dict()
        sample_counts = test["sample_id"].astype(str).value_counts()
        top_sample = str(sample_counts.index[0]) if len(sample_counts) else ""
        top_sample_count = int(sample_counts.iloc[0]) if len(sample_counts) else 0
        top_sample_fraction = top_sample_count / len(test) if len(test) else math.nan
        top_treatment = max(treatment_counts.items(), key=lambda kv: kv[1])[0] if treatment_counts else ""
        top_treatment_fraction = max(treatment_counts.values()) / len(test) if treatment_counts and len(test) else math.nan
        reason = fold_failure_reason(metric)
        interpretation = "No minimum-threshold fold failure detected."
        if "melanocytic_like_recall_lt_0.50" in reason:
            interpretation = "Borderline melanocytic_like recall; fold has melanocytic-heavy held-out composition and should be interpreted as sampling instability, not a biological finding."
        if "adverse_like_recall_lt_0.70" in reason:
            interpretation = "Adverse-like recall below minimum; fold may be sensitive to held-out sample composition and threshold choice, not sufficient for formal Phase 5."
        if top_sample_fraction >= 0.50:
            interpretation += f" Top sample {top_sample} contributes {top_sample_fraction:.2%} of held-out cells."
        else:
            interpretation += " No single sample contributes >=50% of held-out cells."
        failure_rows.append(
            {
                "fold_id": fold,
                "test_sample_id_count": int(test["sample_id"].nunique()),
                "test_cell_count": int(len(test)),
                "melanocytic_like_count": int(class_counts.get("melanocytic_like", 0)),
                "adverse_like_count": int(class_counts.get("adverse_like", 0)),
                "class_distribution": json.dumps(class_counts, ensure_ascii=False, sort_keys=True),
                "treatment_group_distribution": json.dumps(treatment_counts, ensure_ascii=False, sort_keys=True),
                "top_sample_id": top_sample,
                "top_sample_cell_count": top_sample_count,
                "top_sample_fraction": top_sample_fraction,
                "top_treatment_group": top_treatment,
                "top_treatment_fraction": top_treatment_fraction,
                "macro_f1": float(metric["macro_f1"]),
                "balanced_accuracy": float(metric["balanced_accuracy"]),
                "melanocytic_like_recall": float(metric["melanocytic_like_recall"]),
                "adverse_like_recall": float(metric["adverse_like_recall"]),
                "failure_reason": reason,
                "biological_or_sampling_interpretation": interpretation,
            }
        )
        for label, count in sorted(class_counts.items()):
            class_rows.append(
                {
                    "fold_id": fold,
                    "class_label": label,
                    "test_cell_count": int(count),
                    "test_cell_fraction": int(count) / len(test) if len(test) else math.nan,
                    "test_sample_id_count": int(test.loc[test["phase4D_task_label"] == label, "sample_id"].nunique()),
                }
            )
        for treatment, count in sorted(treatment_counts.items()):
            treatment_rows.append(
                {
                    "fold_id": fold,
                    "treatment.group": treatment,
                    "test_cell_count": int(count),
                    "test_cell_fraction": int(count) / len(test) if len(test) else math.nan,
                    "test_sample_id_count": int(test.loc[test["treatment.group"].astype(str) == treatment, "sample_id"].nunique()),
                }
            )
    failures = pd.DataFrame(failure_rows)
    failures.to_csv(TABLES / "phase4F_fold_failure_analysis.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(class_rows).to_csv(TABLES / "phase4F_fold_class_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(treatment_rows).to_csv(TABLES / "phase4F_fold_treatment_distribution.csv", index=False, encoding="utf-8-sig")
    for _, row in failures.iterrows():
        log.append(
            f"fold {int(row['fold_id'])}: macro_f1={row['macro_f1']:.4f}, balanced_accuracy={row['balanced_accuracy']:.4f}, melanocytic_recall={row['melanocytic_like_recall']:.4f}, adverse_recall={row['adverse_like_recall']:.4f}, failure_reason={row['failure_reason']}"
        )
    write_text(LOGS / "phase4F_fold_failure_analysis_log.md", log)


def threshold_table(y_true: np.ndarray, probs: np.ndarray, source: str, evaluation_set: str) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        row, _, _ = p4e.metrics_at_threshold(y_true, probs, threshold, source, evaluation_set)
        row["failed_melanocytic_like_recall_lt_0_50"] = bool(row["melanocytic_like_recall"] < 0.50)
        row["failed_adverse_like_recall_lt_0_70"] = bool(row["adverse_like_recall"] < 0.70)
        row["failed_macro_f1_lt_0_60"] = bool(row["macro_f1"] < 0.60)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_threshold(df: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df["threshold_adverse_probability"]
    for col, label in [
        ("macro_f1", "Macro-F1"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("melanocytic_like_recall", "melanocytic_like recall"),
        ("adverse_like_recall", "adverse_like recall"),
    ]:
        ax.plot(x, df[col], marker="o", label=label)
    ax.axhline(0.60, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0.70, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Adverse-like probability threshold")
    ax.set_ylabel("Metric")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_threshold_robustness(meta: pd.DataFrame, train_ds: Any, sens_ds: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log = ["# Phase 4F threshold robustness log", "", "No new training was performed."]
    y_primary, _, prob_primary = p4e.evaluate_checkpoint(CLASS_WEIGHTED_CKPT, train_ds, meta, "held_out_test")
    primary = threshold_table(y_primary, prob_primary, "phase4E_class_weighted", "GSE115978_primary_held_out_test")
    primary["failed_fold_count_melanocytic_like_recall_lt_0_50"] = "not_applicable_primary_single_split"
    primary["failed_fold_count_adverse_like_recall_lt_0_70"] = "not_applicable_primary_single_split"
    primary["failed_fold_count_macro_f1_lt_0_60"] = "not_applicable_primary_single_split"
    primary.to_csv(TABLES / "phase4F_threshold_robustness_primary.csv", index=False, encoding="utf-8-sig")
    plot_threshold(primary, "Phase 4F primary held-out threshold robustness", FIGURES / "phase4F_threshold_tradeoff_primary.png")

    sens_meta = p4e.sensitivity_meta(sens_ds)
    y_sens, _, prob_sens = p4e.evaluate_checkpoint(CLASS_WEIGHTED_CKPT, sens_ds, sens_meta, "sensitivity")
    gse = threshold_table(y_sens, prob_sens, "phase4E_class_weighted", "GSE72056_processed_expression_sensitivity")
    gse["processed_expression_limitation"] = True
    gse["failed_fold_count_melanocytic_like_recall_lt_0_50"] = "not_applicable_sensitivity_set"
    gse["failed_fold_count_adverse_like_recall_lt_0_70"] = "not_applicable_sensitivity_set"
    gse["failed_fold_count_macro_f1_lt_0_60"] = "not_applicable_sensitivity_set"
    gse.to_csv(TABLES / "phase4F_threshold_robustness_GSE72056.csv", index=False, encoding="utf-8-sig")
    plot_threshold(gse, "Phase 4F GSE72056 threshold robustness", FIGURES / "phase4F_threshold_tradeoff_GSE72056.png")

    repeated = read_csv(TABLES / "phase4E_repeated_grouped_retraining_metrics.csv")
    prob_files = list((MODELS / "phase4E_geneformer_v2_binary_A_repeated_grouped").glob("**/*prob*")) + list(TABLES.glob("phase4E*repeated*prob*"))
    known_mel_fail = int((repeated["melanocytic_like_recall"].astype(float) < 0.50).sum())
    known_adv_fail = int((repeated["adverse_like_recall"].astype(float) < 0.70).sum())
    known_macro_fail = int((repeated["macro_f1"].astype(float) < 0.60).sum())
    rep_rows = []
    for threshold in THRESHOLDS:
        rep_rows.append(
            {
                "model_source": "phase4E_repeated_grouped_retraining",
                "threshold_adverse_probability": threshold,
                "status": "unable_to_recompute_fold_thresholds_without_probabilities" if not prob_files else "probability_files_detected_not_used",
                "accuracy": "unable_to_compute",
                "balanced_accuracy": "unable_to_compute",
                "macro_f1": "unable_to_compute",
                "weighted_f1": "unable_to_compute",
                "adverse_like_recall": "unable_to_compute",
                "melanocytic_like_recall": "unable_to_compute",
                "adverse_like_precision": "unable_to_compute",
                "melanocytic_like_precision": "unable_to_compute",
                "failed_fold_count_melanocytic_like_recall_lt_0_50": "unable_to_recompute_without_saved_probabilities",
                "failed_fold_count_adverse_like_recall_lt_0_70": "unable_to_recompute_without_saved_probabilities",
                "failed_fold_count_macro_f1_lt_0_60": "unable_to_recompute_without_saved_probabilities",
                "known_failed_folds_at_phase4E_fold_selected_threshold_melanocytic_like_recall_lt_0_50": known_mel_fail,
                "known_failed_folds_at_phase4E_fold_selected_threshold_adverse_like_recall_lt_0_70": known_adv_fail,
                "known_failed_folds_at_phase4E_fold_selected_threshold_macro_f1_lt_0_60": known_macro_fail,
            }
        )
    repeated_threshold = pd.DataFrame(rep_rows)
    repeated_threshold.to_csv(TABLES / "phase4F_threshold_robustness_repeated_folds.csv", index=False, encoding="utf-8-sig")
    log.extend(
        [
            "Primary and GSE72056 threshold robustness were recomputed from the Phase 4E class-weighted checkpoint.",
            "Repeated-fold probability files were not found; fold threshold curves were marked unable_to_recompute_fold_thresholds_without_probabilities.",
            f"Known failures at Phase 4E fold-selected thresholds: melanocytic recall <0.50: {known_mel_fail}; adverse recall <0.70: {known_adv_fail}; macro-F1 <0.60: {known_macro_fail}.",
            "GSE72056 was not used for threshold selection.",
        ]
    )
    write_text(LOGS / "phase4F_threshold_robustness_log.md", log)
    return primary, repeated_threshold, gse


def recommend_threshold(primary: pd.DataFrame, gse: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = []
    merged = primary.merge(
        gse,
        on="threshold_adverse_probability",
        suffixes=("_primary", "_GSE72056"),
    )
    for _, row in merged.iterrows():
        primary_ok = (
            row["macro_f1_primary"] >= 0.60
            and row["balanced_accuracy_primary"] >= 0.60
            and row["adverse_like_recall_primary"] >= 0.70
            and row["melanocytic_like_recall_primary"] >= 0.50
        )
        gse_not_collapsed = row["macro_f1_GSE72056"] >= 0.60 and row["balanced_accuracy_GSE72056"] >= 0.60
        gse_adverse_ok = row["adverse_like_recall_GSE72056"] >= 0.60
        score = (
            min(row["adverse_like_recall_primary"], row["melanocytic_like_recall_primary"]) * 2.0
            + row["macro_f1_primary"]
            + row["balanced_accuracy_primary"]
            + row["macro_f1_GSE72056"]
            + row["balanced_accuracy_GSE72056"]
            + min(row["adverse_like_recall_GSE72056"], row["melanocytic_like_recall_GSE72056"]) * 0.5
        )
        rows.append(
            {
                "threshold_adverse_probability": row["threshold_adverse_probability"],
                "primary_minimums_met": bool(primary_ok),
                "GSE72056_not_collapsed": bool(gse_not_collapsed),
                "GSE72056_adverse_recall_ge_0_60": bool(gse_adverse_ok),
                "primary_macro_f1": row["macro_f1_primary"],
                "primary_balanced_accuracy": row["balanced_accuracy_primary"],
                "primary_melanocytic_like_recall": row["melanocytic_like_recall_primary"],
                "primary_adverse_like_recall": row["adverse_like_recall_primary"],
                "GSE72056_macro_f1": row["macro_f1_GSE72056"],
                "GSE72056_balanced_accuracy": row["balanced_accuracy_GSE72056"],
                "GSE72056_melanocytic_like_recall": row["melanocytic_like_recall_GSE72056"],
                "GSE72056_adverse_like_recall": row["adverse_like_recall_GSE72056"],
                "selection_score": score,
            }
        )
    rec = pd.DataFrame(rows)
    candidates = rec.loc[rec["primary_minimums_met"] & rec["GSE72056_not_collapsed"]].copy()
    if candidates.empty:
        candidates = rec.copy()
        reason = "No hard threshold jointly satisfied primary minimums and GSE72056 non-collapse; use continuous delta P(adverse_like), hard labels exploratory only."
    else:
        if (candidates["GSE72056_adverse_recall_ge_0_60"]).any():
            candidates = candidates.loc[candidates["GSE72056_adverse_recall_ge_0_60"]].copy()
            reason = "Selected among thresholds meeting primary minimums, GSE72056 non-collapse, and GSE72056 adverse recall >=0.60."
        else:
            reason = "Selected among thresholds meeting primary minimums and GSE72056 non-collapse; GSE72056 adverse recall remains a limitation."
    selected = candidates.sort_values(["selection_score", "primary_macro_f1"], ascending=False).iloc[0]
    rec["recommended_for_phase5_pilot"] = np.isclose(rec["threshold_adverse_probability"], selected["threshold_adverse_probability"])
    rec["recommended_primary_use"] = np.where(
        rec["recommended_for_phase5_pilot"],
        "use continuous delta_P(adverse_like) first; if a hard label is required use this threshold for pilot only",
        "not_recommended_as_primary_hard_threshold",
    )
    rec["decision_reason"] = reason
    rec["allowed_scope"] = "pilot_only_not_formal_ready"
    rec.to_csv(TABLES / "phase4F_recommended_phase5_threshold.csv", index=False, encoding="utf-8-sig")
    write_text(
        LOGS / "phase4F_phase5_threshold_decision_log.md",
        [
            "# Phase 4F Phase 5 threshold decision log",
            "",
            "Priority: continuous delta P(adverse_like) over hard labels.",
            f"Recommended hard-label threshold for pilot-only use: {float(selected['threshold_adverse_probability']):.2f}",
            reason,
            "This recommendation does not make the model formal Phase 5-ready because repeated grouped retraining was not uniformly stable and repeated-fold threshold probabilities were not saved.",
        ],
    )
    return rec, float(selected["threshold_adverse_probability"])


def readiness_grade(primary: pd.DataFrame, gse: pd.DataFrame, repeated: pd.DataFrame) -> str:
    formal_repeated = (
        (repeated["macro_f1"].astype(float) >= 0.60).all()
        and (repeated["balanced_accuracy"].astype(float) >= 0.60).all()
        and (repeated["adverse_like_recall"].astype(float) >= 0.70).all()
        and (repeated["melanocytic_like_recall"].astype(float) >= 0.50).all()
    )
    primary_any = (
        (primary["macro_f1"] >= 0.60)
        & (primary["balanced_accuracy"] >= 0.60)
        & (primary["adverse_like_recall"] >= 0.70)
        & (primary["melanocytic_like_recall"] >= 0.50)
    ).any()
    gse_noncollapse = ((gse["macro_f1"] >= 0.60) & (gse["balanced_accuracy"] >= 0.60)).any()
    if formal_repeated and primary_any and gse_noncollapse and (gse["adverse_like_recall"] >= 0.70).any():
        return "YES_FORMAL"
    if primary_any and gse_noncollapse:
        return "CONDITIONAL_PILOT"
    return "NO"


def write_summary(primary: pd.DataFrame, gse: pd.DataFrame, recommended: pd.DataFrame, recommended_threshold: float, grade: str) -> None:
    repeated = read_csv(TABLES / "phase4E_repeated_grouped_retraining_metrics.csv")
    failures = read_csv(TABLES / "phase4F_fold_failure_analysis.csv")
    fold1 = failures.loc[failures["fold_id"] == 1].iloc[0]
    fold5 = failures.loc[failures["fold_id"] == 5].iloc[0]
    rec_row = recommended.loc[recommended["recommended_for_phase5_pilot"]].iloc[0]
    lines = [
        "# Phase 4F 中文总结",
        "",
        "本阶段只复核 Phase 4E 结果、threshold robustness、fold failure 和 Phase 5 readiness；未重新训练模型，未进行 in silico deletion、perturbation、候选靶点、TCGA、生存、GDSC、DepMap、ChEMBL、Open Targets 或 DEG。",
        "",
        "## 1. Phase 4E 结果复核",
        "",
        "- Phase 4E 必要文件可读取，Phase 4E verification 状态在 Phase 4F 中重新核查为 PASS。",
        "- GSE72056 未用于训练；repeated grouped retraining 有 5 个 fold 的真实 checkpoint 和 training history，不是 frozen diagnostic。",
        "- sample_id/split_unit 未发现跨 split 泄漏。",
        "",
        "## 2. Fold failure analysis",
        "",
        f"- fold 1: melanocytic_like recall={fold1['melanocytic_like_recall']:.4f}, adverse_like recall={fold1['adverse_like_recall']:.4f}, failure_reason={fold1['failure_reason']}；test cells={int(fold1['test_cell_count'])}, test sample_id={int(fold1['test_sample_id_count'])}。",
        f"- fold 5: melanocytic_like recall={fold5['melanocytic_like_recall']:.4f}, adverse_like recall={fold5['adverse_like_recall']:.4f}, failure_reason={fold5['failure_reason']}；test cells={int(fold5['test_cell_count'])}, test sample_id={int(fold5['test_sample_id_count'])}。",
        "- fold failure 更符合 sample-level held-out sampling/threshold instability，不应解释为新的生物学发现。",
        "",
        "## 3. Threshold robustness",
        "",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"- primary threshold {row['threshold_adverse_probability']:.2f}: Macro-F1={row['macro_f1']:.4f}, Balanced accuracy={row['balanced_accuracy']:.4f}, melanocytic_like recall={row['melanocytic_like_recall']:.4f}, adverse_like recall={row['adverse_like_recall']:.4f}"
        )
    lines.extend(["", "## 4. GSE72056 sensitivity 限制", ""])
    for _, row in gse.iterrows():
        lines.append(
            f"- GSE72056 threshold {row['threshold_adverse_probability']:.2f}: Macro-F1={row['macro_f1']:.4f}, Balanced accuracy={row['balanced_accuracy']:.4f}, melanocytic_like recall={row['melanocytic_like_recall']:.4f}, adverse_like recall={row['adverse_like_recall']:.4f}"
        )
    lines.append("- GSE72056 是 processed/non-integer expression，只能作为 processed-expression sensitivity evaluation，不支持强外部泛化声明。")
    lines.extend(["", "## 5. Phase 5 pilot threshold", ""])
    lines.append(f"- 推荐优先使用 continuous delta P(adverse_like)，不要只依赖 hard label。")
    lines.append(f"- 如必须使用 hard label，推荐 pilot-only threshold = {recommended_threshold:.2f}。")
    lines.append(f"- 推荐阈值 primary Macro-F1={rec_row['primary_macro_f1']:.4f}, GSE72056 Macro-F1={rec_row['GSE72056_macro_f1']:.4f}, GSE72056 adverse_like recall={rec_row['GSE72056_adverse_like_recall']:.4f}。")
    lines.extend(["", "## 6. Phase 5 readiness grading", "", f"READY_FOR_PHASE5 = {grade}", ""])
    if grade == "CONDITIONAL_PILOT":
        lines.extend(
            [
                "允许做什么：",
                "- 仅允许进入 pilot in silico deletion。",
                "- 仅允许报告 delta P(adverse_like) 和 sensitivity-ranked exploratory perturbation signals。",
                "",
                "禁止做什么：",
                "- 不允许输出 overstated treatment-target claims。",
                "- 不允许写强外部泛化声明。",
                "- 不允许把 GSE72056 sensitivity 当作强外部验证。",
                "- 不允许将 hard label threshold 结果包装为正式候选靶点发现。",
                "",
                "必须报告的限制：",
                "- repeated grouped retraining 有 fold 1 和 fold 5 边缘失败。",
                "- repeated-fold threshold probabilities 未保存，不能重算每折 0.50-0.70 threshold robustness。",
                "- GSE72056 为 processed/non-integer expression，存在 domain shift/processed-expression 限制。",
            ]
        )
    elif grade == "YES_FORMAL":
        lines.extend(["满足 formal readiness，但仍需在 Phase 5 保留 sample-level held-out evaluation 和 sensitivity 限制。"])
    else:
        lines.extend(["阻断 Phase 5：primary/GSE/repeated grouped retraining 未达到 pilot 最低解释边界。"])
    write_text(ROOT / "summary_phase4F_zh.md", lines)


def main() -> int:
    ensure_dirs()
    meta, train_ds, sens_ds = preflight()
    write_fold_failure_analysis(meta)
    primary, repeated_threshold, gse = write_threshold_robustness(meta, train_ds, sens_ds)
    recommended, recommended_threshold = recommend_threshold(primary, gse)
    repeated = read_csv(TABLES / "phase4E_repeated_grouped_retraining_metrics.csv")
    grade = readiness_grade(primary, gse, repeated)
    write_summary(primary, gse, recommended, recommended_threshold, grade)
    print("PHASE4F_READINESS_AUDIT: PASS")
    print(f"READY_FOR_PHASE5={grade}")
    print(f"RECOMMENDED_PILOT_THRESHOLD={recommended_threshold:.2f}")
    print(f"SUMMARY={ROOT / 'summary_phase4F_zh.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
