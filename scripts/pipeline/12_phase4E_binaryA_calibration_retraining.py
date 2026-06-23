from __future__ import annotations

import gc
import importlib.util
import json
import math
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_processed"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
SCRIPTS = ROOT / "scripts"

P4D_PATH = SCRIPTS / "10_phase4D_relabel_finetune.py"
spec = importlib.util.spec_from_file_location("phase4d", P4D_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import Phase 4D helpers from {P4D_PATH}")
p4d = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p4d
spec.loader.exec_module(p4d)

LABELS = ["melanocytic_like", "adverse_like"]
MEL = 0
ADV = 1
TASK_DISPLAY = "binary_A_melanocytic_vs_adverse"
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

TRAIN_H5AD = DATA / "GSE115978_malignant_state_phase4D_labeled.h5ad"
SENS_H5AD = DATA / "GSE72056_malignant_state_phase4D_labeled.h5ad"
TRAIN_DS_ORIGINAL = DATA / "tokenized_v2_gc104M" / "GSE115978_malignant_state_labeled_v2.dataset"
SENS_DS_ORIGINAL = DATA / "tokenized_v2_gc104M" / "GSE72056_malignant_state_labeled_v2.dataset"
TRAIN_DS = DATA / "tokenized_v2_gc104M" / "GSE115978_malignant_state_phase4D_labeled_v2.dataset"
SENS_DS = DATA / "tokenized_v2_gc104M" / "GSE72056_malignant_state_phase4D_labeled_v2.dataset"
PHASE4D_BINARY_META = TABLES / "phase4D_binary_A_metadata_with_split.csv"
PHASE4D_CKPT = MODELS / "phase4D_geneformer_v2_binary_A" / "best_model.pt"

CLASS_WEIGHTED_DIR = MODELS / "phase4E_geneformer_v2_binary_A_class_weighted"
FOCAL_DIR = MODELS / "phase4E_geneformer_v2_binary_A_focal_loss"
REPEATED_DIR = MODELS / "phase4E_geneformer_v2_binary_A_repeated_grouped"

SEED = 42
EPOCHS = 3
LR = 1e-5
BATCH_SIZE = 1
GRAD_ACCUM = 8
PATIENCE = 2

REQUIRED_METADATA = [
    "original_cell_id",
    "cell_id",
    "malignant_state",
    "phase4D_binary_A_label",
    "split_unit",
    "split_unit_field",
    "split_unit_type",
    "patient_identity_status",
    "treatment.group",
]


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, LOGS, MODELS, CLASS_WEIGHTED_DIR, FOCAL_DIR, REPEATED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True


def clear_model_dir(path: Path) -> None:
    resolved = path.resolve()
    allowed = MODELS.resolve()
    if not str(resolved).startswith(str(allowed)):
        raise RuntimeError(f"Refusing to clear outside models directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def dataset_columns(path: Path) -> tuple[int | None, list[str]]:
    if not path.exists():
        return None, []
    ds = load_from_disk(str(path))
    return len(ds), list(ds.column_names)


def preflight() -> tuple[Dataset, Dataset, ad.AnnData, ad.AnnData, pd.DataFrame]:
    log = ["# Phase 4E preflight check log", "", f"Timestamp: {datetime.now().isoformat(timespec='seconds')}"]
    rows: list[dict[str, Any]] = []
    required_files = [
        ROOT / "summary_phase4D_zh.md",
        PHASE4D_CKPT,
        TRAIN_H5AD,
        SENS_H5AD,
        TRAIN_DS_ORIGINAL,
        SENS_DS_ORIGINAL,
        TRAIN_DS,
        SENS_DS,
        TABLES / "phase4D_binary_A_primary_test_metrics.csv",
        TABLES / "phase4D_binary_A_per_class_metrics.csv",
        TABLES / "phase4D_GSE72056_binary_A_sensitivity_metrics.csv",
        TABLES / "phase4D_binary_A_baseline_metrics.csv",
        PHASE4D_BINARY_META,
    ]
    missing = []
    for path in required_files:
        ok = path.exists() and (path.is_dir() or path.stat().st_size > 0)
        rows.append({"check_type": "file", "path": rel(path), "status": "ok" if ok else "missing_or_empty"})
        if not ok:
            missing.append(rel(path))

    if missing:
        pd.DataFrame(rows).to_csv(TABLES / "phase4E_input_integrity_check.csv", index=False, encoding="utf-8-sig")
        write_text(LOGS / "phase4E_preflight_check_log.md", log + ["Missing required files:", *[f"- {x}" for x in missing]])
        raise RuntimeError("Phase 4E preflight failed: missing required files.")

    train_h5 = ad.read_h5ad(TRAIN_H5AD)
    sens_h5 = ad.read_h5ad(SENS_H5AD)
    train_ds = load_from_disk(str(TRAIN_DS))
    sens_ds = load_from_disk(str(SENS_DS))
    train_meta = pd.read_csv(PHASE4D_BINARY_META, encoding="utf-8-sig")

    for dataset_name, adata, ds, source in [
        ("GSE115978", train_h5, train_ds, "phase4D_labeled"),
        ("GSE72056", sens_h5, sens_ds, "phase4D_labeled"),
    ]:
        h5_missing = [col for col in REQUIRED_METADATA if col not in adata.obs.columns]
        ds_missing = [col for col in REQUIRED_METADATA if col not in ds.column_names]
        row_match = int(adata.n_obs) == len(ds)
        status = "ok" if not h5_missing and not ds_missing and row_match else "failed"
        rows.append(
            {
                "check_type": "metadata",
                "dataset_id": dataset_name,
                "source": source,
                "h5ad_n_obs": int(adata.n_obs),
                "token_rows": len(ds),
                "row_count_match": row_match,
                "missing_h5ad_fields": ";".join(h5_missing),
                "missing_token_fields": ";".join(ds_missing),
                "status": status,
            }
        )
        if status != "ok":
            pd.DataFrame(rows).to_csv(TABLES / "phase4E_input_integrity_check.csv", index=False, encoding="utf-8-sig")
            write_text(LOGS / "phase4E_preflight_check_log.md", log + [f"{dataset_name} metadata check failed."])
            raise RuntimeError(f"Phase 4E preflight failed for {dataset_name}.")

    # The original tokenized datasets are checked for row consistency, but the
    # Phase 4D-labeled copies are the correct inputs for Phase 4E supervised labels.
    for name, h5, path in [
        ("GSE115978_original_tokenized", train_h5, TRAIN_DS_ORIGINAL),
        ("GSE72056_original_tokenized", sens_h5, SENS_DS_ORIGINAL),
    ]:
        n_rows, cols = dataset_columns(path)
        rows.append(
            {
                "check_type": "original_tokenized_reference",
                "dataset_id": name,
                "token_rows": n_rows,
                "h5ad_n_obs": int(h5.n_obs),
                "row_count_match": n_rows == int(h5.n_obs),
                "missing_token_fields": ";".join([col for col in REQUIRED_METADATA if col not in cols]),
                "status": "reference_only_phase4D_labels_not_expected",
            }
        )

    supervised = train_meta.loc[train_meta["phase4D_supervised_use"].astype(bool)].copy()
    leakage = (
        supervised.groupby("split_unit")["split"].nunique().loc[lambda x: x > 1].index.astype(str).tolist()
        if not supervised.empty
        else []
    )
    rows.append(
        {
            "check_type": "split_unit_leakage",
            "dataset_id": "GSE115978",
            "bad_split_units": ";".join(leakage),
            "status": "ok" if not leakage else "failed",
        }
    )
    if leakage:
        pd.DataFrame(rows).to_csv(TABLES / "phase4E_input_integrity_check.csv", index=False, encoding="utf-8-sig")
        write_text(LOGS / "phase4E_preflight_check_log.md", log + ["split_unit leakage detected.", *leakage])
        raise RuntimeError("Phase 4E preflight failed: split_unit leakage.")

    pd.DataFrame(rows).to_csv(TABLES / "phase4E_input_integrity_check.csv", index=False, encoding="utf-8-sig")
    log.extend(
        [
            "Preflight passed for Phase 4D-labeled H5AD and tokenized datasets.",
            "Original tokenized datasets were checked for row count but are reference-only because Phase 4D labels are stored in phase4D-labeled copies.",
            "GSE72056 will not be used for training or threshold selection.",
        ]
    )
    write_text(LOGS / "phase4E_preflight_check_log.md", log)
    return train_ds, sens_ds, train_h5, sens_h5, train_meta


def threshold_predictions(probs: np.ndarray, adverse_threshold: float) -> np.ndarray:
    return np.where(probs[:, ADV] >= adverse_threshold, ADV, MEL)


def metrics_at_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    model_source: str,
    evaluation_set: str,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    pred = threshold_predictions(probs, threshold)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=[MEL, ADV], zero_division=0
    )
    row: dict[str, Any] = {
        "model_source": model_source,
        "evaluation_set": evaluation_set,
        "threshold_adverse_probability": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "melanocytic_like_precision": float(precision[MEL]),
        "melanocytic_like_recall": float(recall[MEL]),
        "adverse_like_precision": float(precision[ADV]),
        "adverse_like_recall": float(recall[ADV]),
        "melanocytic_like_support": int(support[MEL]),
        "adverse_like_support": int(support[ADV]),
        "n_cells": int(len(y_true)),
        "status": "success",
    }
    if extra:
        row.update(extra)
    per_class = [
        {
            "model_source": model_source,
            "evaluation_set": evaluation_set,
            "threshold_adverse_probability": threshold,
            "label": LABELS[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in [MEL, ADV]
    ]
    return row, per_class, pred


def choose_threshold(table: pd.DataFrame) -> float:
    candidates = table.loc[
        (table["melanocytic_like_recall"] >= 0.50) & (table["adverse_like_recall"] >= 0.70)
    ].copy()
    if candidates.empty:
        candidates = table.copy()
        candidates["constraint_status"] = "constraints_not_satisfied_choose_best_macro_f1"
    else:
        candidates["constraint_status"] = "constraints_satisfied"
    candidates = candidates.sort_values(
        ["macro_f1", "balanced_accuracy", "melanocytic_like_recall", "adverse_like_recall"],
        ascending=[False, False, False, False],
    )
    return float(candidates.iloc[0]["threshold_adverse_probability"])


def threshold_sweep(
    y_true: np.ndarray,
    probs: np.ndarray,
    model_source: str,
    evaluation_set: str,
) -> tuple[pd.DataFrame, float]:
    rows = []
    for threshold in THRESHOLDS:
        row, _, _ = metrics_at_threshold(y_true, probs, threshold, model_source, evaluation_set)
        rows.append(row)
    table = pd.DataFrame(rows)
    selected = choose_threshold(table)
    table["selected_threshold"] = np.isclose(table["threshold_adverse_probability"], selected)
    table["selection_priority"] = "macro_f1_then_balanced_accuracy_with_melanocytic_recall_ge_0.50_and_adverse_recall_ge_0.70"
    table["constraint_satisfied"] = (
        (table["melanocytic_like_recall"] >= 0.50) & (table["adverse_like_recall"] >= 0.70)
    )
    return table, selected


def load_checkpoint(checkpoint: Path, device: torch.device) -> nn.Module:
    model = p4d.GeneformerClassifier(len(LABELS))
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_checkpoint(
    checkpoint: Path,
    ds: Dataset,
    meta: pd.DataFrame,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint, device)
    y_true, y_pred, probs = p4d.evaluate_model(model, ds, meta, split, device, LABELS)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return y_true, y_pred, probs


def prediction_frame(meta: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray, split: str, threshold: float) -> pd.DataFrame:
    sub = meta.loc[(meta["phase4D_supervised_use"].astype(bool)) & (meta["split"] == split)].copy().reset_index(drop=True)
    pred = threshold_predictions(probs, threshold)
    if len(sub) != len(y_true):
        raise RuntimeError(f"Prediction/meta row mismatch for {split}: {len(sub)} vs {len(y_true)}")
    sub["true_label_id"] = y_true.astype(int)
    sub["true_label"] = [LABELS[int(x)] for x in y_true]
    sub["predicted_label_id"] = pred.astype(int)
    sub["predicted_label"] = [LABELS[int(x)] for x in pred]
    sub["prob_melanocytic_like"] = probs[:, MEL]
    sub["prob_adverse_like"] = probs[:, ADV]
    sub["threshold_adverse_probability"] = threshold
    sub["correct"] = sub["true_label_id"] == sub["predicted_label_id"]
    sub["error_type"] = np.where(
        sub["correct"],
        "correct",
        np.where(
            (sub["true_label"] == "melanocytic_like") & (sub["predicted_label"] == "adverse_like"),
            "melanocytic_like_to_adverse_like",
            "adverse_like_to_melanocytic_like",
        ),
    )
    keep = [
        "dataset_id",
        "row_index",
        "original_cell_id",
        "cell_id",
        "malignant_state",
        "split_unit",
        "sample_id",
        "tumor_id",
        "treatment.group",
        "true_label",
        "predicted_label",
        "prob_melanocytic_like",
        "prob_adverse_like",
        "threshold_adverse_probability",
        "correct",
        "error_type",
    ]
    return sub[[col for col in keep if col in sub.columns]]


def aggregate_errors(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for value, sub in df.groupby(by, dropna=False):
        mel = sub.loc[sub["true_label"] == "melanocytic_like"]
        adv = sub.loc[sub["true_label"] == "adverse_like"]
        mel_err = int((mel["error_type"] == "melanocytic_like_to_adverse_like").sum())
        adv_err = int((adv["error_type"] == "adverse_like_to_melanocytic_like").sum())
        rows.append(
            {
                by: value,
                "n_cells": len(sub),
                "n_errors": int((~sub["correct"]).sum()),
                "error_rate": float((~sub["correct"]).mean()) if len(sub) else math.nan,
                "melanocytic_like_cells": len(mel),
                "melanocytic_like_to_adverse_like": mel_err,
                "melanocytic_like_to_adverse_like_rate": mel_err / len(mel) if len(mel) else math.nan,
                "adverse_like_cells": len(adv),
                "adverse_like_to_melanocytic_like": adv_err,
                "adverse_like_to_melanocytic_like_rate": adv_err / len(adv) if len(adv) else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["n_errors", "error_rate"], ascending=[False, False])


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return ece


def write_error_analysis(meta: pd.DataFrame, y_test: np.ndarray, prob_test: np.ndarray) -> None:
    row050, _, pred050 = metrics_at_threshold(y_test, prob_test, 0.50, "phase4D_binary_A", "GSE115978_held_out_test")
    pred_df = prediction_frame(meta, y_test, prob_test, "held_out_test", 0.50)
    pred_df.to_csv(TABLES / "phase4E_binary_A_error_analysis.csv", index=False, encoding="utf-8-sig")
    aggregate_errors(pred_df, "sample_id").to_csv(TABLES / "phase4E_binary_A_error_by_sample_id.csv", index=False, encoding="utf-8-sig")
    aggregate_errors(pred_df, "treatment.group").to_csv(TABLES / "phase4E_binary_A_error_by_treatment_group.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    for label in LABELS:
        vals = pred_df.loc[pred_df["true_label"] == label, "prob_adverse_like"].astype(float)
        ax.hist(vals, bins=20, alpha=0.55, label=f"true {label}")
    ax.axvline(0.50, color="black", linestyle="--", linewidth=1, label="threshold 0.50")
    ax.set_xlabel("Predicted probability of adverse_like")
    ax.set_ylabel("Cell count")
    ax.set_title("Phase 4D binary A held-out probability distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4E_binary_A_probability_distribution.png", dpi=200)
    plt.close(fig)

    cm = confusion_matrix(y_test, pred050, labels=[MEL, ADV])
    mel_to_adv = int(cm[MEL, ADV])
    adv_to_mel = int(cm[ADV, MEL])
    mel_total = int(cm[MEL].sum())
    adv_total = int(cm[ADV].sum())
    ece = expected_calibration_error(y_test, prob_test)
    brier = float(np.mean((prob_test[:, ADV] - (y_test == ADV).astype(float)) ** 2))
    log = [
        "# Phase 4E binary A error analysis log",
        "",
        "Model: Phase 4D Geneformer-V2 binary A checkpoint.",
        "Evaluation set: GSE115978 held-out test only.",
        f"Confusion matrix rows=true, cols=predicted: {cm.tolist()}",
        f"melanocytic_like_to_adverse_like: {mel_to_adv}/{mel_total} = {mel_to_adv / mel_total if mel_total else math.nan:.4f}",
        f"adverse_like_to_melanocytic_like: {adv_to_mel}/{adv_total} = {adv_to_mel / adv_total if adv_total else math.nan:.4f}",
        f"accuracy={row050['accuracy']:.4f}, macro_f1={row050['macro_f1']:.4f}, balanced_accuracy={row050['balanced_accuracy']:.4f}",
        f"ECE_10bin={ece:.4f}; Brier_adverse_probability={brier:.4f}",
        "Error concentration by sample_id and treatment.group was written to tables.",
    ]
    write_text(LOGS / "phase4E_error_analysis_log.md", log)


def write_threshold_calibration(y_val: np.ndarray, prob_val: np.ndarray, model_source: str) -> float:
    table, selected = threshold_sweep(y_val, prob_val, model_source, "GSE115978_validation")
    table.to_csv(TABLES / "phase4E_threshold_calibration_metrics.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["threshold_adverse_probability"], table["macro_f1"], marker="o", label="Macro-F1")
    ax.plot(table["threshold_adverse_probability"], table["balanced_accuracy"], marker="o", label="Balanced accuracy")
    ax.plot(table["threshold_adverse_probability"], table["melanocytic_like_recall"], marker="o", label="melanocytic_like recall")
    ax.plot(table["threshold_adverse_probability"], table["adverse_like_recall"], marker="o", label="adverse_like recall")
    ax.axvline(selected, color="black", linestyle="--", linewidth=1, label=f"selected {selected:.2f}")
    ax.set_xlabel("Adverse-like probability threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.02)
    ax.set_title("Phase 4E threshold calibration on GSE115978 validation")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4E_threshold_tradeoff_curve.png", dpi=200)
    plt.close(fig)

    chosen = table.loc[table["selected_threshold"]].iloc[0]
    write_text(
        LOGS / "phase4E_threshold_calibration_log.md",
        [
            "# Phase 4E threshold calibration log",
            "",
            "Thresholds were selected using GSE115978 validation predictions only.",
            "GSE72056 was not used for threshold selection.",
            f"Selected adverse_like probability threshold = {selected:.2f}",
            f"Validation macro-F1={chosen['macro_f1']:.4f}, balanced_accuracy={chosen['balanced_accuracy']:.4f}, melanocytic_like_recall={chosen['melanocytic_like_recall']:.4f}, adverse_like_recall={chosen['adverse_like_recall']:.4f}",
        ],
    )
    return selected


@dataclass
class TrainOutcome:
    model_source: str
    loss_mode: str
    status: str
    checkpoint: Path | None
    selected_threshold: float | None
    test_metrics_050: dict[str, Any] | None
    test_metrics_calibrated: dict[str, Any] | None
    per_class: list[dict[str, Any]]
    log: list[str]
    gpu_rows: list[dict[str, Any]]
    error: str | None = None


def class_weights_tensor(meta: pd.DataFrame, split: str) -> torch.Tensor:
    return p4d.class_weights(meta, split, len(LABELS))


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, gamma: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, weight=weights, reduction="none")
    pt = torch.softmax(logits, dim=-1).gather(1, targets.view(-1, 1)).squeeze(1).clamp(min=1e-6, max=1.0)
    return (((1.0 - pt) ** gamma) * ce).mean()


def train_binary_model(
    ds: Dataset,
    meta: pd.DataFrame,
    outdir: Path,
    model_source: str,
    loss_mode: str,
    fold_tag: str = "primary",
) -> TrainOutcome:
    clear_model_dir(outdir)
    log = [
        f"# Phase 4E {model_source} training log",
        "",
        "GSE72056 was not used for training or threshold selection.",
        "split_unit = sample_id; no cell-level random split.",
        f"loss_mode = {loss_mode}",
    ]
    gpu_rows: list[dict[str, Any]] = []
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable.")
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_rows.append(p4d.gpu_snapshot("before_model_load", model_source))
        model = p4d.GeneformerClassifier(len(LABELS))
        model.gradient_checkpointing_enable()
        model.to(device)
        gpu_rows.append(p4d.gpu_snapshot("after_model_to_cuda", model_source))
        train_loader = DataLoader(
            p4d.TokenTaskDataset(ds, meta, "train"),
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=p4d.collate,
            generator=torch.Generator().manual_seed(SEED),
        )
        weights = class_weights_tensor(meta, "train").to(device)
        if loss_mode == "class_weighted":
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        elif loss_mode == "focal_loss":
            loss_fn = None
            log.append("focal_loss_gamma = 2.0; alpha implemented through class weights.")
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        best_f1 = -1.0
        best_epoch = -1
        checkpoint = outdir / "best_model.pt"
        history = []
        patience_used = 0
        for epoch in range(1, EPOCHS + 1):
            start = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            steps = 0
            for step, batch in enumerate(train_loader, start=1):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                    labels = batch["labels"].to(device)
                    if loss_mode == "focal_loss":
                        loss = focal_loss(logits, labels, weights, gamma=2.0) / GRAD_ACCUM
                    else:
                        loss = loss_fn(logits, labels) / GRAD_ACCUM
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach().cpu()) * GRAD_ACCUM
                if step % GRAD_ACCUM == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                steps += 1
            y_val, _, prob_val = p4d.evaluate_model(model, ds, meta, "validation", device, LABELS)
            pred_val = prob_val.argmax(axis=1)
            val_metrics = p4d.metric_dict(y_val, pred_val, prob_val, LABELS)
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / max(steps, 1),
                "validation_macro_f1": val_metrics["macro_f1"],
                "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                "seconds": time.time() - start,
            }
            history.append(row)
            log.append(
                f"epoch {epoch}: train_loss={row['train_loss']:.6f}, val_macro_f1={row['validation_macro_f1']:.4f}, val_balanced_accuracy={row['validation_balanced_accuracy']:.4f}, seconds={row['seconds']:.1f}"
            )
            gpu_rows.append(p4d.gpu_snapshot(f"after_epoch_{epoch}", model_source))
            if float(val_metrics["macro_f1"]) > best_f1:
                best_f1 = float(val_metrics["macro_f1"])
                best_epoch = epoch
                patience_used = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "task_key": "binary_A",
                        "labels": LABELS,
                        "best_epoch": best_epoch,
                        "validation_macro_f1": best_f1,
                        "loss_mode": loss_mode,
                        "fold_tag": fold_tag,
                    },
                    checkpoint,
                )
            else:
                patience_used += 1
                if patience_used >= PATIENCE:
                    log.append(f"early stopping at epoch {epoch}")
                    break
        pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False, encoding="utf-8-sig")
        (outdir / "training_config.json").write_text(
            json.dumps(
                {
                    "task": TASK_DISPLAY,
                    "labels": LABELS,
                    "model_path": str(p4d.V2_MODEL),
                    "max_length": 4096,
                    "batch_size": BATCH_SIZE,
                    "gradient_accumulation": GRAD_ACCUM,
                    "epochs": EPOCHS,
                    "learning_rate": LR,
                    "mixed_precision": "fp16/AMP",
                    "loss_mode": loss_mode,
                    "class_weights": [float(x) for x in weights.detach().cpu().numpy()],
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "fold_tag": fold_tag,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        del model
        torch.cuda.empty_cache()
        gc.collect()

        y_val, _, prob_val = evaluate_checkpoint(checkpoint, ds, meta, "validation")
        calib_table, selected = threshold_sweep(y_val, prob_val, model_source, "GSE115978_validation")
        calib_table.to_csv(outdir / "threshold_calibration_metrics.csv", index=False, encoding="utf-8-sig")

        y_test, _, prob_test = evaluate_checkpoint(checkpoint, ds, meta, "held_out_test")
        metrics_050, per_050, pred050 = metrics_at_threshold(
            y_test,
            prob_test,
            0.50,
            model_source,
            "GSE115978_primary_held_out_test",
            {"best_epoch": best_epoch, "loss_mode": loss_mode},
        )
        metrics_cal, per_cal, predcal = metrics_at_threshold(
            y_test,
            prob_test,
            selected,
            model_source,
            "GSE115978_primary_held_out_test",
            {"best_epoch": best_epoch, "loss_mode": loss_mode},
        )
        p4d.save_cm(
            y_test,
            predcal,
            LABELS,
            f"{model_source} calibrated",
            FIGURES / ("phase4E_class_weighted_confusion_matrix_calibrated.png" if loss_mode == "class_weighted" else "phase4E_focal_loss_confusion_matrix_calibrated.png"),
        )
        log.append(f"selected_threshold={selected:.2f}")
        log.append(f"held_out threshold_0.50 macro_f1={metrics_050['macro_f1']:.4f}, balanced_accuracy={metrics_050['balanced_accuracy']:.4f}, melanocytic_recall={metrics_050['melanocytic_like_recall']:.4f}, adverse_recall={metrics_050['adverse_like_recall']:.4f}")
        log.append(f"held_out calibrated macro_f1={metrics_cal['macro_f1']:.4f}, balanced_accuracy={metrics_cal['balanced_accuracy']:.4f}, melanocytic_recall={metrics_cal['melanocytic_like_recall']:.4f}, adverse_recall={metrics_cal['adverse_like_recall']:.4f}")
        gpu_rows.append(p4d.gpu_snapshot("after_test_evaluation", model_source))
        return TrainOutcome(
            model_source=model_source,
            loss_mode=loss_mode,
            status="success",
            checkpoint=checkpoint,
            selected_threshold=selected,
            test_metrics_050=metrics_050,
            test_metrics_calibrated=metrics_cal,
            per_class=per_050 + per_cal,
            log=log,
            gpu_rows=gpu_rows,
        )
    except Exception as exc:
        log.append("TRAINING_FAILED")
        log.append(repr(exc))
        log.append(traceback.format_exc())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return TrainOutcome(model_source, loss_mode, "failed", None, None, None, None, [], log, gpu_rows, repr(exc))


def write_train_outputs(outcome: TrainOutcome, prefix: str) -> None:
    write_text(LOGS / f"{prefix}_training_log.md", outcome.log)
    if outcome.test_metrics_050:
        pd.DataFrame([outcome.test_metrics_050]).to_csv(
            TABLES / f"{prefix}_primary_test_metrics_threshold_050.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame([{"model_source": outcome.model_source, "status": outcome.status, "error": outcome.error}]).to_csv(
            TABLES / f"{prefix}_primary_test_metrics_threshold_050.csv", index=False, encoding="utf-8-sig"
        )
    if outcome.test_metrics_calibrated:
        pd.DataFrame([outcome.test_metrics_calibrated]).to_csv(
            TABLES / f"{prefix}_primary_test_metrics_calibrated.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame([{"model_source": outcome.model_source, "status": outcome.status, "error": outcome.error}]).to_csv(
            TABLES / f"{prefix}_primary_test_metrics_calibrated.csv", index=False, encoding="utf-8-sig"
        )
    if prefix == "phase4E_class_weighted":
        pd.DataFrame(outcome.per_class).to_csv(TABLES / "phase4E_class_weighted_per_class_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(outcome.gpu_rows).to_csv(TABLES / "phase4E_gpu_memory_log.csv", index=False, encoding="utf-8-sig")


def should_run_focal(class_weighted: TrainOutcome) -> bool:
    if not class_weighted.test_metrics_calibrated:
        return False
    return float(class_weighted.test_metrics_calibrated.get("melanocytic_like_recall", 0.0)) < 0.50


def meets_phase5_minimum(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    return (
        float(metrics.get("macro_f1", 0.0)) >= 0.60
        and float(metrics.get("balanced_accuracy", 0.0)) >= 0.60
        and float(metrics.get("adverse_like_recall", 0.0)) >= 0.70
        and float(metrics.get("melanocytic_like_recall", 0.0)) >= 0.50
    )


def best_outcome(class_weighted: TrainOutcome, focal: TrainOutcome | None) -> TrainOutcome:
    candidates = [x for x in [class_weighted, focal] if x and x.test_metrics_calibrated]
    if not candidates:
        return class_weighted
    candidates.sort(
        key=lambda x: (
            meets_phase5_minimum(x.test_metrics_calibrated),
            float(x.test_metrics_calibrated.get("macro_f1", 0.0)),
            float(x.test_metrics_calibrated.get("balanced_accuracy", 0.0)),
        ),
        reverse=True,
    )
    return candidates[0]


def fold_assignments(meta: pd.DataFrame, seed: int = 42) -> list[tuple[int, dict[str, str], dict[str, Any]]]:
    sub = meta.loc[meta["phase4D_supervised_use"].astype(bool)].copy()
    labels = sub["phase4D_task_label"].astype(str).values
    groups = sub["split_unit"].astype(str).values
    folds = []
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (trainval_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(sub)), labels, groups), start=1):
        trainval = sub.iloc[trainval_idx].copy()
        test = sub.iloc[test_idx].copy()
        tv_labels = trainval["phase4D_task_label"].astype(str).values
        tv_groups = trainval["split_unit"].astype(str).values
        try:
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + fold)
            _, val_idx_rel = next(inner.split(np.zeros(len(trainval)), tv_labels, tv_groups))
        except Exception:
            unique_groups = sorted(trainval["split_unit"].astype(str).unique())
            val_groups = set(unique_groups[: max(1, len(unique_groups) // 4)])
            val_idx_rel = np.where(trainval["split_unit"].astype(str).isin(val_groups))[0]
        val = trainval.iloc[val_idx_rel].copy()
        train = trainval.drop(trainval.index[val_idx_rel]).copy()
        assignment: dict[str, str] = {}
        for unit in train["split_unit"].astype(str).unique():
            assignment[unit] = "train"
        for unit in val["split_unit"].astype(str).unique():
            assignment[unit] = "validation"
        for unit in test["split_unit"].astype(str).unique():
            assignment[unit] = "held_out_test"
        fold_detail = {
            "seed": seed,
            "fold": fold,
            "train_sample_id_count": train["split_unit"].nunique(),
            "validation_sample_id_count": val["split_unit"].nunique(),
            "test_sample_id_count": test["split_unit"].nunique(),
            "train_cell_count": len(train),
            "validation_cell_count": len(val),
            "test_cell_count": len(test),
            "train_label_distribution": json.dumps(dict(Counter(train["phase4D_task_label"])), sort_keys=True),
            "validation_label_distribution": json.dumps(dict(Counter(val["phase4D_task_label"])), sort_keys=True),
            "test_label_distribution": json.dumps(dict(Counter(test["phase4D_task_label"])), sort_keys=True),
            "fold_invalid_for_macro_metrics": set(test["phase4D_task_label"]) != set(LABELS),
        }
        folds.append((fold, assignment, fold_detail))
    return folds


def apply_assignment(meta: pd.DataFrame, assignment: dict[str, str]) -> pd.DataFrame:
    out = meta.copy()
    out["split"] = "excluded_from_supervised_training"
    mask = out["phase4D_supervised_use"].astype(bool)
    out.loc[mask, "split"] = out.loc[mask, "split_unit"].astype(str).map(assignment)
    return out


def run_repeated_if_needed(ds: Dataset, base_meta: pd.DataFrame, outcome: TrainOutcome) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = [
        "# Phase 4E repeated grouped retraining log",
        "",
        "Only true retraining counts; frozen-model diagnostics are not used.",
        "split_unit = sample_id; seed = 42; folds = 5.",
    ]
    if not meets_phase5_minimum(outcome.test_metrics_calibrated):
        rows = [
            {
                "seed": 42,
                "fold": "",
                "model_source": outcome.model_source,
                "loss_mode": outcome.loss_mode,
                "retraining_status": "not_run_not_meeting_primary_thresholds",
                "reason": "Primary calibrated run did not meet all minimum Phase 5 entry criteria.",
            }
        ]
        detail = pd.DataFrame(rows)
        metrics = pd.DataFrame(rows)
        metrics.to_csv(TABLES / "phase4E_repeated_grouped_retraining_metrics.csv", index=False, encoding="utf-8-sig")
        detail.to_csv(TABLES / "phase4E_repeated_grouped_fold_details.csv", index=False, encoding="utf-8-sig")
        log.append("Repeated grouped retraining was not run because primary metrics did not meet minimum thresholds.")
        write_text(LOGS / "phase4E_repeated_grouped_retraining_log.md", log)
        return metrics, detail

    fold_metric_rows = []
    fold_detail_rows = []
    for fold, assignment, detail in fold_assignments(base_meta, seed=42):
        detail["model_source"] = outcome.model_source
        detail["loss_mode"] = outcome.loss_mode
        if detail["fold_invalid_for_macro_metrics"]:
            detail["retraining_status"] = "not_run_invalid_fold"
            fold_detail_rows.append(detail)
            fold_metric_rows.append({**detail, "retraining_status": "not_run_invalid_fold"})
            continue
        fold_meta = apply_assignment(base_meta, assignment)
        fold_dir = REPEATED_DIR / f"fold_{fold}"
        fold_outcome = train_binary_model(
            ds,
            fold_meta,
            fold_dir,
            f"phase4E_repeated_{outcome.loss_mode}_fold_{fold}",
            outcome.loss_mode,
            fold_tag=f"fold_{fold}",
        )
        detail["retraining_status"] = fold_outcome.status
        detail["selected_threshold"] = fold_outcome.selected_threshold
        fold_detail_rows.append(detail)
        if fold_outcome.test_metrics_calibrated:
            row = {**detail, **fold_outcome.test_metrics_calibrated}
            row["retraining_status"] = fold_outcome.status
        else:
            row = {**detail, "retraining_status": fold_outcome.status, "error": fold_outcome.error}
        fold_metric_rows.append(row)
        log.append(f"fold {fold}: status={fold_outcome.status}")
        if fold_outcome.test_metrics_calibrated:
            log.append(
                f"fold {fold}: macro_f1={fold_outcome.test_metrics_calibrated['macro_f1']:.4f}, balanced_accuracy={fold_outcome.test_metrics_calibrated['balanced_accuracy']:.4f}, melanocytic_recall={fold_outcome.test_metrics_calibrated['melanocytic_like_recall']:.4f}, adverse_recall={fold_outcome.test_metrics_calibrated['adverse_like_recall']:.4f}"
            )
    metrics = pd.DataFrame(fold_metric_rows)
    detail = pd.DataFrame(fold_detail_rows)
    metrics.to_csv(TABLES / "phase4E_repeated_grouped_retraining_metrics.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(TABLES / "phase4E_repeated_grouped_fold_details.csv", index=False, encoding="utf-8-sig")
    write_text(LOGS / "phase4E_repeated_grouped_retraining_log.md", log)
    return metrics, detail


def sensitivity_meta(ds: Dataset) -> pd.DataFrame:
    meta = p4d.dataset_to_meta(ds, "GSE72056")
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    meta["phase4D_task_label"] = meta["phase4D_binary_A_label"]
    meta["phase4D_label_id"] = meta["phase4D_task_label"].map(label_to_id).fillna(-1).astype(int)
    meta["phase4D_supervised_use"] = meta["phase4D_supervised_include_binary_A"].astype(bool)
    meta["split"] = np.where(meta["phase4D_supervised_use"], "sensitivity", "excluded")
    return meta


def write_sensitivity(ds: Dataset, outcome: TrainOutcome) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = [
        "# Phase 4E GSE72056 sensitivity log",
        "",
        "GSE72056 was not used for training.",
        "GSE72056 was not used for threshold selection.",
        "GSE72056 is processed/non-integer expression.",
        "Sensitivity evaluation only; possible domain shift; not strong external validation.",
    ]
    if not outcome.checkpoint or outcome.selected_threshold is None:
        empty = pd.DataFrame([{"model_source": outcome.model_source, "status": "not_available_no_checkpoint"}])
        empty.to_csv(TABLES / "phase4E_GSE72056_binary_A_sensitivity_threshold_050.csv", index=False, encoding="utf-8-sig")
        empty.to_csv(TABLES / "phase4E_GSE72056_binary_A_sensitivity_calibrated.csv", index=False, encoding="utf-8-sig")
        write_text(LOGS / "phase4E_GSE72056_sensitivity_log.md", log + ["No checkpoint available."])
        return empty, empty
    meta = sensitivity_meta(ds)
    y, _, probs = evaluate_checkpoint(outcome.checkpoint, ds, meta, "sensitivity")
    row050, _, pred050 = metrics_at_threshold(
        y,
        probs,
        0.50,
        outcome.model_source,
        "GSE72056_processed_expression_sensitivity",
        {"processed_expression_limitation": True},
    )
    rowcal, _, predcal = metrics_at_threshold(
        y,
        probs,
        float(outcome.selected_threshold),
        outcome.model_source,
        "GSE72056_processed_expression_sensitivity",
        {"processed_expression_limitation": True},
    )
    df050 = pd.DataFrame([row050])
    dfcal = pd.DataFrame([rowcal])
    df050.to_csv(TABLES / "phase4E_GSE72056_binary_A_sensitivity_threshold_050.csv", index=False, encoding="utf-8-sig")
    dfcal.to_csv(TABLES / "phase4E_GSE72056_binary_A_sensitivity_calibrated.csv", index=False, encoding="utf-8-sig")
    p4d.save_cm(
        y,
        predcal,
        LABELS,
        "Phase 4E GSE72056 binary A calibrated sensitivity",
        FIGURES / "phase4E_GSE72056_binary_A_confusion_matrix_calibrated.png",
    )
    log.append(f"threshold_0.50 macro_f1={row050['macro_f1']:.4f}, balanced_accuracy={row050['balanced_accuracy']:.4f}")
    log.append(f"calibrated threshold={outcome.selected_threshold:.2f}, macro_f1={rowcal['macro_f1']:.4f}, balanced_accuracy={rowcal['balanced_accuracy']:.4f}")
    write_text(LOGS / "phase4E_GSE72056_sensitivity_log.md", log)
    return df050, dfcal


def repeated_stable(repeated: pd.DataFrame) -> bool:
    if repeated.empty or "retraining_status" not in repeated.columns:
        return False
    ran = repeated.loc[repeated["retraining_status"] == "success"].copy()
    if len(ran) < 5:
        return False
    required = ["macro_f1", "balanced_accuracy", "adverse_like_recall", "melanocytic_like_recall"]
    if any(col not in ran.columns for col in required):
        return False
    return bool(
        (ran["macro_f1"].astype(float) >= 0.60).all()
        and (ran["balanced_accuracy"].astype(float) >= 0.60).all()
        and (ran["adverse_like_recall"].astype(float) >= 0.70).all()
        and (ran["melanocytic_like_recall"].astype(float) >= 0.50).all()
    )


def write_summary(
    selected_phase4d_threshold: float,
    class_weighted: TrainOutcome,
    focal: TrainOutcome | None,
    best: TrainOutcome,
    repeated: pd.DataFrame,
    sens050: pd.DataFrame,
    senscal: pd.DataFrame,
) -> str:
    primary_ok = meets_phase5_minimum(best.test_metrics_calibrated)
    repeated_ok = repeated_stable(repeated)
    ready = "CONDITIONAL" if primary_ok and repeated_ok else "NO"
    lines = [
        "# Phase 4E 中文总结",
        "",
        "本阶段只执行 binary A calibration、class-imbalance correction 和 grouped retraining stabilization；未进行 in silico deletion、perturbation、候选靶点、TCGA、生存、GDSC、DepMap、ChEMBL、Open Targets 或 DEG。",
        "",
        "## 1. Preflight",
        "",
        "- Phase 4D-labeled GSE115978/GSE72056 h5ad 与 tokenized dataset 可读取，行数一致。",
        "- 原始 tokenized dataset 行数已核查；Phase 4E 监督标签来自 Phase 4D-labeled tokenized copy。",
        "- split_unit = sample_id；未发现 sample-level split 泄漏。",
        "",
        "## 2. Phase 4D binary A 错误模式",
        "",
        "- held-out prediction probability 已重新从 Phase 4D binary A checkpoint 计算。",
        f"- validation-based selected adverse_like threshold = {selected_phase4d_threshold:.2f}。",
        "- 详细错误按 sample_id 和 treatment.group 输出到 Phase 4E error analysis tables。",
        "",
        "## 3. Class-weighted retraining",
        "",
        f"- training status: {class_weighted.status}",
    ]
    if class_weighted.test_metrics_050:
        m = class_weighted.test_metrics_050
        lines.extend(
            [
                f"- threshold 0.50: Macro-F1={m['macro_f1']:.4f}, Balanced accuracy={m['balanced_accuracy']:.4f}, melanocytic_like recall={m['melanocytic_like_recall']:.4f}, adverse_like recall={m['adverse_like_recall']:.4f}",
            ]
        )
    if class_weighted.test_metrics_calibrated:
        m = class_weighted.test_metrics_calibrated
        lines.append(
            f"- calibrated threshold {class_weighted.selected_threshold:.2f}: Macro-F1={m['macro_f1']:.4f}, Balanced accuracy={m['balanced_accuracy']:.4f}, melanocytic_like recall={m['melanocytic_like_recall']:.4f}, adverse_like recall={m['adverse_like_recall']:.4f}"
        )
    lines.extend(["", "## 4. Focal loss", ""])
    if focal is None:
        lines.append("- focal loss 未执行：class-weighted calibrated melanocytic_like recall 已达到 0.50 门槛或 class-weighted run 未产生可用结果。")
    else:
        lines.append(f"- training status: {focal.status}")
        if focal.test_metrics_calibrated:
            m = focal.test_metrics_calibrated
            lines.append(
                f"- calibrated threshold {focal.selected_threshold:.2f}: Macro-F1={m['macro_f1']:.4f}, Balanced accuracy={m['balanced_accuracy']:.4f}, melanocytic_like recall={m['melanocytic_like_recall']:.4f}, adverse_like recall={m['adverse_like_recall']:.4f}"
            )
    lines.extend(["", "## 5. 最佳 Phase 4E binary A 模型", ""])
    lines.append(f"- selected model: {best.model_source}")
    if best.test_metrics_calibrated:
        m = best.test_metrics_calibrated
        lines.append(
            f"- primary calibrated: Macro-F1={m['macro_f1']:.4f}, Balanced accuracy={m['balanced_accuracy']:.4f}, melanocytic_like recall={m['melanocytic_like_recall']:.4f}, adverse_like recall={m['adverse_like_recall']:.4f}"
        )
    lines.extend(["", "## 6. Repeated grouped retraining", ""])
    if "retraining_status" in repeated.columns:
        for status, n in repeated["retraining_status"].astype(str).value_counts().items():
            lines.append(f"- {status}: {n}")
    if repeated_ok:
        lines.append("- 5-fold repeated grouped retraining 达到最低门槛。")
    else:
        lines.append("- repeated grouped retraining 未达到可支持正式 Phase 5 的稳定性要求，或因 primary thresholds 未满足而未执行。")
    lines.extend(["", "## 7. GSE72056 sensitivity", ""])
    if not sens050.empty and "macro_f1" in sens050.columns:
        m = sens050.iloc[0]
        lines.append(f"- threshold 0.50: Macro-F1={float(m['macro_f1']):.4f}, Balanced accuracy={float(m['balanced_accuracy']):.4f}")
    if not senscal.empty and "macro_f1" in senscal.columns:
        m = senscal.iloc[0]
        lines.append(f"- calibrated: Macro-F1={float(m['macro_f1']):.4f}, Balanced accuracy={float(m['balanced_accuracy']):.4f}")
    lines.append("- GSE72056 是 processed/non-integer expression，只能作为 sensitivity evaluation，不是强外部验证。")
    lines.extend(["", "## 8. Phase 5 readiness", "", f"READY_FOR_PHASE5 = {ready}", ""])
    if ready == "CONDITIONAL":
        lines.extend(
            [
                "必须满足的条件：",
                "- 仅允许进入 pilot in silico deletion，不得作为正式候选靶点发现。",
                "- Phase 5 需要继续保留 sample-level held-out evaluation，并把 GSE72056 解释为 processed-expression sensitivity。",
            ]
        )
    else:
        lines.extend(
            [
                "阻断 Phase 5 的具体问题：",
                "- Primary calibrated binary A 或 repeated grouped retraining 未同时满足 Macro-F1 >= 0.60、Balanced accuracy >= 0.60、adverse_like recall >= 0.70、melanocytic_like recall >= 0.50。",
                "- 当前结果不足以支持 perturbation 或候选靶点输出。",
            ]
        )
    write_text(ROOT / "summary_phase4E_zh.md", lines)
    return ready


def main() -> int:
    ensure_dirs()
    set_seed(SEED)
    train_ds, sens_ds, train_h5, sens_h5, meta = preflight()

    y_val4d, _, prob_val4d = evaluate_checkpoint(PHASE4D_CKPT, train_ds, meta, "validation")
    y_test4d, _, prob_test4d = evaluate_checkpoint(PHASE4D_CKPT, train_ds, meta, "held_out_test")
    write_error_analysis(meta, y_test4d, prob_test4d)
    selected_phase4d_threshold = write_threshold_calibration(y_val4d, prob_val4d, "phase4D_binary_A")

    class_weighted = train_binary_model(
        train_ds,
        meta,
        CLASS_WEIGHTED_DIR,
        "phase4E_class_weighted",
        "class_weighted",
    )
    write_train_outputs(class_weighted, "phase4E_class_weighted")

    focal: TrainOutcome | None = None
    if should_run_focal(class_weighted):
        focal = train_binary_model(train_ds, meta, FOCAL_DIR, "phase4E_focal_loss", "focal_loss")
        write_train_outputs(focal, "phase4E_focal_loss")
    else:
        write_text(
            LOGS / "phase4E_focal_loss_training_log.md",
            [
                "# Phase 4E focal loss training log",
                "",
                "Focal loss not run because class-weighted calibrated melanocytic_like recall reached the 0.50 threshold or no class-weighted calibrated metrics were available.",
            ],
        )
        pd.DataFrame([{"model_source": "phase4E_focal_loss", "status": "not_run_not_needed"}]).to_csv(
            TABLES / "phase4E_focal_loss_primary_test_metrics_threshold_050.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([{"model_source": "phase4E_focal_loss", "status": "not_run_not_needed"}]).to_csv(
            TABLES / "phase4E_focal_loss_primary_test_metrics_calibrated.csv", index=False, encoding="utf-8-sig"
        )

    best = best_outcome(class_weighted, focal)
    repeated_metrics, repeated_details = run_repeated_if_needed(train_ds, meta, best)
    sens050, senscal = write_sensitivity(sens_ds, best)
    ready = write_summary(selected_phase4d_threshold, class_weighted, focal, best, repeated_metrics, sens050, senscal)

    print("PHASE4E_BINARY_A_CALIBRATION_RETRAINING: PASS")
    print(f"BEST_MODEL_SOURCE={best.model_source}")
    print(f"READY_FOR_PHASE5={ready}")
    print(f"SUMMARY={ROOT / 'summary_phase4E_zh.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
