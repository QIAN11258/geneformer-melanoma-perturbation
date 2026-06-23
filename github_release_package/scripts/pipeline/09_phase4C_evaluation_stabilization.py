from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from datasets import Dataset, load_from_disk
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModel


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_processed"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"

V2_MODEL_PATH = Path(r"models/Geneformer\Geneformer-V2-104M_CLcancer")
V2_CKPT = ROOT / "models" / "phase4B_geneformer_v2_clcancer_malignant_state_classifier" / "best_model.pt"
GSE115978_DS = DATA / "tokenized_v2_gc104M" / "GSE115978_malignant_state_labeled_v2.dataset"
GSE72056_DS = DATA / "tokenized_v2_gc104M" / "GSE72056_malignant_state_labeled_v2.dataset"
GSE115978_H5AD = DATA / "GSE115978_malignant_state_labeled.h5ad"
GSE72056_H5AD = DATA / "GSE72056_malignant_state_labeled.h5ad"

TARGET_STATES = [
    "invasive_like",
    "melanocytic_like",
    "cycling_like",
    "stress_hypoxia_like",
]
LABEL_TO_ID = {label: i for i, label in enumerate(TARGET_STATES)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
ZSCORE_COLS = {
    "invasive_like": "invasive_like_zscore",
    "melanocytic_like": "melanocytic_like_zscore",
    "cycling_like": "cycling_like_zscore",
    "stress_hypoxia_like": "stress_hypoxia_like_zscore",
}
REQUIRED_PHASE4B_FILES = [
    ROOT / "summary_phase4B_zh.md",
    TABLES / "phase4B_geneformer_v2_test_metrics.csv",
    TABLES / "phase4B_geneformer_v2_per_class_metrics.csv",
    TABLES / "phase4B_baseline_model_metrics.csv",
    TABLES / "phase4B_GSE72056_sensitivity_metrics.csv",
    TABLES / "phase4B_GSE72056_prediction_distribution.csv",
    TABLES / "phase4B_GSE115978_tokenized_metadata_with_split.csv",
    FIGURES / "phase4B_geneformer_v2_confusion_matrix.png",
]


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


class GeneformerStateClassifier(nn.Module):
    def __init__(self, model_path: Path, n_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        hidden_size = int(getattr(self.encoder.config, "hidden_size"))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = output.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_hidden))


class IndexedTokenDataset(TorchDataset):
    def __init__(self, ds: Dataset, indices: list[int]):
        self.ds = ds
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = self.indices[item]
        row = self.ds[idx]
        state = str(row["malignant_state"])
        label = LABEL_TO_ID.get(state, -1)
        return {
            "row_index": idx,
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_index": [item["row_index"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }


def load_v2_model(device: torch.device) -> GeneformerStateClassifier:
    model = GeneformerStateClassifier(V2_MODEL_PATH, len(TARGET_STATES))
    ckpt = torch.load(V2_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def supervised_indices(ds: Dataset) -> list[int]:
    states = [str(x) for x in ds["malignant_state"]]
    return [i for i, state in enumerate(states) if state in TARGET_STATES]


def predict_indices(ds: Dataset, indices: list[int], batch_size: int = 2) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_v2_model(device)
    loader = DataLoader(IndexedTokenDataset(ds, indices), batch_size=batch_size, shuffle=False, collate_fn=collate)
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            labels = batch["label"].cpu().numpy()
            for pos, row_index in enumerate(batch["row_index"]):
                row = ds[int(row_index)]
                out = {
                    "row_index": int(row_index),
                    "original_cell_id": str(row["original_cell_id"]),
                    "cell_id": str(row["cell_id"]),
                    "sample_id": str(row.get("sample_id", "not_available_in_source")),
                    "tumor_id": str(row.get("tumor_id", "not_available_in_source")),
                    "split_unit": str(row["split_unit"]),
                    "treatment.group": str(row["treatment.group"]),
                    "true_malignant_state": str(row["malignant_state"]),
                    "true_label_id": int(labels[pos]),
                    "predicted_label_id": int(preds[pos]),
                    "predicted_malignant_state": ID_TO_LABEL[int(preds[pos])],
                }
                for class_id, class_name in ID_TO_LABEL.items():
                    out[f"prob_{class_name}"] = float(probs[pos, class_id])
                rows.append(out)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    recall = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_STATES))),
        zero_division=0,
    )[1]
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        **{f"recall_{ID_TO_LABEL[i]}": float(recall[i]) for i in range(len(TARGET_STATES))},
    }


def phase4b_result_audit() -> pd.DataFrame:
    missing = [rel(path) for path in REQUIRED_PHASE4B_FILES if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing required Phase 4B output files: {missing}")

    test_metrics = pd.read_csv(TABLES / "phase4B_geneformer_v2_test_metrics.csv")
    per_class = pd.read_csv(TABLES / "phase4B_geneformer_v2_per_class_metrics.csv")
    split_meta = pd.read_csv(TABLES / "phase4B_GSE115978_tokenized_metadata_with_split.csv")
    test_meta = split_meta.loc[split_meta["split"] == "held_out_test"].copy()
    rows = [
        {
            "audit_item": "held_out_test_n",
            "value": int(test_metrics.loc[0, "n_cells"]),
            "expected_or_source": "tables/phase4B_geneformer_v2_test_metrics.csv",
            "status": "pass" if int(test_metrics.loc[0, "n_cells"]) == len(test_meta) else "fail",
            "note": f"held_out_test metadata rows={len(test_meta)}",
        },
        {
            "audit_item": "held_out_balanced_accuracy",
            "value": float(test_metrics.loc[0, "balanced_accuracy"]),
            "expected_or_source": "Phase 4B metrics",
            "status": "recorded",
            "note": "",
        },
        {
            "audit_item": "held_out_macro_f1",
            "value": float(test_metrics.loc[0, "macro_f1"]),
            "expected_or_source": "Phase 4B metrics",
            "status": "recorded",
            "note": "",
        },
    ]
    for _, row in per_class.iterrows():
        rows.append(
            {
                "audit_item": f"recall_{row['malignant_state']}",
                "value": float(row["recall"]),
                "expected_or_source": "tables/phase4B_geneformer_v2_per_class_metrics.csv",
                "status": "recorded",
                "note": f"support={int(row['support'])}",
            }
        )
    rows.append(
        {
            "audit_item": "confusion_matrix_figure",
            "value": rel(FIGURES / "phase4B_geneformer_v2_confusion_matrix.png"),
            "expected_or_source": "figures/phase4B_geneformer_v2_confusion_matrix.png",
            "status": "pass",
            "note": "Existing Phase 4B figure present; numeric confusion is regenerated in Phase 4C grouped diagnostics.",
        }
    )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase4C_phase4B_result_audit.csv", index=False, encoding="utf-8-sig")
    return df


def grouped_evaluation(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = [
        "# Phase 4C grouped evaluation log",
        "",
        "This is a frozen-model diagnostic using the Phase 4B trained V2 checkpoint.",
        "It does not retrain Geneformer across folds and must not be interpreted as repeated grouped cross-validation training.",
        "split_unit = sample_id; no cell-level random split was introduced.",
        "",
    ]
    y = pred_df["true_label_id"].to_numpy(dtype=int)
    groups = pred_df["split_unit"].astype(str).to_numpy()
    available_groups = len(set(groups))
    n_splits = 5
    seeds = [11, 42, 73] if available_groups >= 5 else [42]
    rows = []
    dist_rows = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups), start=1):
            sub = pred_df.iloc[test_idx].copy()
            metrics = metric_row(
                sub["true_label_id"].to_numpy(dtype=int),
                sub["predicted_label_id"].to_numpy(dtype=int),
            )
            dist = Counter(sub["true_malignant_state"])
            row = {
                "evaluation_type": "frozen_model_grouped_diagnostic",
                "seed": seed,
                "fold": fold,
                "test_sample_id_count": sub["split_unit"].nunique(),
                "test_cell_count": len(sub),
                "malignant_state_distribution": json.dumps(dict(sorted(dist.items())), sort_keys=True),
                **metrics,
                "retraining_performed": False,
                "interpretation": "diagnostic_only_not_retrained_cv",
            }
            rows.append(row)
            for state in TARGET_STATES:
                dist_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "malignant_state": state,
                        "n_cells": int((sub["true_malignant_state"] == state).sum()),
                        "n_sample_id": int(sub.loc[sub["true_malignant_state"] == state, "split_unit"].nunique()),
                    }
                )
            log.append(
                f"seed={seed}, fold={fold}: samples={row['test_sample_id_count']}, cells={len(sub)}, macro_f1={row['macro_f1']:.4f}, balanced_accuracy={row['balanced_accuracy']:.4f}"
            )
    plan = pd.DataFrame(rows)
    dist_df = pd.DataFrame(dist_rows)
    plan.to_csv(TABLES / "phase4C_grouped_evaluation_plan.csv", index=False, encoding="utf-8-sig")
    dist_df.to_csv(TABLES / "phase4C_grouped_eval_label_distribution.csv", index=False, encoding="utf-8-sig")
    write_text(LOGS / "phase4C_grouped_evaluation_log.md", log)
    return plan, dist_df


def state_assignment_from_threshold(obs: pd.DataFrame, top_threshold: float, margin_threshold: float) -> pd.DataFrame:
    scores = obs[[ZSCORE_COLS[state] for state in TARGET_STATES]].copy()
    score_arr = scores.to_numpy(dtype=float)
    order = np.argsort(-score_arr, axis=1)
    top_idx = order[:, 0]
    second_idx = order[:, 1]
    top_score = score_arr[np.arange(score_arr.shape[0]), top_idx]
    second_score = score_arr[np.arange(score_arr.shape[0]), second_idx]
    margin = top_score - second_score
    assigned = []
    for i, idx in enumerate(top_idx):
        if top_score[i] >= top_threshold and margin[i] >= margin_threshold:
            assigned.append(TARGET_STATES[int(idx)])
        else:
            assigned.append("intermediate/ambiguous")
    out = pd.DataFrame(
        {
            "assigned_state": assigned,
            "top_zscore": top_score,
            "margin": margin,
            "sample_id": obs["sample_id"].astype(str).values,
        },
        index=obs.index,
    )
    return out


def label_threshold_sensitivity(adata: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = ["# Phase 4C label threshold sensitivity log", ""]
    rules = [
        ("strict", 0.5, 0.25),
        ("moderate", 0.4, 0.20),
        ("relaxed", 0.3, 0.15),
    ]
    rows = []
    sample_rows = []
    obs = adata.obs.copy()
    if "sample_id" not in obs.columns:
        raise RuntimeError("GSE115978 h5ad missing sample_id.")
    for col in ZSCORE_COLS.values():
        if col not in obs.columns:
            raise RuntimeError(f"GSE115978 h5ad missing {col}.")
    for rule_name, top_thr, margin_thr in rules:
        assigned = state_assignment_from_threshold(obs, top_thr, margin_thr)
        total_labeled = int(assigned["assigned_state"].isin(TARGET_STATES).sum())
        ambiguous = int((assigned["assigned_state"] == "intermediate/ambiguous").sum())
        state_summary = []
        for state in TARGET_STATES:
            sub = assigned.loc[assigned["assigned_state"] == state].copy()
            by_sample = sub["sample_id"].value_counts()
            max_fraction = float(by_sample.iloc[0] / len(sub)) if len(sub) else np.nan
            n_sample = int(sub["sample_id"].nunique())
            dominated = bool((max_fraction >= 0.5) or (n_sample < 3)) if len(sub) else True
            class_lt30 = bool(len(sub) < 30)
            rows.append(
                {
                    "threshold_rule": rule_name,
                    "top_zscore_threshold": top_thr,
                    "margin_threshold": margin_thr,
                    "total_labeled_cells": total_labeled,
                    "ambiguous_cells": ambiguous,
                    "malignant_state": state,
                    "state_cells": len(sub),
                    "state_sample_id_count": n_sample,
                    "max_sample_fraction": max_fraction,
                    "sample_dominance": dominated,
                    "class_lt30_cells": class_lt30,
                    "suitable_for_supervised_training": bool(total_labeled >= 200 and not class_lt30 and not dominated),
                }
            )
            for sample_id, n in by_sample.items():
                sample_rows.append(
                    {
                        "threshold_rule": rule_name,
                        "malignant_state": state,
                        "sample_id": sample_id,
                        "n_cells": int(n),
                        "fraction_within_state": float(n / len(sub)) if len(sub) else np.nan,
                    }
                )
            state_summary.append(f"{state}={len(sub)} cells/{n_sample} samples")
        log.append(f"{rule_name}: labeled={total_labeled}, ambiguous={ambiguous}, " + "; ".join(state_summary))
    df = pd.DataFrame(rows)
    sample_df = pd.DataFrame(sample_rows)
    df.to_csv(TABLES / "phase4C_label_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    sample_df.to_csv(TABLES / "phase4C_state_by_sample_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    plot_label_threshold(df)
    write_text(LOGS / "phase4C_label_threshold_sensitivity_log.md", log)
    return df, sample_df


def plot_label_threshold(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="threshold_rule", columns="malignant_state", values="state_cells").loc[
        ["strict", "moderate", "relaxed"]
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(pivot))
    for state in TARGET_STATES:
        values = pivot[state].to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, label=state)
        bottom += values
    ax.set_ylabel("Labeled cells")
    ax.set_title("Label distribution by threshold rule")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4C_label_distribution_by_threshold.png", dpi=200)
    plt.close(fig)


def alternative_task_feasibility(adata: ad.AnnData) -> pd.DataFrame:
    log = ["# Phase 4C alternative task feasibility log", ""]
    obs = adata.obs.copy()
    tasks = {
        "binary_A_melanocytic_vs_adverse": {
            "mapping": {
                "melanocytic_like": "melanocytic_like",
                "invasive_like": "adverse_like",
                "cycling_like": "adverse_like",
                "stress_hypoxia_like": "adverse_like",
            },
            "priority": "high",
            "rationale": "Uses all supervised labels and avoids the weakest four-class boundary.",
        },
        "binary_B_invasive_vs_melanocytic": {
            "mapping": {
                "invasive_like": "invasive_like",
                "melanocytic_like": "melanocytic_like",
            },
            "priority": "medium",
            "rationale": "Cleaner phenotype contrast but excludes cycling/stress cells.",
        },
        "three_class_melanocytic_invasive_proliferative_stress": {
            "mapping": {
                "melanocytic_like": "melanocytic_like",
                "invasive_like": "invasive_like",
                "cycling_like": "proliferative_stress_like",
                "stress_hypoxia_like": "proliferative_stress_like",
            },
            "priority": "high",
            "rationale": "Keeps invasive and melanocytic separate while merging weak stress/cycling-like boundary.",
        },
    }
    rows = []
    for task_name, cfg in tasks.items():
        mapped = obs["malignant_state"].map(cfg["mapping"])
        sub = obs.loc[mapped.notna()].copy()
        sub["task_label"] = mapped.loc[mapped.notna()].values
        counts = Counter(sub["task_label"])
        sample_counts = {label: int(sub.loc[sub["task_label"] == label, "sample_id"].astype(str).nunique()) for label in counts}
        min_count = min(counts.values()) if counts else 0
        max_count = max(counts.values()) if counts else 0
        imbalance_ratio = float(max_count / min_count) if min_count else np.inf
        imbalance = bool(imbalance_ratio > 3 or min_count < 30)
        suitable = bool(min_count >= 30 and min(sample_counts.values()) >= 3 and imbalance_ratio <= 3)
        rows.append(
            {
                "task": task_name,
                "n_cells": int(len(sub)),
                "class_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
                "class_sample_id_counts": json.dumps(dict(sorted(sample_counts.items())), sort_keys=True),
                "min_class_cells": int(min_count),
                "imbalance_ratio_max_over_min": imbalance_ratio,
                "class_imbalance": imbalance,
                "suitable_for_geneformer_supervised_finetuning": suitable,
                "recommended_priority": cfg["priority"],
                "rationale": cfg["rationale"],
            }
        )
        log.append(f"{task_name}: n={len(sub)}, counts={dict(counts)}, sample_counts={sample_counts}, suitable={suitable}")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase4C_alternative_task_feasibility.csv", index=False, encoding="utf-8-sig")
    write_text(LOGS / "phase4C_alternative_task_log.md", log)
    return df


def stress_hypoxia_failure_analysis(adata: ad.AnnData, pred_df: pd.DataFrame) -> pd.DataFrame:
    log = ["# Phase 4C stress_hypoxia_like failure analysis log", ""]
    obs = adata.obs.copy()
    stress = obs.loc[obs["malignant_state"] == "stress_hypoxia_like"].copy()
    by_sample = stress["sample_id"].astype(str).value_counts()
    max_fraction = float(by_sample.iloc[0] / len(stress)) if len(stress) else np.nan
    sample_dominance = bool(max_fraction >= 0.5 or stress["sample_id"].astype(str).nunique() < 3) if len(stress) else True
    heldout_per_class = pd.read_csv(TABLES / "phase4B_geneformer_v2_per_class_metrics.csv")
    heldout_recall = float(
        heldout_per_class.loc[heldout_per_class["malignant_state"] == "stress_hypoxia_like", "recall"].iloc[0]
    )
    stress_pred = pred_df.loc[pred_df["true_malignant_state"] == "stress_hypoxia_like"].copy()
    pred_counts = Counter(stress_pred["predicted_malignant_state"])
    all_recall = float((stress_pred["predicted_malignant_state"] == "stress_hypoxia_like").mean()) if len(stress_pred) else np.nan
    stress_score = obs["stress_hypoxia_like_zscore"].astype(float)
    score_summary = {
        state: {
            "median": float(obs.loc[obs["malignant_state"] == state, "stress_hypoxia_like_zscore"].median()),
            "mean": float(obs.loc[obs["malignant_state"] == state, "stress_hypoxia_like_zscore"].mean()),
        }
        for state in TARGET_STATES
    }
    stress_median = score_summary["stress_hypoxia_like"]["median"]
    cycling_median = score_summary["cycling_like"]["median"]
    invasive_median = score_summary["invasive_like"]["median"]
    overlap_note = (
        "stress_hypoxia score is not cleanly separated from cycling/invasive"
        if (stress_median - max(cycling_median, invasive_median)) < 0.5
        else "stress_hypoxia score shows median separation but classifier still under-recovers class"
    )
    recommendation = (
        "merge_as_proliferative_stress_like_or_exploratory_state"
        if heldout_recall == 0.0 or all_recall < 0.40
        else "retain_with_caution"
    )
    rows = [
        {"analysis_item": "stress_hypoxia_total_cells", "value": int(len(stress)), "note": ""},
        {"analysis_item": "stress_hypoxia_sample_id_count", "value": int(stress["sample_id"].astype(str).nunique()), "note": ""},
        {"analysis_item": "max_sample_fraction", "value": max_fraction, "note": f"top_sample={by_sample.index[0] if len(by_sample) else 'NA'}"},
        {"analysis_item": "sample_dominance", "value": sample_dominance, "note": "dominance if max fraction >=0.5 or <3 samples"},
        {"analysis_item": "heldout_recall", "value": heldout_recall, "note": "Phase 4B held-out test"},
        {"analysis_item": "all_supervised_frozen_model_recall", "value": all_recall, "note": "Frozen trained model predictions on all supervised cells; diagnostic only"},
        {"analysis_item": "true_stress_prediction_distribution", "value": json.dumps(dict(sorted(pred_counts.items())), sort_keys=True), "note": "Checks confusion with other states"},
        {"analysis_item": "score_overlap_note", "value": overlap_note, "note": json.dumps(score_summary, sort_keys=True)},
        {"analysis_item": "recommendation", "value": recommendation, "note": "Do not use as independent primary class without restabilization"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase4C_stress_hypoxia_failure_analysis.csv", index=False, encoding="utf-8-sig")
    plot_stress_failure(obs, stress_pred)
    log.extend([f"{row['analysis_item']}: {row['value']} ({row['note']})" for row in rows])
    write_text(LOGS / "phase4C_stress_hypoxia_failure_analysis_log.md", log)
    return df


def plot_stress_failure(obs: pd.DataFrame, stress_pred: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    data = [obs.loc[obs["malignant_state"] == state, "stress_hypoxia_like_zscore"].astype(float).to_numpy() for state in TARGET_STATES]
    axes[0].boxplot(data, tick_labels=TARGET_STATES, showfliers=False)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("stress_hypoxia_like_zscore")
    axes[0].set_title("Stress-score overlap by true state")
    pred_counts = Counter(stress_pred["predicted_malignant_state"])
    values = [pred_counts.get(state, 0) for state in TARGET_STATES]
    axes[1].bar(TARGET_STATES, values)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylabel("True stress_hypoxia_like cells")
    axes[1].set_title("Frozen-model predictions for true stress class")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4C_stress_hypoxia_confusion_or_score_overlap.png", dpi=200)
    plt.close(fig)


def gse72056_sensitivity_limitation_summary() -> pd.DataFrame:
    log = ["# Phase 4C GSE72056 sensitivity limitation log", ""]
    log.append("GSE72056 expression matrix is processed/non-integer.")
    log.append("GSE72056 was not used as a primary training dataset.")
    log.append("GSE72056 is not interpreted as strong external validation.")
    log.append("It is only processed-expression sensitivity evaluation.")
    metrics = pd.read_csv(TABLES / "phase4B_GSE72056_sensitivity_metrics.csv")
    pred = pd.read_csv(TABLES / "phase4B_GSE72056_prediction_distribution.csv")
    supervised = pred.loc[pred["true_malignant_state"].isin(TARGET_STATES)].copy()
    y_true = supervised["true_malignant_state"].map(LABEL_TO_ID).to_numpy(dtype=int)
    y_pred = supervised["predicted_malignant_state"].map(LABEL_TO_ID).to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(TARGET_STATES))), zero_division=0
    )
    rows = [
        {
            "row_type": "summary",
            "malignant_state": "all_supervised",
            "support": int(len(supervised)),
            "precision": "",
            "recall": "",
            "f1": "",
            "balanced_accuracy": float(metrics.loc[0, "balanced_accuracy"]),
            "macro_f1": float(metrics.loc[0, "macro_f1"]),
            "prediction_count": json.dumps(dict(sorted(Counter(pred["predicted_malignant_state"]).items())), sort_keys=True),
            "processed_non_integer_expression": True,
            "training_use": "not_used_for_training",
            "interpretation": "processed-expression sensitivity only; not strong external validation",
            "domain_shift_judgment": "possible_domain_shift_or_processed_expression_limitation",
        }
    ]
    for i, state in ID_TO_LABEL.items():
        rows.append(
            {
                "row_type": "per_class",
                "malignant_state": state,
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "balanced_accuracy": "",
                "macro_f1": "",
                "prediction_count": int((pred["predicted_malignant_state"] == state).sum()),
                "processed_non_integer_expression": True,
                "training_use": "not_used_for_training",
                "interpretation": "per-class sensitivity metric",
                "domain_shift_judgment": "possible_domain_shift_or_processed_expression_limitation",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase4C_GSE72056_sensitivity_limitation_summary.csv", index=False, encoding="utf-8-sig")
    log.append(metrics.to_csv(index=False))
    log.append("Prediction distribution:")
    log.append(str(Counter(pred["predicted_malignant_state"])))
    write_text(LOGS / "phase4C_GSE72056_sensitivity_limitation_log.md", log)
    return df


def write_summary(
    audit: pd.DataFrame,
    grouped: pd.DataFrame,
    threshold_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    gse72056_df: pd.DataFrame,
) -> None:
    test_metrics = pd.read_csv(TABLES / "phase4B_geneformer_v2_test_metrics.csv")
    per_class = pd.read_csv(TABLES / "phase4B_geneformer_v2_per_class_metrics.csv")
    grouped_macro_mean = grouped["macro_f1"].mean()
    grouped_macro_min = grouped["macro_f1"].min()
    grouped_bal_mean = grouped["balanced_accuracy"].mean()
    best_alt = alt_df.sort_values(["suitable_for_geneformer_supervised_finetuning", "recommended_priority"], ascending=[False, True]).iloc[0]
    stress_reco = stress_df.loc[stress_df["analysis_item"] == "recommendation", "value"].iloc[0]
    gse_summary = gse72056_df.loc[gse72056_df["row_type"] == "summary"].iloc[0]
    lines = [
        "# Phase 4C 中文总结",
        "",
        "本阶段只做 evaluation stabilization and label-strategy refinement。未进行 in silico deletion、perturbation、候选靶点、TCGA、生存、GDSC、DepMap、ChEMBL、Open Targets 或 DEG 分析。",
        "",
        "## 1. Phase 4B 模型是否支持 Phase 5",
        "",
        f"- Held-out test n = {int(test_metrics.loc[0, 'n_cells'])}",
        f"- Held-out balanced accuracy = {float(test_metrics.loc[0, 'balanced_accuracy']):.4f}",
        f"- Held-out macro-F1 = {float(test_metrics.loc[0, 'macro_f1']):.4f}",
        "- 结论：当前四分类 V2 模型不足以直接支持 Phase 5。",
        "",
        "## 2. Grouped evaluation 是否缓解 held-out test 过小问题",
        "",
        f"- Frozen-model grouped diagnostic folds = {len(grouped)}",
        f"- Diagnostic macro-F1 mean/min = {grouped_macro_mean:.4f}/{grouped_macro_min:.4f}",
        f"- Diagnostic balanced accuracy mean = {grouped_bal_mean:.4f}",
        "- 解释：该分析按 sample_id 分组，但没有重新训练每折模型，因此只能评估预测稳定性，不能替代真正的 repeated grouped fine-tuning。",
        "",
        "## 3. 四分类任务是否仍推荐",
        "",
    ]
    for _, row in per_class.iterrows():
        lines.append(f"- {row['malignant_state']}: held-out recall = {float(row['recall']):.4f}, support = {int(row['support'])}")
    lines.extend(
        [
            "- 结论：四分类暂不推荐作为 Phase 5 primary model，因为 stress_hypoxia_like held-out recall = 0.0，且测试集过小。",
            "",
            "## 4. 二分类或三分类建议",
            "",
        ]
    )
    for _, row in alt_df.iterrows():
        lines.append(
            f"- {row['task']}: n={int(row['n_cells'])}, suitable={row['suitable_for_geneformer_supervised_finetuning']}, priority={row['recommended_priority']}, counts={row['class_counts']}"
        )
    lines.append("- 建议优先重新评估 binary_A_melanocytic_vs_adverse 或 three_class_melanocytic_invasive_proliferative_stress。")
    lines.extend(
        [
            "",
            "## 5. stress_hypoxia_like 是否保留独立 class",
            "",
            f"- 推荐：{stress_reco}",
            "- 当前不建议作为 primary supervised task 的独立 class；更合理的是合并为 proliferative_stress_like 或作为 exploratory state。",
            "",
            "## 6. GSE72056 sensitivity 解释边界",
            "",
            f"- GSE72056 macro-F1 = {float(gse_summary['macro_f1']):.4f}",
            f"- GSE72056 balanced accuracy = {float(gse_summary['balanced_accuracy']):.4f}",
            "- GSE72056 是 processed/non-integer expression，不作为主训练集，也不作为强外部验证，只作为 processed-expression sensitivity evaluation。",
            "",
            "## 7. 是否需要重新运行 Phase 4B fine-tuning",
            "",
            "- 需要。建议先决定 binary/three-class label strategy，再重新运行 sample-level grouped fine-tuning。",
            "- 不建议基于当前四分类 checkpoint 进入 perturbation。",
            "",
            "READY_FOR_PHASE5 = NO",
            "",
            "阻断 Phase 5 的具体问题：held-out test n=12 过小；stress_hypoxia_like recall=0.0；四分类 macro-F1 较低；frozen-model grouped diagnostic 不能替代 repeated grouped retraining；GSE72056 为 processed-expression sensitivity 而非强外部验证；需要先重定标签策略并重新运行 Phase 4B。",
        ]
    )
    write_text(ROOT / "summary_phase4C_zh.md", lines)


def main() -> int:
    ensure_dirs()
    random.seed(42)
    np.random.seed(42)
    log = ["# Phase 4C run log", "", f"Timestamp: {datetime.now().isoformat(timespec='seconds')}"]
    log.append("No perturbation, no deletion, no target ranking, no DEG, no TCGA/drug/dependency analyses.")
    audit = phase4b_result_audit()
    ds = load_from_disk(str(GSE115978_DS))
    pred_path = TABLES / "phase4C_GSE115978_frozen_model_supervised_predictions.csv"
    if pred_path.exists() and pred_path.stat().st_size > 0:
        pred_df = pd.read_csv(pred_path)
    else:
        pred_df = predict_indices(ds, supervised_indices(ds), batch_size=2)
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    grouped, grouped_dist = grouped_evaluation(pred_df)
    adata = ad.read_h5ad(GSE115978_H5AD)
    threshold_df, threshold_sample_df = label_threshold_sensitivity(adata)
    alt_df = alternative_task_feasibility(adata)
    stress_df = stress_hypoxia_failure_analysis(adata, pred_df)
    gse72056_df = gse72056_sensitivity_limitation_summary()
    write_summary(audit, grouped, threshold_df, alt_df, stress_df, gse72056_df)
    log.append("Phase 4C completed.")
    write_text(LOGS / "phase4C_run_log.md", log)
    print("PHASE4C_EVALUATION_STABILIZATION: PASS")
    print(f"SUMMARY={ROOT / 'summary_phase4C_zh.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
