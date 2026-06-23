from __future__ import annotations

import gc
import json
import math
import random
import shutil
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
import scipy.sparse as sp
import torch
from datasets import Dataset, load_from_disk
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModel


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_processed"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
MODELS = ROOT / "models"

V2_MODEL = Path(r"models/Geneformer\Geneformer-V2-104M_CLcancer")
GSE115978_H5AD = DATA / "GSE115978_malignant_state_labeled.h5ad"
GSE72056_H5AD = DATA / "GSE72056_malignant_state_labeled.h5ad"
GSE115978_DS = DATA / "tokenized_v2_gc104M" / "GSE115978_malignant_state_labeled_v2.dataset"
GSE72056_DS = DATA / "tokenized_v2_gc104M" / "GSE72056_malignant_state_labeled_v2.dataset"

GSE115978_H5AD_OUT = DATA / "GSE115978_malignant_state_phase4D_labeled.h5ad"
GSE72056_H5AD_OUT = DATA / "GSE72056_malignant_state_phase4D_labeled.h5ad"
GSE115978_DS_OUT = DATA / "tokenized_v2_gc104M" / "GSE115978_malignant_state_phase4D_labeled_v2.dataset"
GSE72056_DS_OUT = DATA / "tokenized_v2_gc104M" / "GSE72056_malignant_state_phase4D_labeled_v2.dataset"

TASKS = {
    "binary_A": {
        "display": "binary_A_melanocytic_vs_adverse",
        "label_column": "phase4D_binary_A_label",
        "include_column": "phase4D_supervised_include_binary_A",
        "labels": ["melanocytic_like", "adverse_like"],
        "mapping": {
            "melanocytic_like": "melanocytic_like",
            "invasive_like": "adverse_like",
            "cycling_like": "adverse_like",
            "stress_hypoxia_like": "adverse_like",
        },
        "model_dir": MODELS / "phase4D_geneformer_v2_binary_A",
        "figure": FIGURES / "phase4D_binary_A_confusion_matrix.png",
    },
    "three_class": {
        "display": "three_class_melanocytic_invasive_proliferative_stress",
        "label_column": "phase4D_three_class_label",
        "include_column": "phase4D_supervised_include_three_class",
        "labels": ["melanocytic_like", "invasive_like", "proliferative_stress_like"],
        "mapping": {
            "melanocytic_like": "melanocytic_like",
            "invasive_like": "invasive_like",
            "cycling_like": "proliferative_stress_like",
            "stress_hypoxia_like": "proliferative_stress_like",
        },
        "model_dir": MODELS / "phase4D_geneformer_v2_three_class",
        "figure": FIGURES / "phase4D_three_class_confusion_matrix.png",
    },
}
TARGET_ORIGINAL_STATES = [
    "invasive_like",
    "melanocytic_like",
    "cycling_like",
    "stress_hypoxia_like",
]
REQUIRED_METADATA = [
    "original_cell_id",
    "cell_id",
    "malignant_state",
    "split_unit",
    "split_unit_field",
    "split_unit_type",
    "patient_identity_status",
    "treatment.group",
]
SIGNATURE_SCORE_COLUMNS = [
    "invasive_like_score",
    "melanocytic_like_score",
    "cycling_like_score",
    "stress_hypoxia_like_score",
]
MARKER_GENES = {
    "invasive_like": ["AXL", "NGFR", "VIM", "FN1", "ZEB1", "TGFBI"],
    "melanocytic_like": ["MITF", "MLANA", "PMEL", "TYR", "DCT"],
    "cycling_like": ["MKI67", "TOP2A", "PCNA", "MCM2", "STMN1"],
    "stress_hypoxia_like": ["HIF1A", "VEGFA", "CA9", "LDHA"],
}

SEED = 42
EPOCHS = 3
LR = 1e-5
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM = 8
PATIENCE = 2


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, LOGS, MODELS]:
        path.mkdir(parents=True, exist_ok=True)
    for cfg in TASKS.values():
        cfg["model_dir"].mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True


def safe_replace_dir(path: Path) -> None:
    resolved = path.resolve()
    allowed = (DATA / "tokenized_v2_gc104M").resolve()
    if not str(resolved).startswith(str(allowed)):
        raise RuntimeError(f"Refusing to delete outside tokenized_v2_gc104M: {path}")
    if path.exists():
        shutil.rmtree(path)


def map_label(state: str, task_key: str) -> str:
    return TASKS[task_key]["mapping"].get(str(state), "excluded_from_supervised_training")


def preflight() -> tuple[Dataset, Dataset, ad.AnnData, ad.AnnData]:
    lines = ["# Phase 4D preflight check log", "", f"Timestamp: {datetime.now().isoformat(timespec='seconds')}"]
    required = [
        GSE115978_H5AD,
        GSE72056_H5AD,
        GSE115978_DS,
        GSE72056_DS,
        ROOT / "summary_phase4C_zh.md",
        TABLES / "phase4C_alternative_task_feasibility.csv",
        TABLES / "phase4C_stress_hypoxia_failure_analysis.csv",
        TABLES / "phase4C_GSE72056_sensitivity_limitation_summary.csv",
    ]
    missing = [rel(p) for p in required if not p.exists() or (p.is_file() and p.stat().st_size == 0)]
    if missing:
        lines.append("ERROR: missing inputs")
        lines.extend(f"- {item}" for item in missing)
        write_text(LOGS / "phase4D_preflight_check_log.md", lines)
        raise FileNotFoundError(missing)
    train_ds = load_from_disk(str(GSE115978_DS))
    sens_ds = load_from_disk(str(GSE72056_DS))
    train_h5 = ad.read_h5ad(GSE115978_H5AD)
    sens_h5 = ad.read_h5ad(GSE72056_H5AD)
    rows = []
    for dataset_id, ds, h5, ds_path, h5_path in [
        ("GSE115978", train_ds, train_h5, GSE115978_DS, GSE115978_H5AD),
        ("GSE72056", sens_ds, sens_h5, GSE72056_DS, GSE72056_H5AD),
    ]:
        missing_ds = [field for field in REQUIRED_METADATA if field not in ds.column_names]
        missing_h5 = [field for field in REQUIRED_METADATA if field not in h5.obs.columns]
        status = "pass" if len(ds) == h5.n_obs and not missing_ds and not missing_h5 else "fail"
        rows.append(
            {
                "dataset_id": dataset_id,
                "tokenized_dataset": rel(ds_path),
                "h5ad": rel(h5_path),
                "token_rows": len(ds),
                "h5ad_n_obs": h5.n_obs,
                "row_count_match": len(ds) == h5.n_obs,
                "missing_dataset_metadata_fields": ";".join(missing_ds),
                "missing_h5ad_obs_fields": ";".join(missing_h5),
                "status": status,
            }
        )
        lines.append(f"## {dataset_id}")
        lines.append(f"- token rows / h5ad n_obs: {len(ds)} / {h5.n_obs}")
        lines.append(f"- dataset metadata missing: {missing_ds if missing_ds else 'none'}")
        lines.append(f"- h5ad metadata missing: {missing_h5 if missing_h5 else 'none'}")
        lines.append(f"- status: {status}")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase4D_input_integrity_check.csv", index=False, encoding="utf-8-sig")
    write_text(LOGS / "phase4D_preflight_check_log.md", lines)
    if (df["status"] != "pass").any():
        raise RuntimeError("Phase 4D preflight failed.")
    return train_ds, sens_ds, train_h5, sens_h5


def add_phase4d_to_h5ad(adata: ad.AnnData) -> ad.AnnData:
    out = adata.copy()
    states = out.obs["malignant_state"].astype(str)
    for task_key, cfg in TASKS.items():
        labels = states.map(lambda x: map_label(x, task_key)).astype(str)
        out.obs[cfg["label_column"]] = labels.values
        out.obs[cfg["include_column"]] = labels.isin(cfg["labels"]).values
    return out


def add_phase4d_to_dataset(ds: Dataset, output_path: Path) -> Dataset:
    safe_replace_dir(output_path)

    def mapper(example):
        state = str(example["malignant_state"])
        binary = map_label(state, "binary_A")
        three = map_label(state, "three_class")
        example["phase4D_binary_A_label"] = binary
        example["phase4D_three_class_label"] = three
        example["phase4D_supervised_include_binary_A"] = binary in TASKS["binary_A"]["labels"]
        example["phase4D_supervised_include_three_class"] = three in TASKS["three_class"]["labels"]
        return example

    labelled = ds.map(mapper, num_proc=1)
    labelled.save_to_disk(str(output_path))
    return labelled


def construct_labels(train_ds: Dataset, sens_ds: Dataset, train_h5: ad.AnnData, sens_h5: ad.AnnData) -> tuple[Dataset, Dataset, ad.AnnData, ad.AnnData]:
    log = ["# Phase 4D label construction log", "", "Original malignant_state was not modified."]
    train_h5_l = add_phase4d_to_h5ad(train_h5)
    sens_h5_l = add_phase4d_to_h5ad(sens_h5)
    train_h5_l.write_h5ad(GSE115978_H5AD_OUT, compression="gzip")
    sens_h5_l.write_h5ad(GSE72056_H5AD_OUT, compression="gzip")
    train_ds_l = add_phase4d_to_dataset(train_ds, GSE115978_DS_OUT)
    sens_ds_l = add_phase4d_to_dataset(sens_ds, GSE72056_DS_OUT)
    log.append(f"wrote {rel(GSE115978_H5AD_OUT)}")
    log.append(f"wrote {rel(GSE72056_H5AD_OUT)}")
    log.append(f"wrote {rel(GSE115978_DS_OUT)}")
    log.append(f"wrote {rel(GSE72056_DS_OUT)}")
    write_text(LOGS / "phase4D_label_construction_log.md", log)
    write_label_distribution_tables(train_ds_l, sens_ds_l)
    return train_ds_l, sens_ds_l, train_h5_l, sens_h5_l


def dataset_to_meta(ds: Dataset, dataset_id: str) -> pd.DataFrame:
    rows = []
    for i in range(len(ds)):
        row = ds[i]
        out = {
            "dataset_id": dataset_id,
            "row_index": i,
            "malignant_state": str(row["malignant_state"]),
            "split_unit": str(row["split_unit"]),
            "sample_id": str(row.get("sample_id", "not_available_in_source")),
            "tumor_id": str(row.get("tumor_id", "not_available_in_source")),
            "treatment.group": str(row["treatment.group"]),
            "original_cell_id": str(row["original_cell_id"]),
            "cell_id": str(row["cell_id"]),
        }
        for task_key, cfg in TASKS.items():
            out[cfg["label_column"]] = str(row[cfg["label_column"]])
            out[cfg["include_column"]] = bool(row[cfg["include_column"]])
        rows.append(out)
    return pd.DataFrame(rows)


def write_label_distribution_tables(train_ds: Dataset, sens_ds: Dataset) -> None:
    rows_binary, rows_three, sample_rows, treatment_rows = [], [], [], []
    for dataset_id, ds in [("GSE115978", train_ds), ("GSE72056", sens_ds)]:
        meta = dataset_to_meta(ds, dataset_id)
        for task_key, rows in [("binary_A", rows_binary), ("three_class", rows_three)]:
            cfg = TASKS[task_key]
            total = len(meta)
            for label, n in Counter(meta[cfg["label_column"]]).items():
                sub = meta.loc[meta[cfg["label_column"] == label] if False else meta[cfg["label_column"]].eq(label)]
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "task": cfg["display"],
                        "label": label,
                        "n_cells": int(n),
                        "percentage": float(n / total * 100) if total else 0,
                        "n_sample_id": int(sub["split_unit"].nunique()),
                        "include_for_supervised_training": label in cfg["labels"],
                    }
                )
                for sample_id, n_sample in Counter(sub["split_unit"]).items():
                    sample_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "task": cfg["display"],
                            "label": label,
                            "sample_id_or_tumor_id": sample_id,
                            "n_cells": int(n_sample),
                        }
                    )
                for treatment, n_treat in Counter(sub["treatment.group"]).items():
                    treatment_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "task": cfg["display"],
                            "label": label,
                            "treatment.group": treatment,
                            "n_cells": int(n_treat),
                        }
                    )
    pd.DataFrame(rows_binary).to_csv(TABLES / "phase4D_binary_A_label_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_three).to_csv(TABLES / "phase4D_three_class_label_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sample_rows).to_csv(TABLES / "phase4D_label_by_sample_id_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(treatment_rows).to_csv(TABLES / "phase4D_label_by_treatment_group_distribution.csv", index=False, encoding="utf-8-sig")


def make_split_plan(meta: pd.DataFrame, task_key: str, seeds: list[int] = [42, 73, 101]) -> tuple[pd.DataFrame, dict[str, str]]:
    cfg = TASKS[task_key]
    sub = meta.loc[meta[cfg["include_column"]]].copy()
    labels = sub[cfg["label_column"]].astype(str).values
    groups = sub["split_unit"].astype(str).values
    rows = []
    primary_assignment: dict[str, str] | None = None
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (trainval_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(sub)), labels, groups), start=1):
            trainval = sub.iloc[trainval_idx].copy()
            test = sub.iloc[test_idx].copy()
            tv_labels = trainval[cfg["label_column"]].astype(str).values
            tv_groups = trainval["split_unit"].astype(str).values
            val_idx_rel = None
            try:
                inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + fold)
                _, val_idx_rel = next(inner.split(np.zeros(len(trainval)), tv_labels, tv_groups))
            except Exception:
                unique_groups = sorted(trainval["split_unit"].astype(str).unique())
                val_groups = set(unique_groups[: max(1, len(unique_groups) // 4)])
                val_idx_rel = np.where(trainval["split_unit"].astype(str).isin(val_groups))[0]
            val = trainval.iloc[val_idx_rel].copy()
            train = trainval.drop(trainval.index[val_idx_rel]).copy()
            split_parts = {"train": train, "validation": val, "test": test}
            invalid = False
            for split_name, part in split_parts.items():
                if split_name == "test":
                    invalid = invalid or (set(part[cfg["label_column"]]) != set(cfg["labels"]))
            row = {
                "task": cfg["display"],
                "seed": seed,
                "fold": fold,
                "train_sample_id_count": train["split_unit"].nunique(),
                "validation_sample_id_count": val["split_unit"].nunique(),
                "test_sample_id_count": test["split_unit"].nunique(),
                "train_cell_count": len(train),
                "validation_cell_count": len(val),
                "test_cell_count": len(test),
                "train_label_distribution": json.dumps(dict(Counter(train[cfg["label_column"]])), sort_keys=True),
                "validation_label_distribution": json.dumps(dict(Counter(val[cfg["label_column"]])), sort_keys=True),
                "test_label_distribution": json.dumps(dict(Counter(test[cfg["label_column"]])), sort_keys=True),
                "train_treatment_group_distribution": json.dumps(dict(Counter(train["treatment.group"])), sort_keys=True),
                "validation_treatment_group_distribution": json.dumps(dict(Counter(val["treatment.group"])), sort_keys=True),
                "test_treatment_group_distribution": json.dumps(dict(Counter(test["treatment.group"])), sort_keys=True),
                "fold_invalid_for_macro_metrics": invalid,
            }
            rows.append(row)
            if primary_assignment is None and not invalid and seed == 42:
                primary_assignment = {}
                for unit in train["split_unit"].astype(str).unique():
                    primary_assignment[unit] = "train"
                for unit in val["split_unit"].astype(str).unique():
                    primary_assignment[unit] = "validation"
                for unit in test["split_unit"].astype(str).unique():
                    primary_assignment[unit] = "held_out_test"
    if primary_assignment is None:
        raise RuntimeError(f"No valid primary split found for {task_key}.")
    return pd.DataFrame(rows), primary_assignment


def build_meta_for_task(ds: Dataset, task_key: str, assignment: dict[str, str]) -> pd.DataFrame:
    meta = dataset_to_meta(ds, "GSE115978")
    cfg = TASKS[task_key]
    label_to_id = {label: i for i, label in enumerate(cfg["labels"])}
    meta["phase4D_task_label"] = meta[cfg["label_column"]]
    meta["phase4D_label_id"] = meta["phase4D_task_label"].map(label_to_id).fillna(-1).astype(int)
    meta["phase4D_supervised_use"] = meta[cfg["include_column"]]
    meta["split"] = "excluded_from_supervised_training"
    mask = meta["phase4D_supervised_use"]
    meta.loc[mask, "split"] = meta.loc[mask, "split_unit"].map(assignment)
    return meta


class TokenTaskDataset(TorchDataset):
    def __init__(self, ds: Dataset, meta: pd.DataFrame, split: str):
        sub = meta.loc[(meta["phase4D_supervised_use"]) & (meta["split"] == split)].copy()
        self.indices = sub["row_index"].astype(int).tolist()
        self.labels = sub["phase4D_label_id"].astype(int).tolist()
        self.ds = ds

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.ds[self.indices[idx]]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


class GeneformerClassifier(nn.Module):
    def __init__(self, n_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(V2_MODEL), local_files_only=True)
        hidden = int(getattr(self.encoder.config, "hidden_size"))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, n_labels)

    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state[:, 0, :]))


def class_weights(meta: pd.DataFrame, split: str, n_labels: int) -> torch.Tensor:
    sub = meta.loc[(meta["phase4D_supervised_use"]) & (meta["split"] == split)]
    counts = np.array([(sub["phase4D_label_id"] == i).sum() for i in range(n_labels)], dtype=float)
    weights = counts.sum() / (n_labels * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(model: nn.Module, ds: Dataset, meta: pd.DataFrame, split: str, device: torch.device, labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(TokenTaskDataset(ds, meta, split), batch_size=1, shuffle=False, collate_fn=collate)
    y_true, probs = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
            y_true.append(batch["labels"].numpy())
    if not probs:
        return np.array([]), np.array([]), np.array([])
    prob = np.concatenate(probs, axis=0)
    true = np.concatenate(y_true, axis=0)
    pred = prob.argmax(axis=1)
    return true, pred, prob


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, labels: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    try:
        if len(labels) == 2:
            out["macro_auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
            out["macro_auprc"] = float(average_precision_score(y_true, probs[:, 1]))
        else:
            y_bin = label_binarize(y_true, classes=list(range(len(labels))))
            out["macro_auroc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
            out["macro_auprc"] = float(average_precision_score(y_bin, probs, average="macro"))
    except Exception as exc:
        out["macro_auroc"] = f"not_applicable:{type(exc).__name__}"
        out["macro_auprc"] = f"not_applicable:{type(exc).__name__}"
    return out


def per_class_rows(task_display: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], evaluation_set: str) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )
    return [
        {
            "task": task_display,
            "evaluation_set": evaluation_set,
            "label": labels[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    ]


def save_cm(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def gpu_snapshot(event: str, task: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"task": task, "event": event, "cuda_available": False}
    return {
        "task": task,
        "event": event,
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


@dataclass
class TrainResult:
    task_key: str
    status: str
    checkpoint: Path | None
    test_metrics: dict[str, Any] | None
    per_class: list[dict[str, Any]]
    y_true: np.ndarray | None
    y_pred: np.ndarray | None
    probs: np.ndarray | None
    log: list[str]
    gpu_rows: list[dict[str, Any]]
    error: str = ""


def train_task(ds: Dataset, meta: pd.DataFrame, task_key: str) -> TrainResult:
    cfg = TASKS[task_key]
    labels = cfg["labels"]
    outdir = cfg["model_dir"]
    for child in outdir.glob("*"):
        if child.is_file():
            child.unlink()
    log = [f"# Phase 4D {cfg['display']} training log", "", "GSE72056 was not used for training."]
    gpu_rows = []
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable.")
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_rows.append(gpu_snapshot("before_model_load", cfg["display"]))
        model = GeneformerClassifier(len(labels))
        model.gradient_checkpointing_enable()
        model.to(device)
        gpu_rows.append(gpu_snapshot("after_model_to_cuda", cfg["display"]))
        train_loader = DataLoader(
            TokenTaskDataset(ds, meta, "train"),
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(SEED),
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(meta, "train", len(labels)).to(device))
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
                    loss = loss_fn(logits, batch["labels"].to(device)) / GRAD_ACCUM
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach().cpu()) * GRAD_ACCUM
                if step % GRAD_ACCUM == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                steps += 1
            y_val, p_val, prob_val = evaluate_model(model, ds, meta, "validation", device, labels)
            val_metrics = metric_dict(y_val, p_val, prob_val, labels)
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
            gpu_rows.append(gpu_snapshot(f"after_epoch_{epoch}", cfg["display"]))
            if float(val_metrics["macro_f1"]) > best_f1:
                best_f1 = float(val_metrics["macro_f1"])
                best_epoch = epoch
                patience_used = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "task_key": task_key,
                        "labels": labels,
                        "best_epoch": best_epoch,
                        "validation_macro_f1": best_f1,
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
                    "task": cfg["display"],
                    "labels": labels,
                    "model_path": str(V2_MODEL),
                    "max_length": 4096,
                    "batch_size": BATCH_SIZE,
                    "gradient_accumulation": GRAD_ACCUM,
                    "epochs": EPOCHS,
                    "learning_rate": LR,
                    "mixed_precision": "fp16/AMP",
                    "class_imbalance_handling": "weighted cross entropy",
                    "seed": SEED,
                    "best_epoch": best_epoch,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        y_test, p_test, prob_test = evaluate_model(model, ds, meta, "held_out_test", device, labels)
        metrics = metric_dict(y_test, p_test, prob_test, labels)
        metrics.update(
            {
                "task": cfg["display"],
                "evaluation_set": "GSE115978_primary_held_out_test",
                "n_cells": int(len(y_test)),
                "best_epoch": int(best_epoch),
                "status": "success",
            }
        )
        per_class = per_class_rows(cfg["display"], y_test, p_test, labels, "GSE115978_primary_held_out_test")
        save_cm(y_test, p_test, labels, f"{cfg['display']} held-out test", cfg["figure"])
        del model
        torch.cuda.empty_cache()
        return TrainResult(task_key, "success", checkpoint, metrics, per_class, y_test, p_test, prob_test, log, gpu_rows)
    except Exception as exc:
        log.append(f"ERROR: {repr(exc)}")
        log.append(traceback.format_exc())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TrainResult(task_key, "failed", None, None, [], None, None, None, log, gpu_rows, repr(exc))


def write_task_outputs(result: TrainResult) -> None:
    cfg = TASKS[result.task_key]
    prefix = "phase4D_binary_A" if result.task_key == "binary_A" else "phase4D_three_class"
    write_text(LOGS / f"{prefix}_training_log.md", result.log)
    if result.test_metrics:
        pd.DataFrame([result.test_metrics]).to_csv(TABLES / f"{prefix}_primary_test_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(result.per_class).to_csv(TABLES / f"{prefix}_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([{"task": cfg["display"], "status": result.status, "error": result.error}]).to_csv(
            TABLES / f"{prefix}_primary_test_metrics.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([]).to_csv(TABLES / f"{prefix}_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    if result.task_key == "binary_A":
        pd.DataFrame(result.gpu_rows).to_csv(TABLES / "phase4D_binary_A_gpu_memory_log.csv", index=False, encoding="utf-8-sig")


def write_repeated_placeholder(task_key: str, split_plan: pd.DataFrame) -> None:
    prefix = "phase4D_binary_A" if task_key == "binary_A" else "phase4D_three_class"
    df = split_plan.copy()
    df["retraining_status"] = "not_run_resource_deferred"
    df["macro_f1"] = ""
    df["balanced_accuracy"] = ""
    df["reason"] = "Phase 4D completed primary grouped train/validation/test run only; repeated full Geneformer retraining is computationally deferred and not fabricated."
    df.to_csv(TABLES / f"{prefix}_repeated_grouped_metrics.csv", index=False, encoding="utf-8-sig")


def load_checkpoint(task_key: str, checkpoint: Path, device: torch.device) -> nn.Module:
    cfg = TASKS[task_key]
    model = GeneformerClassifier(len(cfg["labels"]))
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_task(model: nn.Module, ds: Dataset, meta: pd.DataFrame, task_key: str, split: str, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    cfg = TASKS[task_key]
    loader = DataLoader(TokenTaskDataset(ds, meta, split), batch_size=1, shuffle=False, collate_fn=collate)
    y_true, probs, rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            pred = prob.argmax(axis=1)
            probs.append(prob)
            y_true.append(batch["labels"].numpy())
    prob_arr = np.concatenate(probs, axis=0)
    true = np.concatenate(y_true, axis=0)
    pred = prob_arr.argmax(axis=1)
    return true, pred, prob_arr, pd.DataFrame()


def sensitivity_eval(task_key: str, checkpoint: Path, sens_ds: Dataset) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, np.ndarray, np.ndarray]:
    cfg = TASKS[task_key]
    meta = dataset_to_meta(sens_ds, "GSE72056")
    label_to_id = {label: i for i, label in enumerate(cfg["labels"])}
    meta["phase4D_task_label"] = meta[cfg["label_column"]]
    meta["phase4D_label_id"] = meta["phase4D_task_label"].map(label_to_id).fillna(-1).astype(int)
    meta["phase4D_supervised_use"] = meta[cfg["include_column"]]
    meta["split"] = np.where(meta["phase4D_supervised_use"], "sensitivity", "excluded")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(task_key, checkpoint, device)
    y_true, y_pred, probs = evaluate_model(model, sens_ds, meta, "sensitivity", device, cfg["labels"])
    metrics = metric_dict(y_true, y_pred, probs, cfg["labels"])
    metrics.update(
        {
            "task": cfg["display"],
            "evaluation_set": "GSE72056_processed_expression_sensitivity",
            "n_cells": int(len(y_true)),
            "processed_expression_limitation": True,
            "status": "success",
        }
    )
    per_class = per_class_rows(cfg["display"], y_true, y_pred, cfg["labels"], "GSE72056_processed_expression_sensitivity")
    pred_rows = []
    sens_indices = meta.loc[meta["phase4D_supervised_use"], "row_index"].astype(int).tolist()
    for pos, row_index in enumerate(sens_indices):
        row = sens_ds[row_index]
        out = {
            "task": cfg["display"],
            "row_index": row_index,
            "original_cell_id": row["original_cell_id"],
            "true_label": cfg["labels"][int(y_true[pos])],
            "predicted_label": cfg["labels"][int(y_pred[pos])],
        }
        for i, label in enumerate(cfg["labels"]):
            out[f"prob_{label}"] = float(probs[pos, i])
        pred_rows.append(out)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, per_class, pd.DataFrame(pred_rows), y_true, y_pred


def h5ad_features(adata: ad.AnnData, row_indices: list[int], genes: list[str]) -> tuple[np.ndarray, list[str]]:
    var_names = pd.Index(adata.var_names.astype(str))
    found = [g for g in genes if g in var_names]
    idx = [var_names.get_loc(g) for g in found]
    x = adata.X[row_indices, :][:, idx]
    if sp.issparse(x):
        x = x.toarray()
    return np.log1p(np.asarray(x, dtype=np.float32)), found


def baseline_for_task(adata: ad.AnnData, meta: pd.DataFrame, task_key: str) -> pd.DataFrame:
    cfg = TASKS[task_key]
    rows = []
    sub = meta.loc[meta["phase4D_supervised_use"]].copy()
    train = sub.loc[sub["split"] == "train"]
    test = sub.loc[sub["split"] == "held_out_test"]
    y_train = train["phase4D_label_id"].to_numpy(dtype=int)
    y_test = test["phase4D_label_id"].to_numpy(dtype=int)
    if all(col in adata.obs.columns for col in SIGNATURE_SCORE_COLUMNS):
        x_train = adata.obs.iloc[train["row_index"].tolist()][SIGNATURE_SCORE_COLUMNS].to_numpy(dtype=np.float32)
        x_test = adata.obs.iloc[test["row_index"].tolist()][SIGNATURE_SCORE_COLUMNS].to_numpy(dtype=np.float32)
        model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))])
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)
        metrics = metric_dict(y_test, pred, prob, cfg["labels"])
        metrics.update({"task": cfg["display"], "model": "signature_score_logistic_regression", "status": "success", "leakage_warning": "high circularity / label-definition leakage risk"})
        rows.append(metrics)
    marker_genes = sorted({g for vals in MARKER_GENES.values() for g in vals})
    try:
        x_all, found = h5ad_features(adata, sub["row_index"].astype(int).tolist(), marker_genes)
        train_pos = [sub.index.get_loc(idx) for idx in train.index]
        test_pos = [sub.index.get_loc(idx) for idx in test.index]
        model = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=1, min_samples_leaf=2)
        model.fit(x_all[train_pos], y_train)
        pred = model.predict(x_all[test_pos])
        prob = model.predict_proba(x_all[test_pos])
        metrics = metric_dict(y_test, pred, prob, cfg["labels"])
        metrics.update({"task": cfg["display"], "model": "marker_gene_random_forest", "status": "success", "n_marker_genes_found": len(found), "leakage_warning": "marker-rule dependency; sanity check only"})
        rows.append(metrics)
    except Exception as exc:
        rows.append({"task": cfg["display"], "model": "marker_gene_random_forest", "status": "failed", "error": repr(exc), "leakage_warning": "marker-rule dependency; sanity check only"})
    return pd.DataFrame(rows)


def plot_model_baseline(task_key: str, geneformer_metrics: dict[str, Any] | None, baseline: pd.DataFrame, path: Path) -> None:
    rows = []
    if geneformer_metrics:
        rows.append({"model": "Geneformer-V2", "macro_f1": geneformer_metrics["macro_f1"], "balanced_accuracy": geneformer_metrics["balanced_accuracy"]})
    for _, row in baseline.iterrows():
        if row["status"] == "success":
            rows.append({"model": row["model"], "macro_f1": row["macro_f1"], "balanced_accuracy": row["balanced_accuracy"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["macro_f1"], width=0.36, label="Macro-F1")
    ax.bar(x + 0.18, df["balanced_accuracy"], width=0.36, label="Balanced accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(TASKS[task_key]["display"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_grouped_split_plans(meta: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict[str, str]]]:
    log = ["# Phase 4D grouped split log", "", "split_unit = sample_id; no cell-level random split."]
    plans = {}
    for task_key in TASKS:
        plan, assignment = make_split_plan(meta, task_key)
        out = TABLES / ("phase4D_binary_A_grouped_split_plan.csv" if task_key == "binary_A" else "phase4D_three_class_grouped_split_plan.csv")
        plan.to_csv(out, index=False, encoding="utf-8-sig")
        plans[task_key] = (plan, assignment)
        invalid = int(plan["fold_invalid_for_macro_metrics"].sum())
        log.append(f"{TASKS[task_key]['display']}: folds={len(plan)}, invalid_folds={invalid}")
    write_text(LOGS / "phase4D_grouped_split_log.md", log)
    return plans


def write_sensitivity_outputs(binary_result: TrainResult, three_result: TrainResult | None, sens_ds: Dataset) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    log = ["# Phase 4D GSE72056 sensitivity log", "", "GSE72056 is processed/non-integer expression.", "Sensitivity evaluation only; not strong external validation; not used for training."]
    pred_all = []
    binary_metrics = None
    three_metrics = None
    if binary_result.status == "success" and binary_result.checkpoint is not None:
        binary_metrics, binary_pc, binary_pred, y, p = sensitivity_eval("binary_A", binary_result.checkpoint, sens_ds)
        pd.DataFrame([binary_metrics]).to_csv(TABLES / "phase4D_GSE72056_binary_A_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
        pred_all.append(binary_pred)
        save_cm(y, p, TASKS["binary_A"]["labels"], "GSE72056 binary A sensitivity", FIGURES / "phase4D_GSE72056_binary_A_confusion_matrix.png")
        log.append(f"binary_A: macro_f1={binary_metrics['macro_f1']:.4f}, balanced_accuracy={binary_metrics['balanced_accuracy']:.4f}")
    else:
        pd.DataFrame([{"task": TASKS["binary_A"]["display"], "status": "not_available_no_checkpoint"}]).to_csv(TABLES / "phase4D_GSE72056_binary_A_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
    if three_result and three_result.status == "success" and three_result.checkpoint is not None:
        three_metrics, three_pc, three_pred, y, p = sensitivity_eval("three_class", three_result.checkpoint, sens_ds)
        pd.DataFrame([three_metrics]).to_csv(TABLES / "phase4D_GSE72056_three_class_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
        pred_all.append(three_pred)
        save_cm(y, p, TASKS["three_class"]["labels"], "GSE72056 three-class sensitivity", FIGURES / "phase4D_GSE72056_three_class_confusion_matrix.png")
        log.append(f"three_class: macro_f1={three_metrics['macro_f1']:.4f}, balanced_accuracy={three_metrics['balanced_accuracy']:.4f}")
    else:
        pd.DataFrame([{"task": TASKS["three_class"]["display"], "status": "not_available_no_checkpoint"}]).to_csv(TABLES / "phase4D_GSE72056_three_class_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
    if pred_all:
        pd.concat(pred_all, ignore_index=True).to_csv(TABLES / "phase4D_GSE72056_prediction_distribution.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([]).to_csv(TABLES / "phase4D_GSE72056_prediction_distribution.csv", index=False, encoding="utf-8-sig")
    write_text(LOGS / "phase4D_GSE72056_sensitivity_log.md", log)
    return binary_metrics, three_metrics


def write_baselines(adata: ad.AnnData, binary_meta: pd.DataFrame, three_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = ["# Phase 4D baseline comparison log", "", "Baselines use the same grouped primary split.", "Signature-score logistic regression has high circularity / label-definition leakage risk.", "Marker-gene random forest has marker-rule dependency and is sanity check only."]
    binary = baseline_for_task(adata, binary_meta, "binary_A")
    three = baseline_for_task(adata, three_meta, "three_class")
    binary.to_csv(TABLES / "phase4D_binary_A_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    three.to_csv(TABLES / "phase4D_three_class_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    write_text(LOGS / "phase4D_baseline_comparison_log.md", log)
    return binary, three


def write_summary(binary_result: TrainResult, three_result: TrainResult | None, binary_base: pd.DataFrame, three_base: pd.DataFrame, binary_sens: dict[str, Any] | None, three_sens: dict[str, Any] | None) -> None:
    ready = "NO"
    blockers = []
    if binary_result.status == "success" and binary_result.test_metrics:
        bm = binary_result.test_metrics
        recalls = {row["label"]: row["recall"] for row in binary_result.per_class}
        binary_min_ok = (
            bm["macro_f1"] >= 0.60
            and bm["balanced_accuracy"] >= 0.60
            and recalls.get("adverse_like", 0) >= 0.60
            and recalls.get("melanocytic_like", 0) >= 0.60
        )
        if binary_min_ok:
            ready = "CONDITIONAL"
            blockers.append("binary A primary run met minimum internal thresholds, but repeated full grouped retraining was not completed; Phase 5 can only be considered as pilot after confirming this limitation.")
        else:
            blockers.append("binary A primary run did not meet minimum Phase 5 thresholds.")
    else:
        blockers.append("binary A Geneformer training failed or no checkpoint was produced.")
    lines = [
        "# Phase 4D 中文总结",
        "",
        "本阶段完成简化标签构建、grouped split 设计、Geneformer-V2 primary fine-tuning、baseline sanity comparison 和 GSE72056 processed-expression sensitivity evaluation。",
        "未进行 in silico deletion、perturbation、候选靶点、TCGA、生存、GDSC、DepMap、ChEMBL、Open Targets 或 DEG 分析。",
        "",
        "## 1. 标签构建",
        "",
        "- binary A 标签构建成功：melanocytic_like vs adverse_like。",
        "- three-class 标签构建成功：melanocytic_like vs invasive_like vs proliferative_stress_like。",
        "- 原始 malignant_state 未修改；intermediate/ambiguous 未强行纳入监督训练。",
        "",
        "## 2. Grouped split",
        "",
        "- split_unit = sample_id；未进行 cell-level random split。",
        "- 已为 binary A 和 three-class 输出 5 folds x 3 seeds grouped split plan。",
        "- repeated grouped retraining 未自动完成，记录为 resource-deferred，未伪造 fold 重训指标。",
        "",
        "## 3. binary A Geneformer-V2",
        "",
        f"- training status: {binary_result.status}",
    ]
    if binary_result.test_metrics:
        bm = binary_result.test_metrics
        lines.extend(
            [
                f"- held-out n = {bm['n_cells']}",
                f"- macro-F1 = {bm['macro_f1']:.4f}",
                f"- balanced accuracy = {bm['balanced_accuracy']:.4f}",
                f"- accuracy = {bm['accuracy']:.4f}",
            ]
        )
        for row in binary_result.per_class:
            lines.append(f"- {row['label']} recall = {row['recall']:.4f}, F1 = {row['f1']:.4f}, support = {row['support']}")
    lines.extend(["", "## 4. three-class Geneformer-V2", ""])
    if three_result:
        lines.append(f"- training status: {three_result.status}")
        if three_result.test_metrics:
            tm = three_result.test_metrics
            lines.extend([f"- macro-F1 = {tm['macro_f1']:.4f}", f"- balanced accuracy = {tm['balanced_accuracy']:.4f}", f"- accuracy = {tm['accuracy']:.4f}"])
            for row in three_result.per_class:
                lines.append(f"- {row['label']} recall = {row['recall']:.4f}, F1 = {row['f1']:.4f}, support = {row['support']}")
    else:
        lines.append("- three-class not run because binary A did not complete.")
    lines.extend(["", "## 5. Baseline comparison", ""])
    for name, df in [("binary A", binary_base), ("three-class", three_base)]:
        for _, row in df.iterrows():
            if row["status"] == "success":
                lines.append(f"- {name} {row['model']}: macro-F1={row['macro_f1']:.4f}, balanced accuracy={row['balanced_accuracy']:.4f}; warning={row.get('leakage_warning','')}")
            else:
                lines.append(f"- {name} {row['model']}: status={row['status']}")
    lines.extend(["", "## 6. GSE72056 sensitivity", ""])
    if binary_sens:
        lines.append(f"- binary A: macro-F1={binary_sens['macro_f1']:.4f}, balanced accuracy={binary_sens['balanced_accuracy']:.4f}")
    if three_sens:
        lines.append(f"- three-class: macro-F1={three_sens['macro_f1']:.4f}, balanced accuracy={three_sens['balanced_accuracy']:.4f}")
    lines.append("- GSE72056 是 processed/non-integer expression，只能作为 sensitivity evaluation，不是强外部验证。")
    lines.extend(["", "## 7. Phase 5 判断", "", f"READY_FOR_PHASE5 = {ready}", ""])
    if ready == "CONDITIONAL":
        lines.append("必须满足的条件：")
    else:
        lines.append("阻断 Phase 5 的具体问题：")
    lines.extend(f"- {item}" for item in blockers)
    write_text(ROOT / "summary_phase4D_zh.md", lines)


def main() -> int:
    ensure_dirs()
    set_seed(SEED)
    train_ds, sens_ds, train_h5, sens_h5 = preflight()
    train_ds_l, sens_ds_l, train_h5_l, sens_h5_l = construct_labels(train_ds, sens_ds, train_h5, sens_h5)
    base_meta = dataset_to_meta(train_ds_l, "GSE115978")
    plans = write_grouped_split_plans(base_meta)
    binary_plan, binary_assignment = plans["binary_A"]
    three_plan, three_assignment = plans["three_class"]
    binary_meta = build_meta_for_task(train_ds_l, "binary_A", binary_assignment)
    three_meta = build_meta_for_task(train_ds_l, "three_class", three_assignment)
    binary_meta.to_csv(TABLES / "phase4D_binary_A_metadata_with_split.csv", index=False, encoding="utf-8-sig")
    three_meta.to_csv(TABLES / "phase4D_three_class_metadata_with_split.csv", index=False, encoding="utf-8-sig")
    write_repeated_placeholder("binary_A", binary_plan)
    write_repeated_placeholder("three_class", three_plan)
    binary_result = train_task(train_ds_l, binary_meta, "binary_A")
    write_task_outputs(binary_result)
    three_result: TrainResult | None = None
    if binary_result.status == "success":
        three_result = train_task(train_ds_l, three_meta, "three_class")
        write_task_outputs(three_result)
    else:
        pd.DataFrame([{"task": TASKS["three_class"]["display"], "status": "not_run_binary_A_failed"}]).to_csv(TABLES / "phase4D_three_class_primary_test_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([]).to_csv(TABLES / "phase4D_three_class_per_class_metrics.csv", index=False, encoding="utf-8-sig")
        write_text(LOGS / "phase4D_three_class_training_log.md", ["# Phase 4D three-class training log", "", "Not run because binary A did not complete successfully."])
    binary_base, three_base = write_baselines(train_h5_l, binary_meta, three_meta)
    plot_model_baseline("binary_A", binary_result.test_metrics, binary_base, FIGURES / "phase4D_geneformer_vs_baseline_binary_A.png")
    if three_result:
        plot_model_baseline("three_class", three_result.test_metrics, three_base, FIGURES / "phase4D_geneformer_vs_baseline_three_class.png")
    else:
        plot_model_baseline("three_class", None, three_base, FIGURES / "phase4D_geneformer_vs_baseline_three_class.png")
    binary_sens, three_sens = write_sensitivity_outputs(binary_result, three_result, sens_ds_l)
    write_summary(binary_result, three_result, binary_base, three_base, binary_sens, three_sens)
    print("PHASE4D_RELABEL_FINE_TUNING: PASS")
    print(f"BINARY_A_STATUS={binary_result.status}")
    print(f"SUMMARY={ROOT / 'summary_phase4D_zh.md'}")
    return 0 if binary_result.status == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
