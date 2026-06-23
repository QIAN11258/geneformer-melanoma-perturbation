from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
import time
import traceback
from collections import Counter, defaultdict
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoConfig, AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
TABLES = PROJECT_ROOT / "tables"
FIGURES = PROJECT_ROOT / "figures"
LOGS = PROJECT_ROOT / "logs"
MODELS = PROJECT_ROOT / "models"

V2_MODEL_PATH = Path(r"models/Geneformer\Geneformer-V2-104M_CLcancer")
V1_MODEL_PATH = Path(r"models/Geneformer\Geneformer-V1-10M")

V2_MAIN_DS = DATA_PROCESSED / "tokenized_v2_gc104M" / "GSE115978_malignant_state_labeled_v2.dataset"
V2_SENS_DS = DATA_PROCESSED / "tokenized_v2_gc104M" / "GSE72056_malignant_state_labeled_v2.dataset"
V1_MAIN_DS = DATA_PROCESSED / "tokenized_v1_gc30M" / "GSE115978_malignant_state_labeled_v1.dataset"
V1_SENS_DS = DATA_PROCESSED / "tokenized_v1_gc30M" / "GSE72056_malignant_state_labeled_v1.dataset"

GSE115978_H5AD = DATA_PROCESSED / "GSE115978_malignant_state_labeled.h5ad"
GSE72056_H5AD = DATA_PROCESSED / "GSE72056_malignant_state_labeled.h5ad"

V2_OUTPUT_DIR = MODELS / "phase4B_geneformer_v2_clcancer_malignant_state_classifier"
V1_OUTPUT_DIR = MODELS / "phase4B_geneformer_v1_backup_malignant_state_classifier"

TARGET_STATES = [
    "invasive_like",
    "melanocytic_like",
    "cycling_like",
    "stress_hypoxia_like",
]
EXCLUDED_STATE = "intermediate/ambiguous"
LABEL_TO_ID = {label: i for i, label in enumerate(TARGET_STATES)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
REQUIRED_METADATA_FIELDS = [
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
EPOCHS = int(os.environ.get("PHASE4B_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("PHASE4B_LR", "1e-5"))
GRADIENT_ACCUMULATION = int(os.environ.get("PHASE4B_GRAD_ACCUM", "8"))
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
EARLY_STOPPING_PATIENCE = 2


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, LOGS, MODELS, V2_OUTPUT_DIR, V1_OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_log(lines: list[str], text: str = "") -> None:
    lines.append(text)


def value_counts_rows(dataset_id: str, ds: Dataset, h5ad: ad.AnnData | None, before: bool) -> list[dict[str, Any]]:
    rows = []
    states = [str(x) for x in ds["malignant_state"]]
    split_units = [str(x) for x in ds["split_unit"]]
    total = len(states)
    for state, n_cells in Counter(states).items():
        indices = [i for i, s in enumerate(states) if s == state]
        n_units = len(set(split_units[i] for i in indices))
        use = (
            "supervised_training_label"
            if state in TARGET_STATES
            else "excluded_from_supervised_training"
        )
        if before:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "malignant_state": state,
                    "n_cells": n_cells,
                    "percentage": n_cells / total * 100 if total else 0,
                    "n_split_units": n_units,
                    "label_use": use,
                    "low_support_class": bool(state in TARGET_STATES and n_cells < 30),
                    "support_rule": "low_support_class if supervised n_cells < 30",
                }
            )
    return rows


def filtered_label_rows(dataset_id: str, ds: Dataset) -> list[dict[str, Any]]:
    states = [str(x) for x in ds["malignant_state"]]
    split_units = [str(x) for x in ds["split_unit"]]
    supervised = [i for i, state in enumerate(states) if state in TARGET_STATES]
    total = len(supervised)
    rows = []
    for state in TARGET_STATES:
        indices = [i for i in supervised if states[i] == state]
        rows.append(
            {
                "dataset_id": dataset_id,
                "malignant_state": state,
                "n_cells": len(indices),
                "percentage_after_filtering": len(indices) / total * 100 if total else 0,
                "n_split_units": len(set(split_units[i] for i in indices)),
                "low_support_class": bool(len(indices) < 30),
                "support_rule": "low_support_class if supervised n_cells < 30",
            }
        )
    return rows


def load_h5ad_backed(path: Path) -> ad.AnnData:
    return ad.read_h5ad(path, backed="r")


def preflight_check() -> tuple[Dataset, Dataset, ad.AnnData, ad.AnnData]:
    log = ["# Phase 4B preflight check log", ""]
    append_log(log, f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    append_log(log, "Restrictions checked: no perturbation, no in silico deletion, no target ranking, no TCGA/drug/dependency analyses.")
    required_files = [
        V2_MAIN_DS,
        V2_SENS_DS,
        GSE115978_H5AD,
        GSE72056_H5AD,
        PROJECT_ROOT / "summary_phase4B0_zh.md",
        TABLES / "phase4B0_v2_tokenization_summary.csv",
        TABLES / "phase4B0_tokenized_metadata_traceability_check.csv",
        TABLES / "phase4B0_gpu_forward_smoke_test.csv",
    ]
    missing = [rel(p) for p in required_files if not p.exists() or (p.is_file() and p.stat().st_size == 0)]
    if missing:
        append_log(log, "ERROR: missing required input files.")
        append_log(log, "\n".join(f"- {item}" for item in missing))
        write_text(LOGS / "phase4B_preflight_check_log.md", log)
        raise FileNotFoundError(f"Missing Phase 4B inputs: {missing}")

    main_ds = load_from_disk(str(V2_MAIN_DS))
    sens_ds = load_from_disk(str(V2_SENS_DS))
    main_h5 = load_h5ad_backed(GSE115978_H5AD)
    sens_h5 = load_h5ad_backed(GSE72056_H5AD)
    checks = []
    for dataset_id, ds, h5, ds_path, h5_path in [
        ("GSE115978_malignant_state_labeled", main_ds, main_h5, V2_MAIN_DS, GSE115978_H5AD),
        ("GSE72056_malignant_state_labeled", sens_ds, sens_h5, V2_SENS_DS, GSE72056_H5AD),
    ]:
        missing_ds = [field for field in REQUIRED_METADATA_FIELDS if field not in ds.column_names]
        missing_h5 = [field for field in REQUIRED_METADATA_FIELDS if field not in h5.obs.columns]
        input_len = len(ds[0]["input_ids"]) if len(ds) else 0
        mask_len = len(ds[0]["attention_mask"]) if len(ds) else 0
        row_match = len(ds) == h5.n_obs
        status = "pass" if not missing_ds and not missing_h5 and row_match and input_len == mask_len == 4096 else "fail"
        checks.append(
            {
                "dataset_id": dataset_id,
                "tokenized_dataset": rel(ds_path),
                "h5ad": rel(h5_path),
                "token_rows": len(ds),
                "h5ad_n_obs": h5.n_obs,
                "row_count_match": row_match,
                "input_ids_length": input_len,
                "attention_mask_length": mask_len,
                "missing_dataset_metadata_fields": ";".join(missing_ds),
                "missing_h5ad_obs_fields": ";".join(missing_h5),
                "status": status,
            }
        )
        append_log(log, f"## {dataset_id}")
        append_log(log, f"- tokenized dataset: {rel(ds_path)}")
        append_log(log, f"- h5ad: {rel(h5_path)}")
        append_log(log, f"- token rows / h5ad n_obs: {len(ds)} / {h5.n_obs}")
        append_log(log, f"- required metadata in dataset: {'PASS' if not missing_ds else 'FAIL ' + str(missing_ds)}")
        append_log(log, f"- required metadata in h5ad.obs: {'PASS' if not missing_h5 else 'FAIL ' + str(missing_h5)}")
        append_log(log, f"- status: {status}")
        append_log(log, "")
    df = pd.DataFrame(checks)
    df.to_csv(TABLES / "phase4B_input_integrity_check.csv", index=False, encoding="utf-8-sig")
    write_text(LOGS / "phase4B_preflight_check_log.md", log)
    if (df["status"] != "pass").any():
        raise RuntimeError("Phase 4B preflight failed; see logs/phase4B_preflight_check_log.md.")
    return main_ds, sens_ds, main_h5, sens_h5


def prepare_supervised_dataframe(ds: Dataset) -> pd.DataFrame:
    rows = []
    for i in range(len(ds)):
        state = str(ds[i]["malignant_state"])
        rows.append(
            {
                "row_index": i,
                "malignant_state": state,
                "label_id": LABEL_TO_ID.get(state, -1),
                "supervised_use": state in TARGET_STATES,
                "split_unit": str(ds[i]["split_unit"]),
                "sample_id": str(ds[i].get("sample_id", "not_available_in_source")),
                "tumor_id": str(ds[i].get("tumor_id", "not_available_in_source")),
                "treatment.group": str(ds[i]["treatment.group"]),
                "original_cell_id": str(ds[i]["original_cell_id"]),
                "cell_id": str(ds[i]["cell_id"]),
            }
        )
    return pd.DataFrame(rows)


def write_label_distributions(main_ds: Dataset, sens_ds: Dataset) -> None:
    before_rows = []
    after_rows = []
    before_rows.extend(value_counts_rows("GSE115978_malignant_state_labeled", main_ds, None, before=True))
    before_rows.extend(value_counts_rows("GSE72056_malignant_state_labeled", sens_ds, None, before=True))
    after_rows.extend(filtered_label_rows("GSE115978_malignant_state_labeled", main_ds))
    after_rows.extend(filtered_label_rows("GSE72056_malignant_state_labeled", sens_ds))
    pd.DataFrame(before_rows).to_csv(TABLES / "phase4B_label_distribution_before_filtering.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(after_rows).to_csv(TABLES / "phase4B_label_distribution_after_filtering.csv", index=False, encoding="utf-8-sig")


def sample_level_split(meta: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    target = meta.loc[meta["supervised_use"]].copy()
    unit_counts = pd.crosstab(target["split_unit"], target["malignant_state"])
    for state in TARGET_STATES:
        if state not in unit_counts.columns:
            unit_counts[state] = 0
    unit_counts = unit_counts[TARGET_STATES]
    units = unit_counts.index.to_numpy()
    n_units = len(units)
    n_test = max(4, round(n_units * 0.20))
    n_val = max(4, round(n_units * 0.20))
    n_train = n_units - n_val - n_test
    if n_train <= 0:
        raise RuntimeError("Not enough split units for sample-level train/validation/test split.")
    total_counts = unit_counts.sum(axis=0).to_numpy(dtype=float)
    global_frac = total_counts / total_counts.sum()
    rng = np.random.default_rng(SEED)
    best_score = math.inf
    best_assignment: dict[str, str] | None = None
    risks = []
    for state in TARGET_STATES:
        present_units = int((unit_counts[state] > 0).sum())
        if present_units < 3:
            risks.append(f"{state}:generalization_risk_n_units_{present_units}")
    for _ in range(100000):
        perm = rng.permutation(units)
        assignment = {
            **{unit: "held_out_test" for unit in perm[:n_test]},
            **{unit: "validation" for unit in perm[n_test : n_test + n_val]},
            **{unit: "train" for unit in perm[n_test + n_val :]},
        }
        score = 0.0
        feasible = True
        for split_name, target_unit_n in [("train", n_train), ("validation", n_val), ("held_out_test", n_test)]:
            split_units = [unit for unit, split in assignment.items() if split == split_name]
            counts = unit_counts.loc[split_units].sum(axis=0).to_numpy(dtype=float)
            if (counts == 0).any():
                feasible = False
                break
            frac = counts / counts.sum()
            score += float(((frac - global_frac) ** 2).sum())
            score += float((len(split_units) / n_units - target_unit_n / n_units) ** 2)
        if feasible and score < best_score:
            best_score = score
            best_assignment = assignment
    if best_assignment is None:
        raise RuntimeError("Could not find sample-level split with all classes represented in each split.")
    meta = meta.copy()
    meta["split"] = "excluded_from_supervised_training"
    meta.loc[meta["supervised_use"], "split"] = meta.loc[meta["supervised_use"], "split_unit"].map(best_assignment)
    return meta, risks


def write_split_outputs(meta: pd.DataFrame, risks: list[str]) -> None:
    split_log = ["# Phase 4B sample-level split log", ""]
    append_log(split_log, "GSE115978 split uses split_unit = sample_id. No cell-level random split was introduced.")
    if risks:
        append_log(split_log, "Generalization risks:")
        split_log.extend(f"- {risk}" for risk in risks)
    else:
        append_log(split_log, "No class was restricted to fewer than three split units.")
    summary_rows = []
    label_rows = []
    treatment_rows = []
    supervised = meta.loc[meta["supervised_use"]].copy()
    for split_name in ["train", "validation", "held_out_test"]:
        sub = supervised.loc[supervised["split"] == split_name]
        summary_rows.append(
            {
                "split": split_name,
                "n_sample_id": sub["split_unit"].nunique(),
                "n_cells": len(sub),
                "malignant_state_distribution": json.dumps(Counter(sub["malignant_state"]), sort_keys=True),
                "treatment_group_distribution": json.dumps(Counter(sub["treatment.group"]), sort_keys=True),
            }
        )
        for state in TARGET_STATES:
            label_rows.append(
                {
                    "split": split_name,
                    "malignant_state": state,
                    "n_cells": int((sub["malignant_state"] == state).sum()),
                    "percentage": float((sub["malignant_state"] == state).mean() * 100) if len(sub) else 0.0,
                }
            )
        for treatment, n in Counter(sub["treatment.group"]).items():
            treatment_rows.append({"split": split_name, "treatment.group": treatment, "n_cells": n})
        append_log(split_log, f"## {split_name}")
        append_log(split_log, f"- sample_id count: {sub['split_unit'].nunique()}")
        append_log(split_log, f"- cell count: {len(sub)}")
        append_log(split_log, f"- malignant_state: {dict(Counter(sub['malignant_state']))}")
        append_log(split_log, f"- treatment.group: {dict(Counter(sub['treatment.group']))}")
    pd.DataFrame(summary_rows).to_csv(TABLES / "phase4B_sample_level_split_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(label_rows).to_csv(TABLES / "phase4B_train_val_test_label_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(treatment_rows).to_csv(TABLES / "phase4B_treatment_group_by_split.csv", index=False, encoding="utf-8-sig")
    write_text(LOGS / "phase4B_split_log.md", split_log)


class TokenRowsDataset(TorchDataset):
    def __init__(self, ds: Dataset, meta: pd.DataFrame, split_name: str):
        sub = meta.loc[(meta["supervised_use"]) & (meta["split"] == split_name)].copy()
        self.indices = sub["row_index"].astype(int).tolist()
        self.labels = sub["label_id"].astype(int).tolist()
        self.ds = ds

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row_idx = self.indices[item]
        row = self.ds[row_idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.labels[item], dtype=torch.long),
        }


class GeneformerStateClassifier(nn.Module):
    def __init__(self, model_path: Path, n_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        hidden_size = int(getattr(self.encoder.config, "hidden_size"))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, n_labels)

    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = output.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_hidden))


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def get_class_weights(meta: pd.DataFrame, split_name: str) -> torch.Tensor:
    sub = meta.loc[(meta["supervised_use"]) & (meta["split"] == split_name)]
    counts = np.array([(sub["label_id"] == i).sum() for i in range(len(TARGET_STATES))], dtype=float)
    weights = counts.sum() / (len(TARGET_STATES) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_torch_model(
    model: nn.Module,
    ds: Dataset,
    meta: pd.DataFrame,
    split_name: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eval_dataset = TokenRowsDataset(ds, meta, split_name)
    loader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
            labels.append(batch["labels"].numpy())
    if not probs:
        return np.array([]), np.array([]), np.array([])
    prob_arr = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0)
    y_pred = prob_arr.argmax(axis=1)
    return y_true, y_pred, prob_arr


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else np.nan,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if len(y_true) else np.nan,
    }
    if probs is None or len(y_true) == 0:
        result["macro_auroc"] = "not_applicable"
        result["macro_auprc"] = "not_applicable"
    else:
        try:
            if len(set(y_true.tolist())) == len(TARGET_STATES):
                y_bin = label_binarize(y_true, classes=list(range(len(TARGET_STATES))))
                result["macro_auroc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
                result["macro_auprc"] = float(average_precision_score(y_bin, probs, average="macro"))
            else:
                result["macro_auroc"] = "not_applicable"
                result["macro_auprc"] = "not_applicable"
        except Exception as exc:
            result["macro_auroc"] = f"not_applicable:{type(exc).__name__}"
            result["macro_auprc"] = f"not_applicable:{type(exc).__name__}"
    return result


def per_class_metrics_rows(model_name: str, evaluation_set: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_STATES))),
        zero_division=0,
    )
    return [
        {
            "model": model_name,
            "evaluation_set": evaluation_set,
            "malignant_state": ID_TO_LABEL[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(TARGET_STATES))
    ]


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_STATES))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(TARGET_STATES)))
    ax.set_yticks(range(len(TARGET_STATES)))
    ax.set_xticklabels(TARGET_STATES, rotation=35, ha="right")
    ax.set_yticklabels(TARGET_STATES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def gpu_memory_snapshot(event: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "event": event,
            "cuda_available": False,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
        }
    return {
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
    status: str
    model_name: str
    output_dir: Path
    best_checkpoint: Path | None
    train_log: list[str]
    memory_rows: list[dict[str, Any]]
    test_metrics: dict[str, Any] | None
    per_class_rows: list[dict[str, Any]]
    y_test: np.ndarray | None
    y_pred: np.ndarray | None
    y_prob: np.ndarray | None
    error: str = ""
    used_v1_backup: bool = False


def train_geneformer_classifier(
    model_path: Path,
    ds: Dataset,
    meta: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    max_length: int,
    is_backup: bool = False,
) -> TrainResult:
    train_log = [f"# Phase 4B {model_name} training log", ""]
    memory_rows: list[dict[str, Any]] = []
    best_checkpoint = output_dir / "best_model.pt"
    try:
        if output_dir.exists():
            for child in output_dir.iterdir():
                if child.is_file():
                    child.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable.")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda")
        append_log(train_log, f"model_path: {model_path}")
        append_log(train_log, f"max_length: {max_length}")
        append_log(train_log, f"batch_size: {TRAIN_BATCH_SIZE}")
        append_log(train_log, f"gradient_accumulation_steps: {GRADIENT_ACCUMULATION}")
        append_log(train_log, f"epochs: {EPOCHS}")
        append_log(train_log, f"learning_rate: {LEARNING_RATE}")
        append_log(train_log, "mixed_precision: fp16 autocast enabled")
        append_log(train_log, "gradient_checkpointing: enabled if supported")
        memory_rows.append({"model": model_name, "phase": "before_model_load", **gpu_memory_snapshot("before_model_load")})

        model = GeneformerStateClassifier(model_path, len(TARGET_STATES))
        model.gradient_checkpointing_enable()
        model.to(device)
        memory_rows.append({"model": model_name, "phase": "after_model_to_cuda", **gpu_memory_snapshot("after_model_to_cuda")})

        train_data = TokenRowsDataset(ds, meta, "train")
        val_data = TokenRowsDataset(ds, meta, "validation")
        test_data = TokenRowsDataset(ds, meta, "held_out_test")
        train_loader = DataLoader(
            train_data,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_batch,
            generator=torch.Generator().manual_seed(SEED),
        )
        class_weights = get_class_weights(meta, "train").to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        best_macro_f1 = -1.0
        best_epoch = -1
        epochs_without_improvement = 0
        training_history = []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            step_count = 0
            start = time.time()
            for step, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(logits, labels) / GRADIENT_ACCUMULATION
                scaler.scale(loss).backward()
                running_loss += float(loss.detach().cpu()) * GRADIENT_ACCUMULATION
                if step % GRADIENT_ACCUMULATION == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                step_count += 1
            y_val, pred_val, prob_val = evaluate_torch_model(model, ds, meta, "validation", device)
            val_metrics = metric_dict(y_val, pred_val, prob_val)
            train_loss = running_loss / max(step_count, 1)
            epoch_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "seconds": time.time() - start,
            }
            training_history.append(epoch_row)
            append_log(
                train_log,
                f"epoch {epoch}: train_loss={train_loss:.6f}, val_balanced_accuracy={val_metrics['balanced_accuracy']:.6f}, val_macro_f1={val_metrics['macro_f1']:.6f}, seconds={epoch_row['seconds']:.1f}",
            )
            memory_rows.append({"model": model_name, "phase": f"after_epoch_{epoch}", **gpu_memory_snapshot(f"after_epoch_{epoch}")})
            if float(val_metrics["macro_f1"]) > best_macro_f1:
                best_macro_f1 = float(val_metrics["macro_f1"])
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "label_to_id": LABEL_TO_ID,
                        "id_to_label": ID_TO_LABEL,
                        "model_path": str(model_path),
                        "model_name": model_name,
                        "best_epoch": best_epoch,
                        "validation_macro_f1": best_macro_f1,
                    },
                    best_checkpoint,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    append_log(train_log, f"early stopping at epoch {epoch}")
                    break

        pd.DataFrame(training_history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
        config = {
            "model_name": model_name,
            "model_path": str(model_path),
            "target_states": TARGET_STATES,
            "label_to_id": LABEL_TO_ID,
            "seed": SEED,
            "epochs_requested": EPOCHS,
            "best_epoch": best_epoch,
            "learning_rate": LEARNING_RATE,
            "batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "mixed_precision": "fp16",
            "gradient_checkpointing": True,
            "save_best_model_only": True,
            "metric_for_best_model": "validation_macro_f1",
        }
        (output_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        y_test, pred_test, prob_test = evaluate_torch_model(model, ds, meta, "held_out_test", device)
        test_metrics = metric_dict(y_test, pred_test, prob_test)
        test_metrics.update(
            {
                "model": model_name,
                "evaluation_set": "GSE115978_held_out_test",
                "n_cells": int(len(y_test)),
                "best_epoch": int(best_epoch),
                "v2_oom": False,
                "status": "success",
            }
        )
        per_class_rows = per_class_metrics_rows(model_name, "GSE115978_held_out_test", y_test, pred_test)
        save_confusion_matrix(
            y_test,
            pred_test,
            f"{model_name} held-out test",
            FIGURES / ("phase4B_geneformer_v2_confusion_matrix.png" if not is_backup else "phase4B_geneformer_v1_backup_confusion_matrix.png"),
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return TrainResult(
            status="success",
            model_name=model_name,
            output_dir=output_dir,
            best_checkpoint=best_checkpoint,
            train_log=train_log,
            memory_rows=memory_rows,
            test_metrics=test_metrics,
            per_class_rows=per_class_rows,
            y_test=y_test,
            y_pred=pred_test,
            y_prob=prob_test,
            used_v1_backup=is_backup,
        )
    except RuntimeError as exc:
        error = str(exc)
        status = "V2_OOM" if "out of memory" in error.lower() and not is_backup else "failed"
        append_log(train_log, f"ERROR: {status}: {error}")
        append_log(train_log, traceback.format_exc())
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TrainResult(
            status=status,
            model_name=model_name,
            output_dir=output_dir,
            best_checkpoint=None,
            train_log=train_log,
            memory_rows=memory_rows,
            test_metrics=None,
            per_class_rows=[],
            y_test=None,
            y_pred=None,
            y_prob=None,
            error=error,
            used_v1_backup=is_backup,
        )
    except Exception as exc:
        error = repr(exc)
        append_log(train_log, f"ERROR: failed: {error}")
        append_log(train_log, traceback.format_exc())
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TrainResult(
            status="failed",
            model_name=model_name,
            output_dir=output_dir,
            best_checkpoint=None,
            train_log=train_log,
            memory_rows=memory_rows,
            test_metrics=None,
            per_class_rows=[],
            y_test=None,
            y_pred=None,
            y_prob=None,
            error=error,
            used_v1_backup=is_backup,
        )


def load_model_for_inference(model_path: Path, checkpoint_path: Path, device: torch.device) -> GeneformerStateClassifier:
    model = GeneformerStateClassifier(model_path, len(TARGET_STATES))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_external_dataset(
    model_path: Path,
    checkpoint_path: Path,
    ds: Dataset,
    dataset_name: str,
    output_prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = prepare_supervised_dataframe(ds)
    meta["split"] = np.where(meta["supervised_use"], "sensitivity", "excluded_from_supervised_metrics")
    model = load_model_for_inference(model_path, checkpoint_path, device)
    y_true, y_pred, probs = evaluate_torch_model(model, ds, meta, "sensitivity", device)
    metrics = metric_dict(y_true, y_pred, probs)
    metrics.update(
        {
            "model": output_prefix,
            "evaluation_set": dataset_name,
            "n_cells": int(len(y_true)),
            "processed_expression_limitation": dataset_name.startswith("GSE72056"),
            "status": "success",
        }
    )
    per_class = per_class_metrics_rows(output_prefix, dataset_name, y_true, y_pred)
    prediction_rows = []
    all_probs = []
    loader_meta = meta.copy()
    # Full prediction distribution, including intermediate/ambiguous.
    full_meta = loader_meta.copy()
    full_meta["split"] = "sensitivity"
    full_meta["supervised_use"] = True
    full_meta["label_id"] = 0
    full_y, full_pred, full_prob = evaluate_torch_model(model, ds, full_meta, "sensitivity", device)
    true_states = [str(x) for x in ds["malignant_state"]]
    for i, pred_id in enumerate(full_pred):
        row = {
            "dataset_id": dataset_name,
            "row_index": i,
            "original_cell_id": str(ds[i]["original_cell_id"]),
            "true_malignant_state": true_states[i],
            "predicted_malignant_state": ID_TO_LABEL[int(pred_id)],
        }
        for class_id, class_name in ID_TO_LABEL.items():
            row[f"prob_{class_name}"] = float(full_prob[i, class_id])
        prediction_rows.append(row)
    pred_df = pd.DataFrame(prediction_rows)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, per_class, pred_df, y_true, y_pred


def h5ad_expression_features(adata: ad.AnnData, row_indices: list[int], genes: list[str]) -> tuple[np.ndarray, list[str]]:
    var_names = pd.Index(adata.var_names.astype(str))
    found = [gene for gene in genes if gene in var_names]
    if not found:
        raise RuntimeError("No marker genes found in h5ad var_names.")
    var_idx = [var_names.get_loc(gene) for gene in found]
    x = adata.X[row_indices, :][:, var_idx]
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    return np.log1p(x), found


def run_baselines(main_h5: ad.AnnData, meta: pd.DataFrame) -> pd.DataFrame:
    log = ["# Phase 4B baseline comparison log", ""]
    append_log(log, "Both baselines use the exact same sample-level split as Geneformer.")
    append_log(log, "signature-score logistic regression has high circularity / label-definition leakage risk.")
    append_log(log, "marker-gene random forest has label-rule dependency risk and is only a sanity check.")
    supervised = meta.loc[meta["supervised_use"]].copy()
    train = supervised.loc[supervised["split"] == "train"]
    test = supervised.loc[supervised["split"] == "held_out_test"]
    y_train = train["label_id"].to_numpy(dtype=int)
    y_test = test["label_id"].to_numpy(dtype=int)
    rows = []

    obs = main_h5.obs
    missing_scores = [col for col in SIGNATURE_SCORE_COLUMNS if col not in obs.columns]
    if missing_scores:
        append_log(log, f"signature-score logistic regression skipped; missing columns: {missing_scores}")
    else:
        x_train = obs.iloc[train["row_index"].tolist()][SIGNATURE_SCORE_COLUMNS].to_numpy(dtype=np.float32)
        x_test = obs.iloc[test["row_index"].tolist()][SIGNATURE_SCORE_COLUMNS].to_numpy(dtype=np.float32)
        clf = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
            ]
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        prob = clf.predict_proba(x_test)
        metrics = metric_dict(y_test, pred, prob)
        metrics.update(
            {
                "model": "signature_score_logistic_regression",
                "evaluation_set": "GSE115978_held_out_test",
                "n_cells": len(y_test),
                "leakage_warning": "high circularity / label-definition leakage risk",
                "status": "success",
            }
        )
        rows.append(metrics)

    marker_genes = sorted({gene for genes in MARKER_GENES.values() for gene in genes})
    try:
        x_all, found = h5ad_expression_features(main_h5, supervised["row_index"].tolist(), marker_genes)
        train_pos = [supervised.index.get_loc(idx) for idx in train.index]
        test_pos = [supervised.index.get_loc(idx) for idx in test.index]
        clf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=SEED,
            min_samples_leaf=2,
            n_jobs=1,
        )
        clf.fit(x_all[train_pos], y_train)
        pred = clf.predict(x_all[test_pos])
        prob = clf.predict_proba(x_all[test_pos])
        metrics = metric_dict(y_test, pred, prob)
        metrics.update(
            {
                "model": "marker_gene_random_forest",
                "evaluation_set": "GSE115978_held_out_test",
                "n_cells": len(y_test),
                "n_marker_genes_found": len(found),
                "marker_genes_found": ";".join(found),
                "leakage_warning": "label-rule dependency risk; sanity check only",
                "status": "success",
            }
        )
        rows.append(metrics)
    except Exception as exc:
        append_log(log, f"marker-gene random forest skipped/failed: {repr(exc)}")
        rows.append(
            {
                "model": "marker_gene_random_forest",
                "evaluation_set": "GSE115978_held_out_test",
                "status": "failed",
                "error": repr(exc),
                "leakage_warning": "label-rule dependency risk; sanity check only",
            }
        )
    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv(TABLES / "phase4B_baseline_model_metrics.csv", index=False, encoding="utf-8-sig")
    write_text(LOGS / "phase4B_baseline_comparison_log.md", log)
    return baseline_df


def plot_model_comparison(geneformer_metrics: dict[str, Any] | None, baseline_df: pd.DataFrame) -> None:
    rows = []
    if geneformer_metrics:
        rows.append(
            {
                "model": "Geneformer-V2",
                "balanced_accuracy": geneformer_metrics.get("balanced_accuracy", np.nan),
                "macro_f1": geneformer_metrics.get("macro_f1", np.nan),
            }
        )
    for _, row in baseline_df.iterrows():
        if row.get("status") == "success":
            rows.append(
                {
                    "model": row["model"],
                    "balanced_accuracy": row.get("balanced_accuracy", np.nan),
                    "macro_f1": row.get("macro_f1", np.nan),
                }
            )
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["balanced_accuracy"].astype(float), width, label="Balanced accuracy")
    ax.bar(x + width / 2, df["macro_f1"].astype(float), width, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Geneformer vs baseline sanity checks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4B_geneformer_vs_baseline_metrics.png", dpi=200)
    plt.close(fig)


def write_training_config_table() -> None:
    rows = [
        {
            "model": "Geneformer-V2-104M_CLcancer",
            "model_path": str(V2_MODEL_PATH),
            "max_length": 4096,
            "batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "early_stopping": True,
            "mixed_precision": "fp16/AMP",
            "seed": SEED,
            "evaluation_strategy": "per epoch",
            "save_best_model_only": True,
            "metric_for_best_model": "macro-F1",
            "class_imbalance_handling": "class-weighted cross entropy",
            "gse72056_used_for_training": False,
        }
    ]
    pd.DataFrame(rows).to_csv(TABLES / "phase4B_geneformer_v2_training_config.csv", index=False, encoding="utf-8-sig")


def write_v2_outputs(train_result: TrainResult) -> None:
    write_text(LOGS / "phase4B_geneformer_v2_training_log.md", train_result.train_log)
    pd.DataFrame(train_result.memory_rows).to_csv(TABLES / "phase4B_gpu_memory_log.csv", index=False, encoding="utf-8-sig")
    if train_result.test_metrics is not None:
        pd.DataFrame([train_result.test_metrics]).to_csv(TABLES / "phase4B_geneformer_v2_test_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(train_result.per_class_rows).to_csv(TABLES / "phase4B_geneformer_v2_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(
            [
                {
                    "model": train_result.model_name,
                    "evaluation_set": "GSE115978_held_out_test",
                    "status": train_result.status,
                    "error": train_result.error,
                }
            ]
        ).to_csv(TABLES / "phase4B_geneformer_v2_test_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([]).to_csv(TABLES / "phase4B_geneformer_v2_per_class_metrics.csv", index=False, encoding="utf-8-sig")


def write_evaluation_log(metrics: dict[str, Any] | None, per_class_rows: list[dict[str, Any]]) -> None:
    lines = ["# Phase 4B Geneformer-V2 evaluation log", ""]
    if metrics is None:
        append_log(lines, "Evaluation not available because V2 training did not produce a checkpoint.")
    else:
        append_log(lines, json.dumps(metrics, indent=2))
        append_log(lines, "")
        append_log(lines, "Per-class metrics:")
        for row in per_class_rows:
            append_log(lines, json.dumps(row, ensure_ascii=False))
    write_text(LOGS / "phase4B_geneformer_v2_evaluation_log.md", lines)


def write_sensitivity_outputs(
    metrics: dict[str, Any] | None,
    per_class_rows: list[dict[str, Any]],
    pred_df: pd.DataFrame | None,
    y_true: np.ndarray | None,
    y_pred: np.ndarray | None,
) -> None:
    lines = ["# Phase 4B GSE72056 sensitivity log", ""]
    append_log(lines, "GSE72056 is sensitivity evaluation only.")
    append_log(lines, "GSE72056 expression matrix is processed/non-integer.")
    append_log(lines, "GSE72056 was not used for training or parameter updates.")
    if metrics is None:
        append_log(lines, "Sensitivity evaluation not available because no V2 checkpoint was produced.")
        pd.DataFrame(
            [{"evaluation_set": "GSE72056_sensitivity", "status": "not_available_no_v2_checkpoint"}]
        ).to_csv(TABLES / "phase4B_GSE72056_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([]).to_csv(TABLES / "phase4B_GSE72056_prediction_distribution.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([metrics]).to_csv(TABLES / "phase4B_GSE72056_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
        pred_df.to_csv(TABLES / "phase4B_GSE72056_prediction_distribution.csv", index=False, encoding="utf-8-sig")
        save_confusion_matrix(y_true, y_pred, "GSE72056 sensitivity evaluation", FIGURES / "phase4B_GSE72056_confusion_matrix.png")
        append_log(lines, json.dumps(metrics, indent=2))
    write_text(LOGS / "phase4B_GSE72056_sensitivity_log.md", lines)


def write_backup_outputs(result: TrainResult | None) -> None:
    if result is None:
        write_text(LOGS / "phase4B_geneformer_v1_backup_training_log.md", ["# Phase 4B V1 backup training log", "", "V1 backup not run because V2 completed successfully."])
        pd.DataFrame([{"model": "Geneformer-V1-10M", "status": "not_run_v2_success"}]).to_csv(
            TABLES / "phase4B_geneformer_v1_backup_test_metrics.csv", index=False, encoding="utf-8-sig"
        )
        return
    write_text(LOGS / "phase4B_geneformer_v1_backup_training_log.md", result.train_log)
    if result.test_metrics is not None:
        pd.DataFrame([result.test_metrics]).to_csv(TABLES / "phase4B_geneformer_v1_backup_test_metrics.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([{"model": result.model_name, "status": result.status, "error": result.error}]).to_csv(
            TABLES / "phase4B_geneformer_v1_backup_test_metrics.csv", index=False, encoding="utf-8-sig"
        )


def write_summary(
    v2_result: TrainResult,
    baseline_df: pd.DataFrame,
    sensitivity_metrics: dict[str, Any] | None,
    v1_result: TrainResult | None,
    risks: list[str],
) -> None:
    geneformer_better = "not_applicable"
    if v2_result.test_metrics is not None and not baseline_df.empty:
        successful = baseline_df.loc[baseline_df["status"] == "success"].copy()
        if not successful.empty:
            gf_macro = float(v2_result.test_metrics["macro_f1"])
            best_base = float(successful["macro_f1"].astype(float).max())
            geneformer_better = "yes" if gf_macro > best_base else "no"
    if v2_result.status != "success":
        ready = "NO"
        ready_reason = "V2 supervised fine-tuning did not produce a usable checkpoint."
    elif v2_result.test_metrics and float(v2_result.test_metrics["macro_f1"]) < 0.40:
        ready = "NO"
        ready_reason = "Held-out macro-F1 is below 0.40, so model quality is not adequate for Phase 5 perturbation."
    elif sensitivity_metrics and sensitivity_metrics.get("status") == "success":
        sens_macro = float(sensitivity_metrics["macro_f1"])
        if sens_macro < 0.35:
            ready = "CONDITIONAL"
            ready_reason = "GSE72056 sensitivity macro-F1 is low, suggesting possible domain shift or processed-expression limitation."
        else:
            ready = "CONDITIONAL"
            ready_reason = "V2 model trained, but Phase 5 should wait for explicit acceptance of GSE72056 processed-expression limitation and model-vs-baseline interpretation."
    else:
        ready = "CONDITIONAL"
        ready_reason = "V2 model trained, but external sensitivity evaluation is not available."

    lines = [
        "# Phase 4B 中文总结",
        "",
        "本阶段执行 Geneformer-V2 supervised fine-tuning、GSE115978 held-out evaluation、baseline sanity comparison 和 GSE72056 sensitivity evaluation。",
        "未执行 in silico deletion、perturbation、候选靶点输出、TCGA、生存分析、GDSC、DepMap、ChEMBL、Open Targets 或 DEG 分析。",
        "",
        "## 1. Preflight",
        "",
        "Phase 4B preflight 已完成。V2 tokenized datasets 与 h5ad 行数一致，required metadata 字段完整。",
        "",
        "## 2. GSE115978 supervised labels",
        "",
    ]
    after = pd.read_csv(TABLES / "phase4B_label_distribution_after_filtering.csv")
    for _, row in after.loc[after["dataset_id"] == "GSE115978_malignant_state_labeled"].iterrows():
        lines.append(f"- {row['malignant_state']}: n_cells={int(row['n_cells'])}, n_split_units={int(row['n_split_units'])}, low_support_class={row['low_support_class']}")
    lines.extend(["", "## 3. Sample-level split", ""])
    split_summary = pd.read_csv(TABLES / "phase4B_sample_level_split_summary.csv")
    for _, row in split_summary.iterrows():
        lines.append(f"- {row['split']}: sample_id={int(row['n_sample_id'])}, cells={int(row['n_cells'])}, malignant_state={row['malignant_state_distribution']}, treatment.group={row['treatment_group_distribution']}")
    lines.append(f"- generalization_risk: {'; '.join(risks) if risks else 'not_detected_by_n_units_threshold'}")
    lines.extend(["", "## 4. Geneformer-V2 fine-tuning", ""])
    lines.append(f"- status: {v2_result.status}")
    if v2_result.test_metrics:
        lines.append(f"- best_epoch: {v2_result.test_metrics.get('best_epoch')}")
        lines.append(f"- held-out accuracy: {v2_result.test_metrics['accuracy']:.4f}")
        lines.append(f"- held-out balanced accuracy: {v2_result.test_metrics['balanced_accuracy']:.4f}")
        lines.append(f"- held-out macro-F1: {v2_result.test_metrics['macro_f1']:.4f}")
        lines.append(f"- held-out weighted-F1: {v2_result.test_metrics['weighted_f1']:.4f}")
    else:
        lines.append(f"- error: {v2_result.error}")
    lines.extend(["", "## 5. Per-class held-out performance", ""])
    for row in v2_result.per_class_rows:
        lines.append(f"- {row['malignant_state']}: precision={row['precision']:.4f}, recall={row['recall']:.4f}, F1={row['f1']:.4f}, support={row['support']}")
    lines.extend(["", "## 6. Baseline comparison", ""])
    for _, row in baseline_df.iterrows():
        if row.get("status") == "success":
            lines.append(f"- {row['model']}: balanced accuracy={float(row['balanced_accuracy']):.4f}, macro-F1={float(row['macro_f1']):.4f}, warning={row.get('leakage_warning', '')}")
        else:
            lines.append(f"- {row['model']}: status={row.get('status')}, error={row.get('error', '')}")
    lines.append(f"- Geneformer better than best baseline by macro-F1: {geneformer_better}")
    lines.extend(["", "## 7. GSE72056 sensitivity evaluation", ""])
    if sensitivity_metrics:
        lines.append(f"- status: {sensitivity_metrics.get('status')}")
        lines.append(f"- balanced accuracy: {float(sensitivity_metrics['balanced_accuracy']):.4f}")
        lines.append(f"- macro-F1: {float(sensitivity_metrics['macro_f1']):.4f}")
        lines.append("- limitation: GSE72056 expression matrix is processed/non-integer; sensitivity evaluation only, not primary training.")
    else:
        lines.append("- status: not_available")
    lines.extend(["", "## 8. Risks and limitations", ""])
    lines.append("- OOM: " + ("not_detected" if v2_result.status == "success" else v2_result.status))
    lines.append("- class imbalance: handled with class-weighted cross entropy; per-class recall must be interpreted directly.")
    lines.append("- sample leakage: no cell-level random split; split was sample-level by split_unit/sample_id.")
    lines.append("- domain shift / processed-expression limitation: present for GSE72056 sensitivity evaluation.")
    if v1_result is None:
        lines.append("- V1 backup: not run because V2 completed successfully.")
    else:
        lines.append(f"- V1 backup: status={v1_result.status}")
    lines.extend(["", f"READY_FOR_PHASE5 = {ready}", "", ready_reason])
    write_text(PROJECT_ROOT / "summary_phase4B_zh.md", lines)


def main() -> int:
    ensure_dirs()
    set_seed(SEED)
    try:
        main_ds, sens_ds, main_h5, sens_h5 = preflight_check()
        write_label_distributions(main_ds, sens_ds)
        meta = prepare_supervised_dataframe(main_ds)
        meta, risks = sample_level_split(meta)
        meta.to_csv(TABLES / "phase4B_GSE115978_tokenized_metadata_with_split.csv", index=False, encoding="utf-8-sig")
        write_split_outputs(meta, risks)
        write_training_config_table()

        v2_result = train_geneformer_classifier(
            V2_MODEL_PATH,
            main_ds,
            meta,
            V2_OUTPUT_DIR,
            "Geneformer-V2-104M_CLcancer",
            4096,
            is_backup=False,
        )
        write_v2_outputs(v2_result)
        write_evaluation_log(v2_result.test_metrics, v2_result.per_class_rows)

        v1_result: TrainResult | None = None
        if v2_result.status != "success":
            v1_ds = load_from_disk(str(V1_MAIN_DS))
            v1_meta = meta.copy()
            v1_result = train_geneformer_classifier(
                V1_MODEL_PATH,
                v1_ds,
                v1_meta,
                V1_OUTPUT_DIR,
                "Geneformer-V1-10M",
                2048,
                is_backup=True,
            )
        write_backup_outputs(v1_result)

        baseline_df = run_baselines(main_h5, meta)
        plot_model_comparison(v2_result.test_metrics, baseline_df)

        sensitivity_metrics = None
        if v2_result.status == "success" and v2_result.best_checkpoint is not None:
            sensitivity_metrics, sensitivity_per_class, pred_df, sens_y, sens_pred = evaluate_external_dataset(
                V2_MODEL_PATH,
                v2_result.best_checkpoint,
                sens_ds,
                "GSE72056_sensitivity",
                "Geneformer-V2-104M_CLcancer",
            )
            write_sensitivity_outputs(sensitivity_metrics, sensitivity_per_class, pred_df, sens_y, sens_pred)
        else:
            write_sensitivity_outputs(None, [], None, None, None)

        write_summary(v2_result, baseline_df, sensitivity_metrics, v1_result, risks)
        print("PHASE4B_SUPERVISED_FINETUNING: PASS" if v2_result.status == "success" else "PHASE4B_SUPERVISED_FINETUNING: V2_FAILED")
        print(f"V2_STATUS={v2_result.status}")
        print(f"SUMMARY={PROJECT_ROOT / 'summary_phase4B_zh.md'}")
        return 0 if v2_result.status == "success" else 2
    except Exception as exc:
        error_log = [
            "# Phase 4B fatal error log",
            "",
            f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
            repr(exc),
            "",
            traceback.format_exc(),
        ]
        write_text(LOGS / "phase4B_fatal_error_log.md", error_log)
        summary = [
            "# Phase 4B 中文总结",
            "",
            "Phase 4B failed before producing a complete supervised Geneformer evaluation.",
            "",
            f"Error: {repr(exc)}",
            "",
            "READY_FOR_PHASE5 = NO",
            "",
            "阻断 Phase 5 的具体问题：Phase 4B supervised fine-tuning/evaluation 未完成。",
        ]
        write_text(PROJECT_ROOT / "summary_phase4B_zh.md", summary)
        print("PHASE4B_SUPERVISED_FINETUNING: FAIL")
        print(repr(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
