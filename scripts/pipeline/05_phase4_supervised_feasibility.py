from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
TOKENIZED_DIR = DATA_PROCESSED / "tokenized"
TABLES = PROJECT_ROOT / "tables"
FIGURES = PROJECT_ROOT / "figures"
LOGS = PROJECT_ROOT / "logs"
MODELS = PROJECT_ROOT / "models" / "phase4_geneformer_malignant_state_classifier"

for directory in (TABLES, FIGURES, LOGS, MODELS):
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TARGET_STATES = [
    "invasive_like",
    "melanocytic_like",
    "cycling_like",
    "stress_hypoxia_like",
]
EXCLUDED_STATE = "intermediate/ambiguous"
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
SIGNATURES = {
    "invasive_like": ["AXL", "NGFR", "VIM", "FN1", "ZEB1", "TGFBI"],
    "melanocytic_like": ["MITF", "MLANA", "PMEL", "TYR", "DCT"],
    "cycling_like": ["MKI67", "TOP2A", "PCNA", "MCM2", "STMN1"],
    "stress_hypoxia_like": ["HIF1A", "VEGFA", "CA9", "LDHA"],
}


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def module_status(name: str) -> str:
    return "available" if importlib.util.find_spec(name) is not None else "missing"


def find_pretrained_model() -> str:
    env_path = os.environ.get("GENEFORMER_PRETRAINED_MODEL_PATH", "").strip()
    if env_path and Path(env_path).exists():
        return env_path
    candidate_roots = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "data_raw" / "geneformer_pretrained",
        Path.home() / ".cache" / "huggingface",
    ]
    for root in candidate_roots:
        if not root.exists():
            continue
        for config in root.rglob("config.json"):
            config_path = str(config).lower()
            if "geneformer" in config_path:
                has_weights = any(
                    sibling.exists()
                    for sibling in [
                        config.parent / "pytorch_model.bin",
                        config.parent / "model.safetensors",
                    ]
                )
                if has_weights:
                    return str(config.parent)
    return "not_available"


def load_phase_inputs() -> dict[str, Any]:
    phase35_summary = PROJECT_ROOT / "summary_phase3_5_zh.md"
    traceability_check = TABLES / "tokenized_metadata_field_check.csv"
    required_files = [
        phase35_summary,
        traceability_check,
        DATA_PROCESSED / "GSE115978_malignant_state_labeled.h5ad",
        DATA_PROCESSED / "GSE72056_malignant_state_labeled.h5ad",
        TOKENIZED_DIR / "GSE115978_malignant_state_labeled_geneformer_ready_tokens.npz",
        TOKENIZED_DIR / "GSE115978_malignant_state_labeled_tokenized_obs_metadata.csv",
        TOKENIZED_DIR / "GSE72056_malignant_state_labeled_geneformer_ready_tokens.npz",
        TOKENIZED_DIR / "GSE72056_malignant_state_labeled_tokenized_obs_metadata.csv",
    ]
    missing = [rel(path) for path in required_files if not path.exists() or path.stat().st_size == 0]
    if missing:
        LOGS.joinpath("phase4_preflight_error_log.md").write_text(
            "# Phase 4 preflight error log\n\n"
            "Phase 4 was stopped because required Phase 3.5 files were missing or empty.\n\n"
            + "\n".join(f"- {item}" for item in missing)
            + "\n",
            encoding="utf-8",
        )
        raise FileNotFoundError(f"Missing required Phase 3.5 files: {missing}")

    return {
        "phase35_summary": phase35_summary.read_text(encoding="utf-8", errors="replace"),
        "traceability_check": pd.read_csv(traceability_check),
    }


def read_dataset(stem: str) -> tuple[ad.AnnData, pd.DataFrame, np.lib.npyio.NpzFile]:
    h5ad = DATA_PROCESSED / f"{stem}.h5ad"
    metadata = TOKENIZED_DIR / f"{stem}_tokenized_obs_metadata.csv"
    tokens = TOKENIZED_DIR / f"{stem}_geneformer_ready_tokens.npz"
    adata = ad.read_h5ad(h5ad)
    meta = pd.read_csv(metadata)
    token_npz = np.load(tokens)
    return adata, meta, token_npz


def validate_metadata(adata: ad.AnnData, meta: pd.DataFrame, token_npz: np.lib.npyio.NpzFile, dataset_id: str) -> None:
    missing_meta = [field for field in REQUIRED_METADATA_FIELDS if field not in meta.columns]
    missing_obs = [field for field in REQUIRED_METADATA_FIELDS if field not in adata.obs.columns]
    problems = []
    if missing_meta:
        problems.append(f"metadata missing fields: {missing_meta}")
    if missing_obs:
        problems.append(f"h5ad obs missing fields: {missing_obs}")
    if len(meta) != adata.n_obs:
        problems.append(f"metadata rows {len(meta)} != h5ad obs {adata.n_obs}")
    if token_npz["input_ids"].shape[0] != len(meta):
        problems.append(f"token rows {token_npz['input_ids'].shape[0]} != metadata rows {len(meta)}")
    if token_npz["input_ids"].shape != token_npz["attention_mask"].shape:
        problems.append("input_ids and attention_mask shapes differ")
    if problems:
        LOGS.joinpath("phase4_preflight_error_log.md").write_text(
            "# Phase 4 preflight error log\n\n"
            f"Dataset `{dataset_id}` failed required metadata/token checks. Phase 4 stopped.\n\n"
            + "\n".join(f"- {problem}" for problem in problems)
            + "\n",
            encoding="utf-8",
        )
        raise ValueError(f"{dataset_id} failed Phase 4 preflight: {problems}")


def write_label_distribution(meta_by_dataset: dict[str, pd.DataFrame]) -> None:
    before_rows = []
    after_rows = []
    for dataset_id, meta in meta_by_dataset.items():
        split_field = "sample_id" if "sample_id" in meta.columns else "tumor_id"
        total = len(meta)
        for state, n_cells in meta["malignant_state"].value_counts(dropna=False).items():
            subset = meta.loc[meta["malignant_state"] == state]
            n_units = subset[split_field].nunique() if split_field in subset.columns else subset["split_unit"].nunique()
            before_rows.append(
                {
                    "dataset_id": dataset_id,
                    "malignant_state": state,
                    "n_cells": int(n_cells),
                    "percentage": float(n_cells / total * 100) if total else 0.0,
                    "n_split_units": int(n_units),
                    "label_use": "excluded_from_primary_training" if state == EXCLUDED_STATE else "candidate_training_label",
                    "low_support_class": bool(state != EXCLUDED_STATE and (n_cells < 50 or n_units < 3)),
                    "support_rule": "low_support_class if n_cells < 50 or n_split_units < 3",
                }
            )
        filtered = meta.loc[meta["malignant_state"].isin(TARGET_STATES)].copy()
        filtered_total = len(filtered)
        for state in TARGET_STATES:
            subset = filtered.loc[filtered["malignant_state"] == state]
            n_cells = len(subset)
            n_units = subset[split_field].nunique() if split_field in subset.columns else subset["split_unit"].nunique()
            after_rows.append(
                {
                    "dataset_id": dataset_id,
                    "malignant_state": state,
                    "n_cells": int(n_cells),
                    "percentage_after_filtering": float(n_cells / filtered_total * 100) if filtered_total else 0.0,
                    "n_split_units": int(n_units),
                    "low_support_class": bool(n_cells < 50 or n_units < 3),
                    "support_rule": "low_support_class if n_cells < 50 or n_split_units < 3",
                }
            )
    pd.DataFrame(before_rows).to_csv(TABLES / "phase4_label_distribution_before_filtering.csv", index=False)
    pd.DataFrame(after_rows).to_csv(TABLES / "phase4_label_distribution_after_filtering.csv", index=False)


def sample_level_split(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = meta.loc[meta["malignant_state"].isin(TARGET_STATES)].copy()
    unit_counts = pd.crosstab(target["sample_id"].astype(str), target["malignant_state"].astype(str))
    for state in TARGET_STATES:
        if state not in unit_counts.columns:
            unit_counts[state] = 0
    unit_counts = unit_counts[TARGET_STATES]
    units = unit_counts.index.to_numpy()
    total_counts = unit_counts.sum(axis=0).to_numpy(dtype=float)
    total_cells = float(total_counts.sum())

    # Unit-level randomized search with a fixed seed. This never splits cells from the
    # same sample_id across train/validation/test.
    rng = np.random.default_rng(RANDOM_SEED)
    n_units = len(units)
    n_test_units = max(4, round(n_units * 0.20))
    n_val_units = max(4, round(n_units * 0.20))
    n_train_units = n_units - n_test_units - n_val_units
    split_targets = {
        "train": n_train_units / n_units,
        "validation": n_val_units / n_units,
        "held_out_test": n_test_units / n_units,
    }
    best_assignment: dict[str, str] | None = None
    best_score = np.inf
    for _ in range(50000):
        perm = rng.permutation(units)
        assignment = {
            **{unit: "held_out_test" for unit in perm[:n_test_units]},
            **{unit: "validation" for unit in perm[n_test_units : n_test_units + n_val_units]},
            **{unit: "train" for unit in perm[n_test_units + n_val_units :]},
        }
        split_counts = {}
        feasible = True
        score = 0.0
        for split_name in ["train", "validation", "held_out_test"]:
            split_units = [unit for unit, split in assignment.items() if split == split_name]
            counts = unit_counts.loc[split_units].sum(axis=0).to_numpy(dtype=float)
            if (counts == 0).any():
                feasible = False
                break
            split_counts[split_name] = counts
            split_frac = counts / counts.sum()
            global_frac = total_counts / total_cells
            score += float(((split_frac - global_frac) ** 2).sum())
            score += float((counts.sum() / total_cells - split_targets[split_name]) ** 2)
        if not feasible:
            continue
        if score < best_score:
            best_score = score
            best_assignment = assignment
    if best_assignment is None:
        raise ValueError("Could not construct a sample-level split with all target classes represented.")

    annotated_target = target.copy()
    annotated_target["phase4_split"] = annotated_target["sample_id"].astype(str).map(best_assignment)
    train = annotated_target.loc[annotated_target["phase4_split"] == "train"].copy()
    val = annotated_target.loc[annotated_target["phase4_split"] == "validation"].copy()
    test = annotated_target.loc[annotated_target["phase4_split"] == "held_out_test"].copy()

    split_map = {}
    for split_name, split_df in [("train", train), ("validation", val), ("held_out_test", test)]:
        for unit in split_df["sample_id"].astype(str).unique():
            split_map[unit] = split_name

    annotated = meta.copy()
    annotated["phase4_label_use"] = np.where(
        annotated["malignant_state"].isin(TARGET_STATES), "supervised_label", "excluded_intermediate_ambiguous"
    )
    annotated["phase4_split"] = annotated["sample_id"].astype(str).map(split_map).fillna("excluded_from_supervised_training")
    annotated.loc[
        annotated["phase4_label_use"] != "supervised_label", "phase4_split"
    ] = "excluded_from_supervised_training"
    annotated.to_csv(TOKENIZED_DIR / "GSE115978_malignant_state_labeled_tokenized_obs_metadata.csv", index=False)
    annotated.to_csv(TABLES / "phase4_GSE115978_tokenized_metadata_with_split.csv", index=False)

    split_rows = []
    for split_name, split_df in [("train", train), ("validation", val), ("held_out_test", test)]:
        class_counts = split_df["malignant_state"].value_counts().to_dict()
        split_rows.append(
            {
                "split": split_name,
                "n_cells": int(len(split_df)),
                "n_split_units": int(split_df["sample_id"].nunique()),
                "split_units": ";".join(sorted(split_df["sample_id"].astype(str).unique())),
                **{f"n_{state}": int(class_counts.get(state, 0)) for state in TARGET_STATES},
            }
        )

    support_rows = []
    ct = pd.crosstab(target["sample_id"].astype(str), target["malignant_state"].astype(str))
    for state in TARGET_STATES:
        state_total = int(ct[state].sum()) if state in ct.columns else 0
        top_unit = ct[state].idxmax() if state in ct.columns and state_total else "not_available"
        top_count = int(ct[state].max()) if state in ct.columns and state_total else 0
        frac = top_count / state_total if state_total else 0.0
        support_rows.append(
            {
                "split": "class_support_risk",
                "malignant_state": state,
                "n_cells": state_total,
                "n_split_units_with_class": int((ct[state] > 0).sum()) if state in ct.columns else 0,
                "top_split_unit": top_unit,
                "top_split_unit_cells": top_count,
                "top_split_unit_fraction": frac,
                "risk_flag": "leakage_or_generalization_risk" if frac >= 0.5 else "no_single_sample_dominance_ge_0.5",
            }
        )

    pd.DataFrame(split_rows).to_csv(TABLES / "phase4_sample_level_split_summary.csv", index=False)

    dist_rows = []
    for split_name, split_df in [("train", train), ("validation", val), ("held_out_test", test)]:
        split_total = len(split_df)
        for state in TARGET_STATES:
            subset = split_df.loc[split_df["malignant_state"] == state]
            dist_rows.append(
                {
                    "split": split_name,
                    "malignant_state": state,
                    "n_cells": int(len(subset)),
                    "percentage_within_split": float(len(subset) / split_total * 100) if split_total else 0.0,
                    "n_split_units": int(subset["sample_id"].nunique()),
                }
            )
    pd.DataFrame(dist_rows).to_csv(TABLES / "phase4_train_val_test_label_distribution.csv", index=False)
    pd.DataFrame(support_rows).to_csv(TABLES / "phase4_sample_state_generalization_risk.csv", index=False)

    LOGS.joinpath("phase4_split_log.md").write_text(
        "# Phase 4 split log\n\n"
        "No cell-level random split was used. Split assignment was performed at `sample_id` level only.\n\n"
        "A fixed-seed unit-level search was used to keep all four target malignant_state classes represented "
        "in train/validation/held-out test while keeping at least four sample_id units in validation and held-out test.\n\n"
        "Intermediate/ambiguous cells were excluded from supervised training labels and retained in metadata as "
        "`excluded_intermediate_ambiguous`.\n\n"
        "Outputs:\n"
        "- `tables/phase4_sample_level_split_summary.csv`\n"
        "- `tables/phase4_train_val_test_label_distribution.csv`\n"
        "- `tables/phase4_sample_state_generalization_risk.csv`\n"
        "- `tables/phase4_GSE115978_tokenized_metadata_with_split.csv`\n",
        encoding="utf-8",
    )
    return train, val, test


def get_gene_symbols(adata: ad.AnnData) -> list[str]:
    if "gene_symbol" in adata.var.columns:
        return adata.var["gene_symbol"].astype(str).tolist()
    return [str(x) for x in adata.var_names]


def extract_marker_matrix(adata: ad.AnnData, obs_names: pd.Index) -> tuple[np.ndarray, list[str]]:
    marker_genes = []
    for genes in SIGNATURES.values():
        marker_genes.extend(genes)
    marker_genes = list(dict.fromkeys(marker_genes))

    symbols = get_gene_symbols(adata)
    symbol_to_index = {}
    for idx, symbol in enumerate(symbols):
        symbol_to_index.setdefault(symbol, idx)
    selected = [(gene, symbol_to_index[gene]) for gene in marker_genes if gene in symbol_to_index]
    if not selected:
        raise ValueError("No marker genes found in AnnData var.")
    row_positions = adata.obs_names.get_indexer(obs_names)
    col_positions = [idx for _, idx in selected]
    matrix = adata.X[row_positions, :][:, col_positions]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32), [gene for gene, _ in selected]


def subset_by_cell_ids(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df.loc[df["phase4_split"] == split].copy()


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    y_proba: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }
    if y_proba is not None and set(labels).issubset(set(y_true)):
        y_bin = label_binarize(y_true, classes=labels)
        try:
            metrics["macro_auroc_ovr"] = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
            metrics["macro_auprc"] = average_precision_score(y_bin, y_proba, average="macro")
        except Exception as exc:  # noqa: BLE001
            metrics["macro_auroc_ovr"] = "not_applicable"
            metrics["macro_auprc"] = f"not_applicable: {exc}"
    else:
        metrics["macro_auroc_ovr"] = "not_applicable"
        metrics["macro_auprc"] = "not_applicable"

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, per_class, cm


def plot_matrix_or_status(
    matrix: np.ndarray | None,
    labels: list[str],
    title: str,
    output: Path,
    status_text: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    if matrix is None:
        ax.axis("off")
        ax.text(0.5, 0.5, status_text or "not available", ha="center", va="center", wrap=True, fontsize=11)
        ax.set_title(title)
    else:
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticklabels(labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run_baselines(adata: ad.AnnData) -> None:
    annotated = pd.read_csv(TOKENIZED_DIR / "GSE115978_malignant_state_labeled_tokenized_obs_metadata.csv")
    supervised = annotated.loc[annotated["phase4_label_use"] == "supervised_label"].copy()
    supervised = supervised.set_index("cell_id", drop=False)
    labels = TARGET_STATES

    train = subset_by_cell_ids(supervised, "train")
    val = subset_by_cell_ids(supervised, "validation")
    test = subset_by_cell_ids(supervised, "held_out_test")
    train_val = pd.concat([train, val], axis=0)

    model_rows = []
    per_class_rows = []

    signature_cols = [f"{state}_score" for state in TARGET_STATES]
    missing_sig = [col for col in signature_cols if col not in adata.obs.columns]
    if missing_sig:
        raise ValueError(f"Missing signature score columns: {missing_sig}")

    obs = adata.obs.copy()
    obs.index = obs["cell_id"].astype(str)
    x_train = obs.loc[train_val["cell_id"].astype(str), signature_cols].astype(float).to_numpy()
    y_train = train_val["malignant_state"].astype(str).to_numpy()
    x_test = obs.loc[test["cell_id"].astype(str), signature_cols].astype(float).to_numpy()
    y_test = test["malignant_state"].astype(str).to_numpy()

    baseline_models: list[tuple[str, Any, np.ndarray, np.ndarray]] = [
        (
            "signature_score_logistic_regression",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            x_train,
            x_test,
        )
    ]

    marker_matrix, marker_genes = extract_marker_matrix(adata, supervised.index)
    marker_df = pd.DataFrame(marker_matrix, index=supervised.index, columns=marker_genes)
    marker_x_train = marker_df.loc[train_val["cell_id"].astype(str)].to_numpy()
    marker_x_test = marker_df.loc[test["cell_id"].astype(str)].to_numpy()
    baseline_models.append(
        (
            "marker_gene_random_forest",
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_SEED,
                class_weight="balanced_subsample",
                n_jobs=-1,
                min_samples_leaf=2,
            ),
            marker_x_train,
            marker_x_test,
        )
    )

    best_baseline_cm = None
    best_baseline_name = None
    best_macro_f1 = -1.0
    for model_name, model, model_x_train, model_x_test in baseline_models:
        model.fit(model_x_train, y_train)
        y_pred = model.predict(model_x_test)
        if hasattr(model, "predict_proba"):
            proba_raw = model.predict_proba(model_x_test)
            model_classes = list(model.classes_)
            proba = np.zeros((len(y_test), len(labels)), dtype=float)
            for col_idx, label in enumerate(labels):
                if label in model_classes:
                    proba[:, col_idx] = proba_raw[:, model_classes.index(label)]
        else:
            proba = None
        metrics, per_class, cm = multiclass_metrics(y_test, y_pred, labels, proba)
        model_rows.append(
            {
                "model": model_name,
                "dataset_id": "GSE115978_malignant_state_labeled",
                "split": "held_out_test",
                "status": "completed",
                "feature_set": "signature_scores" if "signature" in model_name else "marker_genes",
                "interpretation_warning": (
                    "high_circularity_risk_labels_defined_from_signature_scores"
                    if "signature" in model_name
                    else "lower_circularity_than_signature_scores_but_marker_based_labels_remain_related"
                ),
                "n_train_val_cells": int(len(y_train)),
                "n_test_cells": int(len(y_test)),
                **metrics,
            }
        )
        per_class.insert(0, "model", model_name)
        per_class_rows.append(per_class)
        if float(metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(metrics["macro_f1"])
            best_baseline_cm = cm
            best_baseline_name = model_name

    baseline_metrics = pd.DataFrame(model_rows)
    baseline_metrics.to_csv(TABLES / "phase4_baseline_model_metrics.csv", index=False)
    pd.concat(per_class_rows, axis=0).to_csv(TABLES / "phase4_baseline_per_class_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_df = baseline_metrics.melt(
        id_vars=["model"],
        value_vars=["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
        var_name="metric",
        value_name="value",
    )
    pivot = plot_df.pivot(index="metric", columns="model", values="value").loc[
        ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    ]
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Held-out test score")
    ax.set_title("GSE115978 sample-level held-out baseline performance")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(
        0.01,
        0.98,
        "Geneformer: blocked; signature baseline has circularity risk",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4_geneformer_vs_baseline_metrics.png", dpi=220)
    plt.close(fig)

    if best_baseline_cm is not None:
        plot_matrix_or_status(
            best_baseline_cm,
            labels,
            f"Best baseline confusion matrix: {best_baseline_name}",
            FIGURES / "phase4_best_baseline_confusion_matrix.png",
        )


def write_geneformer_blocked_outputs() -> None:
    deps = {
        "torch": module_status("torch"),
        "transformers": module_status("transformers"),
        "datasets": module_status("datasets"),
        "accelerate": module_status("accelerate"),
        "geneformer": module_status("geneformer"),
    }
    pretrained_model = find_pretrained_model()
    missing_deps = [name for name, status in deps.items() if status != "available"]
    training_status = (
        "blocked_missing_dependencies_or_pretrained_model"
        if missing_deps or pretrained_model == "not_available"
        else "ready_but_not_run_by_safety_gate"
    )
    config_rows = [
        {"parameter": "pretrained_model_name_or_path", "value": pretrained_model},
        {"parameter": "geneformer_version", "value": "not_installed" if deps["geneformer"] == "missing" else "installed"},
        {"parameter": "token_dictionary_version", "value": "gc30M"},
        {"parameter": "max_input_length", "value": "4096"},
        {"parameter": "batch_size", "value": "not_run"},
        {"parameter": "learning_rate", "value": "not_run"},
        {"parameter": "epochs", "value": "not_run"},
        {"parameter": "random_seed", "value": str(RANDOM_SEED)},
        {"parameter": "train_val_test_split_unit", "value": "sample_id"},
        {"parameter": "class_imbalance_handling", "value": "planned: class-weighted loss or weighted sampler"},
        {"parameter": "early_stopping", "value": "planned"},
        {"parameter": "training_status", "value": training_status},
        {"parameter": "missing_dependencies", "value": ";".join(missing_deps) if missing_deps else ""},
    ]
    pd.DataFrame(config_rows).to_csv(TABLES / "phase4_training_config.csv", index=False)

    status_payload = {
        "training_status": training_status,
        "missing_dependencies": missing_deps,
        "dependency_status": deps,
        "pretrained_model_name_or_path": pretrained_model,
        "token_dictionary_version": "gc30M",
        "max_input_length": 4096,
        "random_seed": RANDOM_SEED,
        "restriction_compliance": {
            "fine_tuning_completed": False,
            "in_silico_deletion": False,
            "perturbation": False,
            "candidate_target_ranking": False,
            "simulated_or_example_data_used": False,
            "cell_level_random_split_used": False,
        },
    }
    MODELS.joinpath("training_status.json").write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    MODELS.joinpath("README.md").write_text(
        "# Phase 4 Geneformer malignant-state classifier\n\n"
        "Geneformer supervised fine-tuning was not run in this execution because the local environment lacks "
        "`torch`, `transformers`, `datasets`, `accelerate`, `geneformer`, and a Geneformer pretrained model path/cache.\n\n"
        "No substitute model was used as Geneformer.\n",
        encoding="utf-8",
    )

    LOGS.joinpath("phase4_training_log.md").write_text(
        "# Phase 4 training log\n\n"
        "Geneformer supervised fine-tuning was requested, but was not run because required dependencies and "
        "a pretrained Geneformer model were not available in the current environment.\n\n"
        "Dependency status:\n"
        + "\n".join(f"- {name}: {status}" for name, status in deps.items())
        + f"\n- pretrained_model_name_or_path: {pretrained_model}\n\n"
        "No simulated data, example data, substitute transformer, in silico deletion, perturbation, "
        "candidate target ranking, DEG, survival analysis, or drug-resource analysis was performed.\n",
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "model": "Geneformer",
                "dataset_id": "GSE115978_malignant_state_labeled",
                "split": "held_out_test",
                "status": training_status,
                "accuracy": "not_available",
                "balanced_accuracy": "not_available",
                "macro_f1": "not_available",
                "weighted_f1": "not_available",
                "macro_auroc_ovr": "not_available",
                "macro_auprc": "not_available",
                "note": "Geneformer fine-tuning was blocked; no substitute model was used.",
            }
        ]
    ).to_csv(TABLES / "phase4_test_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "Geneformer",
                "class": state,
                "precision": "not_available",
                "recall": "not_available",
                "f1": "not_available",
                "support": "not_available",
                "status": training_status,
            }
            for state in TARGET_STATES
        ]
    ).to_csv(TABLES / "phase4_per_class_metrics.csv", index=False)
    plot_matrix_or_status(
        None,
        TARGET_STATES,
        "Geneformer held-out confusion matrix",
        FIGURES / "phase4_confusion_matrix.png",
        "Geneformer fine-tuning blocked: missing torch/transformers/geneformer/datasets/accelerate and pretrained model. No substitute model used.",
    )
    LOGS.joinpath("phase4_evaluation_log.md").write_text(
        "# Phase 4 evaluation log\n\n"
        "Geneformer held-out test evaluation was not run because Geneformer fine-tuning was blocked. "
        "`tables/phase4_test_metrics.csv` and `tables/phase4_per_class_metrics.csv` record this as `not_available`.\n",
        encoding="utf-8",
    )


def write_sensitivity_blocked_outputs() -> None:
    pd.DataFrame(
        [
            {
                "model": "Geneformer",
                "dataset_id": "GSE72056_malignant_state_labeled",
                "evaluation_type": "processed_expression_sensitivity_only",
                "status": "not_run_geneformer_model_unavailable",
                "accuracy": "not_available",
                "balanced_accuracy": "not_available",
                "macro_f1": "not_available",
                "weighted_f1": "not_available",
                "note": "GSE72056 is processed/likely normalized expression; sensitivity evaluation requires a trained Geneformer classifier, which is unavailable.",
            }
        ]
    ).to_csv(TABLES / "phase4_GSE72056_sensitivity_metrics.csv", index=False)
    plot_matrix_or_status(
        None,
        TARGET_STATES,
        "GSE72056 Geneformer sensitivity confusion matrix",
        FIGURES / "phase4_GSE72056_confusion_matrix.png",
        "Sensitivity evaluation not run: Geneformer classifier unavailable. GSE72056 is processed/likely normalized expression and remains sensitivity-only.",
    )
    LOGS.joinpath("phase4_GSE72056_sensitivity_log.md").write_text(
        "# Phase 4 GSE72056 sensitivity log\n\n"
        "GSE72056 was not used as a main training dataset. It remains a processed-expression sensitivity dataset.\n\n"
        "Sensitivity evaluation was not run because no trained Geneformer classifier is available from Phase 4.\n",
        encoding="utf-8",
    )


def write_summary() -> None:
    baseline = pd.read_csv(TABLES / "phase4_baseline_model_metrics.csv")
    split = pd.read_csv(TABLES / "phase4_sample_level_split_summary.csv")
    risk = pd.read_csv(TABLES / "phase4_sample_state_generalization_risk.csv")
    label_after = pd.read_csv(TABLES / "phase4_label_distribution_after_filtering.csv")
    best = baseline.sort_values("macro_f1", ascending=False).iloc[0]
    risk_flags = risk.loc[risk["risk_flag"] == "leakage_or_generalization_risk"]
    risk_text = (
        "未发现 single-sample dominance >= 0.5 的类别。"
        if risk_flags.empty
        else "存在 leakage/generalization risk: "
        + "; ".join(
            f"{row.malignant_state} top_unit={row.top_split_unit}, fraction={row.top_split_unit_fraction:.2f}"
            for row in risk_flags.itertuples()
        )
    )

    summary = f"""# Phase 4 中文总结

## 1. Phase 4 是否完成 Geneformer supervised fine-tuning

未完成。Phase 4 前置文件和 tokenized metadata 均通过检查，但当前 Python 环境缺少 `torch`、`transformers`、`datasets`、`accelerate`、`geneformer`，且未发现可用的 Geneformer pretrained model path/cache。因此，本轮未运行 Geneformer fine-tuning，也没有使用替代模型冒充 Geneformer。

## 2. 建模标签

主任务为 GSE115978 malignant_state classification。`intermediate/ambiguous` 默认排除出主监督训练标签，仅保留为 excluded/unlabeled 记录。

过滤后训练候选标签分布见 `tables/phase4_label_distribution_after_filtering.csv`：

{label_after.loc[label_after['dataset_id'].eq('GSE115978_malignant_state_labeled'), ['malignant_state', 'n_cells', 'percentage_after_filtering', 'n_split_units', 'low_support_class']].to_string(index=False)}

## 3. GSE115978 sample-level split

使用 `sample_id` 作为 split unit，未使用 cell-level random split。split 概况见 `tables/phase4_sample_level_split_summary.csv`：

{split.to_string(index=False)}

## 4. GSE115978 held-out test performance

Geneformer held-out test performance: not_available，因为 Geneformer fine-tuning 未运行。

## 5. Baseline model performance

已在同一 sample-level split 上完成两个非 Geneformer baseline：

{baseline[['model', 'accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'macro_auroc_ovr', 'macro_auprc']].to_string(index=False)}

当前最佳 baseline 按 macro-F1 为 `{best['model']}`，macro-F1 = {float(best['macro_f1']):.4f}。

注意：`signature_score_logistic_regression` 使用的 signature scores 与 malignant_state 标签定义同源，存在较高 circularity / label-definition leakage 风险，因此只能作为标签规则 sanity check，不能作为独立泛化能力证据。`marker_gene_random_forest` 也仍然使用 marker genes，和标签生成逻辑相关，解释时需要保守。

## 6. Geneformer 是否优于 baseline

无法判断。Geneformer fine-tuning 和 held-out evaluation 未运行，因此不能声称 Geneformer 优于 baseline。

## 7. GSE72056 sensitivity evaluation

未运行 Geneformer sensitivity evaluation，因为没有可用的 trained Geneformer classifier。GSE72056 仍明确标记为 processed/likely normalized expression，仅可作为 sensitivity analysis，不作为主训练数据。

## 8. Leakage/generalization risk

{risk_text}

此外，GSE115978 的 `sample_id` 是否等价于 patient-level split unit 仍为 `needs manual confirmation`。

## 9. 是否具备进入 Phase 5 in silico deletion 的条件

不具备。本阶段没有完成 Geneformer supervised fine-tuning，也没有得到 Geneformer held-out test performance，因此不能进入 Phase 5 in silico deletion。

本阶段未进行 in silico deletion、perturbation、candidate target ranking、TCGA survival analysis、GDSC/DepMap/ChEMBL/Open Targets analysis、DEG analysis，未使用 simulated data/example data，未使用 cell-level random split。

READY_FOR_PHASE5 = NO

阻断 Phase 5 的具体问题：

1. Geneformer fine-tuning 未完成，原因是缺少必要依赖和 pretrained model。
2. Geneformer held-out test performance 不可用，无法评估模型是否可靠。
3. GSE72056 sensitivity evaluation 不可用，因为没有 trained Geneformer classifier。
4. `sample_id` 与真实 patient identity 的关系仍需人工确认。
5. `MCM2` ambiguous mapping 在进入扰动/删除分析前仍需人工 resolve 或正式排除。
"""
    (PROJECT_ROOT / "summary_phase4_zh.md").write_text(summary, encoding="utf-8")


def main() -> None:
    load_phase_inputs()
    gse115_adata, gse115_meta, gse115_tokens = read_dataset("GSE115978_malignant_state_labeled")
    gse720_adata, gse720_meta, gse720_tokens = read_dataset("GSE72056_malignant_state_labeled")
    validate_metadata(gse115_adata, gse115_meta, gse115_tokens, "GSE115978_malignant_state_labeled")
    validate_metadata(gse720_adata, gse720_meta, gse720_tokens, "GSE72056_malignant_state_labeled")

    write_label_distribution(
        {
            "GSE115978_malignant_state_labeled": gse115_meta,
            "GSE72056_malignant_state_labeled": gse720_meta,
        }
    )
    sample_level_split(gse115_meta)
    run_baselines(gse115_adata)
    write_geneformer_blocked_outputs()
    write_sensitivity_blocked_outputs()
    write_summary()
    print("PHASE4_SUPERVISED_FEASIBILITY: completed_with_geneformer_blocked")


if __name__ == "__main__":
    main()
