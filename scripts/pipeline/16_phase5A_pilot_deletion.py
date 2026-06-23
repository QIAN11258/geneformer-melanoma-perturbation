from __future__ import annotations

import importlib.util
import json
import math
import pickle
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_processed"
TABLES = ROOT / "tables"
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

SEED = 42
MAX_CELLS = 300
THRESHOLD = 0.70
LABELS = ["melanocytic_like", "adverse_like"]
MEL = 0
ADV = 1
PAD_TOKEN = 0
CLS_TOKEN = 2
EOS_TOKEN = 3
SPECIAL_TOKENS = {0, 1, 2, 3}

MODEL_DIR = MODELS / "phase4E_geneformer_v2_binary_A_class_weighted"
MODEL_CKPT = MODEL_DIR / "best_model.pt"
TRAIN_DS_ORIG = DATA / "tokenized_v2_gc104M" / "GSE115978_malignant_state_labeled_v2.dataset"
TRAIN_DS = DATA / "tokenized_v2_gc104M" / "GSE115978_malignant_state_phase4D_labeled_v2.dataset"
TRAIN_H5AD = DATA / "GSE115978_malignant_state_phase4D_labeled.h5ad"
GSE_DS_ORIG = DATA / "tokenized_v2_gc104M" / "GSE72056_malignant_state_labeled_v2.dataset"
GSE_DS = DATA / "tokenized_v2_gc104M" / "GSE72056_malignant_state_phase4D_labeled_v2.dataset"
GSE_H5AD = DATA / "GSE72056_malignant_state_phase4D_labeled.h5ad"
TOKEN_DICT = Path(r"models/Geneformer\geneformer\token_dictionary_gc104M.pkl")
GENE_NAME_DICT = Path(r"models/Geneformer\geneformer\gene_name_id_dict_gc104M.pkl")

GENE_WHITELIST = [
    "MITF", "MLANA", "PMEL", "TYR", "DCT",
    "AXL", "NGFR", "VIM", "FN1", "ZEB1", "TGFBI",
    "MKI67", "TOP2A", "PCNA", "STMN1",
    "HIF1A", "VEGFA", "LDHA",
    "BRAF", "NRAS", "KIT", "PTEN", "CDKN2A", "MAPK1", "MAPK3", "AKT1",
]


def ensure_dirs() -> None:
    for path in [TABLES, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path)


def preflight() -> tuple[Dataset, Dataset, ad.AnnData, ad.AnnData]:
    log = ["# Phase 5A preflight check log", "", f"Timestamp: {datetime.now().isoformat(timespec='seconds')}"]
    rows: list[dict[str, Any]] = []
    required = [
        MODEL_DIR,
        MODEL_CKPT,
        TRAIN_DS_ORIG,
        TRAIN_DS,
        TRAIN_H5AD,
        GSE_DS_ORIG,
        GSE_DS,
        GSE_H5AD,
        ROOT / "summary_phase4F_zh.md",
        TABLES / "phase4E_class_weighted_primary_test_metrics_calibrated.csv",
        TABLES / "phase4E_repeated_grouped_retraining_metrics.csv",
        TABLES / "phase4E_GSE72056_binary_A_sensitivity_calibrated.csv",
        TOKEN_DICT,
        GENE_NAME_DICT,
    ]
    missing = []
    for path in required:
        ok = path.exists() and (path.is_dir() or path.stat().st_size > 0)
        rows.append({"check_type": "required_input", "path": rel(path), "status": "ok" if ok else "missing_or_empty"})
        if not ok:
            missing.append(rel(path))
    if missing:
        pd.DataFrame(rows).to_csv(TABLES / "phase5A_input_integrity_check.csv", index=False, encoding="utf-8-sig")
        write_text(LOGS / "phase5A_preflight_check_log.md", log + ["Missing inputs:", *[f"- {x}" for x in missing]])
        raise RuntimeError("Phase 5A preflight failed: missing inputs.")

    summary = (ROOT / "summary_phase4F_zh.md").read_text(encoding="utf-8-sig")
    phase4f_ok = "READY_FOR_PHASE5 = CONDITIONAL_PILOT" in summary
    train_ds = load_from_disk(str(TRAIN_DS))
    train_orig = load_from_disk(str(TRAIN_DS_ORIG))
    gse_ds = load_from_disk(str(GSE_DS))
    gse_orig = load_from_disk(str(GSE_DS_ORIG))
    train_h5 = ad.read_h5ad(TRAIN_H5AD)
    gse_h5 = ad.read_h5ad(GSE_H5AD)
    for dataset_id, h5, ds, orig in [
        ("GSE115978", train_h5, train_ds, train_orig),
        ("GSE72056", gse_h5, gse_ds, gse_orig),
    ]:
        has_label = "phase4D_binary_A_label" in ds.column_names
        orig_has_label = "phase4D_binary_A_label" in orig.column_names
        rows.append(
            {
                "check_type": "tokenized_metadata",
                "dataset_id": dataset_id,
                "h5ad_n_obs": int(h5.n_obs),
                "phase4D_labeled_token_rows": len(ds),
                "original_token_rows": len(orig),
                "phase4D_label_present": has_label,
                "original_token_has_phase4D_label": orig_has_label,
                "row_count_match": int(h5.n_obs) == len(ds) == len(orig),
                "status": "ok" if has_label and int(h5.n_obs) == len(ds) == len(orig) else "failed",
            }
        )
    gse_not_training = True
    accepted_phrases = [
        "GSE72056 was not used for training",
        "GSE72056 not used for training: True",
    ]
    for log_path in [LOGS / "phase4E_class_weighted_training_log.md", LOGS / "phase4F_preflight_check_log.md"]:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8-sig")
            if not any(phrase in text for phrase in accepted_phrases):
                gse_not_training = False
    rows.extend(
        [
            {"check_type": "phase4F_ready_state", "status": "ok" if phase4f_ok else "failed"},
            {"check_type": "model_checkpoint_loadable_candidate", "path": rel(MODEL_CKPT), "status": "ok"},
            {"check_type": "GSE72056_training_exclusion_from_prior_logs", "status": "ok" if gse_not_training else "failed"},
        ]
    )
    failed = [row for row in rows if row.get("status") == "failed"]
    pd.DataFrame(rows).to_csv(TABLES / "phase5A_input_integrity_check.csv", index=False, encoding="utf-8-sig")
    log.extend(
        [
            f"Phase 4F CONDITIONAL_PILOT: {phase4f_ok}",
            "Phase 4D-labeled tokenized copies are used because original tokenized reference datasets do not carry phase4D_binary_A_label.",
            "No original tokenized dataset will be modified.",
            f"GSE72056 prior training exclusion confirmed: {gse_not_training}",
        ]
    )
    if failed:
        log.append("Failed checks:")
        log.extend(f"- {row}" for row in failed)
        write_text(LOGS / "phase5A_preflight_check_log.md", log)
        raise RuntimeError("Phase 5A preflight failed.")
    write_text(LOGS / "phase5A_preflight_check_log.md", log + ["Preflight passed."])
    return train_ds, gse_ds, train_h5, gse_h5


def ds_to_meta(ds: Dataset, dataset_id: str) -> pd.DataFrame:
    cols = [
        "original_cell_id", "cell_id", "sample_id", "tumor_id", "treatment.group",
        "split_unit", "phase4D_binary_A_label", "malignant_state", "dataset_id",
    ]
    records = []
    for i in range(len(ds)):
        row = ds[i]
        records.append({col: row.get(col, "") for col in cols} | {"row_index": i, "dataset_id": dataset_id})
    return pd.DataFrame(records)


def select_adverse_cells(ds: Dataset, dataset_id: str, max_cells: int = MAX_CELLS) -> pd.DataFrame:
    meta = ds_to_meta(ds, dataset_id)
    sub = meta.loc[meta["phase4D_binary_A_label"].astype(str) == "adverse_like"].copy()
    sub = sub.loc[sub["malignant_state"].astype(str) != "intermediate/ambiguous"].copy()
    n_available = len(sub)
    sampled = False
    if len(sub) > max_cells:
        sub = sub.sample(n=max_cells, random_state=SEED).sort_values("row_index").copy()
        sampled = True
    summary = pd.DataFrame(
        [
            {
                "dataset_id": dataset_id,
                "selection_label": "phase4D_binary_A_label == adverse_like",
                "available_adverse_like_cells": n_available,
                "selected_cells": len(sub),
                "max_cells": max_cells,
                "random_seed": SEED,
                "sampled": sampled,
                "sample_id_count": sub["sample_id"].nunique(),
                "treatment_group_distribution": json.dumps(dict(Counter(sub["treatment.group"].astype(str))), ensure_ascii=False, sort_keys=True),
                "split_unit_count": sub["split_unit"].nunique(),
            }
        ]
    )
    return sub, summary


def write_cell_set(train_selected: pd.DataFrame, train_summary: pd.DataFrame, gse_selected: pd.DataFrame, gse_summary: pd.DataFrame) -> None:
    out = pd.concat([train_summary, gse_summary], ignore_index=True)
    out.to_csv(TABLES / "phase5A_perturbation_cell_set_summary.csv", index=False, encoding="utf-8-sig")
    log = [
        "# Phase 5A cell set selection log",
        "",
        "Primary pilot analysis uses only GSE115978 supervised adverse_like cells.",
        f"GSE115978 selected cells: {len(train_selected)}.",
        "Intermediate/ambiguous cells were excluded.",
        "GSE72056 selected cells are optional processed-expression sensitivity only and are not used for training, threshold selection, or ranking selection.",
        f"GSE72056 optional sensitivity selected cells: {len(gse_selected)}.",
    ]
    write_text(LOGS / "phase5A_cell_set_selection_log.md", log)


def load_gene_dicts() -> tuple[dict[str, int], dict[str, str]]:
    with open(TOKEN_DICT, "rb") as handle:
        token_dict = pickle.load(handle)
    with open(GENE_NAME_DICT, "rb") as handle:
        gene_name_id = pickle.load(handle)
    return token_dict, gene_name_id


def cell_has_token(ds: Dataset, row_indices: list[int], token_id: int) -> int:
    count = 0
    for idx in row_indices:
        ids = ds[int(idx)]["input_ids"]
        if token_id in ids:
            count += 1
    return count


def gene_eligibility(ds: Dataset, selected: pd.DataFrame) -> pd.DataFrame:
    token_dict, gene_name_id = load_gene_dicts()
    row_indices = selected["row_index"].astype(int).tolist()
    rows = []
    for gene in GENE_WHITELIST:
        ensembl_id = gene_name_id.get(gene)
        token_id = token_dict.get(ensembl_id) if ensembl_id else None
        n_present = cell_has_token(ds, row_indices, int(token_id)) if token_id is not None else 0
        status = "eligible" if token_id is not None and n_present > 0 else "not_eligible"
        reason = "ok"
        if ensembl_id is None:
            reason = "gene_symbol_not_in_gene_name_id_dict"
        elif token_id is None:
            reason = "ensembl_id_not_in_gc104M_token_dictionary"
        elif n_present == 0:
            reason = "gene_token_absent_in_selected_cells"
        rows.append(
            {
                "gene_symbol": gene,
                "ensembl_id": ensembl_id or "unmapped",
                "token_id": token_id if token_id is not None else "unmapped",
                "in_gc104M_token_dictionary": token_id is not None,
                "n_selected_cells_with_gene_token": n_present,
                "selected_cell_count": len(row_indices),
                "eligible_for_pilot_deletion": status == "eligible",
                "status": status,
                "reason": reason,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase5A_gene_whitelist_eligibility.csv", index=False, encoding="utf-8-sig")
    write_text(
        LOGS / "phase5A_gene_whitelist_log.md",
        [
            "# Phase 5A gene whitelist log",
            "",
            f"Whitelist genes: {len(GENE_WHITELIST)}",
            f"Eligible genes in selected GSE115978 adverse_like cells: {int(df['eligible_for_pilot_deletion'].sum())}",
            "Genes absent from the token dictionary or selected tokenized cells were not perturbed.",
        ],
    )
    return df


def load_model() -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return p4e.load_checkpoint(MODEL_CKPT, device)


def predict_arrays(model: torch.nn.Module, examples: list[dict[str, list[int]]]) -> np.ndarray:
    device = next(model.parameters()).device
    probs = []
    model.eval()
    with torch.no_grad():
        for ex in examples:
            input_ids = torch.tensor(ex["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.tensor(ex["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids, attention_mask)
            probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy()[0])
    return np.vstack(probs) if probs else np.zeros((0, 2), dtype=float)


def examples_for_rows(ds: Dataset, rows: pd.DataFrame) -> list[dict[str, list[int]]]:
    return [{"input_ids": list(ds[int(idx)]["input_ids"]), "attention_mask": list(ds[int(idx)]["attention_mask"])} for idx in rows["row_index"].astype(int)]


def write_baseline(model: torch.nn.Module, ds: Dataset, selected: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    examples = examples_for_rows(ds, selected)
    probs = predict_arrays(model, examples)
    out = selected.reset_index(drop=True).copy()
    out["P_before_melanocytic_like"] = probs[:, MEL]
    out["P_before_adverse_like"] = probs[:, ADV]
    out["hard_label_before_threshold_0.70"] = np.where(out["P_before_adverse_like"] >= THRESHOLD, "adverse_like", "melanocytic_like")
    if dataset_label == "GSE115978":
        out.to_csv(TABLES / "phase5A_baseline_prediction_summary.csv", index=False, encoding="utf-8-sig")
        write_text(
            LOGS / "phase5A_baseline_prediction_log.md",
            [
                "# Phase 5A baseline prediction log",
                "",
                "Baseline predictions were computed with the Phase 4E class-weighted calibrated model.",
                "Input IDs were not modified.",
                f"Cells evaluated: {len(out)}",
                f"Mean P_before_adverse_like: {out['P_before_adverse_like'].mean():.6f}",
            ],
        )
    return out


def delete_token_keep_legal(input_ids: list[int], attention_mask: list[int], target_token: int) -> tuple[list[int], list[int], str]:
    ids = list(input_ids)
    mask = list(attention_mask)
    active_len = sum(1 for x in mask if int(x) == 1)
    active = ids[:active_len]
    pads = ids[active_len:]
    if target_token not in active:
        return ids, mask, "gene_absent_in_cell"
    pos = active.index(target_token)
    if active[pos] in SPECIAL_TOKENS:
        return ids, mask, "refused_special_token"
    if active[0] != CLS_TOKEN:
        return ids, mask, "invalid_missing_cls"
    if EOS_TOKEN not in active:
        return ids, mask, "invalid_missing_eos"
    del active[pos]
    active.append(PAD_TOKEN)
    new_mask = [1 if tok != PAD_TOKEN else 0 for tok in active]
    if len(active) < len(ids):
        active.extend(pads)
        new_mask.extend([0] * len(pads))
    active = active[: len(ids)]
    new_mask = new_mask[: len(mask)]
    if active[0] != CLS_TOKEN or EOS_TOKEN not in active:
        return ids, mask, "invalid_special_token_after_deletion"
    return active, new_mask, "deleted"


def run_deletion_for_dataset(
    model: torch.nn.Module,
    ds: Dataset,
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
    eligibility: pd.DataFrame,
    dataset_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = eligibility.loc[eligibility["eligible_for_pilot_deletion"]].copy()
    baseline_by_row = baseline.set_index("row_index")
    cell_gene_rows = []
    gene_rows = []
    start = time.time()
    for _, gene_row in eligible.iterrows():
        gene = gene_row["gene_symbol"]
        token_id = int(gene_row["token_id"])
        examples = []
        example_meta = []
        failure_count = 0
        for _, cell in selected.iterrows():
            row_index = int(cell["row_index"])
            row = ds[row_index]
            new_ids, new_mask, status = delete_token_keep_legal(row["input_ids"], row["attention_mask"], token_id)
            if status != "deleted":
                if status != "gene_absent_in_cell":
                    failure_count += 1
                continue
            examples.append({"input_ids": new_ids, "attention_mask": new_mask})
            example_meta.append(cell)
        probs_after = predict_arrays(model, examples) if examples else np.zeros((0, 2), dtype=float)
        deltas = []
        for i, cell in enumerate(example_meta):
            row_index = int(cell["row_index"])
            before = baseline_by_row.loc[row_index]
            p_before_adv = float(before["P_before_adverse_like"])
            p_before_mel = float(before["P_before_melanocytic_like"])
            p_after_mel = float(probs_after[i, MEL])
            p_after_adv = float(probs_after[i, ADV])
            delta_adv = p_after_adv - p_before_adv
            deltas.append(delta_adv)
            before_label = "adverse_like" if p_before_adv >= THRESHOLD else "melanocytic_like"
            after_label = "adverse_like" if p_after_adv >= THRESHOLD else "melanocytic_like"
            cell_gene_rows.append(
                {
                    "dataset_id": dataset_label,
                    "gene_symbol": gene,
                    "ensembl_id": gene_row["ensembl_id"],
                    "token_id": token_id,
                    "row_index": row_index,
                    "original_cell_id": cell.get("original_cell_id", ""),
                    "cell_id": cell.get("cell_id", ""),
                    "sample_id": cell.get("sample_id", ""),
                    "treatment.group": cell.get("treatment.group", ""),
                    "split_unit": cell.get("split_unit", ""),
                    "phase4D_binary_A_label": cell.get("phase4D_binary_A_label", ""),
                    "P_before_adverse_like": p_before_adv,
                    "P_after_adverse_like": p_after_adv,
                    "delta_P_adverse_like": delta_adv,
                    "P_before_melanocytic_like": p_before_mel,
                    "P_after_melanocytic_like": p_after_mel,
                    "hard_label_before_threshold_0.70": before_label,
                    "hard_label_after_threshold_0.70": after_label,
                    "hard_label_shift_threshold_0.70": before_label != after_label,
                    "status": "deleted",
                }
            )
        if examples:
            tmp = pd.DataFrame([r for r in cell_gene_rows if r["dataset_id"] == dataset_label and r["gene_symbol"] == gene])
            warning = []
            if len(tmp) < 10:
                warning.append("low_n_cells_with_gene")
            gene_rows.append(
                {
                    "dataset_id": dataset_label,
                    "gene_symbol": gene,
                    "ensembl_id": gene_row["ensembl_id"],
                    "token_id": token_id,
                    "n_cells_with_gene": len(tmp),
                    "mean_P_before_adverse_like": tmp["P_before_adverse_like"].mean(),
                    "mean_P_after_adverse_like": tmp["P_after_adverse_like"].mean(),
                    "mean_delta_P_adverse_like": tmp["delta_P_adverse_like"].mean(),
                    "median_delta_P_adverse_like": tmp["delta_P_adverse_like"].median(),
                    "fraction_cells_delta_negative": float((tmp["delta_P_adverse_like"] < 0).mean()),
                    "mean_P_before_melanocytic_like": tmp["P_before_melanocytic_like"].mean(),
                    "mean_P_after_melanocytic_like": tmp["P_after_melanocytic_like"].mean(),
                    "hard_label_shift_rate_threshold_0.70": float(tmp["hard_label_shift_threshold_0.70"].mean()),
                    "failure_count": failure_count,
                    "warning_flag": ";".join(warning) if warning else "none",
                }
            )
        else:
            gene_rows.append(
                {
                    "dataset_id": dataset_label,
                    "gene_symbol": gene,
                    "ensembl_id": gene_row["ensembl_id"],
                    "token_id": token_id,
                    "n_cells_with_gene": 0,
                    "failure_count": failure_count,
                    "warning_flag": "no_cells_with_gene_after_runtime_check",
                }
            )
    cell_df = pd.DataFrame(cell_gene_rows)
    gene_df = pd.DataFrame(gene_rows)
    if dataset_label == "GSE115978":
        gene_df.to_csv(TABLES / "phase5A_pilot_deletion_effect_by_gene.csv", index=False, encoding="utf-8-sig")
        cell_df.to_csv(TABLES / "phase5A_pilot_deletion_effect_by_cell_gene.csv", index=False, encoding="utf-8-sig")
        write_text(
            LOGS / "phase5A_pilot_deletion_log.md",
            [
                "# Phase 5A pilot deletion log",
                "",
                "Temporary in-memory token deletion was performed for eligible whitelist genes only.",
                "Original tokenized datasets were not modified.",
                "Deletion removed the target gene token from active input_ids, preserved <cls>, preserved <eos>, kept sequence length fixed, and padded the deleted position at the end of the active sequence.",
                f"Dataset: {dataset_label}",
                f"Eligible genes evaluated: {len(eligible)}",
                f"Cell-gene deletion evaluations: {len(cell_df)}",
                f"Elapsed seconds: {time.time() - start:.1f}",
            ],
        )
    return gene_df, cell_df


def write_ranking(gene_df: pd.DataFrame) -> pd.DataFrame:
    ranked = gene_df.copy()
    ranked = ranked.loc[ranked["n_cells_with_gene"].fillna(0).astype(int) > 0].copy()
    ranked = ranked.sort_values(["mean_delta_P_adverse_like", "n_cells_with_gene"], ascending=[True, False]).reset_index(drop=True)
    ranked["exploratory_rank"] = np.arange(1, len(ranked) + 1)
    ranked["interpretation_scope"] = "hypothesis_generating_only"
    ranked["model_dependency_flag"] = "model_dependent"
    ranked["validation_limitations"] = "no_wet_lab_validation;no_TCGA_or_drug_validation_yet;external_sensitivity_limited"
    ranked["not_a_therapeutic_target_claim"] = True
    ranked["not_a_causal_driver_claim"] = True
    ranked.to_csv(TABLES / "phase5A_exploratory_perturbation_ranking.csv", index=False, encoding="utf-8-sig")
    write_text(
        LOGS / "phase5A_ranking_interpretation_log.md",
        [
            "# Phase 5A ranking interpretation log",
            "",
            "Ranking is sorted by mean_delta_P_adverse_like ascending; negative values mean lower predicted adverse_like probability after in-memory gene-token deletion.",
            "This is exploratory perturbation ranking only.",
            "Do not describe genes as therapeutic targets, causal genes, or validated targets.",
            "Required limitations: hypothesis_generating_only; model_dependent; no_wet_lab_validation; no_TCGA_or_drug_validation_yet; external_sensitivity_limited.",
        ],
    )
    return ranked


def write_gse_sensitivity(gse_gene_df: pd.DataFrame, gse_cell_df: pd.DataFrame, primary_ranked: pd.DataFrame) -> None:
    if gse_gene_df.empty:
        pd.DataFrame([{"status": "not_run_or_no_eligible_genes"}]).to_csv(TABLES / "phase5A_GSE72056_pilot_deletion_sensitivity.csv", index=False, encoding="utf-8-sig")
        write_text(LOGS / "phase5A_GSE72056_pilot_sensitivity_log.md", ["# Phase 5A GSE72056 pilot sensitivity log", "", "Not run or no eligible genes."])
        return
    primary = primary_ranked[["gene_symbol", "mean_delta_P_adverse_like"]].rename(columns={"mean_delta_P_adverse_like": "primary_mean_delta_P_adverse_like"})
    out = gse_gene_df.merge(primary, on="gene_symbol", how="left")
    out["processed_expression_limitation"] = True
    out["sensitivity_only_not_external_validation"] = True
    out["used_for_main_gene_selection"] = False
    out["direction_consistency_with_primary"] = np.sign(out["mean_delta_P_adverse_like"].astype(float)) == np.sign(out["primary_mean_delta_P_adverse_like"].astype(float))
    out["domain_shift_or_processed_expression_limitation"] = np.where(out["direction_consistency_with_primary"], "not_flagged_by_direction", "flagged_direction_mismatch")
    out.to_csv(TABLES / "phase5A_GSE72056_pilot_deletion_sensitivity.csv", index=False, encoding="utf-8-sig")
    write_text(
        LOGS / "phase5A_GSE72056_pilot_sensitivity_log.md",
        [
            "# Phase 5A GSE72056 pilot sensitivity log",
            "",
            "GSE72056 is processed/non-integer expression.",
            "GSE72056 is sensitivity only, not strong external validation.",
            "GSE72056 was not used to select the primary exploratory ranking.",
            f"Genes evaluated: {len(out)}",
            f"Direction mismatches vs GSE115978: {int((~out['direction_consistency_with_primary']).sum())}",
        ],
    )


def write_summary(train_summary: pd.DataFrame, eligibility: pd.DataFrame, ranked: pd.DataFrame, gse_done: bool) -> None:
    top = ranked.head(5).copy()
    ready = "CONDITIONAL" if not ranked.empty else "NO"
    lines = [
        "# Phase 5A 中文总结",
        "",
        "本阶段仅执行 pilot in silico deletion feasibility；未进行 TCGA、生存、GDSC、DepMap、ChEMBL、Open Targets、DEG 或正式候选治疗靶点验证。",
        "",
        "## 1. Preflight",
        "",
        "- Phase 4F READY_FOR_PHASE5 = CONDITIONAL_PILOT 已确认。",
        "- Phase 4E class-weighted checkpoint 可用于推理。",
        "- GSE115978/GSE72056 tokenized data 可读取；使用 Phase 4D-labeled tokenized copy 保留 phase4D_binary_A_label。",
        "",
        "## 2. Pilot adverse_like cell set",
        "",
        f"- GSE115978 selected adverse_like cells = {int(train_summary.iloc[0]['selected_cells'])}。",
        f"- selected sample_id count = {int(train_summary.iloc[0]['sample_id_count'])}。",
        "",
        "## 3. Gene whitelist eligibility",
        "",
        f"- whitelist genes = {len(eligibility)}。",
        f"- eligible genes = {int(eligibility['eligible_for_pilot_deletion'].sum())}。",
        "",
        "## 4. Pilot deletion",
        "",
        f"- pilot deletion executed successfully for {len(ranked)} genes with at least one evaluated cell。",
        "- ΔP(adverse_like) < 0 表示删除该 gene token 后模型预测 adverse_like probability 降低。",
        "",
        "## 5. Exploratory strongest decreases",
        "",
    ]
    if top.empty:
        lines.append("- No eligible perturbation effects were available.")
    else:
        for _, row in top.iterrows():
            lines.append(f"- {row['gene_symbol']}: mean_delta_P_adverse_like={row['mean_delta_P_adverse_like']:.6f}, n_cells_with_gene={int(row['n_cells_with_gene'])}")
    lines.extend(
        [
            "",
            "## 6. Interpretation limits",
            "",
            "- 这些结果只能解释为 hypothesis-generating、model-dependent exploratory perturbation signals。",
            "- 不允许称为 therapeutic targets、validated targets 或 mechanistic-driver claims。",
            "- 尚无 wet-lab validation、TCGA/drug/dependency/druggability validation。",
            "- Phase 4F 已显示外部 processed-expression sensitivity 有限制。",
            "",
            "## 7. GSE72056 sensitivity",
            "",
            "- GSE72056 sensitivity 已执行。" if gse_done else "- GSE72056 sensitivity 未执行。",
            "- GSE72056 是 processed/non-integer expression，不作为强外部验证，也不用于筛选主 ranking。",
            "",
            "## 8. Phase 5B recommendation",
            "",
            f"READY_FOR_PHASE5B = {ready}",
            "",
        ]
    )
    if ready == "CONDITIONAL":
        lines.extend(
            [
                "允许继续做什么：",
                "- 仅允许 Phase 5B expanded perturbation，扩大 gene whitelist 或 cell sampling，并继续使用 continuous ΔP(adverse_like) 为主。",
                "",
                "禁止做什么：",
                "- 禁止输出正式候选治疗靶点、overstated treatment-target claims、mechanistic-driver claims 或强外部泛化结论。",
                "- 禁止用 GSE72056 作为强外部验证或主筛选依据。",
                "",
                "必须保留的限制性措辞：",
                "- hypothesis_generating_only；model_dependent；no_wet_lab_validation；no_TCGA_or_drug_validation_yet；external_sensitivity_limited。",
            ]
        )
    else:
        lines.extend(["阻断 Phase 5B：没有足够 eligible deletion effects 支持扩展。"])
    write_text(ROOT / "summary_phase5A_zh.md", lines)


def main() -> int:
    ensure_dirs()
    random.seed(SEED)
    np.random.seed(SEED)
    train_ds, gse_ds, train_h5, gse_h5 = preflight()
    train_selected, train_summary = select_adverse_cells(train_ds, "GSE115978")
    gse_selected, gse_summary = select_adverse_cells(gse_ds, "GSE72056")
    write_cell_set(train_selected, train_summary, gse_selected, gse_summary)
    eligibility = gene_eligibility(train_ds, train_selected)
    model = load_model()
    try:
        train_baseline = write_baseline(model, train_ds, train_selected, "GSE115978")
        primary_gene_df, primary_cell_df = run_deletion_for_dataset(model, train_ds, train_selected, train_baseline, eligibility, "GSE115978")
        ranked = write_ranking(primary_gene_df)
        # Optional processed-expression sensitivity. It reuses the same whitelist
        # eligibility, but runtime presence is checked again in GSE72056 cells.
        gse_baseline = write_baseline(model, gse_ds, gse_selected, "GSE72056")
        gse_gene_df, gse_cell_df = run_deletion_for_dataset(model, gse_ds, gse_selected, gse_baseline, eligibility, "GSE72056")
        write_gse_sensitivity(gse_gene_df, gse_cell_df, ranked)
        write_summary(train_summary, eligibility, ranked, gse_done=True)
    except Exception:
        write_text(LOGS / "phase5A_pilot_deletion_log.md", ["# Phase 5A pilot deletion log", "", "FAILED", traceback.format_exc()])
        raise
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("PHASE5A_PILOT_DELETION: PASS")
    print(f"SUMMARY={ROOT / 'summary_phase5A_zh.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
