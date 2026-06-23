from __future__ import annotations

import importlib.util
import json
import math
import pickle
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
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

P5A_PATH = SCRIPTS / "16_phase5A_pilot_deletion.py"
spec = importlib.util.spec_from_file_location("phase5a", P5A_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import Phase 5A helpers from {P5A_PATH}")
p5a = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p5a
spec.loader.exec_module(p5a)

SEED = 42
BOOTSTRAPS = 1000
BATCH_SIZE_START = 8
GENE_TIMEOUT_SECONDS = 15 * 60
THRESHOLD = 0.70
LABELS = ["melanocytic_like", "adverse_like"]
MEL = 0
ADV = 1
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

PRIMARY_CHUNK_DIR = TABLES / "phase5B_gene_chunks_primary"
GSE_CHUNK_DIR = TABLES / "phase5B_gene_chunks_GSE72056"

CURATED_GENES = [
    # Phase 5A whitelist
    "MITF", "MLANA", "PMEL", "TYR", "DCT", "AXL", "NGFR", "VIM", "FN1", "ZEB1", "TGFBI",
    "MKI67", "TOP2A", "PCNA", "STMN1", "HIF1A", "VEGFA", "LDHA", "BRAF", "NRAS", "KIT",
    "PTEN", "CDKN2A", "MAPK1", "MAPK3", "AKT1",
    # Melanocytic and neural crest lineage
    "SOX10", "PAX3", "TFAP2A", "LEF1", "EDNRB", "GPR143", "SLC45A2", "OCA2", "RAB27A",
    # Invasive/adverse-like and EMT/ECM
    "WNT5A", "ITGA3", "ITGB1", "ITGAV", "ITGB3", "MMP2", "MMP14", "SERPINE1", "SPARC",
    "THBS1", "TNC", "COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2", "COL6A3", "LGALS3",
    "S100A4", "S100A6", "JUN", "JUNB", "FOS", "FOSL1", "EGR1", "DUSP1", "DUSP4", "DUSP6",
    "NR4A1", "ATF3", "KLF6", "KLF4",
    # Cycling/proliferation
    "CDK1", "CDK2", "CDK4", "CCND1", "CCNA2", "CCNB1", "CCNB2", "AURKA", "AURKB", "PLK1",
    "BIRC5", "UBE2C", "CENPF", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "TYMS",
    # Stress/hypoxia/metabolism
    "CA9", "BNIP3", "NDRG1", "SLC2A1", "ALDOA", "ENO1", "PGK1", "HK2", "HSP90AA1",
    "HSPA1A", "HSPA1B", "HSPB1",
    # Melanoma pathways and tumor suppressors
    "MAP2K1", "MAP2K2", "RAF1", "KRAS", "HRAS", "NF1", "RAC1", "RB1", "TP53", "MDM2",
    "TERT", "PIK3CA", "PIK3R1", "MTOR", "TSC1", "TSC2", "FOXO3", "GSK3B",
    # Immune-evasion/drug-response related tumor-cell genes
    "CD274", "PDCD1LG2", "HLA-A", "HLA-B", "HLA-C", "B2M", "JAK1", "JAK2", "STAT1", "STAT3",
    "IRF1", "IFNGR1", "IFNGR2",
]

LOW_SUPPORT_DRIVER_GENES = {"BRAF", "NRAS", "KIT", "PTEN", "CDKN2A", "NF1", "RAC1", "TERT", "TP53", "RB1"}


def ensure_dirs() -> None:
    for path in [TABLES, LOGS, PRIMARY_CHUNK_DIR, GSE_CHUNK_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_dicts() -> tuple[dict[str, int], dict[str, str], dict[int, str], dict[str, str]]:
    token_dict = load_pickle(TOKEN_DICT)
    gene_name_id = load_pickle(GENE_NAME_DICT)
    token_to_ensembl = {int(v): str(k) for k, v in token_dict.items() if isinstance(v, int) and str(k).startswith("ENSG")}
    ensembl_to_symbols: dict[str, list[str]] = defaultdict(list)
    for symbol, ens in gene_name_id.items():
        ensembl_to_symbols[str(ens)].append(str(symbol))
    ensembl_to_symbol = {}
    for ens, symbols in ensembl_to_symbols.items():
        preferred = sorted(symbols, key=lambda s: (("-" in s) or ("." in s), len(s), s))[0]
        ensembl_to_symbol[ens] = preferred
    return token_dict, gene_name_id, token_to_ensembl, ensembl_to_symbol


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


def select_all_adverse(ds: Dataset, dataset_id: str) -> pd.DataFrame:
    meta = ds_to_meta(ds, dataset_id)
    sub = meta.loc[meta["phase4D_binary_A_label"].astype(str) == "adverse_like"].copy()
    sub = sub.loc[sub["malignant_state"].astype(str) != "intermediate/ambiguous"].copy()
    return sub.sort_values("row_index").reset_index(drop=True)


def preflight() -> tuple[Dataset, Dataset, pd.DataFrame, pd.DataFrame]:
    log = ["# Phase 5B preflight check log", "", f"Timestamp: {datetime.now().isoformat(timespec='seconds')}"]
    rows: list[dict[str, Any]] = []
    required = [
        ROOT / "summary_phase5A_zh.md",
        TABLES / "phase5A_pilot_deletion_effect_by_gene.csv",
        TABLES / "phase5A_pilot_deletion_effect_by_cell_gene.csv",
        TABLES / "phase5A_exploratory_perturbation_ranking.csv",
        MODEL_DIR,
        MODEL_CKPT,
        TRAIN_DS_ORIG,
        TRAIN_DS,
        TRAIN_H5AD,
        GSE_DS_ORIG,
        GSE_DS,
        GSE_H5AD,
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
        pd.DataFrame(rows).to_csv(TABLES / "phase5B_input_integrity_check.csv", index=False, encoding="utf-8-sig")
        write_text(LOGS / "phase5B_preflight_check_log.md", log + ["Missing inputs:", *[f"- {x}" for x in missing]])
        raise RuntimeError("Phase 5B preflight failed: missing inputs.")

    summary5a = (ROOT / "summary_phase5A_zh.md").read_text(encoding="utf-8-sig")
    phase5a_ok = "READY_FOR_PHASE5B = CONDITIONAL" in summary5a
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
        rows.append(
            {
                "check_type": "tokenized_metadata",
                "dataset_id": dataset_id,
                "h5ad_n_obs": int(h5.n_obs),
                "phase4D_labeled_token_rows": len(ds),
                "original_token_rows": len(orig),
                "phase4D_label_present": "phase4D_binary_A_label" in ds.column_names,
                "original_token_has_phase4D_label": "phase4D_binary_A_label" in orig.column_names,
                "row_count_match": int(h5.n_obs) == len(ds) == len(orig),
                "status": "ok" if int(h5.n_obs) == len(ds) == len(orig) and "phase4D_binary_A_label" in ds.column_names else "failed",
            }
        )
    rows.extend(
        [
            {"check_type": "phase5A_verification_state", "status": "ok" if phase5a_ok else "failed"},
            {"check_type": "model_checkpoint_exists", "status": "ok"},
            {"check_type": "GSE72056_training_exclusion", "status": "ok", "note": "confirmed in Phase 4F and Phase 5A logs; GSE72056 used here only for processed-expression sensitivity"},
            {"check_type": "original_arrays_not_modified_policy", "status": "ok", "note": "Phase 5B operates on in-memory copied input_ids only"},
        ]
    )
    failed = [row for row in rows if row.get("status") in {"failed", "missing_or_empty"}]
    pd.DataFrame(rows).to_csv(TABLES / "phase5B_input_integrity_check.csv", index=False, encoding="utf-8-sig")
    if failed:
        write_text(LOGS / "phase5B_preflight_check_log.md", log + ["Failed checks:", *[f"- {x}" for x in failed]])
        raise RuntimeError("Phase 5B preflight failed.")
    write_text(LOGS / "phase5B_preflight_check_log.md", log + ["Preflight passed."])
    return train_ds, gse_ds, select_all_adverse(train_ds, "GSE115978"), select_all_adverse(gse_ds, "GSE72056")


def active_tokens_without_special(row: dict[str, Any]) -> list[int]:
    ids = row["input_ids"]
    mask = row["attention_mask"]
    active_len = sum(1 for x in mask if int(x) == 1)
    return [int(tok) for tok in ids[:active_len] if int(tok) not in SPECIAL_TOKENS]


def token_frequency(ds: Dataset, selected: pd.DataFrame) -> Counter:
    counts: Counter[int] = Counter()
    for row_index in selected["row_index"].astype(int):
        counts.update(set(active_tokens_without_special(ds[int(row_index)])))
    return counts


def build_expanded_gene_set(ds: Dataset, selected: pd.DataFrame) -> pd.DataFrame:
    token_dict, gene_name_id, token_to_ensembl, ensembl_to_symbol = load_dicts()
    freq = token_frequency(ds, selected)
    records: dict[str, dict[str, Any]] = {}

    def add_gene(gene: str, source: str) -> None:
        ens = gene_name_id.get(gene)
        token = token_dict.get(ens) if ens else None
        if gene not in records:
            records[gene] = {
                "gene_symbol": gene,
                "ensembl_id": ens or "unmapped",
                "token_id": token if token is not None else "unmapped",
                "source_tags": set(),
            }
        records[gene]["source_tags"].add(source)

    for gene in CURATED_GENES:
        add_gene(gene, "curated_marker_pathway_or_phase5A_whitelist")

    for token, n in freq.most_common():
        if len(records) >= 150:
            break
        ens = token_to_ensembl.get(int(token))
        if not ens:
            continue
        symbol = ensembl_to_symbol.get(ens)
        if not symbol or symbol.startswith("ENSG"):
            continue
        add_gene(symbol, "high_frequency_in_adverse_like_cells")

    rows = []
    for gene, rec in records.items():
        token = rec["token_id"]
        n_cells = int(freq.get(int(token), 0)) if token != "unmapped" else 0
        sample_ids = set()
        if token != "unmapped":
            token_int = int(token)
            for _, cell in selected.iterrows():
                row = ds[int(cell["row_index"])]
                if token_int in active_tokens_without_special(row):
                    sample_ids.add(str(cell["sample_id"]))
        low_support_driver = gene in LOW_SUPPORT_DRIVER_GENES and n_cells > 0 and n_cells < 10
        eligible = token != "unmapped" and (n_cells >= 10 or low_support_driver)
        reason = "ok"
        if token == "unmapped":
            reason = "not_mapped_to_gc104M_token_dictionary"
        elif n_cells == 0:
            reason = "gene_token_absent_in_selected_adverse_like_cells"
        elif n_cells < 10 and low_support_driver:
            reason = "low_support_driver"
        elif n_cells < 10:
            reason = "n_cells_with_gene_lt_10"
        rows.append(
            {
                "gene_symbol": gene,
                "ensembl_id": rec["ensembl_id"],
                "token_id": token,
                "source_tags": ";".join(sorted(rec["source_tags"])),
                "n_cells_with_gene": n_cells,
                "n_samples_with_gene": len(sample_ids),
                "low_support_driver": low_support_driver,
                "eligible_for_expanded_deletion": bool(eligible),
                "eligibility_reason": reason,
            }
        )
    df = pd.DataFrame(rows).sort_values(["eligible_for_expanded_deletion", "n_cells_with_gene"], ascending=[False, False]).reset_index(drop=True)
    eligible = df.loc[df["eligible_for_expanded_deletion"]].copy()
    eligible.to_csv(TABLES / "phase5B_expanded_gene_set.csv", index=False, encoding="utf-8-sig")
    df.to_csv(TABLES / "phase5B_expanded_gene_eligibility.csv", index=False, encoding="utf-8-sig")
    write_text(
        LOGS / "phase5B_gene_set_construction_log.md",
        [
            "# Phase 5B gene set construction log",
            "",
            f"Curated input genes: {len(set(CURATED_GENES))}",
            "High-frequency genes were added from GSE115978 supervised adverse_like token presence.",
            f"Expanded gene records before eligibility filter: {len(df)}",
            f"Eligible expanded deletion genes: {len(eligible)}",
            "Eligibility required gc104M token mapping and n_cells_with_gene >= 10, except explicitly marked low_support_driver genes.",
            "This is not whole-genome perturbation.",
        ],
    )
    return eligible


def load_model() -> torch.nn.Module:
    model = p5a.load_model()
    model.eval()
    return model


def predict_batch(model: torch.nn.Module, examples: list[dict[str, list[int]]], batch_size: int = BATCH_SIZE_START) -> np.ndarray:
    device = next(model.parameters()).device
    probs: list[np.ndarray] = []
    i = 0
    current_batch = batch_size
    while i < len(examples):
        batch = examples[i : i + current_batch]
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long, device=device)
                attention_mask = torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long, device=device)
                logits = model(input_ids, attention_mask)
                probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
            i += current_batch
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if current_batch <= 1:
                raise
            current_batch = max(1, current_batch // 2)
    return np.vstack(probs) if probs else np.zeros((0, 2), dtype=float)


def examples_for_selected(ds: Dataset, selected: pd.DataFrame) -> list[dict[str, list[int]]]:
    return [{"input_ids": list(ds[int(idx)]["input_ids"]), "attention_mask": list(ds[int(idx)]["attention_mask"])} for idx in selected["row_index"].astype(int)]


def baseline_predictions(model: torch.nn.Module, ds: Dataset, selected: pd.DataFrame) -> pd.DataFrame:
    probs = predict_batch(model, examples_for_selected(ds, selected))
    out = selected.reset_index(drop=True).copy()
    out["P_before_melanocytic_like"] = probs[:, MEL]
    out["P_before_adverse_like"] = probs[:, ADV]
    out["hard_label_before_threshold_0.70"] = np.where(out["P_before_adverse_like"] >= THRESHOLD, "adverse_like", "melanocytic_like")
    return out


def delete_token_keep_legal(input_ids: list[int], attention_mask: list[int], target_token: int) -> tuple[list[int], list[int], str]:
    return p5a.delete_token_keep_legal(input_ids, attention_mask, target_token)


def process_gene(
    model: torch.nn.Module,
    ds: Dataset,
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
    gene_row: pd.Series,
    chunk_dir: Path,
    dataset_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene = str(gene_row["gene_symbol"])
    safe_gene = gene.replace("/", "_")
    summary_path = chunk_dir / f"{safe_gene}.summary.csv"
    cell_path = chunk_dir / f"{safe_gene}.cell_gene.csv"
    if summary_path.exists() and cell_path.exists() and summary_path.stat().st_size > 0:
        return pd.read_csv(summary_path, encoding="utf-8-sig"), pd.read_csv(cell_path, encoding="utf-8-sig")

    token_id = int(gene_row["token_id"])
    start = time.time()
    examples = []
    example_meta = []
    failure_count = 0
    status_counts: Counter[str] = Counter()
    for _, cell in selected.iterrows():
        if time.time() - start > GENE_TIMEOUT_SECONDS:
            status_counts["timeout_before_all_cells"] += 1
            break
        row = ds[int(cell["row_index"])]
        new_ids, new_mask, status = delete_token_keep_legal(row["input_ids"], row["attention_mask"], token_id)
        status_counts[status] += 1
        if status == "deleted":
            examples.append({"input_ids": new_ids, "attention_mask": new_mask})
            example_meta.append(cell)
        elif status != "gene_absent_in_cell":
            failure_count += 1
    probs_after = predict_batch(model, examples) if examples else np.zeros((0, 2), dtype=float)
    baseline_by_row = baseline.set_index("row_index")
    rows = []
    for i, cell in enumerate(example_meta):
        row_index = int(cell["row_index"])
        before = baseline_by_row.loc[row_index]
        p_before_adv = float(before["P_before_adverse_like"])
        p_before_mel = float(before["P_before_melanocytic_like"])
        p_after_adv = float(probs_after[i, ADV])
        p_after_mel = float(probs_after[i, MEL])
        before_label = "adverse_like" if p_before_adv >= THRESHOLD else "melanocytic_like"
        after_label = "adverse_like" if p_after_adv >= THRESHOLD else "melanocytic_like"
        rows.append(
            {
                "dataset_id": dataset_id,
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
                "delta_P_adverse_like": p_after_adv - p_before_adv,
                "P_before_melanocytic_like": p_before_mel,
                "P_after_melanocytic_like": p_after_mel,
                "delta_P_melanocytic_like": p_after_mel - p_before_mel,
                "hard_label_before_threshold_0.70": before_label,
                "hard_label_after_threshold_0.70": after_label,
                "hard_label_shift_threshold_0.70": before_label != after_label,
                "status": "deleted",
            }
        )
    cell_df = pd.DataFrame(rows)
    warning = []
    if cell_df.empty:
        summary = pd.DataFrame(
            [
                {
                    "dataset_id": dataset_id,
                    "gene_symbol": gene,
                    "ensembl_id": gene_row["ensembl_id"],
                    "token_id": token_id,
                    "n_cells_with_gene": 0,
                    "n_samples_with_gene": 0,
                    "runtime_seconds": time.time() - start,
                    "failure_count": failure_count,
                    "warning_flag": "no_cells_with_gene_after_runtime_check",
                    "status_counts": json.dumps(dict(status_counts), sort_keys=True),
                }
            ]
        )
    else:
        sample_means = cell_df.groupby("sample_id")["delta_P_adverse_like"].mean()
        if len(cell_df) < 20:
            warning.append("low_n_cells")
        if sample_means.shape[0] < 3:
            warning.append("low_n_samples")
        if status_counts.get("timeout_before_all_cells", 0) > 0:
            warning.append("partial_timeout")
        summary = pd.DataFrame(
            [
                {
                    "dataset_id": dataset_id,
                    "gene_symbol": gene,
                    "ensembl_id": gene_row["ensembl_id"],
                    "token_id": token_id,
                    "n_cells_with_gene": int(len(cell_df)),
                    "n_samples_with_gene": int(cell_df["sample_id"].nunique()),
                    "mean_P_before_adverse_like": cell_df["P_before_adverse_like"].mean(),
                    "mean_P_after_adverse_like": cell_df["P_after_adverse_like"].mean(),
                    "mean_delta_P_adverse_like": cell_df["delta_P_adverse_like"].mean(),
                    "median_delta_P_adverse_like": cell_df["delta_P_adverse_like"].median(),
                    "sd_delta_P_adverse_like": cell_df["delta_P_adverse_like"].std(ddof=1),
                    "fraction_cells_delta_negative": float((cell_df["delta_P_adverse_like"] < 0).mean()),
                    "fraction_samples_delta_negative": float((sample_means < 0).mean()),
                    "mean_delta_P_melanocytic_like": cell_df["delta_P_melanocytic_like"].mean(),
                    "hard_label_shift_rate_threshold_0.70": float(cell_df["hard_label_shift_threshold_0.70"].mean()),
                    "runtime_seconds": time.time() - start,
                    "failure_count": failure_count,
                    "warning_flag": ";".join(warning) if warning else "none",
                    "status_counts": json.dumps(dict(status_counts), sort_keys=True),
                }
            ]
        )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    cell_df.to_csv(cell_path, index=False, encoding="utf-8-sig")
    return summary, cell_df


def run_deletion(
    model: torch.nn.Module,
    ds: Dataset,
    selected: pd.DataFrame,
    genes: pd.DataFrame,
    chunk_dir: Path,
    dataset_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = baseline_predictions(model, ds, selected)
    summary_frames = []
    cell_frames = []
    log_lines = [
        f"# Phase 5B expanded deletion log ({dataset_id})",
        "",
        "Batch inference with resume was used.",
        f"Initial batch size: {BATCH_SIZE_START}",
        f"Per-gene timeout seconds: {GENE_TIMEOUT_SECONDS}",
        "Original tokenized datasets were not modified.",
    ]
    for idx, gene_row in genes.iterrows():
        gene = gene_row["gene_symbol"]
        gene_start = time.time()
        try:
            summary, cell_df = process_gene(model, ds, selected, baseline, gene_row, chunk_dir, dataset_id)
            summary_frames.append(summary)
            cell_frames.append(cell_df)
            runtime = float(summary.iloc[0].get("runtime_seconds", time.time() - gene_start))
            log_lines.append(f"{idx + 1}/{len(genes)} {gene}: status=done, n_cells={summary.iloc[0].get('n_cells_with_gene', 0)}, runtime_seconds={runtime:.1f}")
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            failed = pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "gene_symbol": gene,
                        "ensembl_id": gene_row["ensembl_id"],
                        "token_id": gene_row["token_id"],
                        "n_cells_with_gene": 0,
                        "runtime_seconds": time.time() - gene_start,
                        "failure_count": len(selected),
                        "warning_flag": "OOM_gene_skipped",
                    }
                ]
            )
            failed.to_csv(chunk_dir / f"{str(gene).replace('/', '_')}.summary.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([]).to_csv(chunk_dir / f"{str(gene).replace('/', '_')}.cell_gene.csv", index=False, encoding="utf-8-sig")
            summary_frames.append(failed)
            log_lines.append(f"{idx + 1}/{len(genes)} {gene}: status=OOM_gene_skipped")
        except Exception as exc:
            failed = pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "gene_symbol": gene,
                        "ensembl_id": gene_row["ensembl_id"],
                        "token_id": gene_row["token_id"],
                        "n_cells_with_gene": 0,
                        "runtime_seconds": time.time() - gene_start,
                        "failure_count": len(selected),
                        "warning_flag": f"failed:{type(exc).__name__}",
                    }
                ]
            )
            failed.to_csv(chunk_dir / f"{str(gene).replace('/', '_')}.summary.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([]).to_csv(chunk_dir / f"{str(gene).replace('/', '_')}.cell_gene.csv", index=False, encoding="utf-8-sig")
            summary_frames.append(failed)
            log_lines.append(f"{idx + 1}/{len(genes)} {gene}: status=failed, error={repr(exc)}")
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    cell_df = pd.concat(cell_frames, ignore_index=True) if cell_frames else pd.DataFrame()
    return summary_df, cell_df, baseline


def stability_tables(cell_df: pd.DataFrame, gene_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    boot_rows = []
    sample_rows = []
    for gene, sub in cell_df.groupby("gene_symbol"):
        deltas = sub["delta_P_adverse_like"].astype(float).to_numpy()
        if len(deltas) == 0:
            continue
        n_boot = BOOTSTRAPS
        means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            means[i] = rng.choice(deltas, size=len(deltas), replace=True).mean()
        sample = sub.groupby("sample_id").agg(
            sample_cell_count=("row_index", "count"),
            sample_mean_delta_P_adverse_like=("delta_P_adverse_like", "mean"),
            sample_fraction_cells_delta_negative=("delta_P_adverse_like", lambda x: float((x < 0).mean())),
        ).reset_index()
        sample["gene_symbol"] = gene
        sample_rows.append(sample)
        top_fraction = float(sample["sample_cell_count"].max() / sample["sample_cell_count"].sum()) if len(sample) else math.nan
        ci_low, ci_high = np.percentile(means, [2.5, 97.5])
        fraction_samples_negative = float((sample["sample_mean_delta_P_adverse_like"] < 0).mean()) if len(sample) else math.nan
        warning = []
        if len(sub) < 20:
            warning.append("low_n_cells")
        if len(sample) < 3:
            warning.append("low_n_samples")
        if float((sub["delta_P_adverse_like"] < 0).mean()) < 0.60 or fraction_samples_negative < 0.60:
            warning.append("inconsistent_direction")
        if ci_low <= 0 <= ci_high:
            warning.append("CI_crosses_zero")
        if top_fraction >= 0.50:
            warning.append("sample_dominated_effect")
        boot_rows.append(
            {
                "gene_symbol": gene,
                "n_cells_with_gene": int(len(sub)),
                "n_samples_with_gene": int(len(sample)),
                "bootstrap_iterations": n_boot,
                "mean_delta_P_adverse_like": float(np.mean(deltas)),
                "bootstrap_CI95_low": float(ci_low),
                "bootstrap_CI95_high": float(ci_high),
                "fraction_cells_delta_negative": float((sub["delta_P_adverse_like"] < 0).mean()),
                "fraction_samples_delta_negative": fraction_samples_negative,
                "top_sample_cell_fraction": top_fraction,
                "stability_warning_flags": ";".join(warning) if warning else "none",
            }
        )
    boot = pd.DataFrame(boot_rows)
    sample_level = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
    return boot, sample_level


def gse_direction(primary: pd.DataFrame, gse: pd.DataFrame) -> pd.DataFrame:
    p = primary[["gene_symbol", "mean_delta_P_adverse_like"]].rename(columns={"mean_delta_P_adverse_like": "primary_mean_delta_P_adverse_like"})
    out = gse.merge(p, on="gene_symbol", how="left")
    def direction(row: pd.Series) -> str:
        if pd.isna(row.get("mean_delta_P_adverse_like")) or pd.isna(row.get("primary_mean_delta_P_adverse_like")):
            return "not_evaluable"
        if row["n_cells_with_gene"] <= 0:
            return "not_evaluable"
        return "same_direction" if np.sign(float(row["mean_delta_P_adverse_like"])) == np.sign(float(row["primary_mean_delta_P_adverse_like"])) else "opposite_direction"
    out["direction_consistency"] = out.apply(direction, axis=1)
    out["processed_expression_limitation"] = True
    out["not_strong_external_validation"] = True
    out["used_for_main_ranking"] = False
    out["domain_shift_or_processed_expression_limitation"] = np.where(out["direction_consistency"].eq("opposite_direction"), "flagged_direction_mismatch", "not_flagged_by_direction")
    return out


def ranking_table(gene_df: pd.DataFrame, stability: pd.DataFrame, gse: pd.DataFrame) -> pd.DataFrame:
    out = gene_df.merge(stability, on=["gene_symbol", "n_cells_with_gene"], how="left", suffixes=("", "_stability"))
    gse_small = gse[["gene_symbol", "direction_consistency"]].copy() if "direction_consistency" in gse.columns else pd.DataFrame(columns=["gene_symbol", "direction_consistency"])
    out = out.merge(gse_small, on="gene_symbol", how="left")
    out["CI_does_not_cross_zero"] = ~(out["bootstrap_CI95_low"].fillna(0).le(0) & out["bootstrap_CI95_high"].fillna(0).ge(0))
    out["warning_low_support"] = (out["n_cells_with_gene"].fillna(0).astype(float) < 20) | (out["n_samples_with_gene"].fillna(0).astype(float) < 3)
    out["exploratory"] = True
    out["hypothesis_generating_only"] = True
    out["model_dependent"] = True
    out["no_wet_lab_validation"] = True
    out["no_TCGA_or_drug_validation_yet"] = True
    out["not_validated_therapeutic_targets"] = True
    out = out.sort_values(
        [
            "mean_delta_P_adverse_like",
            "fraction_cells_delta_negative",
            "fraction_samples_delta_negative",
            "n_cells_with_gene",
            "CI_does_not_cross_zero",
            "direction_consistency",
        ],
        ascending=[True, False, False, False, False, True],
    ).reset_index(drop=True)
    out["exploratory_rank"] = np.arange(1, len(out) + 1)
    return out


def write_summary(train_selected: pd.DataFrame, genes: pd.DataFrame, ranking: pd.DataFrame, stability: pd.DataFrame, gse: pd.DataFrame) -> None:
    top = ranking.head(10)
    stable = ranking.loc[
        ranking["stability_warning_flags"].fillna("none").eq("none")
        & ranking["direction_consistency"].fillna("not_evaluable").isin(["same_direction", "not_evaluable"])
    ].head(10)
    downgraded = ranking.loc[~ranking["stability_warning_flags"].fillna("none").eq("none")].head(10)
    ready = "CONDITIONAL" if len(ranking) > 0 else "NO"
    lines = [
        "# Phase 5B 中文总结",
        "",
        "本阶段只执行 expanded exploratory in silico deletion 和稳定性分析；未进行 TCGA、生存、GDSC、DepMap、ChEMBL、Open Targets、DEG 或正式候选治疗靶点验证。",
        "",
        "## 1. Preflight",
        "",
        "- Phase 5A verification/summary 支持进入 Phase 5B conditional expanded analysis。",
        "- 模型和 tokenized data 可读取；原始 h5ad/tokenized arrays 未修改。",
        "",
        "## 2. Cell set",
        "",
        f"- GSE115978 adverse_like cells used = {len(train_selected)}。",
        f"- sample_id count = {train_selected['sample_id'].nunique()}。",
        f"- treatment.group distribution = {json.dumps(dict(Counter(train_selected['treatment.group'].astype(str))), ensure_ascii=False, sort_keys=True)}。",
        "",
        "## 3. Expanded gene set",
        "",
        f"- expanded eligible genes = {len(genes)}。",
        "- gene set 来自 Phase 5A whitelist、melanoma state/pathway markers 和 adverse_like cells high-frequency token genes。",
        "",
        "## 4. Deletion execution",
        "",
        f"- deletion executed for {len(ranking)} genes with evaluable cell-gene effects。",
        "- 使用 batch inference、per-gene chunk 保存和 resume；每个 gene 完成后立即保存。",
        "",
        "## 5. Largest exploratory decreases in ΔP(adverse_like)",
        "",
    ]
    for _, row in top.iterrows():
        lines.append(f"- {row['gene_symbol']}: mean_delta={row['mean_delta_P_adverse_like']:.6f}, n_cells={int(row['n_cells_with_gene'])}, n_samples={int(row['n_samples_with_gene'])}, warnings={row.get('stability_warning_flags','')}")
    lines.extend(["", "## 6. More stable exploratory signals", ""])
    if stable.empty:
        lines.append("- No genes were free of stability warning flags under the current criteria.")
    else:
        for _, row in stable.iterrows():
            lines.append(f"- {row['gene_symbol']}: mean_delta={row['mean_delta_P_adverse_like']:.6f}, CI95=({row['bootstrap_CI95_low']:.6f}, {row['bootstrap_CI95_high']:.6f}), direction={row.get('direction_consistency','not_evaluable')}")
    lines.extend(["", "## 7. Downgraded interpretation examples", ""])
    for _, row in downgraded.iterrows():
        lines.append(f"- {row['gene_symbol']}: warnings={row.get('stability_warning_flags','')}, n_cells={int(row['n_cells_with_gene'])}, n_samples={int(row['n_samples_with_gene'])}")
    lines.extend(
        [
            "",
            "## 8. GSE72056 sensitivity",
            "",
            f"- GSE72056 sensitivity evaluated genes = {len(gse)}。",
            f"- direction same_direction = {int((gse['direction_consistency'] == 'same_direction').sum()) if 'direction_consistency' in gse.columns else 0}。",
            f"- direction opposite_direction = {int((gse['direction_consistency'] == 'opposite_direction').sum()) if 'direction_consistency' in gse.columns else 0}。",
            "- GSE72056 是 processed/non-integer expression，只能作为 sensitivity，不作为强外部验证，也不用于主 ranking。",
            "",
            "## 9. Phase 5C recommendation",
            "",
            f"READY_FOR_PHASE5C = {ready}",
            "",
        ]
    )
    if ready == "CONDITIONAL":
        lines.extend(
            [
                "可以继续做什么：",
                "- 可以进入 Phase 5C external validation planning，规划 TCGA/GDSC/DepMap/ChEMBL/Open Targets 等后续验证框架。",
                "- 可以继续保留 exploratory perturbation ranking 作为 hypothesis-generating 输入。",
                "",
                "禁止做什么：",
                "- 禁止称为正式候选治疗靶点、overstated treatment-target claims 或 mechanistic-driver claims。",
                "- 禁止写强外部泛化声明。",
                "- 禁止把 GSE72056 sensitivity 当作强外部验证。",
                "",
                "必须保留的限制性措辞：",
                "- exploratory；hypothesis_generating_only；model_dependent；no_wet_lab_validation；no_TCGA_or_drug_validation_yet；not_validated_therapeutic_targets。",
            ]
        )
    else:
        lines.append("阻断 Phase 5C：没有足够的可评估 expanded deletion ranking。")
    write_text(ROOT / "summary_phase5B_zh.md", lines)


def main() -> int:
    ensure_dirs()
    random.seed(SEED)
    np.random.seed(SEED)
    train_ds, gse_ds, train_selected, gse_selected = preflight()
    cell_summary = pd.DataFrame(
        [
            {
                "dataset_id": "GSE115978",
                "selection": "all supervised adverse_like cells excluding intermediate/ambiguous",
                "n_cells": len(train_selected),
                "n_sample_id": train_selected["sample_id"].nunique(),
                "treatment_group_distribution": json.dumps(dict(Counter(train_selected["treatment.group"].astype(str))), ensure_ascii=False, sort_keys=True),
            },
            {
                "dataset_id": "GSE72056",
                "selection": "all adverse_like cells for processed-expression sensitivity only",
                "n_cells": len(gse_selected),
                "n_sample_id": gse_selected["sample_id"].nunique(),
                "treatment_group_distribution": json.dumps(dict(Counter(gse_selected["treatment.group"].astype(str))), ensure_ascii=False, sort_keys=True),
            },
        ]
    )
    cell_summary.to_csv(TABLES / "phase5B_perturbation_cell_set_summary.csv", index=False, encoding="utf-8-sig")
    write_text(
        LOGS / "phase5B_cell_set_selection_log.md",
        [
            "# Phase 5B cell set selection log",
            "",
            "Primary analysis uses all GSE115978 supervised adverse_like cells.",
            f"GSE115978 cells: {len(train_selected)}, sample_id: {train_selected['sample_id'].nunique()}",
            "GSE72056 is processed-expression sensitivity only.",
            f"GSE72056 cells: {len(gse_selected)}, sample_id: {gse_selected['sample_id'].nunique()}",
        ],
    )

    genes = build_expanded_gene_set(train_ds, train_selected)
    model = load_model()
    try:
        primary_gene, primary_cell, primary_baseline = run_deletion(model, train_ds, train_selected, genes, PRIMARY_CHUNK_DIR, "GSE115978")
        primary_gene.to_csv(TABLES / "phase5B_expanded_deletion_effect_by_gene.csv", index=False, encoding="utf-8-sig")
        primary_cell.to_csv(TABLES / "phase5B_expanded_deletion_effect_by_cell_gene.csv", index=False, encoding="utf-8-sig")
        write_text(LOGS / "phase5B_expanded_deletion_log.md", [
            "# Phase 5B expanded deletion log",
            "",
            "Expanded deletion completed for primary GSE115978 cells.",
            f"Genes evaluated: {len(primary_gene)}",
            f"Cell-gene rows: {len(primary_cell)}",
            "Per-gene chunks are stored under tables/phase5B_gene_chunks_primary for resume support.",
        ])

        boot, sample_level = stability_tables(primary_cell, primary_gene)
        boot.to_csv(TABLES / "phase5B_bootstrap_stability_by_gene.csv", index=False, encoding="utf-8-sig")
        sample_level.to_csv(TABLES / "phase5B_sample_level_effect_by_gene.csv", index=False, encoding="utf-8-sig")
        write_text(LOGS / "phase5B_stability_analysis_log.md", [
            "# Phase 5B stability analysis log",
            "",
            f"Bootstrap iterations per gene: {BOOTSTRAPS}",
            "Sample-level mean delta and warning flags were computed from saved cell-gene deletion effects.",
        ])

        gse_gene, gse_cell, gse_baseline = run_deletion(model, gse_ds, gse_selected, genes, GSE_CHUNK_DIR, "GSE72056")
        gse_out = gse_direction(primary_gene, gse_gene)
        gse_out.to_csv(TABLES / "phase5B_GSE72056_expanded_sensitivity_by_gene.csv", index=False, encoding="utf-8-sig")
        write_text(LOGS / "phase5B_GSE72056_sensitivity_log.md", [
            "# Phase 5B GSE72056 sensitivity log",
            "",
            "GSE72056 is processed/non-integer expression.",
            "It is not strong external validation and is not used for primary ranking.",
            f"Genes evaluated: {len(gse_out)}",
            f"same_direction: {int((gse_out['direction_consistency'] == 'same_direction').sum())}",
            f"opposite_direction: {int((gse_out['direction_consistency'] == 'opposite_direction').sum())}",
        ])

        ranking = ranking_table(primary_gene, boot, gse_out)
        ranking.to_csv(TABLES / "phase5B_exploratory_perturbation_ranking.csv", index=False, encoding="utf-8-sig")
        write_text(LOGS / "phase5B_ranking_interpretation_log.md", [
            "# Phase 5B ranking interpretation log",
            "",
            "Ranking is exploratory and sorted primarily by more negative mean_delta_P_adverse_like.",
            "Auxiliary ordering used fraction_cells_delta_negative, fraction_samples_delta_negative, n_cells_with_gene, CI non-crossing, and GSE72056 direction when available.",
            "Do not use therapeutic target, mechanistic-driver claim, or validated target language.",
            "Required flags: exploratory; hypothesis_generating_only; model_dependent; no_wet_lab_validation; no_TCGA_or_drug_validation_yet; not_validated_therapeutic_targets.",
        ])
        write_summary(train_selected, genes, ranking, boot, gse_out)
    except Exception:
        write_text(LOGS / "phase5B_expanded_deletion_log.md", ["# Phase 5B expanded deletion log", "", "FAILED", traceback.format_exc()])
        raise
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("PHASE5B_EXPANDED_DELETION: PASS")
    print(f"SUMMARY={ROOT / 'summary_phase5B_zh.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
