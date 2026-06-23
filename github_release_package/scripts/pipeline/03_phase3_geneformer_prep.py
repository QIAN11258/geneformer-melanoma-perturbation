from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import pickle
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
DATA_RAW = PROJECT_ROOT / "data_raw"
TABLES = PROJECT_ROOT / "tables"
FIGURES = PROJECT_ROOT / "figures"
LOGS = PROJECT_ROOT / "logs"

MAPPING_DIR = DATA_RAW / "gene_id_mapping"
GENEFORMER_DIR = DATA_RAW / "geneformer"
for directory in (MAPPING_DIR, GENEFORMER_DIR, TABLES, FIGURES, LOGS):
    directory.mkdir(parents=True, exist_ok=True)

ANNDATA_FILES = {
    "GSE72056_melanoma": DATA_PROCESSED / "GSE72056_melanoma.h5ad",
    "GSE72056_malignant_only": DATA_PROCESSED / "GSE72056_malignant_only.h5ad",
    "GSE115978_melanoma": DATA_PROCESSED / "GSE115978_melanoma.h5ad",
    "GSE115978_malignant_validation": DATA_PROCESSED / "GSE115978_malignant_validation.h5ad",
    "GSE120575_immune_response": DATA_PROCESSED / "GSE120575_immune_response.h5ad",
}

URLS = {
    "hgnc": "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
    "geneformer_token_dict": "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl",
    "geneformer_median_dict": "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl",
    "geneformer_gene_name_id": "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl",
}

SIGNATURES = {
    "invasive_like": ["AXL", "NGFR", "VIM", "FN1", "ZEB1", "TGFBI"],
    "melanocytic_like": ["MITF", "MLANA", "PMEL", "TYR", "DCT"],
    "cycling_like": ["MKI67", "TOP2A", "PCNA", "MCM2", "STMN1"],
    "stress_hypoxia_like": ["HIF1A", "VEGFA", "CA9", "LDHA"],
}


def download_if_missing(url: str, path: Path) -> tuple[str, int | str]:
    if path.exists() and path.stat().st_size > 0:
        return "present", path.stat().st_size
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())
    return "downloaded", path.stat().st_size


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def normalize_ensg(value: str | None) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if not value:
        return ""
    return value.split(".")[0]


def split_hgnc_multi(value: str | float | None) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    value = str(value).strip()
    if not value:
        return []
    parts = re.split(r"\|", value)
    return [part.strip() for part in parts if part.strip()]


def infer_identifier_flags(symbols: list[str]) -> dict[str, int | str]:
    blank = sum(1 for symbol in symbols if not str(symbol).strip())
    ensg = sum(1 for symbol in symbols if re.match(r"^ENSG\d+", str(symbol)))
    loc = sum(1 for symbol in symbols if re.match(r"^LOC\d+", str(symbol)))
    mt = sum(1 for symbol in symbols if re.match(r"^MT-", str(symbol), flags=re.IGNORECASE))
    rps_rpl = sum(1 for symbol in symbols if re.match(r"^RP[SL]\d+", str(symbol), flags=re.IGNORECASE))
    predicted = sum(
        1
        for symbol in symbols
        if re.match(r"^(C\\d+orf\\d+|KIAA\\d+|LINC\\d+)", str(symbol), flags=re.IGNORECASE)
    )
    nonstandard = sum(
        1
        for symbol in symbols
        if str(symbol).strip() and not re.match(r"^[A-Za-z0-9_.:-]+$", str(symbol))
    )
    id_type = "Ensembl ID" if ensg / max(len(symbols), 1) >= 0.8 else "gene symbol"
    return {
        "gene_identifier_type": id_type,
        "blank_symbols": blank,
        "ensembl_like_symbols": ensg,
        "loc_symbols": loc,
        "mitochondrial_symbols": mt,
        "ribosomal_rps_rpl_symbols": rps_rpl,
        "predicted_or_locus_style_symbols": predicted,
        "nonstandard_symbol_count": nonstandard,
    }


def make_hgnc_mapping(hgnc_path: Path, gene_name_id: dict) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str, keep_default_na=False)
    candidates: dict[str, list[dict]] = defaultdict(list)
    current_symbols = set()

    for _, row in hgnc.iterrows():
        approved_symbol = row.get("symbol", "").strip()
        ensembl_id = normalize_ensg(row.get("ensembl_gene_id", ""))
        status = row.get("status", "")
        locus_group = row.get("locus_group", "")
        locus_type = row.get("locus_type", "")
        hgnc_id = row.get("hgnc_id", "")
        if approved_symbol:
            current_symbols.add(approved_symbol)
        if not ensembl_id or not approved_symbol:
            continue
        base = {
            "mapped_symbol": approved_symbol,
            "ensembl_gene_id": ensembl_id,
            "hgnc_id": hgnc_id,
            "hgnc_status": status,
            "locus_group": locus_group,
            "locus_type": locus_type,
        }
        candidates[approved_symbol].append({**base, "mapping_source": "HGNC approved symbol"})
        for prev in split_hgnc_multi(row.get("prev_symbol", "")):
            candidates[prev].append({**base, "mapping_source": "HGNC previous symbol"})
        for alias in split_hgnc_multi(row.get("alias_symbol", "")):
            candidates[alias].append({**base, "mapping_source": "HGNC alias symbol"})

    # Official Geneformer gene_name_id dictionary is used as a conservative fallback.
    if isinstance(gene_name_id, dict):
        for symbol, ensembl_id in gene_name_id.items():
            symbol = str(symbol).strip()
            ensembl_id = normalize_ensg(str(ensembl_id))
            if not symbol or not ensembl_id.startswith("ENSG"):
                continue
            candidates[symbol].append(
                {
                    "mapped_symbol": symbol,
                    "ensembl_gene_id": ensembl_id,
                    "hgnc_id": "",
                    "hgnc_status": "Geneformer dictionary",
                    "locus_group": "",
                    "locus_type": "",
                    "mapping_source": "Geneformer gene_name_id fallback",
                }
            )

    return candidates, {"current_symbols": list(current_symbols)}


def resolve_symbol(symbol: str, candidates: dict[str, list[dict]]) -> dict:
    symbol = str(symbol).strip()
    if not symbol:
        return {
            "gene_symbol": symbol,
            "ensembl_gene_id": "",
            "mapping_status": "unmapped_blank",
            "mapping_source": "",
            "mapped_symbol": "",
            "candidate_ensembl_ids": "",
            "ambiguous_reason": "",
        }
    records = candidates.get(symbol, [])
    if not records:
        return {
            "gene_symbol": symbol,
            "ensembl_gene_id": "",
            "mapping_status": "unmapped",
            "mapping_source": "",
            "mapped_symbol": "",
            "candidate_ensembl_ids": "",
            "ambiguous_reason": "",
        }
    # Deduplicate exact same Ensembl IDs, prioritizing HGNC approved > previous > alias > fallback.
    by_ens: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_ens[record["ensembl_gene_id"]].append(record)
    if len(by_ens) > 1:
        return {
            "gene_symbol": symbol,
            "ensembl_gene_id": "",
            "mapping_status": "ambiguous",
            "mapping_source": ";".join(sorted({r["mapping_source"] for r in records})),
            "mapped_symbol": ";".join(sorted({r["mapped_symbol"] for r in records})),
            "candidate_ensembl_ids": ";".join(sorted(by_ens)),
            "ambiguous_reason": "symbol maps to multiple Ensembl gene IDs across HGNC/Geneformer sources",
        }
    ensembl_id = next(iter(by_ens))
    recs = by_ens[ensembl_id]
    priority = {
        "HGNC approved symbol": 0,
        "HGNC previous symbol": 1,
        "HGNC alias symbol": 2,
        "Geneformer gene_name_id fallback": 3,
    }
    best = sorted(recs, key=lambda r: priority.get(r["mapping_source"], 9))[0]
    status_map = {
        "HGNC approved symbol": "mapped_approved",
        "HGNC previous symbol": "mapped_previous",
        "HGNC alias symbol": "mapped_alias",
        "Geneformer gene_name_id fallback": "mapped_geneformer_fallback",
    }
    return {
        "gene_symbol": symbol,
        "ensembl_gene_id": ensembl_id,
        "mapping_status": status_map.get(best["mapping_source"], "mapped"),
        "mapping_source": best["mapping_source"],
        "mapped_symbol": best["mapped_symbol"],
        "candidate_ensembl_ids": ensembl_id,
        "ambiguous_reason": "",
    }


def get_gene_symbols(adata: ad.AnnData) -> list[str]:
    if "gene_symbol" in adata.var.columns:
        return adata.var["gene_symbol"].astype(str).tolist()
    return [str(value) for value in adata.var_names]


def sparse_stats(adata: ad.AnnData) -> dict:
    x = adata.X
    if sparse.issparse(x):
        data = x.data
        total = np.asarray(x.sum(axis=1)).ravel()
        detected = np.asarray(x.getnnz(axis=1)).ravel()
        gene_detected_fraction = np.asarray(x.getnnz(axis=0)).ravel() / x.shape[0]
    else:
        arr = np.asarray(x)
        data = arr.ravel()
        total = arr.sum(axis=1)
        detected = (arr > 0).sum(axis=1)
        gene_detected_fraction = (arr > 0).sum(axis=0) / arr.shape[0]
    nonzero_data = data[data != 0]
    if nonzero_data.size:
        negative_values = int((nonzero_data < 0).sum())
        integer_fraction = float(np.mean(np.isclose(nonzero_data, np.rint(nonzero_data))))
        min_value = float(nonzero_data.min())
        max_value = float(nonzero_data.max())
    else:
        negative_values = 0
        integer_fraction = 1.0
        min_value = 0.0
        max_value = 0.0
    return {
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "negative_values": negative_values,
        "nonzero_integer_fraction": integer_fraction,
        "min_nonzero_value": min_value,
        "max_value": max_value,
        "total_expression_min": float(np.min(total)),
        "total_expression_median": float(np.median(total)),
        "total_expression_mean": float(np.mean(total)),
        "total_expression_max": float(np.max(total)),
        "detected_genes_min": int(np.min(detected)),
        "detected_genes_median": float(np.median(detected)),
        "detected_genes_mean": float(np.mean(detected)),
        "detected_genes_max": int(np.max(detected)),
        "gene_detected_fraction_median": float(np.median(gene_detected_fraction)),
        "gene_detected_fraction_mean": float(np.mean(gene_detected_fraction)),
        "total_expression": total,
        "detected_genes": detected,
    }


def classify_expression(anndata_id: str, stats: dict) -> tuple[str, str]:
    if anndata_id.startswith("GSE115978") and stats["nonzero_integer_fraction"] >= 0.999:
        return "raw count-like integer matrix", "suitable technical pilot for Geneformer tokenization after Ensembl mapping"
    if anndata_id.startswith("GSE120575"):
        return "TPM processed expression", "immune response validation only; not malignant discovery input"
    if anndata_id.startswith("GSE72056"):
        return "unknown processed expression / likely normalized", "processed-expression sensitivity analysis only unless raw counts are recovered"
    if stats["negative_values"] > 0:
        return "scaled expression or transformed matrix", "not suitable for Geneformer tokenization without raw count recovery"
    return "unknown processed expression", "needs manual confirmation"


def build_figures(hist_data: dict[str, dict]) -> None:
    for key, path, xlabel in [
        ("detected_genes", FIGURES / "detected_genes_per_cell_histogram.png", "Detected genes per cell"),
        ("total_expression", FIGURES / "total_expression_per_cell_histogram.png", "Total expression per cell"),
    ]:
        n = len(hist_data)
        fig, axes = plt.subplots(n, 1, figsize=(9, max(3, 2.6 * n)), constrained_layout=True)
        if n == 1:
            axes = [axes]
        for ax, (dataset_id, values) in zip(axes, hist_data.items()):
            arr = np.asarray(values[key], dtype=float)
            ax.hist(arr, bins=60, color="#3366aa", alpha=0.85)
            ax.set_title(dataset_id)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Cells")
        fig.savefig(path, dpi=180)
        plt.close(fig)


def dense_rows_for_indices(adata: ad.AnnData, indices: np.ndarray) -> np.ndarray:
    subset = adata.X[indices, :]
    if sparse.issparse(subset):
        return subset.toarray()
    return np.asarray(subset)


def tokenization_dry_run(
    anndata_id: str,
    path: Path,
    mapping_df: pd.DataFrame,
    token_dict: dict,
    median_dict: dict,
    seed: int = 20260621,
) -> tuple[dict, list[str]]:
    adata = ad.read_h5ad(path)
    rng = np.random.default_rng(seed)
    n = min(100, adata.n_obs)
    if adata.n_obs <= n:
        sampled = np.arange(adata.n_obs)
    else:
        sampled = np.sort(rng.choice(adata.n_obs, size=n, replace=False))
    genes = get_gene_symbols(adata)
    map_by_symbol = mapping_df.drop_duplicates("gene_symbol").set_index("gene_symbol").to_dict("index")
    gene_info = []
    duplicate_token_symbols = Counter()
    for idx, symbol in enumerate(genes):
        rec = map_by_symbol.get(symbol, {})
        ensg = rec.get("ensembl_gene_id", "")
        if not ensg or rec.get("mapping_status") == "ambiguous":
            gene_info.append((idx, symbol, "", None, None))
            continue
        token = token_dict.get(ensg)
        median = median_dict.get(ensg, 1.0) if isinstance(median_dict, dict) else 1.0
        try:
            median = float(median)
        except Exception:
            median = 1.0
        if not median or not np.isfinite(median):
            median = 1.0
        gene_info.append((idx, symbol, ensg, token, median))
        if token is not None:
            duplicate_token_symbols[ensg] += 1

    rows = dense_rows_for_indices(adata, sampled)
    lengths = []
    v1_lengths = []
    v2_lengths = []
    empty = 0
    short = 0
    unmapped_excluded = 0
    attention_mask_ok = True
    token_id_ok = True

    for row in rows:
        token_scores: dict[int, float] = {}
        for idx, symbol, ensg, token, median in gene_info:
            expr = float(row[idx])
            if expr <= 0:
                continue
            if token is None:
                unmapped_excluded += 1
                continue
            score = expr / median
            if token not in token_scores or score > token_scores[token]:
                token_scores[token] = score
        ranked = sorted(token_scores.items(), key=lambda kv: kv[1], reverse=True)
        sequence = [int(token) for token, _ in ranked]
        if not sequence:
            empty += 1
        if len(sequence) < 100:
            short += 1
        if any(token is None or int(token) < 0 for token in sequence):
            token_id_ok = False
        mask = [1] * min(len(sequence), 4096)
        if len(mask) != min(len(sequence), 4096) or any(value != 1 for value in mask):
            attention_mask_ok = False
        lengths.append(len(sequence))
        v1_lengths.append(min(len(sequence), 2048))
        v2_lengths.append(min(len(sequence), 4096))

    special_keys = [key for key in token_dict if isinstance(key, str) and key.startswith("<")]
    summary = {
        "dataset_id": anndata_id,
        "source_h5ad": str(path.relative_to(PROJECT_ROOT)),
        "sampled_cells": int(n),
        "token_dictionary_size": len(token_dict),
        "median_dictionary_size": len(median_dict) if isinstance(median_dict, dict) else 0,
        "special_token_keys_detected": ";".join(map(str, special_keys)) if special_keys else "none_detected",
        "sequence_length_min": int(np.min(lengths)) if lengths else 0,
        "sequence_length_median": float(np.median(lengths)) if lengths else 0,
        "sequence_length_max": int(np.max(lengths)) if lengths else 0,
        "v1_2048_length_median": float(np.median(v1_lengths)) if v1_lengths else 0,
        "v2_4096_length_median": float(np.median(v2_lengths)) if v2_lengths else 0,
        "empty_sequences": empty,
        "abnormally_short_sequences_lt100": short,
        "attention_mask_check": "pass" if attention_mask_ok else "fail",
        "token_id_check": "pass" if token_id_ok else "fail",
        "unmapped_positive_gene_events_excluded": int(unmapped_excluded),
        "dry_run_status": "pass" if empty == 0 and token_id_ok and attention_mask_ok else "review_required",
        "notes": "Dry run only; no fine-tuning, perturbation, deletion, or candidate target ranking.",
    }
    log = [
        f"## {anndata_id}",
        "",
        f"- Sampled cells: {n}",
        f"- Raw tokenized sequence length min/median/max: {summary['sequence_length_min']}/{summary['sequence_length_median']}/{summary['sequence_length_max']}",
        f"- V1 2048 clipped median length: {summary['v1_2048_length_median']}",
        f"- V2 4096 clipped median length: {summary['v2_4096_length_median']}",
        f"- Empty sequences: {empty}",
        f"- Short sequences <100: {short}",
        f"- Attention mask check: {summary['attention_mask_check']}",
        f"- Token ID check: {summary['token_id_check']}",
        f"- Unmapped positive gene events excluded: {unmapped_excluded}",
        "",
    ]
    return summary, log


def score_signatures() -> None:
    path = ANNDATA_FILES["GSE72056_malignant_only"]
    adata = ad.read_h5ad(path)
    gene_symbols = pd.Series(get_gene_symbols(adata), index=np.arange(adata.n_vars))
    symbol_to_indices = defaultdict(list)
    for idx, symbol in enumerate(gene_symbols):
        symbol_to_indices[symbol].append(idx)

    marker_rows = []
    score_arrays = {}
    for signature, markers in SIGNATURES.items():
        present_indices = []
        for marker in markers:
            indices = symbol_to_indices.get(marker, [])
            marker_rows.append(
                {
                    "signature": signature,
                    "marker": marker,
                    "present_in_var": bool(indices),
                    "n_matching_features": len(indices),
                    "feature_indices": ";".join(map(str, indices)) if indices else "",
                }
            )
            present_indices.extend(indices)
        if present_indices:
            sub = adata.X[:, present_indices]
            if sparse.issparse(sub):
                score = np.asarray(sub.mean(axis=1)).ravel()
            else:
                score = np.asarray(sub).mean(axis=1)
        else:
            score = np.full(adata.n_obs, np.nan)
        score_arrays[signature] = score
        adata.obs[f"{signature}_score"] = score
        mean = np.nanmean(score)
        sd = np.nanstd(score)
        z = (score - mean) / sd if sd > 0 else np.zeros_like(score)
        adata.obs[f"{signature}_zscore"] = z

    zmat = np.vstack([adata.obs[f"{sig}_zscore"].to_numpy(dtype=float) for sig in SIGNATURES]).T
    labels = []
    sig_names = list(SIGNATURES)
    for row in zmat:
        order = np.argsort(row)[::-1]
        top = order[0]
        second = order[1] if len(order) > 1 else order[0]
        top_score = row[top]
        margin = top_score - row[second]
        if np.isfinite(top_score) and top_score >= 0.5 and margin >= 0.25:
            labels.append(sig_names[top])
        else:
            labels.append("intermediate/ambiguous")
    adata.obs["preliminary_malignant_state"] = labels
    adata.uns["phase3_malignant_state_label_rule"] = (
        "Mean available marker expression per signature; z-scored across malignant cells. "
        "Label assigned only when top signature z-score >= 0.5 and margin to second >= 0.25; otherwise intermediate/ambiguous."
    )
    adata.write_h5ad(DATA_PROCESSED / "GSE72056_malignant_state_labeled.h5ad", compression="gzip")

    pd.DataFrame(marker_rows).to_csv(
        TABLES / "GSE72056_malignant_state_signature_marker_check.csv", index=False
    )
    score_summary_rows = []
    for signature in SIGNATURES:
        score = adata.obs[f"{signature}_score"].to_numpy(dtype=float)
        zscore = adata.obs[f"{signature}_zscore"].to_numpy(dtype=float)
        score_summary_rows.append(
            {
                "signature": signature,
                "n_markers_requested": len(SIGNATURES[signature]),
                "n_markers_present": int(
                    sum(1 for marker in SIGNATURES[signature] if marker in symbol_to_indices)
                ),
                "score_min": float(np.nanmin(score)),
                "score_median": float(np.nanmedian(score)),
                "score_mean": float(np.nanmean(score)),
                "score_max": float(np.nanmax(score)),
                "zscore_min": float(np.nanmin(zscore)),
                "zscore_median": float(np.nanmedian(zscore)),
                "zscore_mean": float(np.nanmean(zscore)),
                "zscore_max": float(np.nanmax(zscore)),
            }
        )
    pd.DataFrame(score_summary_rows).to_csv(
        TABLES / "GSE72056_malignant_state_score_summary.csv", index=False
    )
    dist = adata.obs["preliminary_malignant_state"].value_counts().rename_axis("state").reset_index(name="n_cells")
    dist["percentage"] = dist["n_cells"] / adata.n_obs * 100
    dist.to_csv(TABLES / "GSE72056_malignant_state_label_distribution.csv", index=False)

    lines = [
        "# Phase 3 malignant-state label log",
        "",
        "No DEG, model training, or perturbation was performed.",
        "",
        "Label rule:",
        "",
        "- Mean available marker expression per signature.",
        "- Scores are z-scored across GSE72056 malignant cells.",
        "- Label assigned only when top z-score >= 0.5 and margin to second signature >= 0.25.",
        "- Otherwise cells are labelled `intermediate/ambiguous`.",
        "",
    ]
    for _, row in dist.iterrows():
        lines.append(f"- {row['state']}: {int(row['n_cells'])} cells ({row['percentage']:.2f}%)")
    (LOGS / "phase3_malignant_state_label_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def deterministic_split(units: list[str]) -> dict[str, str]:
    units = sorted(set(units))
    ranked = sorted(units, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    n = len(ranked)
    n_test = max(1, round(n * 0.2)) if n >= 5 else max(1, n // 5)
    n_val = max(1, round(n * 0.2)) if n >= 5 else max(1, n // 5)
    assignments = {}
    for idx, unit in enumerate(ranked):
        if idx < n_test:
            split = "held-out test"
        elif idx < n_test + n_val:
            split = "validation"
        else:
            split = "train"
        assignments[unit] = split
    return assignments


def build_patient_splits() -> None:
    rows = []
    log = [
        "# Phase 3 patient-level split log",
        "",
        "Cell-level random split is not used.",
        "Splits are deterministic hashes of available patient/sample units.",
        "",
    ]
    split_specs = {
        "GSE72056_melanoma": ("tumor_id", "tumor_id from embedded matrix metadata; patient identity needs manual confirmation"),
        "GSE115978_melanoma": ("sample_id", "sample_id from cell.annotations; true patient-level identity needs manual confirmation"),
    }
    for dataset_id, (field, note) in split_specs.items():
        adata = ad.read_h5ad(ANNDATA_FILES[dataset_id], backed="r")
        obs = adata.obs.copy()
        adata.file.close()
        if field not in obs.columns:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "split": "needs manual confirmation",
                    "patient_or_sample_id": "needs manual confirmation",
                    "n_cells": "needs manual confirmation",
                    "n_malignant_cells": "needs manual confirmation",
                    "source_field": field,
                    "patient_id_status": "missing",
                    "notes": note,
                }
            )
            continue
        units = obs[field].astype(str).tolist()
        assignments = deterministic_split(units)
        for unit, split in sorted(assignments.items(), key=lambda kv: (kv[1], kv[0])):
            sub = obs[obs[field].astype(str) == unit]
            if "malignant_label" in sub.columns:
                malignant = int((sub["malignant_label"].astype(str) == "malignant").sum())
            elif "cell_type" in sub.columns:
                malignant = int((sub["cell_type"].astype(str) == "Mal").sum())
            else:
                malignant = "needs manual confirmation"
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "split": split,
                    "patient_or_sample_id": unit,
                    "n_cells": int(sub.shape[0]),
                    "n_malignant_cells": malignant,
                    "source_field": field,
                    "patient_id_status": "available_but_needs_manual_confirmation",
                    "notes": note,
                }
            )
        counts = Counter(assignments.values())
        log.append(f"## {dataset_id}")
        log.append("")
        log.append(f"- Source field: {field}")
        log.append(f"- Unit count: {len(assignments)}")
        log.append(f"- Train units: {counts.get('train', 0)}")
        log.append(f"- Validation units: {counts.get('validation', 0)}")
        log.append(f"- Held-out test units: {counts.get('held-out test', 0)}")
        log.append(f"- Note: {note}")
        log.append("")
    pd.DataFrame(rows).to_csv(TABLES / "patient_level_split_plan.csv", index=False)
    (LOGS / "phase3_patient_split_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")


def main() -> None:
    download_rows = []
    hgnc_path = MAPPING_DIR / "hgnc_complete_set.txt"
    token_path = GENEFORMER_DIR / "token_dictionary_gc30M.pkl"
    median_path = GENEFORMER_DIR / "gene_median_dictionary_gc30M.pkl"
    name_id_path = GENEFORMER_DIR / "gene_name_id_dict_gc30M.pkl"
    for label, url, path in [
        ("HGNC complete set", URLS["hgnc"], hgnc_path),
        ("Geneformer token dictionary gc30M", URLS["geneformer_token_dict"], token_path),
        ("Geneformer gene median dictionary gc30M", URLS["geneformer_median_dict"], median_path),
        ("Geneformer gene_name_id dictionary gc30M", URLS["geneformer_gene_name_id"], name_id_path),
    ]:
        status, size = download_if_missing(url, path)
        download_rows.append({"resource": label, "url": url, "path": str(path.relative_to(PROJECT_ROOT)), "status": status, "bytes": size})
    pd.DataFrame(download_rows).to_csv(TABLES / "phase3_resource_downloads.csv", index=False)

    token_dict = load_pickle(token_path)
    median_dict = load_pickle(median_path)
    gene_name_id = load_pickle(name_id_path)
    geneformer_vocab = {normalize_ensg(key) for key in token_dict.keys() if str(key).startswith("ENSG")}
    candidates, _ = make_hgnc_mapping(hgnc_path, gene_name_id)

    all_symbols = set()
    anndata_symbols = {}
    var_check_rows = []
    for anndata_id, path in ANNDATA_FILES.items():
        adata = ad.read_h5ad(path, backed="r")
        symbols = get_gene_symbols(adata)
        adata.file.close()
        anndata_symbols[anndata_id] = symbols
        all_symbols.update(symbols)
        flags = infer_identifier_flags(symbols)
        counts = Counter(symbols)
        var_check_rows.append(
            {
                "dataset_id": anndata_id,
                "n_gene_rows": len(symbols),
                "n_unique_symbols": len(counts),
                "n_duplicated_symbols": sum(1 for _, c in counts.items() if c > 1),
                "n_duplicated_gene_rows": sum(c for _, c in counts.items() if c > 1),
                **flags,
            }
        )
    pd.DataFrame(var_check_rows).to_csv(TABLES / "phase3_var_name_check_summary.csv", index=False)

    resolved = {symbol: resolve_symbol(symbol, candidates) for symbol in sorted(all_symbols)}
    mapping_rows = list(resolved.values())
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(TABLES / "gene_symbol_to_ensembl_mapping_phase3.csv", index=False)

    summary_rows = []
    unmapped_rows = []
    duplicated_rows = []
    ambiguous_rows = []
    overlap_rows = []
    for anndata_id, symbols in anndata_symbols.items():
        counts = Counter(symbols)
        unique_symbols = sorted(counts)
        mapped_ens = []
        in_vocab = []
        mapped_status_counts = Counter()
        duplicate_resolved = 0
        duplicate_unresolved = 0
        for symbol in unique_symbols:
            rec = resolved[symbol]
            mapped_status_counts[rec["mapping_status"]] += 1
            ensg = rec["ensembl_gene_id"]
            if ensg:
                mapped_ens.append(ensg)
                if ensg in geneformer_vocab:
                    in_vocab.append(ensg)
            if rec["mapping_status"].startswith("unmapped"):
                unmapped_rows.append({"dataset_id": anndata_id, **rec})
            if rec["mapping_status"] == "ambiguous":
                ambiguous_rows.append({"dataset_id": anndata_id, **rec})
        for symbol, count in counts.items():
            if count <= 1:
                continue
            rec = resolved[symbol]
            status = "resolved" if rec["ensembl_gene_id"] and rec["mapping_status"] != "ambiguous" else "unresolved"
            if status == "resolved":
                duplicate_resolved += count
            else:
                duplicate_unresolved += count
            duplicated_rows.append(
                {
                    "dataset_id": anndata_id,
                    "gene_symbol": symbol,
                    "occurrences": count,
                    "mapping_status": rec["mapping_status"],
                    "ensembl_gene_id": rec["ensembl_gene_id"],
                    "duplicate_resolution_status": status,
                    "phase3_rule": "no gene rows deleted; duplicates only resolved for tokenization dry run if one unique Ensembl ID exists",
                }
            )
        mapped_unique = len(set(mapped_ens))
        vocab_unique = len(set(in_vocab))
        total = len(unique_symbols)
        overlap_pct = vocab_unique / total * 100 if total else 0
        unmapped_pct = (total - mapped_unique) / total * 100 if total else 0
        if overlap_pct < 70:
            risk = "high risk"
        elif overlap_pct <= 85:
            risk = "moderate risk"
        else:
            risk = "acceptable"
        summary_rows.append(
            {
                "dataset_id": anndata_id,
                "total_gene_rows": len(symbols),
                "unique_gene_symbols": total,
                "mapped_ensembl_genes": mapped_unique,
                "geneformer_vocab_genes": vocab_unique,
                "overlap_percentage": overlap_pct,
                "unmapped_percentage": unmapped_pct,
                "duplicate_resolved_count": duplicate_resolved,
                "duplicate_unresolved_count": duplicate_unresolved,
                "ambiguous_symbol_count": mapped_status_counts["ambiguous"],
                "unmapped_symbol_count": mapped_status_counts["unmapped"] + mapped_status_counts["unmapped_blank"],
                "risk_label": risk,
            }
        )
        overlap_rows.append(summary_rows[-1])

    pd.DataFrame(summary_rows).to_csv(TABLES / "gene_id_mapping_summary.csv", index=False)
    pd.DataFrame(unmapped_rows).to_csv(TABLES / "genes_unmapped_by_dataset.csv", index=False)
    pd.DataFrame(duplicated_rows).to_csv(TABLES / "duplicated_genes_by_dataset.csv", index=False)
    pd.DataFrame(ambiguous_rows).to_csv(TABLES / "ambiguous_gene_mappings.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(TABLES / "geneformer_vocab_overlap.csv", index=False)

    # Expression value type and figures
    expr_rows = []
    hist_data = {}
    for anndata_id, path in ANNDATA_FILES.items():
        adata = ad.read_h5ad(path)
        stats = sparse_stats(adata)
        expr_type, recommendation = classify_expression(anndata_id, stats)
        hist_data[anndata_id] = {
            "detected_genes": stats.pop("detected_genes"),
            "total_expression": stats.pop("total_expression"),
        }
        expr_rows.append({"dataset_id": anndata_id, "expression_value_type": expr_type, "phase3_recommendation": recommendation, **stats})
    pd.DataFrame(expr_rows).to_csv(TABLES / "expression_value_type_summary.csv", index=False)
    build_figures(hist_data)

    # Tokenization dry run on requested datasets.
    dry_summaries = []
    dry_log = [
        "# Phase 3 tokenization dry run log",
        "",
        "Dry run only. No fine-tuning, in silico deletion, perturbation, or candidate target ranking was performed.",
        "",
        "Method: map gene symbols to Ensembl IDs, retain Ensembl IDs found in Geneformer gc30M token dictionary, divide expression by Geneformer gc30M non-zero median when available, rank genes by normalized expression, and inspect at most 100 sampled cells.",
        "",
    ]
    for anndata_id in ["GSE115978_malignant_validation", "GSE72056_malignant_only"]:
        summary, log_lines = tokenization_dry_run(
            anndata_id,
            ANNDATA_FILES[anndata_id],
            mapping_df,
            token_dict,
            median_dict,
        )
        dry_summaries.append(summary)
        dry_log.extend(log_lines)
    pd.DataFrame(dry_summaries).to_csv(TABLES / "tokenization_dry_run_summary.csv", index=False)
    (LOGS / "phase3_tokenization_dry_run_log.md").write_text("\n".join(dry_log) + "\n", encoding="utf-8")

    score_signatures()
    build_patient_splits()

    log_lines = [
        "# Phase 3 gene ID mapping and vocabulary log",
        "",
        "Sources:",
        "",
        f"- HGNC complete set: {URLS['hgnc']}",
        f"- Geneformer token dictionary: {URLS['geneformer_token_dict']}",
        f"- Geneformer median dictionary: {URLS['geneformer_median_dict']}",
        f"- Geneformer gene_name_id dictionary: {URLS['geneformer_gene_name_id']}",
        "",
        "Rules:",
        "",
        "- Gene symbols are first mapped with HGNC approved symbols.",
        "- HGNC previous and alias symbols are used only when they map to one unique Ensembl ID.",
        "- Geneformer gene_name_id dictionary is used as fallback.",
        "- Ambiguous symbols are not forced into one mapping.",
        "- Unmapped, duplicated, and ambiguous genes are exported separately.",
        "- No genes are deleted from AnnData objects in Phase 3.",
        "",
    ]
    for row in overlap_rows:
        log_lines.append(
            f"- {row['dataset_id']}: overlap {row['overlap_percentage']:.2f}%, risk {row['risk_label']}, mapped {row['mapped_ensembl_genes']}/{row['unique_gene_symbols']}."
        )
    (LOGS / "phase3_gene_id_mapping_log.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
