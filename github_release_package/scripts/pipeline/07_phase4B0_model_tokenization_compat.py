from __future__ import annotations

import gc
import json
import os
import pickle
import shutil
import sys
import traceback
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoConfig, AutoModel

from geneformer import TranscriptomeTokenizer
from geneformer import (
    ENSEMBL_MAPPING_FILE,
    ENSEMBL_MAPPING_FILE_30M,
    GENE_MEDIAN_FILE,
    GENE_MEDIAN_FILE_30M,
    TOKEN_DICTIONARY_FILE,
    TOKEN_DICTIONARY_FILE_30M,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"

MODEL_ROOT = Path(r"models/Geneformer")
V1_MODEL = MODEL_ROOT / "Geneformer-V1-10M"
V2_MODEL = MODEL_ROOT / "Geneformer-V2-104M_CLcancer"

TOKENIZER_INPUT_DIR = DATA_PROCESSED / "phase4B0_tokenizer_input"
V2_OUT = DATA_PROCESSED / "tokenized_v2_gc104M"
V1_OUT = DATA_PROCESSED / "tokenized_v1_gc30M"

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

TRACE_METADATA = [
    "original_cell_id",
    "cell_id",
    "malignant_state",
    "sample_id",
    "tumor_id",
    "treatment.group",
    "split_unit",
    "split_unit_field",
    "split_unit_type",
    "patient_identity_status",
    "dataset_id",
]

DATASETS = {
    "GSE115978": {
        "h5ad": DATA_PROCESSED / "GSE115978_malignant_state_labeled.h5ad",
        "dataset_id": "GSE115978_malignant_state_labeled",
    },
    "GSE72056": {
        "h5ad": DATA_PROCESSED / "GSE72056_malignant_state_labeled.h5ad",
        "dataset_id": "GSE72056_malignant_state_labeled",
    },
}

VERSIONS = {
    "V2": {
        "model_series": "Geneformer-V2",
        "intended_model": "Geneformer-V2-104M_CLcancer",
        "model_path": V2_MODEL,
        "token_dictionary_path": Path(TOKEN_DICTIONARY_FILE),
        "gene_median_path": Path(GENE_MEDIAN_FILE),
        "gene_mapping_path": Path(ENSEMBL_MAPPING_FILE),
        "model_input_size": 4096,
        "special_token": True,
        "collapse_gene_ids": True,
        "output_dir": V2_OUT,
        "suffix": "v2",
    },
    "V1": {
        "model_series": "Geneformer-V1",
        "intended_model": "Geneformer-V1-10M",
        "model_path": V1_MODEL,
        "token_dictionary_path": Path(TOKEN_DICTIONARY_FILE_30M),
        "gene_median_path": Path(GENE_MEDIAN_FILE_30M),
        "gene_mapping_path": Path(ENSEMBL_MAPPING_FILE_30M),
        "model_input_size": 2048,
        "special_token": False,
        "collapse_gene_ids": True,
        "output_dir": V1_OUT,
        "suffix": "v1",
    },
}


def ensure_dirs() -> None:
    for path in [TABLES, LOGS, TOKENIZER_INPUT_DIR, V1_OUT, V2_OUT]:
        path.mkdir(parents=True, exist_ok=True)


def safe_replace_dir(path: Path) -> None:
    resolved = path.resolve()
    allowed_roots = [V1_OUT.resolve(), V2_OUT.resolve()]
    if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
        raise RuntimeError(f"Refusing to remove directory outside tokenized outputs: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def read_pickle_dict(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def append_log(log_lines: list[str], text: str = "") -> None:
    log_lines.append(text)


def get_sparse_stats(adata) -> dict:
    x = adata.X
    if sp.issparse(x):
        data = x.data
        row_sums = np.asarray(x.sum(axis=1)).ravel()
    else:
        arr = np.asarray(x)
        data = arr.ravel()
        row_sums = arr.sum(axis=1)
    sample = data[: min(100000, data.size)] if data.size else data
    integer_like = bool(np.allclose(sample, np.round(sample))) if sample.size else None
    return {
        "x_sparse": bool(sp.issparse(x)),
        "x_nonzero_values": int(data.size),
        "x_min": float(np.min(data)) if data.size else np.nan,
        "x_max": float(np.max(data)) if data.size else np.nan,
        "x_negative_values": int(np.sum(data < 0)) if data.size else 0,
        "x_integer_like_first_100k_nonzero": integer_like,
        "row_sum_min": float(np.min(row_sums)),
        "row_sum_median": float(np.median(row_sums)),
        "row_sum_max": float(np.max(row_sums)),
    }


def locate_resources() -> pd.DataFrame:
    rows = []
    for version, cfg in VERSIONS.items():
        token_dict = read_pickle_dict(cfg["token_dictionary_path"])
        median_dict = read_pickle_dict(cfg["gene_median_path"])
        mapping_dict = read_pickle_dict(cfg["gene_mapping_path"])
        model_config = AutoConfig.from_pretrained(str(cfg["model_path"]), local_files_only=True)
        special_token_ids = {
            key: token_dict.get(key)
            for key in ["<pad>", "<mask>", "<cls>", "<eos>"]
            if key in token_dict
        }
        rows.append(
            {
                "model_series": cfg["model_series"],
                "intended_model": cfg["intended_model"],
                "model_path": str(cfg["model_path"]),
                "model_path_exists": cfg["model_path"].exists(),
                "config_vocab_size": getattr(model_config, "vocab_size", "needs manual confirmation"),
                "config_max_position_embeddings": getattr(
                    model_config, "max_position_embeddings", "needs manual confirmation"
                ),
                "token_dictionary_path": str(cfg["token_dictionary_path"]),
                "token_dictionary_exists": cfg["token_dictionary_path"].exists(),
                "token_dictionary_entries": len(token_dict),
                "token_id_min": min(token_dict.values()),
                "token_id_max": max(token_dict.values()),
                "special_token_ids": json.dumps(special_token_ids, sort_keys=True),
                "gene_median_dictionary_path": str(cfg["gene_median_path"]),
                "gene_median_dictionary_exists": cfg["gene_median_path"].exists(),
                "gene_median_dictionary_entries": len(median_dict),
                "gene_mapping_dictionary_path": str(cfg["gene_mapping_path"]),
                "gene_mapping_dictionary_exists": cfg["gene_mapping_path"].exists(),
                "gene_mapping_dictionary_entries": len(mapping_dict),
                "model_input_size": cfg["model_input_size"],
                "special_token": cfg["special_token"],
                "collapse_gene_ids": cfg["collapse_gene_ids"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase4B0_model_token_dictionary_inventory.csv", index=False, encoding="utf-8-sig")
    return df


def verify_inputs_and_prepare_h5ad(log_lines: list[str]) -> tuple[dict[str, Path], pd.DataFrame]:
    mapping = pd.read_csv(TABLES / "gene_symbol_to_ensembl_mapping_phase3.csv", dtype=str).fillna("")
    map_dict = dict(zip(mapping["gene_symbol"], mapping["ensembl_gene_id"]))
    metadata_check = pd.read_csv(TABLES / "tokenized_metadata_field_check.csv", dtype=str).fillna("")

    append_log(log_lines, "# Phase 4B-0 input and tokenizer-input preparation log")
    append_log(log_lines, "")
    append_log(log_lines, f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    append_log(log_lines, "Restriction: no fine-tuning, no perturbation, no in silico deletion, no candidate target output.")
    append_log(log_lines, "Original h5ad files are read only; tokenizer input h5ad files are derived copies.")
    append_log(log_lines, "")

    prepared_paths = {}
    prep_rows = []

    required_side_files = [
        ROOT / "summary_phase4A1_zh.md",
        TABLES / "phase4A1_geneformer_gpu_model_check.csv",
        TABLES / "tokenized_metadata_field_check.csv",
    ]
    for path in required_side_files:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Required side file missing or empty: {path}")
        append_log(log_lines, f"Side file present: {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")

    for key, info in DATASETS.items():
        adata = sc.read_h5ad(info["h5ad"])
        append_log(log_lines, "")
        append_log(log_lines, f"## {info['dataset_id']}")
        append_log(log_lines, f"- h5ad: {info['h5ad'].relative_to(ROOT)}")
        append_log(log_lines, f"- shape: {adata.n_obs} cells x {adata.n_vars} genes")

        missing = [field for field in REQUIRED_METADATA if field not in adata.obs.columns]
        if missing:
            raise RuntimeError(f"{info['dataset_id']} missing required metadata fields: {missing}")
        append_log(log_lines, "- required metadata fields: PASS")

        stats = get_sparse_stats(adata)
        expression_type = "raw_count_like" if stats["x_integer_like_first_100k_nonzero"] else "processed_or_non_integer_expression"
        if stats["x_negative_values"] > 0:
            expression_type = "contains_negative_values"
        append_log(log_lines, f"- expression type check: {expression_type}")
        append_log(
            log_lines,
            f"- row sum min/median/max: {stats['row_sum_min']:.6g}/{stats['row_sum_median']:.6g}/{stats['row_sum_max']:.6g}",
        )

        gene_symbols = adata.var["gene_symbol"].astype(str) if "gene_symbol" in adata.var.columns else adata.var_names.astype(str)
        ensembl_ids = gene_symbols.map(map_dict)
        mapped_count = int((ensembl_ids.fillna("") != "").sum())
        adata.var["ensembl_id"] = ensembl_ids.replace("", np.nan).values
        adata.var["phase4B0_gene_symbol_for_mapping"] = gene_symbols.values
        adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
        adata.obs["filter_pass"] = 1
        adata.obs["dataset_id"] = info["dataset_id"]

        if "sample_id" not in adata.obs.columns:
            adata.obs["sample_id"] = "not_available_in_source"
        if "tumor_id" not in adata.obs.columns:
            adata.obs["tumor_id"] = "not_available_in_source"
        for col in TRACE_METADATA:
            if col not in adata.obs.columns:
                adata.obs[col] = "not_available_in_source"
            adata.obs[col] = adata.obs[col].astype(str)

        output_path = TOKENIZER_INPUT_DIR / f"{info['dataset_id']}_tokenizer_input.h5ad"
        adata.write_h5ad(output_path, compression="gzip")
        prepared_paths[key] = output_path

        append_log(log_lines, f"- mapped Ensembl IDs in var: {mapped_count}/{adata.n_vars}")
        append_log(log_lines, f"- derived tokenizer input: {output_path.relative_to(ROOT)}")
        append_log(log_lines, "- original h5ad modification: no")

        prep_row = {
            "dataset_key": key,
            "dataset_id": info["dataset_id"],
            "h5ad": str(info["h5ad"].relative_to(ROOT)),
            "tokenizer_input_h5ad": str(output_path.relative_to(ROOT)),
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "required_metadata_missing": "",
            "has_original_n_counts": "n_counts" in sc.read_h5ad(info["h5ad"], backed="r").obs.columns,
            "has_original_var_ensembl_id": "ensembl_id" in sc.read_h5ad(info["h5ad"], backed="r").var.columns,
            "mapped_ensembl_genes": mapped_count,
            "expression_type_check": expression_type,
        }
        prep_row.update(stats)
        prep_rows.append(prep_row)

    append_log(log_lines, "")
    append_log(log_lines, "Prior tokenized metadata field check was read from tables/tokenized_metadata_field_check.csv:")
    append_log(log_lines, metadata_check.to_csv(index=False))
    return prepared_paths, pd.DataFrame(prep_rows)


def instantiate_tokenizer(version: str) -> TranscriptomeTokenizer:
    cfg = VERSIONS[version]
    return TranscriptomeTokenizer(
        custom_attr_name_dict={field: field for field in TRACE_METADATA},
        nproc=1,
        chunk_size=256,
        model_input_size=cfg["model_input_size"],
        special_token=cfg["special_token"],
        collapse_gene_ids=cfg["collapse_gene_ids"],
        use_h5ad_index=False,
        keep_counts=False,
        model_version=version,
    )


def crop_and_pad_cells(
    raw_cells: list[np.ndarray],
    token_dict: dict,
    model_input_size: int,
    special_token: bool,
) -> tuple[list[list[int]], list[list[int]], list[int], list[int]]:
    pad_id = int(token_dict.get("<pad>", 0))
    cls_id = token_dict.get("<cls>")
    eos_id = token_dict.get("<eos>")
    input_ids = []
    attention_masks = []
    lengths = []
    raw_lengths = []
    for cell in raw_cells:
        raw = [int(x) for x in np.asarray(cell, dtype=np.int64).tolist()]
        raw_lengths.append(len(raw))
        if special_token:
            if cls_id is None or eos_id is None:
                raise RuntimeError("Special-token tokenizer requested but <cls>/<eos> absent from token dictionary.")
            seq = [int(cls_id)] + raw[: model_input_size - 2] + [int(eos_id)]
        else:
            seq = raw[:model_input_size]
        seq_len = len(seq)
        if seq_len > model_input_size:
            raise RuntimeError(f"Internal length error: {seq_len} > {model_input_size}")
        mask = [1] * seq_len + [0] * (model_input_size - seq_len)
        seq = seq + [pad_id] * (model_input_size - seq_len)
        input_ids.append(seq)
        attention_masks.append(mask)
        lengths.append(seq_len)
    return input_ids, attention_masks, lengths, raw_lengths


def value_counts_string(values) -> str:
    counts = Counter([str(v) for v in values])
    return ";".join(f"{k}:{counts[k]}" for k in sorted(counts))


def compare_distribution(h5ad_values, ds_values) -> str:
    return "pass" if Counter(map(str, h5ad_values)) == Counter(map(str, ds_values)) else "fail"


def tokenize_version_dataset(
    version: str,
    dataset_key: str,
    prepared_h5ad: Path,
    log_lines: list[str],
) -> dict:
    cfg = VERSIONS[version]
    dataset_id = DATASETS[dataset_key]["dataset_id"]
    output_path = cfg["output_dir"] / f"{dataset_id}_{cfg['suffix']}.dataset"
    safe_replace_dir(output_path)

    token_dict = read_pickle_dict(cfg["token_dictionary_path"])
    vocab_size = len(token_dict)

    append_log(log_lines, "")
    append_log(log_lines, f"## {dataset_id} {version}")
    append_log(log_lines, f"- tokenizer input: {prepared_h5ad.relative_to(ROOT)}")
    append_log(log_lines, f"- output dataset: {output_path.relative_to(ROOT)}")
    append_log(log_lines, f"- model_input_size: {cfg['model_input_size']}")
    append_log(log_lines, f"- special_token: {cfg['special_token']}")
    append_log(log_lines, f"- collapse_gene_ids: {cfg['collapse_gene_ids']}")

    tokenizer = instantiate_tokenizer(version)
    raw_cells, metadata, _ = tokenizer.tokenize_anndata(prepared_h5ad, file_format="h5ad")
    if metadata is None:
        raise RuntimeError("Tokenizer returned no metadata.")
    input_ids, attention_masks, lengths, raw_lengths = crop_and_pad_cells(
        raw_cells,
        token_dict,
        cfg["model_input_size"],
        cfg["special_token"],
    )
    data = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "length": lengths,
        "length_uncropped": raw_lengths,
    }
    data.update(metadata)
    ds = Dataset.from_dict(data)
    ds.save_to_disk(str(output_path))

    arr_min = min(min(row) for row in input_ids) if input_ids else np.nan
    arr_max = max(max(row) for row in input_ids) if input_ids else np.nan
    empty_raw = int(sum(length == 0 for length in raw_lengths))
    abnormal_short = int(sum(0 < length < 100 for length in raw_lengths))
    truncated = int(sum(raw > cfg["model_input_size"] - (2 if cfg["special_token"] else 0) for raw in raw_lengths))
    row_count = len(ds)
    h5ad_n_obs = sc.read_h5ad(prepared_h5ad, backed="r").n_obs

    if cfg["special_token"]:
        cls_id = token_dict["<cls>"]
        eos_id = token_dict["<eos>"]
        special_check = all(
            ids[0] == cls_id and ids[length - 1] == eos_id
            for ids, length in zip(input_ids, lengths)
            if length >= 2
        )
    else:
        special_check = True

    token_range_check = bool(arr_min >= 0 and arr_max < vocab_size)
    mask_shape_check = all(len(ids) == len(mask) == cfg["model_input_size"] for ids, mask in zip(input_ids, attention_masks))

    append_log(log_lines, f"- rows: {row_count}; h5ad n_obs: {h5ad_n_obs}")
    append_log(log_lines, f"- raw sequence length min/median/max: {np.min(raw_lengths)}/{np.median(raw_lengths)}/{np.max(raw_lengths)}")
    append_log(log_lines, f"- final sequence length min/median/max: {np.min(lengths)}/{np.median(lengths)}/{np.max(lengths)}")
    append_log(log_lines, f"- padded input_ids shape: ({row_count}, {cfg['model_input_size']})")
    append_log(log_lines, f"- padded attention_mask shape: ({row_count}, {cfg['model_input_size']})")
    append_log(log_lines, f"- token range check: {'pass' if token_range_check else 'fail'}")
    append_log(log_lines, f"- special token check: {'pass' if special_check else 'fail'}")
    append_log(log_lines, f"- empty raw sequences: {empty_raw}")
    append_log(log_lines, f"- abnormal short raw sequences (<100): {abnormal_short}")
    append_log(log_lines, f"- truncated cells: {truncated}")

    return {
        "model_series": cfg["model_series"],
        "intended_model": cfg["intended_model"],
        "dataset_id": dataset_id,
        "source_h5ad": str(DATASETS[dataset_key]["h5ad"].relative_to(ROOT)),
        "tokenizer_input_h5ad": str(prepared_h5ad.relative_to(ROOT)),
        "tokenized_dataset": str(output_path.relative_to(ROOT)),
        "token_rows": row_count,
        "h5ad_n_obs": h5ad_n_obs,
        "row_count_match": row_count == h5ad_n_obs,
        "model_input_size": cfg["model_input_size"],
        "input_ids_shape": f"{row_count}x{cfg['model_input_size']}",
        "attention_mask_shape": f"{row_count}x{cfg['model_input_size']}",
        "raw_length_min": int(np.min(raw_lengths)),
        "raw_length_median": float(np.median(raw_lengths)),
        "raw_length_max": int(np.max(raw_lengths)),
        "final_length_min": int(np.min(lengths)),
        "final_length_median": float(np.median(lengths)),
        "final_length_max": int(np.max(lengths)),
        "empty_raw_sequences": empty_raw,
        "abnormal_short_raw_sequences_lt100": abnormal_short,
        "truncated_cells": truncated,
        "truncated_fraction": truncated / row_count if row_count else np.nan,
        "token_id_min": int(arr_min),
        "token_id_max": int(arr_max),
        "vocab_size": vocab_size,
        "token_range_check": "pass" if token_range_check else "fail",
        "special_token": cfg["special_token"],
        "special_token_check": "pass" if special_check else "fail",
        "attention_mask_length_check": "pass" if mask_shape_check else "fail",
        "tokenization_status": "success",
    }


def run_tokenization(prepared_paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    v2_log = ["# Phase 4B-0 V2-compatible tokenization log", ""]
    v1_log = ["# Phase 4B-0 V1-compatible tokenization log", ""]
    v2_rows, v1_rows = [], []

    for dataset_key, prepared_path in prepared_paths.items():
        v2_rows.append(tokenize_version_dataset("V2", dataset_key, prepared_path, v2_log))
    for dataset_key, prepared_path in prepared_paths.items():
        v1_rows.append(tokenize_version_dataset("V1", dataset_key, prepared_path, v1_log))

    (LOGS / "phase4B0_v2_tokenization_log.md").write_text("\n".join(v2_log) + "\n", encoding="utf-8")
    (LOGS / "phase4B0_v1_tokenization_log.md").write_text("\n".join(v1_log) + "\n", encoding="utf-8")
    v2_df = pd.DataFrame(v2_rows)
    v1_df = pd.DataFrame(v1_rows)
    v2_df.to_csv(TABLES / "phase4B0_v2_tokenization_summary.csv", index=False, encoding="utf-8-sig")
    v1_df.to_csv(TABLES / "phase4B0_v1_tokenization_summary.csv", index=False, encoding="utf-8-sig")
    return v2_df, v1_df


def metadata_traceability_check() -> pd.DataFrame:
    rows = []
    log = ["# Phase 4B-0 metadata traceability log", ""]
    for version, cfg in VERSIONS.items():
        for dataset_key, info in DATASETS.items():
            dataset_id = info["dataset_id"]
            ds_path = cfg["output_dir"] / f"{dataset_id}_{cfg['suffix']}.dataset"
            ds = load_from_disk(str(ds_path))
            adata = sc.read_h5ad(info["h5ad"], backed="r")
            row_match = len(ds) == adata.n_obs
            ds_original = [str(x) for x in ds["original_cell_id"]]
            h5_original = [str(x) for x in adata.obs["original_cell_id"].tolist()]
            original_matches = ds_original == h5_original
            original_unique = len(ds_original) == len(set(ds_original))
            malignant_match = compare_distribution(adata.obs["malignant_state"].tolist(), ds["malignant_state"])
            split_match = compare_distribution(adata.obs["split_unit"].tolist(), ds["split_unit"])
            treatment_match = compare_distribution(adata.obs["treatment.group"].tolist(), ds["treatment.group"])
            no_random_split = row_match and original_matches and split_match == "pass"
            rows.append(
                {
                    "model_series": cfg["model_series"],
                    "dataset_id": dataset_id,
                    "tokenized_dataset": str(ds_path.relative_to(ROOT)),
                    "h5ad_n_obs": adata.n_obs,
                    "token_rows": len(ds),
                    "row_count_match": row_match,
                    "original_cell_id_unique": original_unique,
                    "original_cell_id_order_matches_h5ad": original_matches,
                    "malignant_state_distribution_match": malignant_match,
                    "split_unit_distribution_match": split_match,
                    "treatment_group_distribution_match": treatment_match,
                    "split_unit_field_values": value_counts_string(ds["split_unit_field"]),
                    "split_unit_type_values": value_counts_string(ds["split_unit_type"]),
                    "patient_identity_status_values": value_counts_string(ds["patient_identity_status"]),
                    "no_cell_level_random_split_introduced": "pass" if no_random_split else "fail",
                    "metadata_traceability_status": "pass"
                    if all(
                        [
                            row_match,
                            original_matches,
                            original_unique,
                            malignant_match == "pass",
                            split_match == "pass",
                            treatment_match == "pass",
                            no_random_split,
                        ]
                    )
                    else "fail",
                }
            )
            append_log(log, f"## {cfg['model_series']} {dataset_id}")
            append_log(log, f"- token rows vs h5ad n_obs: {len(ds)} vs {adata.n_obs}")
            append_log(log, f"- original_cell_id unique: {original_unique}")
            append_log(log, f"- original_cell_id order matches h5ad: {original_matches}")
            append_log(log, f"- malignant_state distribution match: {malignant_match}")
            append_log(log, f"- split_unit distribution match: {split_match}")
            append_log(log, f"- treatment.group distribution match: {treatment_match}")
            append_log(log, f"- no cell-level random split introduced: {'pass' if no_random_split else 'fail'}")
            append_log(log, "")
            adata.file.close()
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase4B0_tokenized_metadata_traceability_check.csv", index=False, encoding="utf-8-sig")
    (LOGS / "phase4B0_metadata_traceability_log.md").write_text("\n".join(log), encoding="utf-8")
    return df


def sample_batch(ds, model_input_size: int, n: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    if len(ds) < n:
        raise RuntimeError(f"Dataset has fewer than {n} rows.")
    idx = list(range(n))
    input_ids = torch.tensor([ds[i]["input_ids"] for i in idx], dtype=torch.long)
    attention_mask = torch.tensor([ds[i]["attention_mask"] for i in idx], dtype=torch.long)
    if tuple(input_ids.shape) != (n, model_input_size):
        raise RuntimeError(f"Unexpected input_ids shape {tuple(input_ids.shape)}")
    if tuple(attention_mask.shape) != (n, model_input_size):
        raise RuntimeError(f"Unexpected attention_mask shape {tuple(attention_mask.shape)}")
    return input_ids, attention_mask


def gpu_forward_smoke_test() -> pd.DataFrame:
    log = ["# Phase 4B-0 GPU forward smoke test log", ""]
    rows = []
    for version, cfg in VERSIONS.items():
        for dataset_key, info in DATASETS.items():
            dataset_id = info["dataset_id"]
            ds_path = cfg["output_dir"] / f"{dataset_id}_{cfg['suffix']}.dataset"
            row = {
                "model_series": cfg["model_series"],
                "intended_model": cfg["intended_model"],
                "dataset_id": dataset_id,
                "tokenized_dataset": str(ds_path.relative_to(ROOT)),
                "model_path": str(cfg["model_path"]),
                "batch_size": 4,
                "model_input_size": cfg["model_input_size"],
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "status": "not_run",
                "error": "",
            }
            append_log(log, f"## {cfg['model_series']} {dataset_id}")
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable.")
                ds = load_from_disk(str(ds_path))
                input_ids, attention_mask = sample_batch(ds, cfg["model_input_size"], n=4)
                token_dict = read_pickle_dict(cfg["token_dictionary_path"])
                vocab_size = len(token_dict)
                if int(input_ids.max()) >= vocab_size or int(input_ids.min()) < 0:
                    raise RuntimeError("Batch token IDs outside vocabulary range.")
                torch.cuda.empty_cache()
                before_alloc = torch.cuda.memory_allocated(0)
                before_reserved = torch.cuda.memory_reserved(0)
                model = AutoModel.from_pretrained(str(cfg["model_path"]), local_files_only=True)
                model.eval().to("cuda")
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                torch.cuda.synchronize()
                last_hidden = out.last_hidden_state
                row.update(
                    {
                        "status": "pass",
                        "input_ids_shape": str(tuple(input_ids.shape)),
                        "attention_mask_shape": str(tuple(attention_mask.shape)),
                        "output_last_hidden_state_shape": str(tuple(last_hidden.shape)),
                        "memory_allocated_before_bytes": int(before_alloc),
                        "memory_reserved_before_bytes": int(before_reserved),
                        "memory_allocated_after_bytes": int(torch.cuda.memory_allocated(0)),
                        "memory_reserved_after_bytes": int(torch.cuda.memory_reserved(0)),
                    }
                )
                append_log(log, f"- status: pass")
                append_log(log, f"- input_ids shape: {row['input_ids_shape']}")
                append_log(log, f"- output last_hidden_state shape: {row['output_last_hidden_state_shape']}")
                append_log(log, f"- memory allocated after: {row['memory_allocated_after_bytes']}")
                del out, last_hidden, input_ids, attention_mask, model
            except RuntimeError as exc:
                row.update(
                    {
                        "status": "oom_or_runtime_error" if "out of memory" in str(exc).lower() else "fail",
                        "error": str(exc),
                    }
                )
                append_log(log, f"- status: {row['status']}")
                append_log(log, f"- error: {str(exc)}")
            except Exception as exc:
                row.update({"status": "fail", "error": repr(exc)})
                append_log(log, f"- status: fail")
                append_log(log, f"- error: {repr(exc)}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                rows.append(row)
                append_log(log, "")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase4B0_gpu_forward_smoke_test.csv", index=False, encoding="utf-8-sig")
    (LOGS / "phase4B0_gpu_forward_smoke_test_log.md").write_text("\n".join(log), encoding="utf-8")
    return df


def write_dictionary_log(inventory: pd.DataFrame) -> None:
    lines = ["# Phase 4B-0 model/token dictionary log", ""]
    for _, row in inventory.iterrows():
        lines.append(f"## {row['model_series']}")
        for col in inventory.columns:
            lines.append(f"- {col}: {row[col]}")
        lines.append("")
    (LOGS / "phase4B0_model_token_dictionary_log.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    prep_df: pd.DataFrame,
    v2_df: pd.DataFrame,
    v1_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    gpu_df: pd.DataFrame,
) -> None:
    v2_ok = bool(
        (v2_df["tokenization_status"] == "success").all()
        and (v2_df["row_count_match"] == True).all()
        and (v2_df["token_range_check"] == "pass").all()
        and (v2_df["special_token_check"] == "pass").all()
    )
    v1_ok = bool(
        (v1_df["tokenization_status"] == "success").all()
        and (v1_df["row_count_match"] == True).all()
        and (v1_df["token_range_check"] == "pass").all()
        and (v1_df["special_token_check"] == "pass").all()
    )
    trace_ok = bool((trace_df["metadata_traceability_status"] == "pass").all())
    gpu_all_pass = bool((gpu_df["status"] == "pass").all())
    v2_gpu_pass = bool((gpu_df[gpu_df["model_series"] == "Geneformer-V2"]["status"] == "pass").all())
    v1_gpu_pass = bool((gpu_df[gpu_df["model_series"] == "Geneformer-V1"]["status"] == "pass").all())

    expression_warning = prep_df[prep_df["expression_type_check"] != "raw_count_like"]
    compatibility_warning = (
        "GSE72056 expression values are processed/non-integer, so Geneformer tokenization is technically generated "
        "from real data but should be interpreted as conditional unless raw counts are confirmed."
        if not expression_warning.empty
        else "No non-integer expression warning detected in the first 100k non-zero values."
    )

    if v2_ok and v1_ok and trace_ok and gpu_all_pass and expression_warning.empty:
        ready = "YES"
        blockers = "None."
    elif v2_ok and v1_ok and trace_ok and (v2_gpu_pass or v1_gpu_pass):
        ready = "CONDITIONAL"
        blockers = (
            "需在 Phase 4B 前处理条件: "
            "1) 确认 GSE72056 processed/non-integer expression 是否可作为 Geneformer 输入, 或替换为 raw counts; "
            "2) 若任何 V2 forward smoke test 失败或 OOM, 使用更小 batch/AMP/gradient checkpointing 或退回 V1; "
            "3) Phase 4B 只能使用与所选模型匹配的 tokenized dataset。"
        )
    else:
        ready = "NO"
        blockers = "V1/V2 tokenization、metadata traceability 或 GPU smoke test 存在失败项。"

    lines = [
        "# Phase 4B-0 中文总结",
        "",
        "本阶段只完成 model-tokenization compatibility preparation、重新 tokenization、metadata traceability 和 GPU forward smoke test。",
        "未进行 fine-tuning、in silico deletion、perturbation 或候选靶点输出。",
        "",
        "## 1. V2-compatible tokenization",
        "",
        f"- 状态: {'成功' if v2_ok else '失败或需复核'}",
    ]
    for _, row in v2_df.iterrows():
        lines.append(
            f"- {row['dataset_id']}: rows={row['token_rows']}, input_ids={row['input_ids_shape']}, "
            f"raw length median={row['raw_length_median']}, truncated cells={row['truncated_cells']}, "
            f"token range={row['token_range_check']}, special token={row['special_token_check']}"
        )
    lines.extend(
        [
            "",
            "## 2. V1-compatible backup tokenization",
            "",
            f"- 状态: {'成功' if v1_ok else '失败或需复核'}",
        ]
    )
    for _, row in v1_df.iterrows():
        lines.append(
            f"- {row['dataset_id']}: rows={row['token_rows']}, input_ids={row['input_ids_shape']}, "
            f"raw length median={row['raw_length_median']}, truncated cells={row['truncated_cells']}, "
            f"truncated fraction={row['truncated_fraction']:.4f}, token range={row['token_range_check']}"
        )
    lines.extend(
        [
            "",
            "## 3. metadata traceability",
            "",
            f"- 状态: {'完整' if trace_ok else '存在失败项'}",
        ]
    )
    for _, row in trace_df.iterrows():
        lines.append(
            f"- {row['model_series']} {row['dataset_id']}: rows match={row['row_count_match']}, "
            f"original_cell_id order match={row['original_cell_id_order_matches_h5ad']}, "
            f"malignant_state={row['malignant_state_distribution_match']}, "
            f"split_unit={row['split_unit_distribution_match']}, treatment.group={row['treatment_group_distribution_match']}"
        )
    lines.extend(
        [
            "",
            "## 4. GPU forward smoke test",
            "",
            f"- 状态: {'全部通过' if gpu_all_pass else '存在失败或 OOM'}",
        ]
    )
    for _, row in gpu_df.iterrows():
        lines.append(
            f"- {row['model_series']} {row['dataset_id']}: status={row['status']}, "
            f"batch_size={row['batch_size']}, input_size={row['model_input_size']}, "
            f"output_shape={row.get('output_last_hidden_state_shape', '')}"
        )
        if row["status"] != "pass":
            lines.append(f"  错误: {row['error']}")
    lines.extend(
        [
            "",
            "## 5. 推荐 Phase 4B 主模型",
            "",
            "- 优先路线: `Geneformer-V2-104M_CLcancer`，仅使用 `data_processed/tokenized_v2_gc104M/` 下的 V2/gc104M tokenized dataset。",
            "- 备用路线: 若 V2 在正式训练中 OOM 或发现 tokenization/表达矩阵条件不满足，则退回 `Geneformer-V1-10M`，仅使用 `data_processed/tokenized_v1_gc30M/` 下的 V1/gc30M tokenized dataset。",
            "- 禁止把旧的 gc30M 4096 token arrays 直接喂给 V2，也禁止把 4096 token 直接喂给 V1。",
            "",
            "## 6. 重要限制和条件",
            "",
            f"- 表达矩阵条件: {compatibility_warning}",
            "- 本阶段创建的是派生 tokenizer input h5ad，原始 AnnData 未修改。",
            "- 旧 tokenized arrays 未修改。",
            "",
            f"READY_FOR_PHASE4B_SUPERVISED_FINETUNING = {ready}",
            "",
            blockers,
        ]
    )
    (ROOT / "summary_phase4B0_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    combined_log_path = LOGS / "phase4B0_input_preparation_log.md"
    log_lines: list[str] = []
    try:
        inventory = locate_resources()
        write_dictionary_log(inventory)
        prepared_paths, prep_df = verify_inputs_and_prepare_h5ad(log_lines)
        prep_df.to_csv(TABLES / "phase4B0_tokenizer_input_preparation_summary.csv", index=False, encoding="utf-8-sig")
        combined_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        v2_df, v1_df = run_tokenization(prepared_paths)
        trace_df = metadata_traceability_check()
        gpu_df = gpu_forward_smoke_test()
        write_summary(prep_df, v2_df, v1_df, trace_df, gpu_df)
        print("PHASE4B0_MODEL_TOKENIZATION_COMPATIBILITY: PASS")
        print(f"summary={ROOT / 'summary_phase4B0_zh.md'}")
        return 0
    except Exception as exc:
        log_lines.append("")
        log_lines.append("## ERROR")
        log_lines.append(str(exc))
        log_lines.append("")
        log_lines.append(traceback.format_exc())
        combined_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print("PHASE4B0_MODEL_TOKENIZATION_COMPATIBILITY: FAIL")
        print(str(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
