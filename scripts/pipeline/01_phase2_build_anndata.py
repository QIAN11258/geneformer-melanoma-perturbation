from __future__ import annotations

import csv
import gzip
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data_raw"
METADATA_DIR = DATA_RAW / "metadata"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
TABLES = PROJECT_ROOT / "tables"
LOGS = PROJECT_ROOT / "logs"

for directory in (DATA_PROCESSED, TABLES, LOGS):
    directory.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "GSE72056": {
        "role": "discovery scRNA-seq dataset",
        "matrix": DATA_RAW / "GSE72056_melanoma_single_cell_revised_v2.txt.gz",
        "delimiter": "\t",
        "metadata": None,
        "h5ad": DATA_PROCESSED / "GSE72056_melanoma.h5ad",
        "malignant_h5ad": DATA_PROCESSED / "GSE72056_malignant_only.h5ad",
    },
    "GSE115978": {
        "role": "external scRNA-seq validation dataset",
        "matrix": DATA_RAW / "GSE115978_counts.csv.gz",
        "delimiter": ",",
        "metadata": METADATA_DIR / "GSE115978_cell.annotations.csv.gz",
        "h5ad": DATA_PROCESSED / "GSE115978_melanoma.h5ad",
        "malignant_h5ad": DATA_PROCESSED / "GSE115978_malignant_validation.h5ad",
    },
    "GSE120575": {
        "role": "immunotherapy-response-related validation dataset",
        "matrix": DATA_RAW / "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz",
        "delimiter": "\t",
        "metadata": METADATA_DIR / "GSE120575_patient_ID_single_cells.txt.gz",
        "h5ad": DATA_PROCESSED / "GSE120575_immune_response.h5ad",
        "malignant_h5ad": None,
    },
}

VALIDATION_RESOURCES = {
    "TCGA-SKCM": "bulk prognosis validation dataset",
    "GDSC": "drug sensitivity validation resource",
    "DepMap/CCLE": "dependency and compound vulnerability validation resource",
    "ChEMBL": "druggability and bioactivity validation resource",
    "Open Targets": "target-disease, tractability and safety validation resource",
}


def strip_quotes(value: str) -> str:
    return value.strip().strip('"').strip("'")


def infer_identifier_type(genes: list[str]) -> str:
    if not genes:
        return "needs manual confirmation"
    ensembl_like = sum(1 for gene in genes if re.match(r"^ENS[A-Z]*G\d+", str(gene)))
    symbol_like = sum(1 for gene in genes if re.match(r"^[A-Za-z0-9_.:-]+$", str(gene)))
    if ensembl_like / len(genes) >= 0.8:
        return "Ensembl ID"
    if symbol_like / len(genes) >= 0.8:
        return "gene symbol"
    return "needs manual confirmation"


def make_unique_feature_ids(genes: list[str]) -> list[str]:
    seen: dict[str, int] = defaultdict(int)
    feature_ids: list[str] = []
    for gene in genes:
        seen[gene] += 1
        if seen[gene] == 1:
            feature_ids.append(gene)
        else:
            feature_ids.append(f"{gene}__dup{seen[gene]}")
    return feature_ids


def read_gse115978_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path, compression="gzip")
    meta = meta.rename(
        columns={
            "cells": "cell_id",
            "samples": "sample_id",
            "cell.types": "cell_type",
            "treatment.group": "treatment_group",
            "Cohort": "cohort",
            "no.of.genes": "n_genes_reported",
            "no.of.reads": "n_reads_reported",
        }
    )
    meta["cell_id"] = meta["cell_id"].astype(str)
    return meta.set_index("cell_id", drop=False)


def read_gse120575_metadata(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not re.match(r"^Sample\s+\d+\t", line):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            rows.append(
                {
                    "sample_name": parts[0],
                    "cell_id": parts[1],
                    "source_name": parts[2],
                    "organism": parts[3],
                    "patient_timepoint": parts[4],
                    "response": parts[5],
                    "therapy": parts[6],
                }
            )
    meta = pd.DataFrame(rows)
    if meta.empty:
        return pd.DataFrame(columns=["cell_id"]).set_index("cell_id", drop=False)
    meta["cell_id"] = meta["cell_id"].astype(str)
    return meta.set_index("cell_id", drop=False)


def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_numeric_values(
    values: list[str],
    expected_cells: int,
    dataset_id: str,
    row_name: str,
    row_number: int,
    qc: dict,
) -> tuple[list[int], list[float]]:
    if len(values) > expected_cells and all(v == "" for v in values[expected_cells:]):
        values = values[:expected_cells]
    if len(values) != expected_cells:
        qc["row_length_mismatch"] += 1
        qc["row_length_examples"].append(
            {
                "dataset_id": dataset_id,
                "row_number": row_number,
                "row_name": row_name,
                "observed_values": len(values),
                "expected_values": expected_cells,
            }
        )
        if len(values) < expected_cells:
            values = values + [""] * (expected_cells - len(values))
        else:
            values = values[:expected_cells]

    indices: list[int] = []
    parsed: list[float] = []
    for idx, token in enumerate(values):
        token = token.strip()
        if token == "" or token.lower() in {"na", "nan", "null"}:
            qc["missing_values"] += 1
            continue
        try:
            value = float(token)
        except ValueError:
            qc["non_numeric_values"] += 1
            continue
        if not math.isfinite(value):
            qc["missing_values"] += 1
            continue
        if value < 0:
            qc["negative_values"] += 1
        if abs(value - round(value)) > 1e-8:
            qc["has_non_integer_values"] = True
        qc["numeric_values"] += 1
        if qc["min_value"] is None or value < qc["min_value"]:
            qc["min_value"] = value
        if qc["max_value"] is None or value > qc["max_value"]:
            qc["max_value"] = value
        if value != 0:
            indices.append(idx)
            parsed.append(value)
    return indices, parsed


def matrix_to_csc(
    dataset_id: str,
    path: Path,
    delimiter: str,
) -> tuple[sparse.csc_matrix, list[str], list[str], pd.DataFrame, dict, list[dict]]:
    qc = {
        "missing_values": 0,
        "negative_values": 0,
        "non_numeric_values": 0,
        "numeric_values": 0,
        "has_non_integer_values": False,
        "row_length_mismatch": 0,
        "row_length_examples": [],
        "min_value": None,
        "max_value": None,
        "metadata_rows_in_matrix": 0,
    }

    data = array("f")
    indices = array("i")
    indptr = array("i", [0])
    genes: list[str] = []
    obs_from_matrix = pd.DataFrame()
    row_length_examples: list[dict] = []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader)
        if dataset_id == "GSE115978" and header and header[0] == "":
            cell_ids = [strip_quotes(value) for value in header[1:]]
        elif dataset_id in {"GSE72056", "GSE120575"}:
            cell_ids = [strip_quotes(value) for value in header[1:]]
        else:
            cell_ids = [strip_quotes(value) for value in header[1:]]

        expected_cells = len(cell_ids)
        obs_records: dict[str, list[str]] = {}

        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            row_name = strip_quotes(row[0])

            if dataset_id == "GSE72056" and row_number in {2, 3, 4}:
                key = {
                    "tumor": "tumor_id",
                    "malignant(1=no,2=yes,0=unresolved)": "malignant_code",
                    "non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)": "non_malignant_code",
                }.get(row_name, row_name)
                obs_records[key] = [strip_quotes(value) for value in row[1 : 1 + expected_cells]]
                qc["metadata_rows_in_matrix"] += 1
                continue

            if dataset_id == "GSE120575" and row_number == 2 and row_name == "":
                obs_records["matrix_patient_timepoint"] = [
                    strip_quotes(value) for value in row[1 : 1 + expected_cells]
                ]
                qc["metadata_rows_in_matrix"] += 1
                continue

            if row_name == "":
                qc["non_numeric_values"] += max(0, len(row) - 1)
                continue

            values = row[1:]
            nz_indices, nz_values = parse_numeric_values(
                values, expected_cells, dataset_id, row_name, row_number, qc
            )
            genes.append(row_name)
            indices.extend(nz_indices)
            data.extend(nz_values)
            indptr.append(len(indices))

    if qc["row_length_examples"]:
        row_length_examples = qc["row_length_examples"][:25]

    if obs_records:
        obs_from_matrix = pd.DataFrame(obs_records, index=cell_ids)
        obs_from_matrix.index.name = "cell_id"
    else:
        obs_from_matrix = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))

    data_np = np.frombuffer(data, dtype=np.float32)
    indices_np = np.frombuffer(indices, dtype=np.int32)
    indptr_np = np.frombuffer(indptr, dtype=np.int32)
    matrix = sparse.csc_matrix(
        (data_np, indices_np, indptr_np),
        shape=(len(cell_ids), len(genes)),
        dtype=np.float32,
    )
    return matrix, cell_ids, genes, obs_from_matrix, qc, row_length_examples


def parse_numeric_line_fast(
    values_text: str,
    expected_cells: int,
    delimiter: str,
    dataset_id: str,
    row_name: str,
    row_number: int,
    qc: dict,
) -> tuple[np.ndarray, np.ndarray]:
    values_text = values_text.rstrip("\r\n")
    while values_text.endswith(delimiter):
        values_text = values_text[: -len(delimiter)]
    arr = np.fromstring(values_text, sep=delimiter, dtype=np.float32)

    if arr.size != expected_cells:
        qc["row_length_mismatch"] += 1
        qc["row_length_examples"].append(
            {
                "dataset_id": dataset_id,
                "row_number": row_number,
                "row_name": row_name,
                "observed_values": int(arr.size),
                "expected_values": expected_cells,
            }
        )
        values = values_text.split(delimiter) if values_text else []
        nz_indices, nz_values = parse_numeric_values(
            values, expected_cells, dataset_id, row_name, row_number, qc
        )
        return np.asarray(nz_indices, dtype=np.int32), np.asarray(nz_values, dtype=np.float32)

    if arr.size:
        finite = np.isfinite(arr)
        if not bool(finite.all()):
            qc["missing_values"] += int((~finite).sum())
            arr = np.where(finite, arr, 0)
        qc["numeric_values"] += int(arr.size)
        negative_n = int((arr < 0).sum())
        qc["negative_values"] += negative_n
        if not qc["has_non_integer_values"]:
            qc["has_non_integer_values"] = bool(np.any(np.abs(arr - np.rint(arr)) > 1e-8))
        row_min = float(arr.min())
        row_max = float(arr.max())
        if qc["min_value"] is None or row_min < qc["min_value"]:
            qc["min_value"] = row_min
        if qc["max_value"] is None or row_max > qc["max_value"]:
            qc["max_value"] = row_max

    nz = np.flatnonzero(arr != 0).astype(np.int32, copy=False)
    return nz, arr[nz].astype(np.float32, copy=False)


def matrix_to_csc_fast(
    dataset_id: str,
    path: Path,
    delimiter: str,
) -> tuple[sparse.csc_matrix, list[str], list[str], pd.DataFrame, dict, list[dict]]:
    qc = {
        "missing_values": 0,
        "negative_values": 0,
        "non_numeric_values": 0,
        "numeric_values": 0,
        "has_non_integer_values": False,
        "row_length_mismatch": 0,
        "row_length_examples": [],
        "min_value": None,
        "max_value": None,
        "metadata_rows_in_matrix": 0,
    }

    data_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    indptr_values: list[int] = [0]
    genes: list[str] = []
    obs_records: dict[str, list[str]] = {}

    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        header = handle.readline().rstrip("\n")
        header_parts = header.split(delimiter)
        cell_ids = [strip_quotes(value) for value in header_parts[1:] if value != ""]
        expected_cells = len(cell_ids)

        for row_number, line in enumerate(handle, start=2):
            if not line:
                continue
            line = line.rstrip("\n")
            if delimiter not in line:
                continue
            row_name, values_text = line.split(delimiter, 1)
            row_name = strip_quotes(row_name)

            if dataset_id == "GSE72056" and row_number in {2, 3, 4}:
                key = {
                    "tumor": "tumor_id",
                    "malignant(1=no,2=yes,0=unresolved)": "malignant_code",
                    "non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)": "non_malignant_code",
                }.get(row_name, row_name)
                values = values_text.split(delimiter)
                obs_records[key] = [strip_quotes(value) for value in values[:expected_cells]]
                qc["metadata_rows_in_matrix"] += 1
                continue

            if dataset_id == "GSE120575" and row_number == 2 and row_name == "":
                values = values_text.split(delimiter)
                obs_records["matrix_patient_timepoint"] = [
                    strip_quotes(value) for value in values[:expected_cells]
                ]
                qc["metadata_rows_in_matrix"] += 1
                continue

            if row_name == "":
                qc["non_numeric_values"] += 1
                continue

            nz_indices, nz_values = parse_numeric_line_fast(
                values_text, expected_cells, delimiter, dataset_id, row_name, row_number, qc
            )
            genes.append(row_name)
            index_chunks.append(nz_indices)
            data_chunks.append(nz_values)
            indptr_values.append(indptr_values[-1] + int(nz_indices.size))

    if data_chunks:
        data_np = np.concatenate(data_chunks).astype(np.float32, copy=False)
        indices_np = np.concatenate(index_chunks).astype(np.int32, copy=False)
    else:
        data_np = np.asarray([], dtype=np.float32)
        indices_np = np.asarray([], dtype=np.int32)
    indptr_np = np.asarray(indptr_values, dtype=np.int64)
    matrix = sparse.csc_matrix(
        (data_np, indices_np, indptr_np),
        shape=(len(cell_ids), len(genes)),
        dtype=np.float32,
    )
    if obs_records:
        obs_from_matrix = pd.DataFrame(obs_records, index=cell_ids)
        obs_from_matrix.index.name = "cell_id"
    else:
        obs_from_matrix = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
    return matrix, cell_ids, genes, obs_from_matrix, qc, qc["row_length_examples"][:25]


def annotate_gse72056_obs(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.copy()
    malignant_map = {"0": "unresolved", "1": "non-malignant", "2": "malignant"}
    non_malignant_map = {
        "0": "not_applicable_or_malignant",
        "1": "T",
        "2": "B",
        "3": "Macrophage",
        "4": "Endothelial",
        "5": "CAF",
        "6": "NK",
    }
    obs["cell_id"] = obs.index.astype(str)
    obs["malignant_label"] = obs["malignant_code"].astype(str).map(malignant_map).fillna(
        "needs manual confirmation"
    )
    obs["non_malignant_cell_type"] = obs["non_malignant_code"].astype(str).map(
        non_malignant_map
    ).fillna("needs manual confirmation")
    return obs


def build_var(dataset_id: str, genes: list[str]) -> pd.DataFrame:
    counts = Counter(genes)
    feature_ids = make_unique_feature_ids(genes)
    var = pd.DataFrame(index=pd.Index(feature_ids, name="feature_id"))
    var["gene_symbol"] = genes
    var["gene_identifier_type"] = infer_identifier_type(genes)
    var["gene_symbol_is_duplicate"] = [counts[gene] > 1 for gene in genes]
    var["dataset_id"] = dataset_id
    return var


def expression_type(dataset_id: str, qc: dict) -> str:
    if dataset_id == "GSE115978" and not qc["has_non_integer_values"]:
        return "raw/count-like integer count matrix from counts.csv.gz"
    if dataset_id == "GSE120575":
        return "TPM processed expression by GEO file name; log status needs manual confirmation"
    if dataset_id == "GSE72056":
        return "processed normalized expression; exact normalization needs manual confirmation"
    return "needs manual confirmation"


def align_metadata(dataset_id: str, cell_ids: list[str], obs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cfg = DATASETS[dataset_id]
    summary = {
        "dataset_id": dataset_id,
        "matrix_cells": len(cell_ids),
        "metadata_rows": 0,
        "matched_cells": len(cell_ids),
        "missing_in_metadata": 0,
        "metadata_not_in_matrix": 0,
        "alignment_key": "embedded matrix metadata",
        "alignment_status": "matched",
        "notes": "",
    }

    if dataset_id == "GSE115978":
        meta = read_gse115978_metadata(cfg["metadata"])
        summary["metadata_rows"] = int(meta.shape[0])
        summary["alignment_key"] = "matrix column ID to cell.annotations cells"
        missing = [cell for cell in cell_ids if cell not in meta.index]
        extra = [cell for cell in meta.index if cell not in set(cell_ids)]
        summary["matched_cells"] = len(cell_ids) - len(missing)
        summary["missing_in_metadata"] = len(missing)
        summary["metadata_not_in_matrix"] = len(extra)
        summary["alignment_status"] = "matched" if not missing else "partial"
        obs = obs.join(meta.reindex(cell_ids), how="left")
        obs["dataset_role"] = DATASETS[dataset_id]["role"]
        if "cell_type" in obs.columns:
            obs["malignant_label"] = np.where(obs["cell_type"].astype(str) == "Mal", "malignant", "non-malignant/context")
        return obs, summary

    if dataset_id == "GSE120575":
        meta = read_gse120575_metadata(cfg["metadata"])
        summary["metadata_rows"] = int(meta.shape[0])
        summary["alignment_key"] = "matrix column ID to patient_ID metadata title"
        missing = [cell for cell in cell_ids if cell not in meta.index]
        extra = [cell for cell in meta.index if cell not in set(cell_ids)]
        summary["matched_cells"] = len(cell_ids) - len(missing)
        summary["missing_in_metadata"] = len(missing)
        summary["metadata_not_in_matrix"] = len(extra)
        summary["alignment_status"] = "matched" if not missing else "partial"
        obs = obs.join(meta.reindex(cell_ids), how="left")
        obs["dataset_role"] = DATASETS[dataset_id]["role"]
        obs["malignant_discovery_use"] = "do_not_use_immune_only"
        if "matrix_patient_timepoint" in obs.columns and "patient_timepoint" in obs.columns:
            mismatches = (
                obs["matrix_patient_timepoint"].astype(str)
                != obs["patient_timepoint"].astype(str)
            ).sum()
            summary["notes"] = f"matrix patient/timepoint mismatches: {int(mismatches)}"
        return obs, summary

    if dataset_id == "GSE72056":
        obs = annotate_gse72056_obs(obs)
        obs["dataset_role"] = DATASETS[dataset_id]["role"]
        summary["metadata_rows"] = int(obs.shape[0])
        summary["alignment_key"] = "matrix column metadata rows"
        return obs, summary

    return obs, summary


def safe_str_obs(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.copy()
    for column in obs.columns:
        if pd.api.types.is_object_dtype(obs[column]) or pd.api.types.is_string_dtype(obs[column]):
            obs[column] = obs[column].where(obs[column].notna(), "needs manual confirmation").astype(str)
    return obs


def write_anndata(dataset_id: str, matrix: sparse.csc_matrix, obs: pd.DataFrame, var: pd.DataFrame) -> ad.AnnData:
    obs = safe_str_obs(obs)
    obs.index.name = None
    var.index.name = None
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.uns["dataset_id"] = dataset_id
    adata.uns["dataset_role"] = DATASETS[dataset_id]["role"]
    adata.uns["phase2_rule"] = (
        "No genes were deleted or merged; duplicated gene symbols are preserved in var['gene_symbol'] "
        "and unique feature IDs are used only for AnnData index compatibility."
    )
    adata.write_h5ad(DATASETS[dataset_id]["h5ad"], compression="gzip")
    return adata


def subset_malignant(dataset_id: str, adata: ad.AnnData) -> tuple[int, str]:
    out = DATASETS[dataset_id].get("malignant_h5ad")
    if out is None:
        return 0, "not applicable"
    if dataset_id == "GSE72056":
        mask = adata.obs["malignant_code"].astype(str) == "2"
        note = "malignant_code == 2"
    elif dataset_id == "GSE115978":
        mask = adata.obs["cell_type"].astype(str) == "Mal"
        note = "cell_type == Mal"
    else:
        return 0, "not applicable"
    malignant = adata[mask].copy()
    malignant.uns["subset_rule"] = note
    malignant.write_h5ad(out, compression="gzip")
    return int(mask.sum()), note


def add_distribution(rows: list[dict], dataset_id: str, obs: pd.DataFrame, field: str) -> None:
    if field not in obs.columns:
        rows.append(
            {
                "dataset_id": dataset_id,
                "field": field,
                "label": "needs manual confirmation",
                "n_cells": "needs manual confirmation",
            }
        )
        return
    counts = obs[field].astype(str).fillna("needs manual confirmation").value_counts(dropna=False)
    for label, n in counts.items():
        rows.append({"dataset_id": dataset_id, "field": field, "label": label, "n_cells": int(n)})


def query_mygene(symbols: list[str]) -> tuple[dict[str, str], str]:
    mappings: dict[str, str] = {}
    if not symbols:
        return mappings, "no symbols"

    endpoint = "https://rest.ensembl.org/lookup/symbol/homo_sapiens?content-type=application/json"
    priority_symbols = [
        "MITF",
        "SOX10",
        "AXL",
        "NGFR",
        "PMEL",
        "MLANA",
        "TYR",
        "TP53",
        "CD8A",
        "PDCD1",
        "CTLA4",
        "LAG3",
        "HAVCR2",
    ]
    symbol_set = set(symbols)
    chunk = [symbol for symbol in priority_symbols if symbol in symbol_set]
    status = (
        "limited Phase 2 Ensembl REST sanity check only; full gene-symbol-to-Ensembl "
        "mapping is pending and required before Geneformer tokenization"
    )
    if not chunk:
        return mappings, status
    payload = json.dumps({"symbols": chunk}).encode("utf-8")
    request = urllib.request.Request(endpoint, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return mappings, f"limited mapping sanity check failed: {exc}; full mapping pending"
    if isinstance(parsed, dict):
        for query, result in parsed.items():
            if isinstance(result, dict) and result.get("id"):
                mappings[str(query)] = str(result["id"])
    return mappings, status


def main() -> None:
    inventory = pd.read_csv(PROJECT_ROOT / "dataset_inventory.csv")
    role_rows = []
    for dataset_id, role in {**{k: v["role"] for k, v in DATASETS.items()}, **VALIDATION_RESOURCES}.items():
        inv_match = inventory[inventory["dataset_id"] == dataset_id]
        role_rows.append(
            {
                "dataset_id": dataset_id,
                "confirmed_phase2_role": role,
                "phase1_recommended_use": inv_match["recommended_use"].iloc[0]
                if not inv_match.empty
                else "needs manual confirmation",
            }
        )
    pd.DataFrame(role_rows).to_csv(TABLES / "dataset_role_confirmation.csv", index=False)

    download_lines = [
        "# Phase 2 download log",
        "",
        "No FASTQ, SRA, BAM, or raw sequencing archives were downloaded.",
        "",
        "| dataset | file | bytes | status |",
        "|---|---:|---:|---|",
    ]
    for dataset_id, cfg in DATASETS.items():
        matrix_path = cfg["matrix"]
        status = "present" if matrix_path.exists() else "missing"
        bytes_size = matrix_path.stat().st_size if matrix_path.exists() else "needs manual confirmation"
        download_lines.append(f"| {dataset_id} | {matrix_path.name} | {bytes_size} | {status} |")
        metadata_path = cfg.get("metadata")
        if metadata_path is not None:
            meta_status = "present" if metadata_path.exists() else "missing"
            meta_size = metadata_path.stat().st_size if metadata_path.exists() else "needs manual confirmation"
            download_lines.append(f"| {dataset_id} | {metadata_path.name} | {meta_size} | {meta_status} |")
    write_md(LOGS / "phase2_download_log.md", download_lines)

    expression_qc_rows = []
    alignment_rows = []
    cell_type_rows = []
    malignant_summary_rows = []
    build_log_lines = [
        "# Phase 2 AnnData build log",
        "",
        "Rule: no gene rows were deleted or merged in Phase 2. Duplicate gene symbols were preserved.",
        "",
    ]
    matrix_check_lines = ["# Phase 2 matrix format check", ""]
    all_dataset_genes: dict[str, list[str]] = {}
    all_obs: dict[str, pd.DataFrame] = {}
    h5ad_status: dict[str, str] = {}

    for dataset_id, cfg in DATASETS.items():
        matrix_path = cfg["matrix"]
        if not matrix_path.exists():
            h5ad_status[dataset_id] = "failed: matrix missing"
            continue
        matrix, cell_ids, genes, obs_from_matrix, qc, row_examples = matrix_to_csc_fast(
            dataset_id, matrix_path, cfg["delimiter"]
        )
        obs, alignment = align_metadata(dataset_id, cell_ids, obs_from_matrix)
        var = build_var(dataset_id, genes)
        all_dataset_genes[dataset_id] = genes
        all_obs[dataset_id] = obs

        adata = write_anndata(dataset_id, matrix, obs, var)
        malignant_n, malignant_rule = subset_malignant(dataset_id, adata)
        h5ad_status[dataset_id] = "success"

        gene_counts = Counter(genes)
        duplicate_symbols = sum(1 for _, count in gene_counts.items() if count > 1)
        duplicate_rows = sum(count for _, count in gene_counts.items() if count > 1)
        value_type = expression_type(dataset_id, qc)
        identifier_type = infer_identifier_type(genes)
        n_cells, n_genes = matrix.shape
        nnz = int(matrix.nnz)
        density = nnz / (n_cells * n_genes) if n_cells and n_genes else 0

        expression_qc_rows.append(
            {
                "dataset_id": dataset_id,
                "file_name": matrix_path.name,
                "file_size_bytes": matrix_path.stat().st_size,
                "matrix_orientation": "gene x cell in source; AnnData X is cell x gene",
                "n_cells": n_cells,
                "n_genes": n_genes,
                "delimiter": "tab" if cfg["delimiter"] == "\t" else "comma",
                "gene_identifier_type": identifier_type,
                "expression_value_type": value_type,
                "min_value": qc["min_value"],
                "max_value": qc["max_value"],
                "nonzero_values": nnz,
                "density": density,
                "has_non_integer_values": qc["has_non_integer_values"],
                "missing_values": qc["missing_values"],
                "negative_values": qc["negative_values"],
                "non_numeric_values": qc["non_numeric_values"],
                "row_length_mismatch": qc["row_length_mismatch"],
                "duplicated_gene_symbols": duplicate_symbols,
                "duplicated_gene_rows": duplicate_rows,
                "metadata_rows_in_matrix": qc["metadata_rows_in_matrix"],
                "metadata_alignment_status": alignment["alignment_status"],
                "h5ad_build_status": "success",
            }
        )
        alignment_rows.append(alignment)

        for field in [
            "malignant_label",
            "non_malignant_cell_type",
            "cell_type",
            "treatment_group",
            "response",
            "therapy",
            "patient_timepoint",
        ]:
            add_distribution(cell_type_rows, dataset_id, obs, field)

        if dataset_id == "GSE72056":
            malignant_cells = int((obs["malignant_code"].astype(str) == "2").sum())
            non_malignant_cells = int((obs["malignant_code"].astype(str) == "1").sum())
            unresolved_cells = int((obs["malignant_code"].astype(str) == "0").sum())
            malignant_summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "total_cells": n_cells,
                    "malignant_cells": malignant_cells,
                    "non_malignant_reference_context_cells": non_malignant_cells,
                    "unresolved_cells": unresolved_cells,
                    "subset_file": cfg["malignant_h5ad"].name,
                    "subset_rule": malignant_rule,
                    "discovery_model_use": "malignant-only h5ad; non-malignant retained only as reference/context",
                }
            )
        elif dataset_id == "GSE115978":
            malignant_cells = int((obs["cell_type"].astype(str) == "Mal").sum())
            malignant_summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "total_cells": n_cells,
                    "malignant_cells": malignant_cells,
                    "non_malignant_reference_context_cells": n_cells - malignant_cells,
                    "unresolved_cells": "needs manual confirmation",
                    "subset_file": cfg["malignant_h5ad"].name,
                    "subset_rule": malignant_rule,
                    "discovery_model_use": "malignant-only validation h5ad; not primary discovery",
                }
            )
        else:
            malignant_summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "total_cells": n_cells,
                    "malignant_cells": 0,
                    "non_malignant_reference_context_cells": n_cells,
                    "unresolved_cells": 0,
                    "subset_file": "not applicable",
                    "subset_rule": "immune-only response validation dataset",
                    "discovery_model_use": "do not use for malignant discovery",
                }
            )

        matrix_check_lines.extend(
            [
                f"## {dataset_id}",
                "",
                f"- File: `{matrix_path.name}`",
                f"- File size: {matrix_path.stat().st_size} bytes",
                f"- Source orientation: gene x cell",
                f"- AnnData orientation: cell x gene",
                f"- Shape: {n_cells} cells x {n_genes} genes",
                f"- Gene identifier type: {identifier_type}",
                f"- Expression value type: {value_type}",
                f"- Missing values: {qc['missing_values']}",
                f"- Negative values: {qc['negative_values']}",
                f"- Non-numeric values inside expression rows: {qc['non_numeric_values']}",
                f"- Row length mismatches: {qc['row_length_mismatch']}",
                f"- Duplicate gene symbols: {duplicate_symbols} symbols / {duplicate_rows} rows",
                f"- Metadata alignment: {alignment['alignment_status']} ({alignment['matched_cells']} matched cells)",
                "",
            ]
        )
        if row_examples:
            matrix_check_lines.append("Row length mismatch examples were written into logs only; no gene rows were dropped.")
            matrix_check_lines.append("")

        build_log_lines.extend(
            [
                f"## {dataset_id}",
                "",
                f"- Full AnnData: `{cfg['h5ad'].relative_to(PROJECT_ROOT)}`",
                f"- Build status: success",
                f"- Full shape: {n_cells} cells x {n_genes} genes",
                f"- Malignant subset rule: {malignant_rule}",
                f"- Malignant subset cells: {malignant_n}",
                "",
            ]
        )
        if dataset_id == "GSE115978":
            observed_treatment = sorted(obs["treatment_group"].astype(str).dropna().unique().tolist())
            build_log_lines.append(f"- Observed treatment.group labels: {', '.join(observed_treatment)}")
            build_log_lines.append("")
        if dataset_id == "GSE120575":
            build_log_lines.append("- Marked as immune-only response validation; not used for malignant discovery.")
            build_log_lines.append("")

    pd.DataFrame(expression_qc_rows).to_csv(TABLES / "expression_matrix_qc_summary.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(TABLES / "metadata_alignment_summary.csv", index=False)
    pd.DataFrame(cell_type_rows).to_csv(TABLES / "cell_type_distribution.csv", index=False)
    pd.DataFrame(malignant_summary_rows).to_csv(TABLES / "malignant_cell_summary.csv", index=False)

    if "GSE120575" in all_obs:
        obs = all_obs["GSE120575"].copy()
        response_summary = (
            obs.groupby(["response", "therapy"], dropna=False)
            .agg(
                n_cells=("cell_id", "count") if "cell_id" in obs.columns else ("therapy", "count"),
                n_patient_timepoints=("patient_timepoint", "nunique"),
            )
            .reset_index()
        )
        response_summary.to_csv(TABLES / "GSE120575_response_therapy_summary.csv", index=False)

    write_md(LOGS / "phase2_matrix_format_check.md", matrix_check_lines)
    write_md(LOGS / "phase2_anndata_build_log.md", build_log_lines)

    duplicated_rows = []
    for dataset_id, genes in all_dataset_genes.items():
        feature_ids = make_unique_feature_ids(genes)
        counts = Counter(genes)
        grouped_ids: dict[str, list[str]] = defaultdict(list)
        for gene, feature_id in zip(genes, feature_ids):
            grouped_ids[gene].append(feature_id)
        for gene, count in counts.items():
            if count > 1:
                duplicated_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "gene_symbol": gene,
                        "occurrences": count,
                        "feature_ids": ";".join(grouped_ids[gene]),
                        "phase2_rule": "kept all rows; no deletion or merge",
                    }
                )
    pd.DataFrame(duplicated_rows).to_csv(TABLES / "duplicated_genes.csv", index=False)

    unique_symbols = sorted({gene for genes in all_dataset_genes.values() for gene in genes if gene})
    mapping, mapping_status = query_mygene(unique_symbols)
    mapping_rows = []
    unmapped_rows = []
    mapping_summary_rows = []
    for symbol in unique_symbols:
        mapping_rows.append(
            {
                "gene_symbol": symbol,
                "ensembl_gene_id": mapping.get(symbol, ""),
                "mapping_source": "Ensembl REST lookup/symbol homo_sapiens",
                "mapping_status": "mapped" if symbol in mapping else "unmapped",
            }
        )
    pd.DataFrame(mapping_rows).to_csv(TABLES / "gene_symbol_to_ensembl_mapping.csv", index=False)

    for dataset_id, genes in all_dataset_genes.items():
        counts = Counter(genes)
        unique = sorted(counts)
        unmapped = [gene for gene in unique if gene not in mapping]
        for gene in unmapped:
            unmapped_rows.append(
                {
                    "dataset_id": dataset_id,
                    "gene_symbol": gene,
                    "reason": "not confirmed by limited Phase 2 mapping prep; full Ensembl/biomaRt mapping required before Geneformer tokenization",
                }
            )
        mapping_summary_rows.append(
            {
                "dataset_id": dataset_id,
                "n_gene_rows": len(genes),
                "n_unique_gene_symbols": len(unique),
                "n_symbols_with_ensembl_mapping": len(unique) - len(unmapped),
                "n_symbols_unmapped": len(unmapped),
                "n_duplicate_gene_rows": sum(count for _, count in counts.items() if count > 1),
                "mapping_source": "Ensembl REST lookup/symbol homo_sapiens",
                "mapping_status": mapping_status,
                "phase2_gene_rule": "no gene deletion or merging; duplicate rows preserved",
            }
        )
    pd.DataFrame(unmapped_rows).to_csv(TABLES / "genes_unmapped.csv", index=False)
    pd.DataFrame(mapping_summary_rows).to_csv(TABLES / "gene_id_mapping_summary.csv", index=False)

    gene_identifier_rows = []
    for row in mapping_summary_rows:
        dataset_id = row["dataset_id"]
        genes = all_dataset_genes[dataset_id]
        counts = Counter(genes)
        gene_identifier_rows.append(
            {
                "dataset_id": dataset_id,
                "gene_identifier_type": infer_identifier_type(genes),
                "n_gene_rows": len(genes),
                "n_unique_gene_symbols": len(counts),
                "n_duplicated_symbols": sum(1 for _, count in counts.items() if count > 1),
                "n_duplicated_gene_rows": row["n_duplicate_gene_rows"],
                "n_symbols_with_ensembl_mapping": row["n_symbols_with_ensembl_mapping"],
                "n_symbols_unmapped": row["n_symbols_unmapped"],
                "mapping_ready_for_geneformer": "partial" if row["n_symbols_unmapped"] else "yes",
                "notes": "Unmapped symbols require manual review before Geneformer tokenization.",
            }
        )
    pd.DataFrame(gene_identifier_rows).to_csv(TABLES / "gene_identifier_summary.csv", index=False)

    gene_log_lines = [
        "# Phase 2 gene ID mapping log",
        "",
        f"Mapping status: {mapping_status}",
        "",
        "Rules:",
        "",
        "- No gene rows were deleted.",
        "- No duplicated symbols were merged.",
        "- Unique AnnData feature IDs were created only to make `.var_names` stable.",
        "- Original gene symbols are preserved in `.var['gene_symbol']`.",
        "- Unmapped symbols are written to `tables/genes_unmapped.csv` and require manual confirmation before Geneformer tokenization.",
        "",
    ]
    for row in mapping_summary_rows:
        gene_log_lines.extend(
            [
                f"## {row['dataset_id']}",
                "",
                f"- Gene rows: {row['n_gene_rows']}",
                f"- Unique symbols: {row['n_unique_gene_symbols']}",
                f"- Mapped symbols: {row['n_symbols_with_ensembl_mapping']}",
                f"- Unmapped symbols: {row['n_symbols_unmapped']}",
                f"- Duplicate gene rows: {row['n_duplicate_gene_rows']}",
                "",
            ]
        )
    write_md(LOGS / "phase2_gene_id_mapping_log.md", gene_log_lines)

    final_status = pd.DataFrame(
        [
            {
                "dataset_id": dataset_id,
                "role": DATASETS[dataset_id]["role"],
                "h5ad": str(DATASETS[dataset_id]["h5ad"].relative_to(PROJECT_ROOT)),
                "status": h5ad_status.get(dataset_id, "not attempted"),
            }
            for dataset_id in DATASETS
        ]
    )
    final_status.to_csv(TABLES / "phase2_anndata_status.csv", index=False)


if __name__ == "__main__":
    main()
