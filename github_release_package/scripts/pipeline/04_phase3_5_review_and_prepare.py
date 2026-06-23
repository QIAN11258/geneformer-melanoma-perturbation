from __future__ import annotations

import hashlib
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
TABLES = PROJECT_ROOT / "tables"
FIGURES = PROJECT_ROOT / "figures"
LOGS = PROJECT_ROOT / "logs"
TOKENIZED_DIR = DATA_PROCESSED / "tokenized"
for directory in (TABLES, FIGURES, LOGS, TOKENIZED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

H5ADS = {
    "GSE72056_melanoma": DATA_PROCESSED / "GSE72056_melanoma.h5ad",
    "GSE72056_malignant_state_labeled": DATA_PROCESSED / "GSE72056_malignant_state_labeled.h5ad",
    "GSE115978_melanoma": DATA_PROCESSED / "GSE115978_melanoma.h5ad",
    "GSE115978_malignant_validation": DATA_PROCESSED / "GSE115978_malignant_validation.h5ad",
}

KEY_GENES = [
    "MITF", "MLANA", "PMEL", "TYR", "DCT",
    "AXL", "NGFR", "VIM", "FN1", "ZEB1", "TGFBI",
    "MKI67", "TOP2A", "PCNA", "MCM2", "STMN1",
    "HIF1A", "VEGFA", "CA9", "LDHA",
    "BRAF", "NRAS", "KIT", "PTEN", "CDKN2A", "MAPK1", "MAPK3", "AKT1", "CTNNB1",
]

SIGNATURES = {
    "invasive_like": ["AXL", "NGFR", "VIM", "FN1", "ZEB1", "TGFBI"],
    "melanocytic_like": ["MITF", "MLANA", "PMEL", "TYR", "DCT"],
    "cycling_like": ["MKI67", "TOP2A", "PCNA", "MCM2", "STMN1"],
    "stress_hypoxia_like": ["HIF1A", "VEGFA", "CA9", "LDHA"],
}


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def normalize_ensg(value: str) -> str:
    value = str(value).strip()
    return value.split(".")[0] if value else ""


def get_symbols(adata: ad.AnnData) -> list[str]:
    if "gene_symbol" in adata.var.columns:
        return adata.var["gene_symbol"].astype(str).tolist()
    return [str(x) for x in adata.var_names]


def load_resources():
    mapping = pd.read_csv(TABLES / "gene_symbol_to_ensembl_mapping_phase3.csv", dtype=str).fillna("")
    token_dict = load_pickle(PROJECT_ROOT / "data_raw" / "geneformer" / "token_dictionary_gc30M.pkl")
    median_dict = load_pickle(PROJECT_ROOT / "data_raw" / "geneformer" / "gene_median_dictionary_gc30M.pkl")
    hgnc = pd.read_csv(
        PROJECT_ROOT / "data_raw" / "gene_id_mapping" / "hgnc_complete_set.txt",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    hgnc_status = hgnc.set_index("symbol")["status"].to_dict()
    return mapping, token_dict, median_dict, hgnc_status


def mapping_lookup(mapping: pd.DataFrame) -> dict[str, dict]:
    return mapping.drop_duplicates("gene_symbol").set_index("gene_symbol").to_dict("index")


def key_gene_mapping_check(mapping: pd.DataFrame, token_vocab: set[str], hgnc_status: dict[str, str]) -> None:
    lookup = mapping_lookup(mapping)
    rows = []
    for gene in KEY_GENES:
        rec = lookup.get(gene, {})
        ensg = normalize_ensg(rec.get("ensembl_gene_id", ""))
        status = rec.get("mapping_status", "unmapped") if rec else "unmapped"
        if status == "ambiguous":
            note = "needs manual review; ambiguous mapping was not auto-selected"
        elif not ensg:
            note = "unmapped; retained in audit table but excluded from Geneformer tokenization input"
        elif ensg not in token_vocab:
            note = "mapped but absent from Geneformer vocabulary"
        else:
            note = "mapped and present in Geneformer vocabulary"
        rows.append(
            {
                "gene_symbol": gene,
                "mapped_ensembl_id": ensg,
                "hgnc_status": hgnc_status.get(gene, "needs manual confirmation"),
                "in_geneformer_vocab": bool(ensg and ensg in token_vocab),
                "mapping_status": status,
                "note": note,
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "key_melanoma_gene_mapping_check.csv", index=False)


def mean_by_feature(adata: ad.AnnData) -> np.ndarray:
    if sparse.issparse(adata.X):
        return np.asarray(adata.X.mean(axis=0)).ravel()
    return np.asarray(adata.X).mean(axis=0)


def candidate_gene_universe(
    dataset_id: str,
    adata: ad.AnnData,
    mapping_by_symbol: dict[str, dict],
    token_vocab: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    symbols = get_symbols(adata)
    means = mean_by_feature(adata)
    symbol_counts = Counter(symbols)
    rows = []
    retained = []
    for i, symbol in enumerate(symbols):
        rec = mapping_by_symbol.get(symbol, {})
        ensg = normalize_ensg(rec.get("ensembl_gene_id", ""))
        mapping_status = rec.get("mapping_status", "unmapped") if rec else "unmapped"
        in_vocab = bool(ensg and ensg in token_vocab)
        exclusion_reason = ""
        if not ensg:
            exclusion_reason = "unmapped"
        elif mapping_status == "ambiguous":
            exclusion_reason = "ambiguous_mapping"
        elif not in_vocab:
            exclusion_reason = "not_in_geneformer_vocab"
        decision = "retain_candidate" if not exclusion_reason else "exclude_before_tokenization"
        row = {
            "dataset_id": dataset_id,
            "feature_index": i,
            "gene_symbol": symbol,
            "ensembl_gene_id": ensg,
            "mapping_status": mapping_status,
            "in_geneformer_vocab": in_vocab,
            "mean_expression": float(means[i]),
            "duplicate_gene_symbol_count": symbol_counts[symbol],
            "decision": decision,
            "exclusion_reason": exclusion_reason,
        }
        rows.append(row)
        if decision == "retain_candidate":
            retained.append(row)

    by_ens = defaultdict(list)
    for row in retained:
        by_ens[row["ensembl_gene_id"]].append(row)

    duplicate_report = []
    selected_indices = set()
    duplicate_ensembl_ids = 0
    for ensg, group in by_ens.items():
        if len(group) == 1:
            selected_indices.add(group[0]["feature_index"])
            continue
        duplicate_ensembl_ids += 1
        best = sorted(group, key=lambda r: (-r["mean_expression"], r["gene_symbol"], r["feature_index"]))[0]
        selected_indices.add(best["feature_index"])
        for row in group:
            duplicate_report.append(
                {
                    "dataset_id": dataset_id,
                    "ensembl_gene_id": ensg,
                    "gene_symbol": row["gene_symbol"],
                    "feature_index": row["feature_index"],
                    "mean_expression": row["mean_expression"],
                    "resolution_rule": "max mean expression selected for tokenization; AnnData unchanged",
                    "selected_for_tokenization": row["feature_index"] == best["feature_index"],
                }
            )

    for row in rows:
        if row["decision"] == "retain_candidate":
            row["selected_for_tokenization"] = row["feature_index"] in selected_indices
            if not row["selected_for_tokenization"]:
                row["decision"] = "exclude_duplicate_ensembl_after_report"
                row["exclusion_reason"] = "duplicate_ensembl_id_not_selected"
        else:
            row["selected_for_tokenization"] = False

    summary = {
        "dataset_id": dataset_id,
        "total_gene_rows": len(symbols),
        "mapped_to_ensembl_rows": sum(bool(r["ensembl_gene_id"]) for r in rows),
        "in_geneformer_vocab_rows": sum(r["in_geneformer_vocab"] for r in rows),
        "candidate_rows_before_duplicate_resolution": len(retained),
        "duplicate_gene_symbol_rows": sum(c for _, c in symbol_counts.items() if c > 1),
        "duplicate_gene_symbols": sum(1 for _, c in symbol_counts.items() if c > 1),
        "duplicate_ensembl_ids": duplicate_ensembl_ids,
        "selected_tokenization_genes": sum(r["selected_for_tokenization"] for r in rows),
        "excluded_rows": sum(not r["selected_for_tokenization"] for r in rows),
        "unmapped_rows": sum(r["exclusion_reason"] == "unmapped" for r in rows),
        "ambiguous_rows": sum(r["exclusion_reason"] == "ambiguous_mapping" for r in rows),
        "not_in_vocab_rows": sum(r["exclusion_reason"] == "not_in_geneformer_vocab" for r in rows),
    }
    return pd.DataFrame(rows), pd.DataFrame(duplicate_report), summary


def write_filtering_rules() -> None:
    rules = [
        ("R1", "retain", "Keep only genes mapped to one Ensembl gene ID and present in Geneformer vocabulary."),
        ("R2", "exclude_with_log", "Exclude unmapped genes from tokenization input, but keep them in audit tables."),
        ("R3", "exclude_with_log", "Exclude ambiguous mappings unless manually reviewed and resolved."),
        ("R4", "report", "Report duplicate gene symbols and duplicate Ensembl IDs before tokenization."),
        ("R5", "resolve_for_tokenization_only", "For duplicate Ensembl IDs, select the feature with max mean expression for tokenization output; do not modify AnnData."),
        ("R6", "no_training", "No fine-tuning, perturbation, in silico deletion, survival analysis, or drug-resource analysis in Phase 3.5."),
    ]
    pd.DataFrame(rules, columns=["rule_id", "action", "decision_rule"]).to_csv(
        TABLES / "geneformer_gene_filtering_rules.csv", index=False
    )


def get_marker_indices(symbols: list[str], markers: list[str]) -> dict[str, list[int]]:
    idx = defaultdict(list)
    for i, symbol in enumerate(symbols):
        if symbol in markers:
            idx[symbol].append(i)
    return idx


def compute_signature_scores(adata: ad.AnnData) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    symbols = get_symbols(adata)
    all_markers = [m for markers in SIGNATURES.values() for m in markers]
    marker_indices = get_marker_indices(symbols, all_markers)
    marker_rows = []
    scores = {}
    for sig, markers in SIGNATURES.items():
        indices = []
        for marker in markers:
            present = marker_indices.get(marker, [])
            marker_rows.append(
                {
                    "signature": sig,
                    "marker": marker,
                    "present_in_var": bool(present),
                    "n_matching_features": len(present),
                    "feature_indices": ";".join(map(str, present)),
                }
            )
            indices.extend(present)
        if indices:
            sub = adata.X[:, indices]
            score = np.asarray(sub.mean(axis=1)).ravel() if sparse.issparse(sub) else np.asarray(sub).mean(axis=1)
        else:
            score = np.full(adata.n_obs, np.nan)
        scores[sig] = score
    return pd.DataFrame(marker_rows), scores


def assign_state_labels(scores: dict[str, np.ndarray]) -> tuple[pd.DataFrame, list[str]]:
    sigs = list(SIGNATURES)
    zscores = {}
    for sig in sigs:
        x = scores[sig].astype(float)
        sd = np.nanstd(x)
        zscores[sig] = (x - np.nanmean(x)) / sd if sd > 0 else np.zeros_like(x)
    zmat = np.vstack([zscores[sig] for sig in sigs]).T
    labels = []
    for row in zmat:
        order = np.argsort(row)[::-1]
        top, second = order[0], order[1]
        if row[top] >= 0.5 and row[top] - row[second] >= 0.25:
            labels.append(sigs[top])
        else:
            labels.append("intermediate/ambiguous")
    obs_scores = pd.DataFrame({f"{sig}_score": scores[sig] for sig in sigs})
    for sig in sigs:
        obs_scores[f"{sig}_zscore"] = zscores[sig]
    obs_scores["preliminary_malignant_state"] = labels
    return obs_scores, labels


def plot_umap_and_validation() -> None:
    adata = ad.read_h5ad(H5ADS["GSE72056_malignant_state_labeled"])
    if "X_umap_phase3_5" not in adata.obsm:
        x = adata.X
        mean = np.asarray(x.mean(axis=0)).ravel() if sparse.issparse(x) else np.asarray(x).mean(axis=0)
        sq = x.copy()
        if sparse.issparse(sq):
            sq.data **= 2
            second = np.asarray(sq.mean(axis=0)).ravel()
        else:
            second = np.asarray(x**2).mean(axis=0)
        var = second - mean**2
        top = np.argsort(var)[-min(2000, adata.n_vars):]
        x_top = x[:, top]
        n_comp = min(30, x_top.shape[0] - 1, x_top.shape[1] - 1)
        emb = TruncatedSVD(n_components=n_comp, random_state=20260621).fit_transform(x_top)
        coords = umap.UMAP(n_neighbors=20, min_dist=0.35, random_state=20260621).fit_transform(emb)
        adata.obsm["X_umap_phase3_5"] = coords
        adata.write_h5ad(H5ADS["GSE72056_malignant_state_labeled"], compression="gzip")
    coords = adata.obsm["X_umap_phase3_5"]

    def scatter_category(field: str, path: Path, title: str) -> None:
        labels = adata.obs[field].astype(str)
        cats = sorted(labels.unique())
        cmap = plt.get_cmap("tab20")
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        for i, cat in enumerate(cats):
            mask = labels == cat
            ax.scatter(coords[mask, 0], coords[mask, 1], s=10, color=cmap(i % 20), label=cat, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    scatter_category("preliminary_malignant_state", FIGURES / "GSE72056_umap_malignant_state.png", "GSE72056 malignant states")
    scatter_category("tumor_id", FIGURES / "GSE72056_umap_tumor_id.png", "GSE72056 tumor_id")

    score_cols = [
        ("invasive_like_score", "AXL/invasive score"),
        ("melanocytic_like_score", "MITF/melanocytic score"),
        ("cycling_like_score", "Cycling score"),
        ("stress_hypoxia_like_score", "Hypoxia/stress score"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, (col, title) in zip(axes.ravel(), score_cols):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=adata.obs[col].astype(float), s=10, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(FIGURES / "GSE72056_signature_score_umaps.png", dpi=180)
    plt.close(fig)

    marker_summary(adata, "GSE72056", group_field="preliminary_malignant_state")
    signature_violin(adata, FIGURES / "GSE72056_signature_score_violin.png")
    marker_dotplot(adata, FIGURES / "GSE72056_marker_dotplot.png")
    tumor_distribution_and_dominance(adata)


def marker_summary(adata: ad.AnnData, prefix: str, group_field: str) -> pd.DataFrame:
    symbols = get_symbols(adata)
    all_markers = [m for markers in SIGNATURES.values() for m in markers]
    marker_to_indices = get_marker_indices(symbols, all_markers)
    groups = adata.obs[group_field].astype(str)
    rows = []
    for group in sorted(groups.unique()):
        mask = np.where(groups == group)[0]
        for sig, markers in SIGNATURES.items():
            sig_scores = adata.obs.loc[groups == group, f"{sig}_score"].astype(float)
            rows.append(
                {
                    "dataset_id": prefix,
                    "group_field": group_field,
                    "group": group,
                    "feature_type": "signature_score",
                    "signature": sig,
                    "marker": "",
                    "mean_expression": "",
                    "detected_fraction": "",
                    "mean_signature_score": float(sig_scores.mean()),
                    "median_signature_score": float(sig_scores.median()),
                    "n_cells": int(len(mask)),
                }
            )
            for marker in markers:
                idxs = marker_to_indices.get(marker, [])
                if idxs:
                    sub = adata.X[mask, :][:, idxs]
                    arr = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
                    mean_expr = float(arr.mean())
                    frac = float((arr > 0).any(axis=1).mean())
                else:
                    mean_expr = np.nan
                    frac = np.nan
                rows.append(
                    {
                        "dataset_id": prefix,
                        "group_field": group_field,
                        "group": group,
                        "feature_type": "marker",
                        "signature": sig,
                        "marker": marker,
                        "mean_expression": mean_expr,
                        "detected_fraction": frac,
                        "mean_signature_score": "",
                        "median_signature_score": "",
                        "n_cells": int(len(mask)),
                    }
                )
    df = pd.DataFrame(rows)
    if prefix == "GSE72056":
        df.to_csv(TABLES / "GSE72056_state_marker_expression_summary.csv", index=False)
        df[df["feature_type"] == "signature_score"].to_csv(
            TABLES / "GSE72056_state_signature_score_summary.csv", index=False
        )
    return df


def marker_dotplot(adata: ad.AnnData, path: Path) -> None:
    symbols = get_symbols(adata)
    markers = [m for group in SIGNATURES.values() for m in group]
    marker_to_indices = get_marker_indices(symbols, markers)
    groups = sorted(adata.obs["preliminary_malignant_state"].astype(str).unique())
    mean_vals, frac_vals = [], []
    for group in groups:
        mask = np.where(adata.obs["preliminary_malignant_state"].astype(str) == group)[0]
        means, fracs = [], []
        for marker in markers:
            idxs = marker_to_indices.get(marker, [])
            if idxs:
                sub = adata.X[mask, :][:, idxs]
                arr = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
                means.append(float(arr.mean()))
                fracs.append(float((arr > 0).any(axis=1).mean()))
            else:
                means.append(0.0)
                fracs.append(0.0)
        mean_vals.append(means)
        frac_vals.append(fracs)
    mean_vals = np.asarray(mean_vals)
    frac_vals = np.asarray(frac_vals)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for y, group in enumerate(groups):
        for x, marker in enumerate(markers):
            ax.scatter(x, y, s=30 + 260 * frac_vals[y, x], c=[mean_vals[y, x]], cmap="viridis", vmin=mean_vals.min(), vmax=mean_vals.max())
    ax.set_xticks(range(len(markers)))
    ax.set_xticklabels(markers, rotation=60, ha="right")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_title("Marker dotplot by malignant state")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=mean_vals.min(), vmax=mean_vals.max()))
    fig.colorbar(sm, ax=ax, label="Mean expression")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def signature_violin(adata: ad.AnnData, path: Path) -> None:
    groups = sorted(adata.obs["preliminary_malignant_state"].astype(str).unique())
    score_cols = list(SIGNATURES)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, sig in zip(axes.ravel(), score_cols):
        data = [
            adata.obs.loc[adata.obs["preliminary_malignant_state"].astype(str) == g, f"{sig}_score"].astype(float).to_numpy()
            for g in groups
        ]
        ax.violinplot(data, showmeans=True, showextrema=False)
        ax.set_title(sig)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Signature score")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def tumor_distribution_and_dominance(adata: ad.AnnData) -> None:
    tab = (
        adata.obs.groupby(["tumor_id", "preliminary_malignant_state"], observed=False)
        .size()
        .reset_index(name="n_cells")
    )
    tab["tumor_total_cells"] = tab.groupby("tumor_id")["n_cells"].transform("sum")
    tab["state_fraction_within_tumor"] = tab["n_cells"] / tab["tumor_total_cells"]
    tab.to_csv(TABLES / "GSE72056_state_by_tumor_id_distribution.csv", index=False)

    dom_rows = []
    for state, sub in tab.groupby("preliminary_malignant_state"):
        state_total = int(sub["n_cells"].sum())
        top = sub.sort_values("n_cells", ascending=False).iloc[0]
        dom_rows.append(
            {
                "state": state,
                "state_total_cells": state_total,
                "top_tumor_id": top["tumor_id"],
                "top_tumor_cells": int(top["n_cells"]),
                "top_tumor_fraction_of_state": float(top["n_cells"] / state_total) if state_total else 0,
                "dominance_flag": "single_tumor_dominated" if state_total and top["n_cells"] / state_total >= 0.5 else "not_single_tumor_dominated",
            }
        )
    pd.DataFrame(dom_rows).to_csv(TABLES / "GSE72056_state_patient_dominance_check.csv", index=False)


def label_gse115978() -> None:
    adata = ad.read_h5ad(H5ADS["GSE115978_malignant_validation"])
    marker_check, scores = compute_signature_scores(adata)
    obs_scores, labels = assign_state_labels(scores)
    for col in obs_scores.columns:
        adata.obs[col] = obs_scores[col].to_numpy()
    adata.uns["phase3_5_label_rule"] = "Same marker signatures and thresholds as GSE72056: top z-score >= 0.5 and margin >= 0.25; otherwise intermediate/ambiguous."
    out = DATA_PROCESSED / "GSE115978_malignant_state_labeled.h5ad"
    adata.write_h5ad(out, compression="gzip")

    marker_check.to_csv(TABLES / "GSE115978_malignant_state_signature_marker_check.csv", index=False)
    dist = adata.obs["preliminary_malignant_state"].value_counts().rename_axis("state").reset_index(name="n_cells")
    dist["percentage"] = dist["n_cells"] / adata.n_obs * 100
    dist.to_csv(TABLES / "GSE115978_malignant_state_label_distribution.csv", index=False)
    sample_tab = (
        adata.obs.groupby(["sample_id", "preliminary_malignant_state"], observed=False)
        .size()
        .reset_index(name="n_cells")
    )
    sample_tab["sample_total_cells"] = sample_tab.groupby("sample_id")["n_cells"].transform("sum")
    sample_tab["state_fraction_within_sample"] = sample_tab["n_cells"] / sample_tab["sample_total_cells"]
    sample_tab.to_csv(TABLES / "GSE115978_state_by_sample_id_distribution.csv", index=False)
    treatment_tab = (
        adata.obs.groupby(["treatment_group", "preliminary_malignant_state"], observed=False)
        .size()
        .reset_index(name="n_cells")
    )
    treatment_tab["treatment_total_cells"] = treatment_tab.groupby("treatment_group")["n_cells"].transform("sum")
    treatment_tab["state_fraction_within_treatment_group"] = treatment_tab["n_cells"] / treatment_tab["treatment_total_cells"]
    treatment_tab.to_csv(TABLES / "GSE115978_state_by_treatment_group_distribution.csv", index=False)


def deterministic_split(units: list[str]) -> dict[str, str]:
    ranked = sorted(set(units), key=lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())
    n = len(ranked)
    n_test = max(1, round(n * 0.2))
    n_val = max(1, round(n * 0.2))
    result = {}
    for i, unit in enumerate(ranked):
        if i < n_test:
            result[unit] = "held-out test"
        elif i < n_test + n_val:
            result[unit] = "validation"
        else:
            result[unit] = "train"
    return result


def split_verification() -> None:
    specs = {
        "GSE72056_melanoma": ("tumor_id", "tumor-level split", "tumor_id available, true patient identity needs manual confirmation"),
        "GSE115978_melanoma": ("sample_id", "sample-level split", "sample_id available, true patient identity needs manual confirmation"),
    }
    summary_rows = []
    plan_rows = []
    lines = [
        "# Phase 3.5 split unit verification log",
        "",
        "Cell-level random split is not used as the primary training/validation scheme.",
        "",
    ]
    for dataset_id, (field, split_type, note) in specs.items():
        adata = ad.read_h5ad(H5ADS[dataset_id], backed="r")
        obs = adata.obs.copy()
        adata.file.close()
        available = field in obs.columns
        units = sorted(obs[field].astype(str).unique()) if available else []
        assignments = deterministic_split(units) if units else {}
        summary_rows.append(
            {
                "dataset_id": dataset_id,
                "source_field": field,
                "field_available": available,
                "n_units": len(units),
                "recommended_split_level": split_type if available else "needs manual confirmation",
                "patient_identity_status": "needs manual confirmation",
                "usable_without_cell_level_random_split": bool(available),
                "note": note,
            }
        )
        for unit, split in sorted(assignments.items(), key=lambda kv: (kv[1], kv[0])):
            sub = obs[obs[field].astype(str) == unit]
            plan_rows.append(
                {
                    "dataset_id": dataset_id,
                    "split": split,
                    "split_unit": unit,
                    "split_level": split_type,
                    "n_cells": int(sub.shape[0]),
                    "source_field": field,
                    "patient_identity_status": "needs manual confirmation",
                    "note": note,
                }
            )
        counts = Counter(assignments.values())
        lines += [
            f"## {dataset_id}",
            "",
            f"- Source field: {field}",
            f"- Recommended split level: {split_type if available else 'needs manual confirmation'}",
            f"- Units: {len(units)}",
            f"- Train/validation/test units: {counts.get('train', 0)}/{counts.get('validation', 0)}/{counts.get('held-out test', 0)}",
            f"- Note: {note}",
            "",
        ]
    pd.DataFrame(summary_rows).to_csv(TABLES / "split_unit_verification_summary.csv", index=False)
    pd.DataFrame(plan_rows).to_csv(TABLES / "patient_or_sample_level_split_plan_phase3_5.csv", index=False)
    (LOGS / "phase3_5_split_unit_verification_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_tokenization(
    dataset_id: str,
    path: Path,
    mapping_by_symbol: dict[str, dict],
    token_dict: dict,
    median_dict: dict,
    max_length: int = 4096,
) -> dict:
    adata = ad.read_h5ad(path)
    token_vocab = {normalize_ensg(k) for k in token_dict if str(k).startswith("ENSG")}
    universe, duplicate_report, summary = candidate_gene_universe(dataset_id, adata, mapping_by_symbol, token_vocab)
    selected = universe[universe["selected_for_tokenization"] == True].copy()
    feature_token = np.full(adata.n_vars, -1, dtype=np.int64)
    feature_median = np.ones(adata.n_vars, dtype=np.float32)
    for _, row in selected.iterrows():
        idx = int(row["feature_index"])
        ensg = row["ensembl_gene_id"]
        feature_token[idx] = int(token_dict[ensg])
        try:
            med = float(median_dict.get(ensg, 1.0))
        except Exception:
            med = 1.0
        feature_median[idx] = med if np.isfinite(med) and med > 0 else 1.0

    x = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    pad_token = int(token_dict.get("<pad>", 0))
    mask_token_valid = "<mask>" in token_dict
    input_ids = np.full((adata.n_obs, max_length), pad_token, dtype=np.int32)
    attention = np.zeros((adata.n_obs, max_length), dtype=np.uint8)
    raw_lengths = np.zeros(adata.n_obs, dtype=np.int32)
    empty = 0
    short = 0
    for i in range(adata.n_obs):
        row = x.getrow(i)
        idx = row.indices
        data = row.data.astype(np.float32)
        valid = (data > 0) & (feature_token[idx] >= 0)
        if not np.any(valid):
            empty += 1
            continue
        idx = idx[valid]
        data = data[valid]
        scores = data / feature_median[idx]
        tokens = feature_token[idx]
        # feature_token has one feature per Ensembl after duplicate resolution, but collapse defensively.
        best = {}
        for token, score in zip(tokens, scores):
            token = int(token)
            score = float(score)
            if token not in best or score > best[token]:
                best[token] = score
        ordered = [token for token, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]
        raw_lengths[i] = len(ordered)
        if len(ordered) < 100:
            short += 1
        clipped = ordered[:max_length]
        input_ids[i, : len(clipped)] = clipped
        attention[i, : len(clipped)] = 1

    out_npz = TOKENIZED_DIR / f"{dataset_id}_geneformer_ready_tokens.npz"
    np.savez_compressed(
        out_npz,
        input_ids=input_ids,
        attention_mask=attention,
        raw_sequence_lengths=raw_lengths,
        pad_token=np.array([pad_token], dtype=np.int32),
        max_length=np.array([max_length], dtype=np.int32),
    )
    obs_cols = [c for c in ["cell_id", "sample_id", "tumor_id", "treatment_group", "preliminary_malignant_state"] if c in adata.obs.columns]
    adata.obs[obs_cols].to_csv(TOKENIZED_DIR / f"{dataset_id}_tokenized_obs_metadata.csv")
    selected.to_csv(TOKENIZED_DIR / f"{dataset_id}_tokenized_gene_selection.csv", index=False)
    if not duplicate_report.empty:
        duplicate_report.to_csv(TOKENIZED_DIR / f"{dataset_id}_tokenized_duplicate_resolution_report.csv", index=False)

    attn_valid = bool(((attention == 0) | (attention == 1)).all())
    token_valid = bool((input_ids >= 0).all())
    return {
        "dataset_id": dataset_id,
        "source_h5ad": str(path.relative_to(PROJECT_ROOT)),
        "tokenized_npz": str(out_npz.relative_to(PROJECT_ROOT)),
        "cell_number": adata.n_obs,
        "genes_retained": int(summary["selected_tokenization_genes"]),
        "genes_excluded": int(summary["excluded_rows"]),
        "candidate_rows_before_duplicate_resolution": int(summary["candidate_rows_before_duplicate_resolution"]),
        "median_token_sequence_length": float(np.median(raw_lengths)),
        "min_token_sequence_length": int(raw_lengths.min()),
        "max_token_sequence_length": int(raw_lengths.max()),
        "empty_sequence_count": int(empty),
        "abnormal_short_sequence_count_lt100": int(short),
        "attention_mask_validity": "pass" if attn_valid else "fail",
        "special_token_validity": "pass" if "<pad>" in token_dict and mask_token_valid else "review_required",
        "pad_token": pad_token,
        "max_length": max_length,
        "notes": "Full tokenization preparation only; no fine-tuning or perturbation.",
    }


def full_tokenization_prep(mapping: pd.DataFrame, token_dict: dict, median_dict: dict) -> None:
    mapping_by_symbol = mapping_lookup(mapping)
    datasets = {
        "GSE115978_malignant_state_labeled": DATA_PROCESSED / "GSE115978_malignant_state_labeled.h5ad",
        "GSE72056_malignant_state_labeled": DATA_PROCESSED / "GSE72056_malignant_state_labeled.h5ad",
    }
    rows = []
    log = [
        "# Phase 3.5 full tokenization preparation log",
        "",
        "No fine-tuning, in silico deletion, perturbation, candidate target output, survival analysis, or drug-resource analysis was performed.",
        "",
        "Gene filtering follows `tables/geneformer_gene_filtering_rules.csv`.",
        "",
    ]
    for dataset_id, path in datasets.items():
        summary = prepare_tokenization(dataset_id, path, mapping_by_symbol, token_dict, median_dict)
        rows.append(summary)
        log += [
            f"## {dataset_id}",
            "",
            f"- Cells: {summary['cell_number']}",
            f"- Genes retained: {summary['genes_retained']}",
            f"- Genes excluded: {summary['genes_excluded']}",
            f"- Median token sequence length: {summary['median_token_sequence_length']}",
            f"- Empty sequences: {summary['empty_sequence_count']}",
            f"- Attention mask validity: {summary['attention_mask_validity']}",
            f"- Special token validity: {summary['special_token_validity']}",
            f"- Output: `{summary['tokenized_npz']}`",
            "",
        ]
    pd.DataFrame(rows).to_csv(TABLES / "full_tokenization_preparation_summary.csv", index=False)
    (LOGS / "phase3_5_full_tokenization_preparation_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")


def write_state_validation_log() -> None:
    dist = pd.read_csv(TABLES / "GSE72056_malignant_state_label_distribution.csv")
    dom = pd.read_csv(TABLES / "GSE72056_state_patient_dominance_check.csv")
    lines = [
        "# Phase 3.5 state label validation log",
        "",
        "No DEG, fine-tuning, perturbation, or causal interpretation was performed.",
        "",
        "GSE72056 malignant-state label distribution:",
        "",
    ]
    for _, row in dist.iterrows():
        lines.append(f"- {row['state']}: {int(row['n_cells'])} cells ({float(row['percentage']):.2f}%)")
    lines += ["", "Single tumor dominance check:"]
    for _, row in dom.iterrows():
        lines.append(
            f"- {row['state']}: top tumor {row['top_tumor_id']} contributes {float(row['top_tumor_fraction_of_state'])*100:.2f}% ({row['dominance_flag']})"
        )
    (LOGS / "phase3_5_state_label_validation_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary() -> None:
    key = pd.read_csv(TABLES / "key_melanoma_gene_mapping_check.csv")
    overlap = pd.read_csv(TABLES / "geneformer_vocab_overlap.csv")
    gse72056_dist = pd.read_csv(TABLES / "GSE72056_malignant_state_label_distribution.csv")
    gse115978_dist = pd.read_csv(TABLES / "GSE115978_malignant_state_label_distribution.csv")
    tok = pd.read_csv(TABLES / "full_tokenization_preparation_summary.csv")
    split = pd.read_csv(TABLES / "split_unit_verification_summary.csv")
    key_pass = bool((key["in_geneformer_vocab"].astype(str).str.lower() == "true").all())
    lines = [
        "# Phase 3.5 中文总结",
        "",
        "## 1. 关键 melanoma marker mapping",
        "",
        f"- 关键 marker 全部在 Geneformer vocabulary 中：{'是' if key_pass else '否'}。",
        "- 详细表：`tables/key_melanoma_gene_mapping_check.csv`。",
        "",
        "## 2. Geneformer gene filtering/collapsing 规则",
        "",
        "- 仅保留成功映射到唯一 Ensembl ID 且存在于 Geneformer vocabulary 的基因。",
        "- unmapped genes 排除于 tokenization，但保留日志。",
        "- ambiguous genes 排除，除非人工确认唯一映射。",
        "- duplicate Ensembl IDs 仅在 tokenization 输出中按 max mean expression 选择一个 feature；AnnData 不改动。",
        "- 规则表：`tables/geneformer_gene_filtering_rules.csv`。",
        "",
        "## 3. GSE72056 malignant-state labels 生物学合理性",
        "",
        "- 已生成 UMAP、marker dotplot、signature violin、state-by-tumor distribution 和 dominance check。",
        "- marker signatures 与预期生物学轴一致，但 GSE72056 是 processed expression，因此仅建议 sensitivity analysis。",
        "- 初步 label 分布：",
    ]
    for _, row in gse72056_dist.iterrows():
        lines.append(f"  - {row['state']}: {int(row['n_cells'])} cells ({float(row['percentage']):.2f}%)")
    lines += [
        "",
        "## 4. GSE115978 独立 malignant-state labels",
        "",
        "- 已用同一 marker set 和同一阈值规则生成独立 labels。",
        "- 详细输出：`data_processed/GSE115978_malignant_state_labeled.h5ad`。",
        "- 初步 label 分布：",
    ]
    for _, row in gse115978_dist.iterrows():
        lines.append(f"  - {row['state']}: {int(row['n_cells'])} cells ({float(row['percentage']):.2f}%)")
    gse115978_overlap = overlap.loc[overlap["dataset_id"] == "GSE115978_malignant_validation", "overlap_percentage"].iloc[0]
    lines += [
        "",
        "## 5. GSE115978 是否可作为 Phase 4 fine-tuning 主输入",
        "",
        f"- 技术上最合适：raw count-like integer matrix，Geneformer overlap {gse115978_overlap:.2f}%，full tokenization preparation 已完成。",
        "- 但仍需人工确认 sample_id 是否可代表 patient-level split unit，以及 ambiguous/unmapped gene mapping 是否可接受。",
        "",
        "## 6. GSE72056 是否仅作为 sensitivity analysis",
        "",
        "- 是。GSE72056 malignant-state labels 可用于 biological sensitivity analysis，但表达矩阵是 processed/likely normalized，不建议作为主 fine-tuning 输入。",
        "",
        "## 7. patient/sample-level split 是否可用",
        "",
    ]
    for _, row in split.iterrows():
        lines.append(
            f"- {row['dataset_id']}: {row['recommended_split_level']}；{row['patient_identity_status']}；{row['note']}"
        )
    lines += [
        "",
        "## 8. 是否具备进入 Phase 4 supervised fine-tuning 条件",
        "",
        "- 暂不建议直接进入 Phase 4。",
        "- GSE115978 可以作为最优 technical pilot，但 Phase 4 前需要人工确认 split unit、mapping ambiguity 和监督标签策略。",
        "",
        "## 9. Phase 4 前仍需人工确认",
        "",
        "- GSE115978 sample_id 与真实 patient identity 的关系。",
        "- GSE72056 tumor_id 是否等价于 patient-level unit。",
        "- ambiguous gene mappings 是否人工 resolve。",
        "- duplicate Ensembl ID 使用 max mean expression 规则是否可接受。",
        "- malignant-state labels 是否符合领域专家对黑色素瘤状态的解释。",
        "- 是否接受 GSE72056 processed expression 仅作 sensitivity analysis。",
        "",
        "## 10. Full tokenization preparation",
        "",
    ]
    for _, row in tok.iterrows():
        lines.append(
            f"- {row['dataset_id']}: cells={int(row['cell_number'])}, retained genes={int(row['genes_retained'])}, median length={float(row['median_token_sequence_length']):.1f}, empty sequences={int(row['empty_sequence_count'])}."
        )
    (PROJECT_ROOT / "summary_phase3_5_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    mapping, token_dict, median_dict, hgnc_status = load_resources()
    token_vocab = {normalize_ensg(k) for k in token_dict if str(k).startswith("ENSG")}
    mapping_by_symbol = mapping_lookup(mapping)

    key_gene_mapping_check(mapping, token_vocab, hgnc_status)
    write_filtering_rules()
    label_gse115978()

    candidate_summaries = []
    duplicate_reports = []
    for dataset_id, path in {
        "GSE72056_malignant_state_labeled": H5ADS["GSE72056_malignant_state_labeled"],
        "GSE115978_malignant_state_labeled": DATA_PROCESSED / "GSE115978_malignant_state_labeled.h5ad",
    }.items():
        adata = ad.read_h5ad(path)
        _, dup, summary = candidate_gene_universe(dataset_id, adata, mapping_by_symbol, token_vocab)
        candidate_summaries.append(summary)
        if not dup.empty:
            duplicate_reports.append(dup)
    pd.DataFrame(candidate_summaries).to_csv(TABLES / "geneformer_candidate_gene_universe_summary.csv", index=False)
    if duplicate_reports:
        pd.concat(duplicate_reports, ignore_index=True).to_csv(TABLES / "geneformer_duplicate_resolution_report.csv", index=False)
    else:
        pd.DataFrame(columns=["dataset_id", "ensembl_gene_id", "gene_symbol", "feature_index", "mean_expression", "resolution_rule", "selected_for_tokenization"]).to_csv(
            TABLES / "geneformer_duplicate_resolution_report.csv", index=False
        )

    filter_log = [
        "# Phase 3.5 gene filtering decision log",
        "",
        "No rules were applied to formal model training in Phase 3.5.",
        "",
        "- Retain only uniquely mapped Ensembl genes found in Geneformer vocabulary.",
        "- Exclude unmapped and ambiguous genes from tokenization input, with audit tables retained.",
        "- Duplicate Ensembl IDs are resolved for tokenization output only by max mean expression.",
        "- Original AnnData objects are not modified by filtering/collapsing rules.",
        "",
    ]
    for row in candidate_summaries:
        filter_log.append(
            f"- {row['dataset_id']}: selected {row['selected_tokenization_genes']} / {row['total_gene_rows']} gene rows after filtering and duplicate resolution."
        )
    (LOGS / "phase3_5_gene_filtering_decision_log.md").write_text("\n".join(filter_log) + "\n", encoding="utf-8")

    plot_umap_and_validation()
    write_state_validation_log()
    split_verification()
    full_tokenization_prep(mapping, token_dict, median_dict)
    write_summary()


if __name__ == "__main__":
    main()
