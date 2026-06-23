from __future__ import annotations

import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

import anndata as ad
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE2_PATH = PROJECT_ROOT / "scripts" / "01_phase2_build_anndata.py"

spec = importlib.util.spec_from_file_location("phase2", PHASE2_PATH)
phase2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(phase2)


def read_genes_from_h5ad(path: Path) -> list[str]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        genes = adata.var["gene_symbol"].astype(str).tolist()
    finally:
        adata.file.close()
    return genes


def main() -> None:
    all_dataset_genes = {}
    for dataset_id, cfg in phase2.DATASETS.items():
        all_dataset_genes[dataset_id] = read_genes_from_h5ad(cfg["h5ad"])

    duplicated_rows = []
    for dataset_id, genes in all_dataset_genes.items():
        feature_ids = phase2.make_unique_feature_ids(genes)
        counts = Counter(genes)
        grouped_ids = defaultdict(list)
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
    pd.DataFrame(duplicated_rows).to_csv(phase2.TABLES / "duplicated_genes.csv", index=False)

    unique_symbols = sorted({gene for genes in all_dataset_genes.values() for gene in genes if gene})
    mapping, mapping_status = phase2.query_mygene(unique_symbols)

    mapping_rows = [
        {
            "gene_symbol": symbol,
            "ensembl_gene_id": mapping.get(symbol, ""),
            "mapping_source": "Ensembl REST lookup/symbol homo_sapiens limited sanity check",
            "mapping_status": "mapped_limited_check" if symbol in mapping else "pending_full_mapping",
        }
        for symbol in unique_symbols
    ]
    pd.DataFrame(mapping_rows).to_csv(phase2.TABLES / "gene_symbol_to_ensembl_mapping.csv", index=False)

    unmapped_rows = []
    mapping_summary_rows = []
    for dataset_id, genes in all_dataset_genes.items():
        counts = Counter(genes)
        unique = sorted(counts)
        pending = [gene for gene in unique if gene not in mapping]
        for gene in pending:
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
                "n_symbols_with_limited_ensembl_mapping": len(unique) - len(pending),
                "n_symbols_pending_full_mapping": len(pending),
                "n_duplicate_gene_rows": sum(count for _, count in counts.items() if count > 1),
                "mapping_source": "Ensembl REST lookup/symbol homo_sapiens limited sanity check",
                "mapping_status": mapping_status,
                "phase2_gene_rule": "no gene deletion or merging; duplicate rows preserved",
            }
        )
    pd.DataFrame(unmapped_rows).to_csv(phase2.TABLES / "genes_unmapped.csv", index=False)
    pd.DataFrame(mapping_summary_rows).to_csv(phase2.TABLES / "gene_id_mapping_summary.csv", index=False)

    gene_identifier_rows = []
    for row in mapping_summary_rows:
        dataset_id = row["dataset_id"]
        genes = all_dataset_genes[dataset_id]
        counts = Counter(genes)
        gene_identifier_rows.append(
            {
                "dataset_id": dataset_id,
                "gene_identifier_type": phase2.infer_identifier_type(genes),
                "n_gene_rows": len(genes),
                "n_unique_gene_symbols": len(counts),
                "n_duplicated_symbols": sum(1 for _, count in counts.items() if count > 1),
                "n_duplicated_gene_rows": row["n_duplicate_gene_rows"],
                "n_symbols_with_limited_ensembl_mapping": row["n_symbols_with_limited_ensembl_mapping"],
                "n_symbols_pending_full_mapping": row["n_symbols_pending_full_mapping"],
                "mapping_ready_for_geneformer": "no",
                "notes": "Full symbol-to-Ensembl mapping is required before Geneformer tokenization.",
            }
        )
    pd.DataFrame(gene_identifier_rows).to_csv(phase2.TABLES / "gene_identifier_summary.csv", index=False)

    lines = [
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
        "- Phase 2 performs a limited Ensembl REST sanity check only.",
        "- Full gene-symbol-to-Ensembl mapping is still required before Geneformer tokenization.",
        "- Pending symbols are written to `tables/genes_unmapped.csv`; this means pending full mapping, not proven absent from Ensembl.",
        "",
    ]
    if mapping:
        lines.append("Limited sanity-check mappings:")
        lines.append("")
        for symbol, ensembl_id in sorted(mapping.items()):
            lines.append(f"- {symbol}: {ensembl_id}")
        lines.append("")
    for row in mapping_summary_rows:
        lines.extend(
            [
                f"## {row['dataset_id']}",
                "",
                f"- Gene rows: {row['n_gene_rows']}",
                f"- Unique symbols: {row['n_unique_gene_symbols']}",
                f"- Limited-check mapped symbols: {row['n_symbols_with_limited_ensembl_mapping']}",
                f"- Pending full mapping: {row['n_symbols_pending_full_mapping']}",
                f"- Duplicate gene rows: {row['n_duplicate_gene_rows']}",
                "",
            ]
        )
    phase2.write_md(phase2.LOGS / "phase2_gene_id_mapping_log.md", lines)


if __name__ == "__main__":
    main()
