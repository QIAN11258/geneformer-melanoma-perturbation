from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"
FIG_DIR = ROOT / "figures" / "phase5C_TCGA_SKCM_survival_plots"

REQUIRED_FILES = [
    TABLES / "phase5C_input_integrity_check.csv",
    TABLES / "phase5C_candidate_gene_panel.csv",
    TABLES / "phase5C_artifact_essentiality_filter.csv",
    TABLES / "phase5C_TCGA_SKCM_expression_summary.csv",
    TABLES / "phase5C_TCGA_SKCM_survival_association.csv",
    TABLES / "phase5C_DepMap_melanoma_dependency.csv",
    TABLES / "phase5C_DepMap_pan_cancer_dependency_context.csv",
    TABLES / "phase5C_GDSC_drug_sensitivity_association.csv",
    TABLES / "phase5C_druggability_tractability_summary.csv",
    TABLES / "phase5C_integrated_exploratory_evidence_matrix.csv",
    LOGS / "phase5C_preflight_check_log.md",
    LOGS / "phase5C_candidate_panel_log.md",
    LOGS / "phase5C_artifact_filter_log.md",
    LOGS / "phase5C_TCGA_SKCM_log.md",
    LOGS / "phase5C_DepMap_log.md",
    LOGS / "phase5C_GDSC_log.md",
    LOGS / "phase5C_druggability_log.md",
    LOGS / "phase5C_evidence_integration_log.md",
    ROOT / "summary_phase5C_zh.md",
]


def main() -> None:
    failures: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            failures.append(f"missing: {path.relative_to(ROOT)}")
        elif path.stat().st_size <= 0:
            failures.append(f"empty: {path.relative_to(ROOT)}")

    dfs = {}
    for path in REQUIRED_FILES:
        if path.suffix == ".csv" and path.exists() and path.stat().st_size > 0:
            dfs[path.stem] = pd.read_csv(path)

    expected_genes = {
        "PABPC1", "FOS", "RPL15", "HSP90AB1", "RPL8", "ACTG1", "RACK1",
        "FN1", "TGFBI", "JUN", "ATF3", "COL1A2", "HLA-B",
    }

    integrity = dfs.get("phase5C_input_integrity_check", pd.DataFrame())
    if integrity.empty or not integrity["status"].eq("ok").all():
        failures.append("input integrity check has non-ok rows")

    for stem in [
        "phase5C_candidate_gene_panel",
        "phase5C_artifact_essentiality_filter",
        "phase5C_TCGA_SKCM_expression_summary",
        "phase5C_DepMap_melanoma_dependency",
        "phase5C_DepMap_pan_cancer_dependency_context",
        "phase5C_druggability_tractability_summary",
        "phase5C_integrated_exploratory_evidence_matrix",
    ]:
        df = dfs.get(stem, pd.DataFrame())
        if len(df) != 13:
            failures.append(f"{stem} row count != 13: {len(df)}")
        elif set(df["gene_symbol"]) != expected_genes:
            failures.append(f"{stem} gene set mismatch")

    survival = dfs.get("phase5C_TCGA_SKCM_survival_association", pd.DataFrame())
    if survival.empty:
        failures.append("TCGA survival association table is empty")
    else:
        required_cols = {"gene_symbol", "endpoint", "model", "status", "interpretation_scope"}
        if not required_cols.issubset(survival.columns):
            failures.append("TCGA survival table missing required columns")
        if "FDR" not in survival.columns:
            failures.append("TCGA survival table missing FDR column")

    gdsc = dfs.get("phase5C_GDSC_drug_sensitivity_association", pd.DataFrame())
    if gdsc.empty:
        failures.append("GDSC association table is empty")
    else:
        if "FDR" not in gdsc.columns:
            failures.append("GDSC table missing FDR column")
        if gdsc["drug"].nunique() < 3:
            failures.append("GDSC table contains fewer than 3 selected drug features")

    depmap = dfs.get("phase5C_DepMap_melanoma_dependency", pd.DataFrame())
    if not depmap.empty:
        if depmap["melanoma_model_n"].min() <= 0:
            failures.append("DepMap melanoma model count is zero for at least one gene")
        if "expression_dependency_FDR" not in depmap.columns:
            failures.append("DepMap table missing expression_dependency_FDR")

    drug = dfs.get("phase5C_druggability_tractability_summary", pd.DataFrame())
    if not drug.empty and not drug["drugbank_status"].astype(str).str.contains("needs manual confirmation", case=False, na=False).all():
        failures.append("DrugBank manual confirmation status is not consistently marked")

    integrated = dfs.get("phase5C_integrated_exploratory_evidence_matrix", pd.DataFrame())
    if not integrated.empty:
        if not integrated["allowed_interpretation"].astype(str).str.contains("exploratory", case=False, na=False).all():
            failures.append("integrated matrix missing exploratory interpretation flag")
        if not integrated["forbidden_interpretation"].astype(str).str.contains("do_not_present", case=False, na=False).all():
            failures.append("integrated matrix missing forbidden interpretation flag")

    summary_path = ROOT / "summary_phase5C_zh.md"
    summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    if "READY_FOR_PHASE5D = CONDITIONAL" not in summary:
        failures.append("summary missing READY_FOR_PHASE5D = CONDITIONAL")
    if "exploratory" not in summary.lower() and "探索" not in summary:
        failures.append("summary missing exploratory boundary wording")

    pngs = list(FIG_DIR.glob("*.png")) if FIG_DIR.exists() else []
    if len(pngs) < 10:
        failures.append(f"expected at least 10 KM plot PNGs, found {len(pngs)}")

    result = {
        "verification": "PASS" if not failures else "FAIL",
        "failures": failures,
        "tables": {name: list(df.shape) for name, df in dfs.items()},
        "km_plot_png_count": len(pngs),
        "integrated_priority_counts": integrated["final_exploratory_priority"].value_counts().to_dict() if not integrated.empty else {},
        "gdsc_drug_count": int(gdsc["drug"].nunique()) if not gdsc.empty else 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)
    print("PHASE5C_VERIFICATION: PASS")


if __name__ == "__main__":
    main()
