from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"


def ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, encoding="utf-8-sig")


required = [
    ROOT / "summary_phase5A_zh.md",
    LOGS / "phase5A_preflight_check_log.md",
    LOGS / "phase5A_cell_set_selection_log.md",
    LOGS / "phase5A_gene_whitelist_log.md",
    LOGS / "phase5A_baseline_prediction_log.md",
    LOGS / "phase5A_pilot_deletion_log.md",
    LOGS / "phase5A_ranking_interpretation_log.md",
    LOGS / "phase5A_GSE72056_pilot_sensitivity_log.md",
    TABLES / "phase5A_input_integrity_check.csv",
    TABLES / "phase5A_perturbation_cell_set_summary.csv",
    TABLES / "phase5A_gene_whitelist_eligibility.csv",
    TABLES / "phase5A_baseline_prediction_summary.csv",
    TABLES / "phase5A_pilot_deletion_effect_by_gene.csv",
    TABLES / "phase5A_pilot_deletion_effect_by_cell_gene.csv",
    TABLES / "phase5A_exploratory_perturbation_ranking.csv",
    TABLES / "phase5A_GSE72056_pilot_deletion_sensitivity.csv",
]

missing = [str(path.relative_to(ROOT)) for path in required if not ok(path)]
integrity = read_csv("phase5A_input_integrity_check.csv")
cell_set = read_csv("phase5A_perturbation_cell_set_summary.csv")
elig = read_csv("phase5A_gene_whitelist_eligibility.csv")
baseline = read_csv("phase5A_baseline_prediction_summary.csv")
by_gene = read_csv("phase5A_pilot_deletion_effect_by_gene.csv")
by_cell = read_csv("phase5A_pilot_deletion_effect_by_cell_gene.csv")
ranking = read_csv("phase5A_exploratory_perturbation_ranking.csv")
gse = read_csv("phase5A_GSE72056_pilot_deletion_sensitivity.csv")
summary = (ROOT / "summary_phase5A_zh.md").read_text(encoding="utf-8-sig")
ready_line = next((line.strip() for line in summary.splitlines() if line.strip().startswith("READY_FOR_PHASE5B")), "")

report = {
    "missing_or_empty_files": missing,
    "integrity_status_counts": integrity["status"].astype(str).value_counts().to_dict() if "status" in integrity.columns else {},
    "cell_set": cell_set.to_dict(orient="records"),
    "eligible_gene_count": int(elig["eligible_for_pilot_deletion"].astype(bool).sum()) if "eligible_for_pilot_deletion" in elig.columns else 0,
    "baseline_cell_count": int(len(baseline)),
    "by_gene_count": int(len(by_gene)),
    "by_cell_gene_count": int(len(by_cell)),
    "top_ranked": ranking.head(10)[["exploratory_rank", "gene_symbol", "mean_delta_P_adverse_like", "n_cells_with_gene"]].to_dict(orient="records") if not ranking.empty else [],
    "ranking_scope_values": sorted(ranking["interpretation_scope"].astype(str).unique().tolist()) if "interpretation_scope" in ranking.columns else [],
    "ranking_target_claim_values": sorted(ranking["not_a_therapeutic_target_claim"].astype(str).unique().tolist()) if "not_a_therapeutic_target_claim" in ranking.columns else [],
    "gse_rows": int(len(gse)),
    "gse_sensitivity_only_values": sorted(gse["sensitivity_only_not_external_validation"].astype(str).unique().tolist()) if "sensitivity_only_not_external_validation" in gse.columns else [],
    "summary_ready_line": ready_line,
}

errors = []
if missing:
    errors.append("missing_or_empty_files")
if (integrity["status"].astype(str).isin(["failed", "missing_or_empty"])).any():
    errors.append("integrity_failed")
if report["eligible_gene_count"] == 0:
    errors.append("no_eligible_genes")
if len(baseline) == 0:
    errors.append("empty_baseline")
if len(by_gene) == 0 or len(by_cell) == 0:
    errors.append("empty_deletion_outputs")
if "hypothesis_generating_only" not in report["ranking_scope_values"]:
    errors.append("ranking_scope_missing")
if "True" not in report["ranking_target_claim_values"]:
    errors.append("target_claim_guard_missing")
if len(gse) == 0:
    errors.append("empty_gse_sensitivity")
if ready_line not in {"READY_FOR_PHASE5B = YES", "READY_FOR_PHASE5B = NO", "READY_FOR_PHASE5B = CONDITIONAL"}:
    errors.append("invalid_ready_line")

print(json.dumps(report, ensure_ascii=False, indent=2))
if errors:
    raise SystemExit("PHASE5A_VERIFICATION_FAILED: " + "; ".join(errors))
print("PHASE5A_VERIFICATION: PASS")
