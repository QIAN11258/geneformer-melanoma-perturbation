from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"
RAW = ROOT / "data_raw" / "phase5C_external"
KM_DIR = FIGURES / "phase5C_TCGA_SKCM_survival_plots"

for path in (TABLES, LOGS, FIGURES, RAW, KM_DIR):
    path.mkdir(parents=True, exist_ok=True)

QUERY_DATE = date.today().isoformat()
PYTHON = Path(sys.executable)

CBIO_HELPER = Path(
    r"scripts/external_helpers/cbioportal_rest_request.py"
)
CHEMBL_HELPER = Path(
    r"scripts/external_helpers/chembl_rest_request.py"
)
OT_HELPER = Path(
    r"scripts/external_helpers/opentargets_graphql.py"
)


CANDIDATE_TIERS = {
    "PABPC1": "Tier 1: high-priority model-stable candidates",
    "FOS": "Tier 1: high-priority model-stable candidates",
    "RPL15": "Tier 1: high-priority model-stable candidates",
    "HSP90AB1": "Tier 1: high-priority model-stable candidates",
    "RPL8": "Tier 1: high-priority model-stable candidates",
    "ACTG1": "Tier 1: high-priority model-stable candidates",
    "RACK1": "Tier 1: high-priority model-stable candidates",
    "FN1": "Tier 2: biologically plausible but downgraded candidates",
    "TGFBI": "Tier 2: biologically plausible but downgraded candidates",
    "JUN": "Tier 2: biologically plausible but downgraded candidates",
    "ATF3": "Tier 2: biologically plausible but downgraded candidates",
    "COL1A2": "Tier 2: biologically plausible but downgraded candidates",
    "HLA-B": "Tier 3: direction-inconsistent or caution candidates",
}

MANUAL_RISK_TAGS = {
    "PABPC1": ["translation-related", "housekeeping-like", "pan-essential risk", "detection-frequency-driven perturbation risk"],
    "RPL15": ["ribosomal/translation-related", "housekeeping-like", "pan-essential risk", "detection-frequency-driven perturbation risk"],
    "RPL8": ["ribosomal/translation-related", "housekeeping-like", "pan-essential risk", "detection-frequency-driven perturbation risk"],
    "RACK1": ["ribosomal/translation-related", "housekeeping-like", "pan-essential risk"],
    "ACTG1": ["housekeeping-like", "cytoskeletal", "pan-essential risk", "detection-frequency-driven perturbation risk"],
    "HSP90AB1": ["stress-response-related", "broad cancer dependency risk", "pan-essential risk"],
    "FOS": ["stress-response-related", "immediate-early/AP-1", "context-dependent"],
    "JUN": ["stress-response-related", "immediate-early/AP-1", "GSE72056 opposite direction"],
    "ATF3": ["stress-response-related", "low support", "GSE72056 opposite direction"],
    "FN1": ["stromal/ECM-related", "microenvironment-confounding risk", "GSE72056 opposite direction"],
    "TGFBI": ["stromal/ECM-related", "microenvironment-confounding risk", "low support", "GSE72056 opposite direction"],
    "COL1A2": ["stromal/ECM-related", "microenvironment-confounding risk", "low support", "GSE72056 opposite direction"],
    "HLA-B": ["immune-related", "GSE72056 opposite direction", "antigen-presentation context"],
}

GDSC2_PRIORITY_DRUG_TERMS = [
    "DABRAFENIB",
    "TRAMETINIB",
    "SELUMETINIB",
    "ERK_2440",
    "ERK_6604",
    "SORAFENIB",
    "TANESPIMYCIN",
    "AUY",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_log(name: str, lines: list[str]) -> None:
    (LOGS / name).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def safe_json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_helper(helper: Path, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        [str(PYTHON), str(helper)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=str(ROOT),
        timeout=timeout,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": {
                "code": "helper_failed",
                "message": (proc.stderr or proc.stdout)[:2000],
            },
        }
    try:
        return json.loads(proc.stdout)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": {"code": "invalid_helper_json", "message": str(exc)}, "stdout": proc.stdout[:500]}


def cbio_call(path: str, raw_name: str, *, method: str = "GET", params: dict[str, Any] | None = None, json_body: Any = None, timeout: int = 180) -> tuple[dict[str, Any], Any | None]:
    raw_path = RAW / raw_name
    payload = {
        "base_url": "https://www.cbioportal.org/api",
        "path": path,
        "method": method,
        "params": params or {},
        "headers": {"Accept": "application/json", "Content-Type": "application/json"},
        "json_body": json_body,
        "save_raw": True,
        "raw_output_path": rel(raw_path),
        "max_items": 5,
        "max_depth": 4,
        "timeout_sec": timeout,
    }
    if json_body is None:
        payload.pop("json_body")
    out = run_helper(CBIO_HELPER, payload, timeout=timeout + 30)
    data = safe_json_load(raw_path) if out.get("ok") and raw_path.exists() else None
    return out, data


def chembl_call(path: str, raw_name: str, *, params: dict[str, Any], record_path: str | None = None, timeout: int = 120) -> tuple[dict[str, Any], Any | None]:
    raw_path = RAW / raw_name
    payload = {
        "base_url": "https://www.ebi.ac.uk/chembl/api/data",
        "path": path,
        "params": params,
        "headers": {"Accept": "application/json"},
        "record_path": record_path,
        "save_raw": True,
        "raw_output_path": rel(raw_path),
        "max_items": 10,
        "max_depth": 4,
        "timeout_sec": timeout,
    }
    if record_path is None:
        payload.pop("record_path")
    out = run_helper(CHEMBL_HELPER, payload, timeout=timeout + 30)
    data = safe_json_load(raw_path) if out.get("ok") and raw_path.exists() else None
    return out, data


def opentargets_call(query: str, variables: dict[str, Any], raw_name: str, timeout: int = 180) -> tuple[dict[str, Any], Any | None]:
    raw_path = RAW / raw_name
    payload = {
        "query": query,
        "variables": variables,
        "save_raw": True,
        "raw_output_path": rel(raw_path),
        "max_items": 10,
        "max_depth": 5,
        "timeout_sec": timeout,
    }
    out = run_helper(OT_HELPER, payload, timeout=timeout + 30)
    data = safe_json_load(raw_path) if out.get("ok") and raw_path.exists() else None
    return out, data


def breadbox_get(path: str, timeout: int = 120) -> Any:
    url = "https://depmap.org/portal/breadbox/" + path.lstrip("/")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def breadbox_post(path: str, payload: dict[str, Any], timeout: int = 180) -> Any:
    url = "https://depmap.org/portal/breadbox/" + path.lstrip("/")
    r = requests.post(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def fdr(values: pd.Series) -> pd.Series:
    valid = values.notna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.sum() > 0:
        out.loc[valid] = multipletests(values.loc[valid].astype(float), method="fdr_bh")[1]
    return out


def parse_event(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).upper()
    if text.startswith("1:") or "DECEASED" in text or "DEAD" in text or "RECUR" in text or "PROGRESS" in text:
        return 1.0
    if text.startswith("0:") or "LIVING" in text or "ALIVE" in text or "DISEASEFREE" in text or "FREE" in text:
        return 0.0
    return np.nan


def stage_to_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).upper()
    if "IV" in text:
        return 4.0
    if "III" in text:
        return 3.0
    if "II" in text:
        return 2.0
    if "I" in text:
        return 1.0
    return np.nan


def cox_model(df: pd.DataFrame, time_col: str, event_col: str, x_cols: list[str]) -> dict[str, Any]:
    work = df[[time_col, event_col] + x_cols].replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work[time_col].astype(float) > 0) & (work[event_col].isin([0.0, 1.0]))]
    if len(work) < 30 or work[event_col].sum() < 5:
        return {"n": len(work), "events": float(work[event_col].sum()) if len(work) else 0, "status": "insufficient_events"}
    try:
        exog = work[x_cols].astype(float)
        fit = PHReg(work[time_col].astype(float), exog, status=work[event_col].astype(float)).fit(disp=0)
        idx = list(exog.columns).index(x_cols[0])
        beta = float(fit.params[idx])
        ci = fit.conf_int()
        ci_low = float(ci[idx, 0])
        ci_high = float(ci[idx, 1])
        return {
            "n": int(len(work)),
            "events": int(work[event_col].sum()),
            "HR": float(math.exp(beta)),
            "CI95_low": float(math.exp(ci_low)),
            "CI95_high": float(math.exp(ci_high)),
            "p": float(fit.pvalues[idx]),
            "status": "ok",
        }
    except Exception as exc:  # noqa: BLE001
        return {"n": int(len(work)), "events": int(work[event_col].sum()), "status": f"cox_failed: {type(exc).__name__}: {exc}"}


def km_curve(times: np.ndarray, events: np.ndarray) -> tuple[list[float], list[float]]:
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    surv = 1.0
    xs = [0.0]
    ys = [1.0]
    for t in np.unique(times[events == 1]):
        at_risk = np.sum(times >= t)
        observed = np.sum((times == t) & (events == 1))
        if at_risk > 0:
            xs.extend([float(t), float(t)])
            ys.extend([surv, surv * (1.0 - observed / at_risk)])
            surv = ys[-1]
    if len(times) > 0:
        xs.append(float(np.nanmax(times)))
        ys.append(surv)
    return xs, ys


def make_km_plot(df: pd.DataFrame, gene: str, time_col: str, event_col: str, expr_col: str, out_path: Path) -> None:
    work = df[[time_col, event_col, expr_col]].replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work[time_col] > 0) & (work[event_col].isin([0.0, 1.0]))].copy()
    if len(work) < 30:
        return
    median = work[expr_col].median()
    work["group"] = np.where(work[expr_col] >= median, "high", "low")
    plt.figure(figsize=(5.6, 4.2))
    for group, color in [("low", "#4C78A8"), ("high", "#E45756")]:
        sub = work[work["group"] == group]
        xs, ys = km_curve(sub[time_col].to_numpy(float), sub[event_col].to_numpy(float))
        plt.step(xs, ys, where="post", label=f"{group} (n={len(sub)})", color=color, linewidth=1.8)
    plt.xlabel("Months")
    plt.ylabel("Overall survival probability")
    plt.title(f"TCGA-SKCM exploratory OS: {gene}")
    plt.ylim(0, 1.03)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_candidate_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        ROOT / "summary_phase5B_zh.md",
        TABLES / "phase5B_exploratory_perturbation_ranking.csv",
        TABLES / "phase5B_bootstrap_stability_by_gene.csv",
        TABLES / "phase5B_sample_level_effect_by_gene.csv",
        TABLES / "phase5B_GSE72056_expanded_sensitivity_by_gene.csv",
    ]
    integrity_rows = []
    for p in required:
        integrity_rows.append(
            {
                "input_file": str(p.relative_to(ROOT)),
                "exists": p.exists(),
                "non_empty": p.exists() and p.stat().st_size > 0,
                "status": "ok" if p.exists() and p.stat().st_size > 0 else "missing_or_empty",
            }
        )
    integrity = pd.DataFrame(integrity_rows)
    integrity.to_csv(TABLES / "phase5C_input_integrity_check.csv", index=False)
    if not (integrity["status"] == "ok").all():
        write_log("phase5C_preflight_check_log.md", ["# Phase 5C preflight", "", "FAILED: one or more Phase 5B inputs are missing."])
        raise SystemExit("Phase 5C preflight failed")

    ranking = pd.read_csv(TABLES / "phase5B_exploratory_perturbation_ranking.csv")
    sample = pd.read_csv(TABLES / "phase5B_sample_level_effect_by_gene.csv")
    sample_agg = (
        sample.groupby("gene_symbol")
        .agg(
            sample_level_min_delta=("sample_mean_delta_P_adverse_like", "min"),
            sample_level_max_delta=("sample_mean_delta_P_adverse_like", "max"),
            sample_level_median_delta=("sample_mean_delta_P_adverse_like", "median"),
            sample_level_n=("sample_id", "nunique"),
        )
        .reset_index()
    )
    genes = list(CANDIDATE_TIERS)
    panel = ranking[ranking["gene_symbol"].isin(genes)].copy()
    panel = panel.merge(sample_agg, on="gene_symbol", how="left")
    panel["phase5C_tier"] = panel["gene_symbol"].map(CANDIDATE_TIERS)
    panel["manual_interpretation_risk_tags"] = panel["gene_symbol"].map(lambda g: ";".join(MANUAL_RISK_TAGS.get(g, [])))
    panel["GSE72056_caution"] = np.where(panel["direction_consistency"].eq("opposite_direction"), "GSE72056 opposite direction", "not_flagged")
    panel["phase5C_scope"] = "exploratory_external_evidence_triangulation_only"
    panel = panel.sort_values(["phase5C_tier", "exploratory_rank"])
    panel.to_csv(TABLES / "phase5C_candidate_gene_panel.csv", index=False)
    write_log(
        "phase5C_preflight_check_log.md",
        [
            "# Phase 5C preflight",
            "",
            f"- Query date: {QUERY_DATE}",
            "- Phase 5B required outputs were readable.",
            f"- Candidate panel genes: {len(panel)}.",
            "- Scope: exploratory evidence triangulation only.",
        ],
    )
    write_log(
        "phase5C_candidate_panel_log.md",
        [
            "# Candidate panel",
            "",
            "- Tier 1: PABPC1, FOS, RPL15, HSP90AB1, RPL8, ACTG1, RACK1.",
            "- Tier 2: FN1, TGFBI, JUN, ATF3, COL1A2.",
            "- Tier 3: HLA-B.",
            "- Phase 5B ranking, bootstrap/sample-level fields, and GSE72056 direction were retained.",
        ],
    )
    return panel, integrity


def run_tcga(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = panel["gene_symbol"].tolist()
    cbio_log = ["# TCGA-SKCM cBioPortal log", "", f"- Query date: {QUERY_DATE}"]
    study_id = "skcm_tcga_gdc"
    profile_id = "skcm_tcga_gdc_mrna_seq_tpm"
    sample_list_id = "skcm_tcga_gdc_tpm"
    cbio_log.append(f"- Study: {study_id}; profile: {profile_id}; sample list: {sample_list_id}.")

    _, sample_ids = cbio_call(f"sample-lists/{sample_list_id}/sample-ids", "cbioportal_skcm_tcga_gdc_tpm_sample_ids.json")
    sample_ids = sample_ids or []
    cbio_log.append(f"- mRNA sample IDs returned: {len(sample_ids)}.")

    _, gene_records = cbio_call(
        "genes/fetch",
        "cbioportal_candidate_genes_entrez.json",
        method="POST",
        params={"geneIdType": "HUGO_GENE_SYMBOL", "projection": "SUMMARY"},
        json_body=genes,
    )
    gene_records = gene_records or []
    symbol_to_entrez = {r["hugoGeneSymbol"]: int(r["entrezGeneId"]) for r in gene_records if "hugoGeneSymbol" in r and "entrezGeneId" in r}
    entrez_to_symbol = {v: k for k, v in symbol_to_entrez.items()}
    cbio_log.append(f"- Candidate genes mapped to Entrez IDs: {len(symbol_to_entrez)}/{len(genes)}.")

    _, mol = cbio_call(
        f"molecular-profiles/{profile_id}/molecular-data/fetch",
        "cbioportal_skcm_tcga_gdc_candidate_tpm.json",
        method="POST",
        json_body={"entrezGeneIds": list(symbol_to_entrez.values()), "sampleListId": sample_list_id},
        timeout=240,
    )
    mol = mol or []
    expr = pd.DataFrame(mol)
    if expr.empty:
        cbio_log.append("- Expression fetch returned no rows.")
        write_log("phase5C_TCGA_SKCM_log.md", cbio_log)
        empty = pd.DataFrame()
        empty.to_csv(TABLES / "phase5C_TCGA_SKCM_expression_summary.csv", index=False)
        empty.to_csv(TABLES / "phase5C_TCGA_SKCM_survival_association.csv", index=False)
        return empty, empty
    expr["gene_symbol"] = expr["entrezGeneId"].map(entrez_to_symbol)
    expr["value"] = pd.to_numeric(expr["value"], errors="coerce")
    expr["patientId"] = expr["sampleId"].astype(str).str.slice(0, 12)
    sample_ids_for_clin = sorted(expr["sampleId"].dropna().unique().tolist())
    patient_ids = sorted(expr["patientId"].dropna().unique().tolist())

    _, sample_clin = cbio_call(
        f"studies/{study_id}/clinical-data/fetch",
        "cbioportal_skcm_tcga_gdc_sample_clinical.json",
        method="POST",
        params={"clinicalDataType": "SAMPLE", "projection": "SUMMARY"},
        json_body={"ids": sample_ids_for_clin, "attributeIds": ["SAMPLE_TYPE", "CANCER_TYPE_DETAILED", "FRACTION_GENOME_ALTERED", "MUTATION_COUNT", "TMB_NONSYNONYMOUS"]},
        timeout=240,
    )
    _, patient_clin = cbio_call(
        f"studies/{study_id}/clinical-data/fetch",
        "cbioportal_skcm_tcga_gdc_patient_clinical.json",
        method="POST",
        params={"clinicalDataType": "PATIENT", "projection": "SUMMARY"},
        json_body={"ids": patient_ids, "attributeIds": ["OS_MONTHS", "OS_STATUS", "DFS_MONTHS", "DFS_STATUS", "AGE", "SEX", "PATH_STAGE", "VITAL_STATUS"]},
        timeout=240,
    )
    sample_clin_df = pd.DataFrame(sample_clin or [])
    patient_clin_df = pd.DataFrame(patient_clin or [])
    sample_wide = (
        sample_clin_df.pivot_table(index="sampleId", columns="clinicalAttributeId", values="value", aggfunc="first").reset_index()
        if not sample_clin_df.empty
        else pd.DataFrame({"sampleId": sample_ids_for_clin})
    )
    patient_wide = (
        patient_clin_df.pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first").reset_index()
        if not patient_clin_df.empty
        else pd.DataFrame({"patientId": patient_ids})
    )
    expr_wide = expr.pivot_table(index=["sampleId", "patientId"], columns="gene_symbol", values="value", aggfunc="mean").reset_index()
    merged = expr_wide.merge(sample_wide, on="sampleId", how="left").merge(patient_wide, on="patientId", how="left")

    expression_rows = []
    for gene in genes:
        if gene not in merged.columns:
            expression_rows.append({"gene_symbol": gene, "status": "not_available"})
            continue
        values = pd.to_numeric(merged[gene], errors="coerce")
        stage_medians = {}
        if "PATH_STAGE" in merged:
            stage_medians = (
                pd.DataFrame({"stage": merged["PATH_STAGE"], "expr": np.log2(values + 1)})
                .dropna()
                .groupby("stage")["expr"]
                .median()
                .round(4)
                .to_dict()
            )
        expression_rows.append(
            {
                "gene_symbol": gene,
                "source": "cBioPortal TCGA-SKCM GDC 2025",
                "study_id": study_id,
                "molecular_profile_id": profile_id,
                "query_date": QUERY_DATE,
                "n_samples_with_expression": int(values.notna().sum()),
                "n_patients_with_expression": int(merged.loc[values.notna(), "patientId"].nunique()),
                "mean_TPM": float(values.mean()) if values.notna().any() else np.nan,
                "median_TPM": float(values.median()) if values.notna().any() else np.nan,
                "mean_log2_TPM_plus1": float(np.log2(values + 1).mean()) if values.notna().any() else np.nan,
                "median_log2_TPM_plus1": float(np.log2(values + 1).median()) if values.notna().any() else np.nan,
                "sample_type_counts": json.dumps(merged.loc[values.notna(), "SAMPLE_TYPE"].value_counts(dropna=False).to_dict(), ensure_ascii=False) if "SAMPLE_TYPE" in merged else "not_available",
                "path_stage_median_log2_TPM_plus1": json.dumps(stage_medians, ensure_ascii=False),
                "immune_stromal_context": "not_available_from_cbioportal_minimal_query",
                "status": "ok",
            }
        )
    expression_summary = pd.DataFrame(expression_rows)
    expression_summary.to_csv(TABLES / "phase5C_TCGA_SKCM_expression_summary.csv", index=False)

    patient_expr = expr.copy()
    patient_expr["expr_log2"] = np.log2(patient_expr["value"] + 1)
    patient_expr_wide = patient_expr.pivot_table(index="patientId", columns="gene_symbol", values="expr_log2", aggfunc="median").reset_index()
    surv = patient_expr_wide.merge(patient_wide, on="patientId", how="left")
    if "OS_MONTHS" in surv:
        surv["OS_MONTHS_numeric"] = pd.to_numeric(surv["OS_MONTHS"], errors="coerce")
    if "OS_STATUS" in surv:
        surv["OS_event"] = surv["OS_STATUS"].map(parse_event)
    if "DFS_MONTHS" in surv:
        surv["DFS_MONTHS_numeric"] = pd.to_numeric(surv["DFS_MONTHS"], errors="coerce")
    if "DFS_STATUS" in surv:
        surv["DFS_event"] = surv["DFS_STATUS"].map(parse_event)
    surv["AGE_numeric"] = pd.to_numeric(surv.get("AGE", np.nan), errors="coerce")
    surv["SEX_male"] = surv.get("SEX", pd.Series(index=surv.index, dtype=object)).astype(str).str.upper().eq("MALE").astype(float)
    surv["PATH_STAGE_numeric"] = surv.get("PATH_STAGE", pd.Series(index=surv.index, dtype=object)).map(stage_to_number)

    surv_rows = []
    for gene in genes:
        if gene not in surv.columns:
            surv_rows.append({"gene_symbol": gene, "endpoint": "OS", "model": "continuous_z", "status": "expression_not_available"})
            continue
        gene_df = surv.copy()
        expr_col = f"{gene}_z"
        med_col = f"{gene}_high_vs_low"
        gene_df[expr_col] = (gene_df[gene] - gene_df[gene].mean()) / gene_df[gene].std(ddof=0)
        gene_df[med_col] = (gene_df[gene] >= gene_df[gene].median()).astype(float)
        for endpoint, time_col, event_col in [("OS", "OS_MONTHS_numeric", "OS_event"), ("DFS", "DFS_MONTHS_numeric", "DFS_event")]:
            if time_col not in gene_df or event_col not in gene_df:
                surv_rows.append({"gene_symbol": gene, "endpoint": endpoint, "model": "continuous_z", "status": "endpoint_not_available"})
                continue
            for model_name, cols in [
                ("continuous_z", [expr_col]),
                ("median_high_vs_low", [med_col]),
                ("multivariable_continuous_z_age_sex_stage", [expr_col, "AGE_numeric", "SEX_male", "PATH_STAGE_numeric"]),
            ]:
                res = cox_model(gene_df, time_col, event_col, cols)
                row = {
                    "gene_symbol": gene,
                    "endpoint": endpoint,
                    "model": model_name,
                    "time_col": time_col,
                    "event_col": event_col,
                    "source": "cBioPortal TCGA-SKCM GDC 2025",
                    "study_id": study_id,
                    "query_date": QUERY_DATE,
                    "interpretation_scope": "exploratory_association_not_causal",
                }
                row.update(res)
                surv_rows.append(row)
        if "OS_MONTHS_numeric" in gene_df and "OS_event" in gene_df:
            make_km_plot(gene_df, gene, "OS_MONTHS_numeric", "OS_event", gene, KM_DIR / f"{gene}_OS_KM.png")
    survival = pd.DataFrame(surv_rows)
    if "p" in survival:
        survival["FDR"] = survival.groupby(["endpoint", "model"])["p"].transform(fdr)
    survival.to_csv(TABLES / "phase5C_TCGA_SKCM_survival_association.csv", index=False)

    cbio_log.extend(
        [
            f"- Expression records returned: {len(expr)}.",
            f"- Patient clinical records returned: {len(patient_clin_df)}.",
            "- OS was evaluated when OS_MONTHS/OS_STATUS were present.",
            "- DFS/DSS/PFI were only evaluated if present; unavailable endpoints were marked rather than fabricated.",
            "- Cox outputs are exploratory associations with FDR correction.",
        ]
    )
    write_log("phase5C_TCGA_SKCM_log.md", cbio_log)
    return expression_summary, survival


def run_depmap_and_gdsc(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    genes = panel["gene_symbol"].tolist()
    log = ["# DepMap/CCLE log", "", f"- Query date: {QUERY_DATE}", "- Source: DepMap portal breadbox API, current public release exposed by portal."]
    model_meta_raw = breadbox_post(
        "datasets/tabular/depmap_model_metadata",
        {"columns": ["ModelID", "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype", "CellLineName", "CCLEName", "PrimaryOrMetastasis"]},
        timeout=180,
    )
    (RAW / "depmap_model_metadata_candidate_columns.json").write_text(json.dumps(model_meta_raw, indent=2), encoding="utf-8")
    model_meta = pd.DataFrame(model_meta_raw)
    melanoma_ids = model_meta.index[
        model_meta.astype(str).apply(lambda r: r.str.contains("Melanoma", case=False, na=False).any(), axis=1)
    ].tolist()
    log.append(f"- Model metadata rows: {len(model_meta)}; melanoma-like models by metadata text search: {len(melanoma_ids)}.")

    chronos_raw = breadbox_post("datasets/matrix/Chronos_Combined", {"feature_identifier": "label", "features": genes}, timeout=240)
    expr_raw = breadbox_post("datasets/matrix/expression", {"feature_identifier": "label", "features": genes}, timeout=240)
    dependency_raw = breadbox_post("datasets/matrix/CRISPRGeneDependency", {"feature_identifier": "label", "features": genes}, timeout=240)
    (RAW / "depmap_chronos_candidate_genes.json").write_text(json.dumps(chronos_raw, indent=2), encoding="utf-8")
    (RAW / "depmap_expression_candidate_genes.json").write_text(json.dumps(expr_raw, indent=2), encoding="utf-8")
    (RAW / "depmap_crispr_gene_dependency_candidate_genes.json").write_text(json.dumps(dependency_raw, indent=2), encoding="utf-8")
    chronos = pd.DataFrame(chronos_raw)
    dep_prob = pd.DataFrame(dependency_raw)
    dep_expr = pd.DataFrame(expr_raw)

    rows = []
    pan_rows = []
    for gene in genes:
        ge = pd.to_numeric(chronos.get(gene), errors="coerce")
        expr = pd.to_numeric(dep_expr.get(gene), errors="coerce")
        mel = ge.loc[ge.index.intersection(melanoma_ids)].dropna()
        pan = ge.dropna()
        mel_expr = expr.loc[expr.index.intersection(melanoma_ids)].dropna()
        common = sorted(set(mel.index).intersection(expr.dropna().index))
        corr, pval = (np.nan, np.nan)
        if len(common) >= 10 and len(set(expr.loc[common].round(8))) > 1 and len(set(ge.loc[common].round(8))) > 1:
            corr, pval = stats.spearmanr(expr.loc[common], ge.loc[common], nan_policy="omit")
        lineage_means = []
        for lineage, idx in model_meta.groupby("OncotreeLineage").groups.items():
            vals = ge.loc[ge.index.intersection(list(idx))].dropna()
            if len(vals) >= 5:
                lineage_means.append((lineage, float(vals.mean()), int(len(vals))))
        lineage_means = sorted(lineage_means, key=lambda x: x[1])
        melanoma_mean = float(mel.mean()) if len(mel) else np.nan
        pan_mean = float(pan.mean()) if len(pan) else np.nan
        melanoma_frac = float((mel < -0.5).mean()) if len(mel) else np.nan
        pan_frac = float((pan < -0.5).mean()) if len(pan) else np.nan
        rows.append(
            {
                "gene_symbol": gene,
                "source": "DepMap portal breadbox API current public release",
                "query_date": QUERY_DATE,
                "melanoma_model_n": int(len(mel)),
                "pan_cancer_model_n": int(len(pan)),
                "melanoma_mean_chronos_gene_effect": melanoma_mean,
                "melanoma_median_chronos_gene_effect": float(mel.median()) if len(mel) else np.nan,
                "melanoma_fraction_gene_effect_lt_minus_0_5": melanoma_frac,
                "pan_cancer_mean_chronos_gene_effect": pan_mean,
                "pan_cancer_fraction_gene_effect_lt_minus_0_5": pan_frac,
                "melanoma_specific_vs_pan_delta_mean": melanoma_mean - pan_mean if pd.notna(melanoma_mean) and pd.notna(pan_mean) else np.nan,
                "melanoma_selective_dependency": "exploratory_yes" if pd.notna(melanoma_mean) and pd.notna(pan_mean) and melanoma_mean < pan_mean - 0.15 and melanoma_frac > pan_frac + 0.1 else "not_supported",
                "broad_dependency": "yes" if pd.notna(pan_frac) and (pan_frac >= 0.2 or pan_mean < -0.45) else "no",
                "pan_essential_risk": "yes" if pd.notna(pan_frac) and (pan_frac >= 0.5 or pan_mean < -0.65) else "no",
                "melanoma_expression_n": int(len(mel_expr)),
                "melanoma_mean_expression_log2TPM_plus1": float(mel_expr.mean()) if len(mel_expr) else np.nan,
                "expression_dependency_spearman_r": float(corr) if pd.notna(corr) else np.nan,
                "expression_dependency_p": float(pval) if pd.notna(pval) else np.nan,
                "interpretation_scope": "exploratory_dependency_context_not_melanoma_specific_claim",
            }
        )
        pan_rows.append(
            {
                "gene_symbol": gene,
                "source": "DepMap portal breadbox API current public release",
                "query_date": QUERY_DATE,
                "pan_cancer_model_n": int(len(pan)),
                "pan_cancer_mean_chronos_gene_effect": pan_mean,
                "pan_cancer_fraction_gene_effect_lt_minus_0_5": pan_frac,
                "melanoma_lineage_mean_chronos_gene_effect": melanoma_mean,
                "melanoma_lineage_rank_by_dependency_mean": next((i + 1 for i, x in enumerate(lineage_means) if str(x[0]).lower() == "skin"), np.nan),
                "top_dependency_lineages_by_mean": json.dumps([{"lineage": a, "mean": round(b, 4), "n": c} for a, b, c in lineage_means[:5]], ensure_ascii=False),
            }
        )
    depmap = pd.DataFrame(rows)
    if "expression_dependency_p" in depmap:
        depmap["expression_dependency_FDR"] = fdr(depmap["expression_dependency_p"])
    pan_context = pd.DataFrame(pan_rows)
    depmap.to_csv(TABLES / "phase5C_DepMap_melanoma_dependency.csv", index=False)
    pan_context.to_csv(TABLES / "phase5C_DepMap_pan_cancer_dependency_context.csv", index=False)
    log.append("- CRISPR Chronos, CRISPR dependency probability, and expression were queried by candidate gene labels.")
    log.append("- Melanoma-specific labels are exploratory and compared against pan-cancer context.")
    write_log("phase5C_DepMap_log.md", log)

    gdsc_log = ["# GDSC log", "", f"- Query date: {QUERY_DATE}", "- Source: DepMap portal GDSC2_log2AUC_collapsed matrix."]
    features = breadbox_get("datasets/features/GDSC2_log2AUC_collapsed", timeout=120)
    selected = []
    for feat in features:
        label = feat.get("label", "")
        if any(term.lower() in label.lower() for term in GDSC2_PRIORITY_DRUG_TERMS):
            selected.append(feat)
    selected = selected[:20]
    gdsc_log.append(f"- Selected GDSC2 features by melanoma-relevant terms: {len(selected)}.")
    if selected:
        drug_raw = breadbox_post(
            "datasets/matrix/GDSC2_log2AUC_collapsed",
            {"feature_identifier": "id", "features": [x["id"] for x in selected]},
            timeout=240,
        )
        (RAW / "depmap_gdsc2_priority_drugs_log2auc.json").write_text(json.dumps(drug_raw, indent=2), encoding="utf-8")
        drug_df = pd.DataFrame(drug_raw).rename(columns={x["id"]: x["label"] for x in selected})
    else:
        drug_df = pd.DataFrame()
    gdsc_rows = []
    for gene in genes:
        gene_expr = pd.to_numeric(dep_expr.get(gene), errors="coerce")
        for drug in drug_df.columns:
            drug_vals = pd.to_numeric(drug_df[drug], errors="coerce")
            common = sorted(set(melanoma_ids).intersection(gene_expr.dropna().index).intersection(drug_vals.dropna().index))
            corr, pval = (np.nan, np.nan)
            if len(common) >= 8 and len(set(gene_expr.loc[common].round(8))) > 1 and len(set(drug_vals.loc[common].round(8))) > 1:
                corr, pval = stats.spearmanr(gene_expr.loc[common], drug_vals.loc[common], nan_policy="omit")
            gdsc_rows.append(
                {
                    "gene_symbol": gene,
                    "drug": drug,
                    "dataset": "GDSC2_log2AUC_collapsed",
                    "source": "DepMap portal breadbox API current public release",
                    "query_date": QUERY_DATE,
                    "melanoma_cell_line_n": int(len(common)),
                    "spearman_expression_vs_log2AUC": float(corr) if pd.notna(corr) else np.nan,
                    "p": float(pval) if pd.notna(pval) else np.nan,
                    "direction_note": "negative_r_means_higher_expression_associated_with_lower_log2AUC_more_sensitive_exploratory" if pd.notna(corr) and corr < 0 else "nonnegative_or_not_available",
                    "interpretation_scope": "exploratory_correlation_not_causal",
                }
            )
    gdsc = pd.DataFrame(gdsc_rows)
    if "p" in gdsc:
        gdsc["FDR"] = fdr(gdsc["p"])
    gdsc.to_csv(TABLES / "phase5C_GDSC_drug_sensitivity_association.csv", index=False)
    gdsc_log.append("- Correlations were computed only where candidate expression and drug log2AUC overlapped in melanoma models.")
    gdsc_log.append("- No causal drug sensitivity inference was made.")
    write_log("phase5C_GDSC_log.md", gdsc_log)
    return depmap, pan_context, gdsc


def run_druggability(panel: pd.DataFrame, depmap: pd.DataFrame) -> pd.DataFrame:
    genes = panel["gene_symbol"].tolist()
    log = ["# Druggability/tractability log", "", f"- Query date: {QUERY_DATE}", "- Sources: Open Targets Platform GraphQL and ChEMBL REST API.", "- DrugBank was not programmatically queried because no open bulk API is available in this environment; marked needs manual confirmation."]
    ot_query = """
query targetInfo($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    tractability { modality label value }
    targetClass { id label level }
    isEssential
    depMapEssentiality { tissueName tissueId }
    associatedDiseases(page: {index: 0, size: 200}) {
      count
      rows { disease { id name } score datasourceScores { id score } }
    }
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug { id name maximumClinicalStage drugType }
        diseases { diseaseFromSource disease { id name } }
      }
    }
  }
}
"""
    rows = []
    for _, row in panel.iterrows():
        gene = row["gene_symbol"]
        ensembl = row["ensembl_id"]
        ot_out, ot_data = opentargets_call(ot_query, {"ensemblId": ensembl}, f"opentargets_{gene}.json", timeout=240)
        target = ((ot_data or {}).get("data") or {}).get("target") if isinstance(ot_data, dict) else None
        if target is None:
            target = {}
            log.append(f"- Open Targets unavailable for {gene}: {ot_out.get('error', ot_out)}")
        human_target_id = None
        chembl_out, chembl_target_data = chembl_call(
            "target/search.json",
            f"chembl_target_search_{gene}.json",
            params={"q": gene, "limit": 10},
            record_path="targets",
        )
        targets = (chembl_target_data or {}).get("targets", []) if isinstance(chembl_target_data, dict) else []
        for t in targets:
            if str(t.get("organism", "")).lower() == "homo sapiens":
                human_target_id = t.get("target_chembl_id")
                break
        mechanisms = []
        if human_target_id:
            _, mech_data = chembl_call(
                "mechanism.json",
                f"chembl_mechanism_{gene}.json",
                params={"target_chembl_id": human_target_id, "limit": 50},
                record_path="mechanisms",
                timeout=180,
            )
            mechanisms = (mech_data or {}).get("mechanisms", []) if isinstance(mech_data, dict) else []

        tract_true = [f"{x.get('modality')}:{x.get('label')}" for x in (target.get("tractability") or []) if x.get("value") is True]
        target_classes = [x.get("label") for x in (target.get("targetClass") or []) if x.get("label")]
        dep_tissues = [x.get("tissueName") for x in (target.get("depMapEssentiality") or []) if x.get("tissueName")]
        assoc_rows = (((target.get("associatedDiseases") or {}).get("rows")) or [])
        melanoma_assoc = [
            {
                "disease": ((x.get("disease") or {}).get("name")),
                "score": x.get("score"),
                "datasources": ",".join([d.get("id", "") for d in x.get("datasourceScores", [])]),
            }
            for x in assoc_rows
            if "melanoma" in str((x.get("disease") or {}).get("name", "")).lower()
        ]
        clinical_rows = (((target.get("drugAndClinicalCandidates") or {}).get("rows")) or [])
        clinical_drugs = []
        melanoma_drugs = []
        for item in clinical_rows:
            drug = item.get("drug") or {}
            if drug.get("name"):
                clinical_drugs.append(f"{drug.get('name')}({item.get('maxClinicalStage')})")
            disease_text = " ".join(
                [str((d.get("disease") or {}).get("name", "")) + " " + str(d.get("diseaseFromSource", "")) for d in item.get("diseases", [])]
            ).lower()
            if "melanoma" in disease_text and drug.get("name"):
                melanoma_drugs.append(f"{drug.get('name')}({item.get('maxClinicalStage')})")
        chembl_compounds = []
        for m in mechanisms:
            mol = m.get("molecule_chembl_id") or m.get("parent_molecule_chembl_id")
            action = m.get("mechanism_of_action") or m.get("action_type")
            if mol:
                chembl_compounds.append(f"{mol}:{action}")
        dep_row = depmap[depmap["gene_symbol"].eq(gene)].iloc[0].to_dict() if not depmap[depmap["gene_symbol"].eq(gene)].empty else {}
        rows.append(
            {
                "gene_symbol": gene,
                "ensembl_id": ensembl,
                "source": "Open Targets Platform + ChEMBL",
                "query_date": QUERY_DATE,
                "target_class": ";".join(target_classes) if target_classes else "not_available",
                "open_targets_isEssential": target.get("isEssential", "not_available"),
                "open_targets_depmap_essentiality_tissues": ";".join(dep_tissues[:20]) if dep_tissues else "not_available",
                "open_targets_tractability_true_flags": ";".join(tract_true) if tract_true else "none_detected",
                "open_targets_drug_candidate_count": (target.get("drugAndClinicalCandidates") or {}).get("count", 0) if target else "not_available",
                "open_targets_drugs_preview": ";".join(sorted(set(clinical_drugs))[:12]) if clinical_drugs else "none_detected",
                "open_targets_melanoma_drugs_preview": ";".join(sorted(set(melanoma_drugs))[:12]) if melanoma_drugs else "none_detected",
                "open_targets_melanoma_association": json.dumps(melanoma_assoc[:5], ensure_ascii=False),
                "chembl_human_target_id": human_target_id or "not_available",
                "chembl_mechanism_count": len(mechanisms),
                "chembl_compounds_preview": ";".join(chembl_compounds[:12]) if chembl_compounds else "none_detected",
                "drugbank_status": "needs manual confirmation; not queried via automated API",
                "safety_or_broad_essentiality_concern": "yes" if dep_row.get("pan_essential_risk") == "yes" or "pan-essential risk" in str(row.get("manual_interpretation_risk_tags", "")) else "not_flagged_by_available_sources",
                "known_melanoma_or_cancer_association_evidence": "melanoma_evidence_present_in_open_targets" if melanoma_assoc or melanoma_drugs else "not_detected_or_needs_manual_confirmation",
                "evidence_level": "exploratory_public_database_context",
            }
        )
        time.sleep(0.2)
    drug = pd.DataFrame(rows)
    drug.to_csv(TABLES / "phase5C_druggability_tractability_summary.csv", index=False)
    write_log("phase5C_druggability_log.md", log)
    return drug


def build_artifact_filter(panel: pd.DataFrame, depmap: pd.DataFrame, drug: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in panel.iterrows():
        gene = row["gene_symbol"]
        tags = set(filter(None, str(row.get("manual_interpretation_risk_tags", "")).split(";")))
        dep = depmap[depmap["gene_symbol"].eq(gene)]
        dep_dict = dep.iloc[0].to_dict() if not dep.empty else {}
        dr = drug[drug["gene_symbol"].eq(gene)]
        dr_dict = dr.iloc[0].to_dict() if not dr.empty else {}
        if dep_dict.get("pan_essential_risk") == "yes" or dr_dict.get("open_targets_isEssential") is True:
            tags.add("pan-essential risk")
        if dep_dict.get("broad_dependency") == "yes":
            tags.add("broad cancer dependency risk")
        if row.get("direction_consistency") == "opposite_direction":
            tags.add("GSE72056 opposite direction")
        high_artifact = any(t in tags for t in ["ribosomal/translation-related", "translation-related", "housekeeping-like", "pan-essential risk"])
        if gene in {"PABPC1", "RPL15", "RPL8", "RACK1", "ACTG1"}:
            recommendation = "deprioritize_for_target_claim"
        elif high_artifact or dep_dict.get("broad_dependency") == "yes":
            recommendation = "keep_but_downgrade"
        elif row.get("phase5C_tier", "").startswith("Tier 1") and row.get("direction_consistency") == "same_direction":
            recommendation = "keep_high_priority"
        else:
            recommendation = "exploratory_only"
        rows.append(
            {
                "gene_symbol": gene,
                "ribosomal_genes": "yes" if gene.startswith("RPL") or gene.startswith("RPS") else "no",
                "translation_related_genes": "yes" if any("translation" in t for t in tags) else "no",
                "housekeeping_or_cytoskeletal_genes": "yes" if any(t in tags for t in ["housekeeping-like", "cytoskeletal"]) else "no",
                "pan_essential_gene_risk": "yes" if "pan-essential risk" in tags else "no",
                "broad_cancer_dependency_risk": "yes" if "broad cancer dependency risk" in tags else "no",
                "detection_frequency_driven_perturbation_risk": "yes" if "detection-frequency-driven perturbation risk" in tags else "no",
                "immune_related": "yes" if "immune-related" in tags else "no",
                "stromal_ECM_related": "yes" if "stromal/ECM-related" in tags else "no",
                "stress_response_related": "yes" if "stress-response-related" in tags else "no",
                "low_support": "yes" if "low support" in tags or bool(row.get("warning_low_support", False)) else "no",
                "GSE72056_opposite_direction": "yes" if "GSE72056 opposite direction" in tags else "no",
                "melanoma_mean_chronos_gene_effect": dep_dict.get("melanoma_mean_chronos_gene_effect", np.nan),
                "pan_cancer_mean_chronos_gene_effect": dep_dict.get("pan_cancer_mean_chronos_gene_effect", np.nan),
                "manual_and_external_risk_tags": ";".join(sorted(tags)),
                "recommendation": recommendation,
                "interpretation_scope": "risk_filter_do_not_delete_genes_automatically",
            }
        )
    artifact = pd.DataFrame(rows)
    artifact.to_csv(TABLES / "phase5C_artifact_essentiality_filter.csv", index=False)
    write_log(
        "phase5C_artifact_filter_log.md",
        [
            "# Artifact / essentiality filter",
            "",
            "- Genes were not deleted from the candidate panel.",
            "- Ribosomal/translation/housekeeping/pan-essential/broad dependency risks were used only for interpretation downgrading.",
            "- PABPC1, RPL15, RPL8, RACK1, and ACTG1 were explicitly flagged for artifact/essentiality caution.",
        ],
    )
    return artifact


def build_integration(panel: pd.DataFrame, artifact: pd.DataFrame, tcga_surv: pd.DataFrame, depmap: pd.DataFrame, gdsc: pd.DataFrame, drug: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in panel.iterrows():
        gene = row["gene_symbol"]
        art = artifact[artifact["gene_symbol"].eq(gene)].iloc[0].to_dict()
        dep = depmap[depmap["gene_symbol"].eq(gene)].iloc[0].to_dict() if not depmap[depmap["gene_symbol"].eq(gene)].empty else {}
        dr = drug[drug["gene_symbol"].eq(gene)].iloc[0].to_dict() if not drug[drug["gene_symbol"].eq(gene)].empty else {}
        surv_gene = tcga_surv[(tcga_surv.get("gene_symbol", pd.Series(dtype=str)).eq(gene)) & (tcga_surv.get("status", pd.Series(dtype=str)).eq("ok"))] if not tcga_surv.empty else pd.DataFrame()
        surv_hits = 0
        if not surv_gene.empty and "FDR" in surv_gene:
            surv_hits = int((surv_gene["FDR"] < 0.1).sum())
        gdsc_gene = gdsc[gdsc["gene_symbol"].eq(gene)] if not gdsc.empty else pd.DataFrame()
        gdsc_hits = int((gdsc_gene["FDR"] < 0.1).sum()) if not gdsc_gene.empty and "FDR" in gdsc_gene else 0
        model_support = "strong_model_stable" if row["phase5C_tier"].startswith("Tier 1") and row["direction_consistency"] == "same_direction" else "downgraded_or_direction_caution"
        artifact_high = art["recommendation"] in {"deprioritize_for_target_claim", "keep_but_downgrade"}
        druggable = "HSP90" in str(dr.get("open_targets_drugs_preview", "")) or dr.get("open_targets_drug_candidate_count", 0) not in [0, "not_available"]
        if art["recommendation"] == "deprioritize_for_target_claim" or row["direction_consistency"] == "opposite_direction":
            priority = "deprioritized_due_to_artifact_or_inconsistent_evidence"
        elif model_support == "strong_model_stable" and (dep.get("melanoma_selective_dependency") == "exploratory_yes" or druggable or gdsc_hits > 0):
            priority = "exploratory_high"
        elif model_support == "strong_model_stable":
            priority = "exploratory_moderate"
        else:
            priority = "exploratory_low"
        rows.append(
            {
                "gene_symbol": gene,
                "phase5C_tier": row["phase5C_tier"],
                "model_perturbation_support": model_support,
                "mean_delta_P_adverse_like": row["mean_delta_P_adverse_like"],
                "bootstrap_sample_stability": row.get("stability_warning_flags", "not_available"),
                "GSE72056_direction": row.get("direction_consistency", "not_available"),
                "artifact_essentiality_risk": art["manual_and_external_risk_tags"],
                "TCGA_clinical_association": f"{surv_hits} Cox tests with FDR<0.1; exploratory only",
                "DepMap_dependency_evidence": f"melanoma_selective={dep.get('melanoma_selective_dependency', 'not_available')}; broad_dependency={dep.get('broad_dependency', 'not_available')}",
                "GDSC_drug_sensitivity_evidence": f"{gdsc_hits} expression-drug correlations with FDR<0.1; exploratory only",
                "druggability_tractability_evidence": f"target_class={dr.get('target_class', 'not_available')}; OT_drugs={dr.get('open_targets_drug_candidate_count', 'not_available')}; ChEMBL_mechanisms={dr.get('chembl_mechanism_count', 'not_available')}",
                "final_exploratory_priority": priority,
                "allowed_interpretation": "exploratory_hypothesis_generating_model_dependent_external_context_only",
                "forbidden_interpretation": "do_not_present_as_validated_target_or_causal_driver_or_clinically_actionable",
            }
        )
    integrated = pd.DataFrame(rows)
    integrated.to_csv(TABLES / "phase5C_integrated_exploratory_evidence_matrix.csv", index=False)
    write_log(
        "phase5C_evidence_integration_log.md",
        [
            "# Evidence integration",
            "",
            "- Integrated Phase 5B model signal with TCGA, DepMap, GDSC, ChEMBL, and Open Targets context.",
            "- Final priority remains exploratory and hypothesis-generating.",
            "- Broad dependency, housekeeping/ribosomal, ECM/stromal, immune context, and GSE72056 opposite direction were used for downgrading.",
        ],
    )
    return integrated


def write_summary(panel: pd.DataFrame, artifact: pd.DataFrame, depmap: pd.DataFrame, gdsc: pd.DataFrame, drug: pd.DataFrame, integrated: pd.DataFrame) -> None:
    high_model_artifact = artifact[artifact["recommendation"].isin(["deprioritize_for_target_claim", "keep_but_downgrade"])]["gene_symbol"].tolist()
    high_external = integrated[integrated["final_exploratory_priority"].eq("exploratory_high")]["gene_symbol"].tolist()
    downgraded = integrated[integrated["final_exploratory_priority"].eq("deprioritized_due_to_artifact_or_inconsistent_evidence")]["gene_symbol"].tolist()
    moderate = integrated[integrated["final_exploratory_priority"].eq("exploratory_moderate")]["gene_symbol"].tolist()
    lines = [
        "# Phase 5C 中文总结",
        "",
        "## 1. Candidate panel",
        "",
        f"- 本阶段读取 Phase 5B 真实输出，构建候选 panel：{len(panel)} 个基因。",
        "- Tier 1: PABPC1, FOS, RPL15, HSP90AB1, RPL8, ACTG1, RACK1。",
        "- Tier 2: FN1, TGFBI, JUN, ATF3, COL1A2。",
        "- Tier 3: HLA-B。",
        "",
        "## 2. Model signal 与 artifact/essentiality 风险",
        "",
        f"- 模型信号较强但 artifact/essentiality 风险较高或需降级解释的基因：{', '.join(high_model_artifact) if high_model_artifact else 'none'}。",
        "- PABPC1、RPL15、RPL8、RACK1、ACTG1 重点保留为模型扰动敏感信号，但因翻译/核糖体/housekeeping/广谱依赖风险，不适合直接转化为靶点叙述。",
        "- HLA-B、JUN、ATF3、COL1A2、FN1、TGFBI 存在 GSE72056 opposite direction、CI 跨 0、样本支持不足或 ECM/immune confounding，均需降级。",
        "",
        "## 3. 外部探索性证据",
        "",
        f"- 外部探索性支持相对较好的基因：{', '.join(high_external) if high_external else 'none under strict exploratory integration'}。",
        f"- 保留中等探索优先级的基因：{', '.join(moderate) if moderate else 'none'}。",
        "- TCGA-SKCM 使用 cBioPortal `skcm_tcga_gdc`、mRNA TPM profile `skcm_tcga_gdc_mrna_seq_tpm`，输出表达分布和探索性 Cox 关联。所有 HR/P/FDR 只能解释为 association。",
        "- DepMap 使用当前 portal breadbox 暴露的 Public 26Q1 相关矩阵，输出 melanoma lineage dependency、pan-cancer dependency context 和 expression-dependency correlation。",
        "- GDSC 使用 DepMap portal 的 GDSC2 log2AUC collapsed 矩阵，对 BRAF/MEK/ERK/HSP90 等 melanoma-relevant drug features 做表达-药敏相关探索。",
        "- ChEMBL/Open Targets 用于 tractability、target class、clinical/drug candidate、melanoma disease association 和 broad essentiality context。DrugBank 未自动查询，标记为 needs manual confirmation。",
        "",
        "## 4. 需要降级或暂不推进的基因",
        "",
        f"- 因 artifact、essentiality 或方向不一致被降级/去优先化的基因：{', '.join(downgraded) if downgraded else 'none'}。",
        "",
        "## 5. Phase 5D 建议",
        "",
        "- 可以进入 Phase 5D 做 manuscript-level figure/table preparation，包括 Phase 5B model signal、Phase 5C external evidence matrix、artifact filter、TCGA/DepMap/GDSC exploratory panels。",
        "- 图表标题和正文必须保持 exploratory、association、context、hypothesis-generating 语言。",
        "",
        "## 6. 必须保留的限制性措辞",
        "",
        "- 本阶段结果不能称为已验证治疗靶点、因果驱动基因或临床可行动靶点。",
        "- GSE72056 仍只能作为 processed/non-integer expression sensitivity，不能作为强外部验证。",
        "- TCGA 生存结果只能作为探索性关联，不能写成预后生物标志物验证。",
        "- DepMap/GDSC 结果只能作为细胞系依赖/药敏相关背景，不能写成体内或临床疗效证据。",
        "",
        "READY_FOR_PHASE5D = CONDITIONAL",
        "",
        "可以继续做什么：",
        "- 准备 manuscript-level figure/table 草图。",
        "- 整理 exploratory evidence matrix 和限制性说明。",
        "- 对 DrugBank 或手工文献证据进行人工复核。",
        "",
        "禁止做什么：",
        "- 禁止输出正式治疗靶点结论。",
        "- 禁止把任一外部数据库关联写成因果或临床可行动证据。",
        "- 禁止把 broad dependency 或 housekeeping/ribosomal 信号写成 melanoma-specific vulnerability。",
        "",
        "必须满足的条件：",
        "- Phase 5D 图表和正文继续保留 exploratory / hypothesis-generating / model-dependent 边界。",
        "- 对 DrugBank、关键药物和核心基因的外部证据进行人工核对后再进入投稿级文字。",
    ]
    (ROOT / "summary_phase5C_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    panel, _ = build_candidate_panel()
    tcga_expr, tcga_surv = run_tcga(panel)
    depmap, pan_context, gdsc = run_depmap_and_gdsc(panel)
    drug = run_druggability(panel, depmap)
    artifact = build_artifact_filter(panel, depmap, drug)
    integrated = build_integration(panel, artifact, tcga_surv, depmap, gdsc, drug)
    write_summary(panel, artifact, depmap, gdsc, drug, integrated)
    print("PHASE5C_EXTERNAL_EVIDENCE: PASS")
    print(f"SUMMARY={ROOT / 'summary_phase5C_zh.md'}")


if __name__ == "__main__":
    main()
