# -*- coding: utf-8 -*-
"""Regenerate main figures from the public result tables.

This script uses only files included in the GitHub release package. It does not
rerun model training, in silico deletion or external database queries.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results_tables"
FIGURES = ROOT / "figures_final"
FIGURES.mkdir(exist_ok=True)


def clean(x):
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def num(x, default=np.nan):
    try:
        if clean(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def wrap(text, width=24):
    return "\n".join(textwrap.wrap(clean(text), width=width, break_long_words=False))


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_1():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    roles = [
        ("GSE115978", "primary Geneformer\nmodeling"),
        ("GSE72056", "processed-expression\nsensitivity only"),
        ("GSE120575", "immune-response\ncontext only"),
        ("External resources", "TCGA/DepMap/GDSC2\nChEMBL/Open Targets"),
    ]
    colors = ["#DDEBF7", "#E2F0D9", "#FFF2CC", "#EADCF8"]
    for i, (label, role) in enumerate(roles):
        x = 0.05 + i * 0.23
        ax.add_patch(plt.Rectangle((x, 0.72), 0.19, 0.16, fc=colors[i], ec="#333333", lw=1.1))
        ax.text(x + 0.095, 0.825, label, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 0.095, 0.765, role, ha="center", va="center", fontsize=9)
    steps = ["AnnData and\nmetadata QC", "Geneformer\nfine-tuning", "Expanded deletion", "Evidence\nintegration", "Exploratory\nhypotheses"]
    xs = np.linspace(0.08, 0.84, len(steps))
    for i, step in enumerate(steps):
        ax.add_patch(plt.Rectangle((xs[i], 0.36), 0.15, 0.13, fc="#F7F7F7", ec="#444444", lw=1))
        ax.text(xs[i] + 0.075, 0.425, step, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1], 0.425), xytext=(xs[i] + 0.15, 0.425), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.5, 0.18, "Exploratory computational framework; no clinical or mechanistic claim.", ha="center", fontsize=10)
    ax.set_title("Study workflow", fontsize=15, fontweight="bold")
    save(fig, "Figure_1_workflow")


def figure_2():
    df = pd.read_csv(TABLES / "model_performance_CBAC.csv")
    df = df[df["macro_f1"].notna() & (df["macro_f1"].astype(str) != "")]
    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    labels = [wrap(x, 20) for x in df["analysis"]]
    x = np.arange(len(labels))
    width = 0.24
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, [num(v, 0) for v in df[metric]], width, label=metric.replace("_", " "))
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    ax.set_title("Geneformer binary model evaluation", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "Figure_2_model_evaluation")


def figure_3():
    df = pd.read_csv(TABLES / "top_perturbation_genes_CBAC.csv").head(15).copy()
    df["mean_delta"] = df["mean_delta_P_adverse_like"].map(num)
    df["ci_low"] = df["bootstrap_CI95_low"].map(num)
    df["ci_high"] = df["bootstrap_CI95_high"].map(num)
    df = df.sort_values("mean_delta")
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(df))
    vals = df["mean_delta"].values
    ax.barh(y, vals, color="#2C7FB8", edgecolor="#333333", alpha=0.9)
    ax.errorbar(vals, y, xerr=[vals - df["ci_low"].values, df["ci_high"].values - vals], fmt="none", ecolor="#222222", lw=0.8)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene_symbol"], fontsize=9)
    ax.set_xlabel("Mean delta P(adverse_like)")
    ax.set_title("Expanded in silico deletion ranking", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    save(fig, "Figure_3_perturbation_ranking")


def evidence_scores(row):
    delta = num(row.get("mean_delta_P_adverse_like"))
    model = 1.0 if np.isfinite(delta) and delta < 0 else 0.0
    gse = 1.0 if row.get("GSE72056_direction") == "same_direction" else -1.0 if row.get("GSE72056_direction") == "opposite_direction" else 0.0
    broad = -1.0 if row.get("broad_dependency") == "yes" else 0.5
    pan = -1.0 if row.get("pan_essential_risk") == "yes" else 0.5
    tract = 1.0 if num(row.get("chembl_mechanism_count"), 0) > 0 or num(row.get("open_targets_drug_candidate_count"), 0) > 0 else 0.0
    priority = clean(row.get("final_exploratory_priority"))
    pr = 1.0 if "exploratory_high" in priority else 0.5 if "moderate" in priority else -1.0 if "deprioritized" in priority else 0.0
    return [model, gse, broad, pan, tract, pr]


def figure_4():
    df = pd.read_csv(TABLES / "integrated_exploratory_evidence_CBAC.csv")
    matrix = np.array([evidence_scores(row) for _, row in df.iterrows()])
    cols = ["Model\neffect", "GSE72056\ndirection", "Broad\ndependency", "Pan-essential\nrisk", "Tractability\ncontext", "Exploratory\npriority"]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(matrix, cmap="RdYlBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["gene_symbol"], fontsize=9)
    ax.set_title("Integrated exploratory evidence matrix", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    save(fig, "Figure_4_integrated_evidence_matrix")


def figure_5():
    df = pd.read_csv(TABLES / "integrated_exploratory_evidence_CBAC.csv")
    h = df[df["gene_symbol"] == "HSP90AB1"].iloc[0]
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(["HSP90AB1"], [num(h["mean_delta_P_adverse_like"])], color="#2C7FB8", edgecolor="#333333")
    ax1.axhline(0, color="#333333")
    ax1.set_ylabel("Mean delta P(adverse_like)")
    ax1.set_title("Model-dependent deletion effect", fontsize=11, fontweight="bold")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(["Melanoma\nDepMap", "Pan-cancer\nDepMap"], [num(h["melanoma_mean_chronos_gene_effect"]), num(h["pan_cancer_mean_chronos_gene_effect"])], color=["#8DD3C7", "#BEBADA"], edgecolor="#333333")
    ax2.axhline(0, color="#333333")
    ax2.set_ylabel("Mean Chronos gene effect")
    ax2.set_title("Dependency context", fontsize=11, fontweight="bold")
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    notes = [
        f"Final priority: {h.get('final_exploratory_priority')}",
        f"GSE72056 direction: {h.get('GSE72056_direction')}",
        f"Risk tags: {h.get('manual_and_external_risk_tags')}",
        "Interpretation: exploratory cross-evidence signal with dependency and stress-response caveats.",
    ]
    ax3.text(0.02, 0.95, "Evidence summary", fontsize=12, fontweight="bold", va="top")
    ax3.text(0.02, 0.82, "\n".join("- " + wrap(n, 115).replace("\n", "\n  ") for n in notes), fontsize=9, va="top")
    fig.suptitle("HSP90AB1-focused exploratory evidence panel", fontsize=14, fontweight="bold")
    save(fig, "Figure_5_HSP90AB1_panel")


def graphical_abstract():
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")
    panels = [
        ("Public melanoma data\nand tokenization", "GSE115978 primary\nGSE72056 sensitivity only\nAnnData, Ensembl IDs,\nmetadata preserved"),
        ("Binary Geneformer model\nand deletion readout", "melanocytic_like vs adverse_like\nsample-level splits\ndelta P(adverse_like)\nmodel-dependent ranking"),
        ("External evidence\ntriangulation", "TCGA-SKCM, DepMap/CCLE,\nGDSC2, ChEMBL, Open Targets\nHSP90AB1 exploratory signal\nwith caveats"),
    ]
    xs = [0.05, 0.37, 0.69]
    colors = ["#DDEBF7", "#E2F0D9", "#FFF2CC"]
    for i, (title, body) in enumerate(panels):
        ax.add_patch(plt.Rectangle((xs[i], 0.18), 0.26, 0.62, fc=colors[i], ec="#333333", lw=1.2))
        ax.text(xs[i] + 0.13, 0.67, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(xs[i] + 0.13, 0.42, body, ha="center", va="center", fontsize=10)
        if i < 2:
            ax.annotate("", xy=(xs[i + 1] - 0.03, 0.49), xytext=(xs[i] + 0.28, 0.49), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.5, 0.05, "Exploratory computational framework; no clinical or mechanistic claim.", ha="center", fontsize=10)
    fig.suptitle("Geneformer-guided exploratory prioritization in melanoma", fontsize=15, fontweight="bold")
    save(fig, "graphical_abstract_CBAC")


def main():
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    graphical_abstract()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
