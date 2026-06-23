# Geneformer-guided exploratory prioritization of adverse malignant cell-state regulators in melanoma

## Overview

This repository contains a public, lightweight reproducibility package for the project:

**Geneformer-guided in silico perturbation identifies exploratory regulators of adverse malignant cell states in melanoma.**

The repository is prepared for code and result-table transparency in support of a Computational Biology and Chemistry submission. It contains reproducibility scripts, public data-source documentation, main result tables, supplementary table index files, final figure-generation code and manuscript-support files. It does **not** contain raw sequencing data, processed AnnData objects, tokenized arrays, Geneformer checkpoints, Hugging Face caches or local model caches.

All interpretation should remain exploratory, model-dependent and association/context bounded. HSP90AB1 is reported as an exploratory cross-evidence signal with broad dependency and stress-response caveats, not as a treatment claim.

## Public datasets and resources

- GSE115978: primary melanoma single-cell dataset for supervised Geneformer modeling and expanded deletion analysis.
- GSE72056: processed-expression single-cell sensitivity resource only.
- GSE120575: immune-response-related melanoma single-cell resource; not used for malignant discovery.
- TCGA-SKCM / GDC and cBioPortal: exploratory bulk expression and clinical association context.
- DepMap / CCLE: melanoma-lineage and pan-cancer dependency context.
- GDSC2: exploratory drug-sensitivity association context.
- ChEMBL and Open Targets: tractability and target-context resources.
- DrugBank was not queried automatically and requires manual confirmation.

See `data_sources.md` for access links and release notes.

## Software environment

Create an environment with either:

```bash
conda env create -f environment.yml
conda activate melanoma-geneformer-public
```

or:

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA/CPU environment. The original GPU validation used a CUDA 12.8 PyTorch wheel on Windows, but this repository does not assume that local hardware.

## Analysis workflow

The staged workflow is documented in `scripts/pipeline/`:

1. Public metadata and processed matrix acquisition from GEO and external resources.
2. AnnData construction and expression-matrix quality checks.
3. Gene symbol to Ensembl mapping and Geneformer-compatible tokenization preparation.
4. Sample-level Geneformer malignant-state modeling and grouped evaluation.
5. Exploratory in silico deletion, bootstrap stability and sample-level stability summaries.
6. External evidence triangulation using TCGA-SKCM, DepMap/CCLE, GDSC2, ChEMBL and Open Targets.
7. Manuscript-level table and figure preparation.

Some upstream scripts require large public data files, model weights and token dictionaries that are intentionally excluded from the repository. See `reproducibility_notes.md`.

## Reproducing main tables and figures

The main result tables are in `results_tables/`.

To regenerate the main figures from the published result tables:

```bash
python scripts/reproduce_main_figures.py
```

This writes PNG and PDF outputs to `figures_final/`.

## Limitations

- This repository does not include raw FASTQ/SRA files, AnnData objects, tokenized datasets, model checkpoints or caches.
- GSE72056 is used only as processed-expression sensitivity analysis.
- External TCGA, DepMap, GDSC2, ChEMBL and Open Targets resources are context layers, not validation of mechanism or clinical use.
- DrugBank was not queried automatically and requires manual confirmation.
- Repository outputs support reproducibility of the reported computational workflow and figures; independent raw-count single-cell cohorts and wet-lab perturbation studies are required for stronger biological interpretation.

## Citation

If this repository is used, cite the associated manuscript once available:

Qian J. Geneformer-guided in silico perturbation identifies exploratory regulators of adverse malignant cell states in melanoma. Manuscript in preparation/submission.

Reference details, DOI and repository archive DOI should be updated after journal or Zenodo/GitHub release.

## Contact

Jiatian Qian  
Department of Radiology, Shanghai General Hospital, Shanghai Jiao Tong University School of Medicine, Shanghai, China.  
Email: qjt840116225@gmail.com  
ORCID: 0009-0007-0791-9211
