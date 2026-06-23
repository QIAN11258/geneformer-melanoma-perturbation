# Reproducibility notes

## Files intentionally not uploaded

The following file classes are excluded from the public repository because they are large, redistributability-limited, cache-derived or machine-specific:

- raw FASTQ/SRA files;
- downloaded GEO matrix archives and large processed expression matrices;
- AnnData `.h5ad` files;
- tokenized Geneformer datasets and NumPy arrays;
- Geneformer model checkpoints and Hugging Face caches;
- local GPU environment directories and model-cache paths;
- cell-level expanded deletion intermediate tables that can be regenerated from the pipeline;
- DOCX submission files and cover-letter files.

## How to regenerate omitted files

1. Download public processed matrices and metadata from the accessions listed in `data_sources.md`.
2. Run the staged scripts in `scripts/pipeline/` in numerical order.
3. Download or cache the required Geneformer pretrained model and token dictionaries from the official Geneformer resources.
4. Use sample-level split units only; do not use cell-level random splitting for supervised evaluation.
5. Re-run figure generation with:

```bash
python scripts/reproduce_main_figures.py
```

## Expected release-package contents

The release package includes:

- clean README and reproducibility documentation;
- environment files;
- cleaned pipeline scripts with relative paths or documented placeholder paths;
- main result tables and supplementary table index;
- final figure files and a script to regenerate them from included tables;
- manuscript-support files needed for transparent reporting.

## Interpretation boundary

This is an exploratory computational framework. It does not establish treatment use, clinical actionability or mechanism. HSP90AB1 is retained as an exploratory cross-evidence signal requiring independent experimental follow-up.
