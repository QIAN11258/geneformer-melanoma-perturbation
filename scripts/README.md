# Scripts

`reproduce_main_figures.py` regenerates the main figures from `results_tables/`.

`pipeline/` contains cleaned versions of the staged analysis scripts used during the project. Large input files, tokenized datasets, model checkpoints and local caches are not included. Scripts that depend on Geneformer weights expect the user to provide model files under relative paths such as `models/Geneformer/` or adapt the path variables before execution.

`external_helpers/` is a placeholder for optional REST/API helper scripts. The original local helper paths were removed from this public package.
