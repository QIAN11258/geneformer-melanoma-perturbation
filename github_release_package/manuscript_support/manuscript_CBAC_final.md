# Geneformer-guided in silico perturbation identifies exploratory regulators of adverse malignant cell states in melanoma

## Highlights

- Geneformer prioritized adverse-like melanoma state hypotheses
- Grouped evaluation supported delta P(adverse_like) readouts
- Expanded deletion ranked 143 genes in 392 adverse-like cells
- Evidence triangulation highlighted HSP90AB1 with caveats
- Artifact filters downgraded broad-dependency model signals

## Abstract

Single-cell foundation models offer a route for perturbation hypothesis generation in heterogeneous malignant states, but their outputs require careful calibration, sample-aware evaluation and contextual interpretation. We developed an exploratory Geneformer-V2 framework for adverse-like malignant cell-state analysis in melanoma. Public datasets were assigned non-overlapping roles: GSE115978 supported a binary melanocytic-like versus adverse-like modeling task, whereas GSE72056 was retained only for processed-expression sensitivity analysis. Tokenized inputs preserved cell identifiers, sample identifiers, treatment-group metadata and split units. A class-weighted calibrated Geneformer model was evaluated by grouped held-out testing, repeated grouped retraining and sensitivity analysis. The primary grouped test showed accuracy 0.714, balanced accuracy 0.733 and macro-F1 0.714, supporting continuous delta P(adverse_like) as a perturbation readout rather than hard-label cross-dataset interpretation. Expanded in silico deletion ranked 143 eligible genes across 392 GSE115978 adverse-like cells, followed by bootstrap and sample-level stability checks. External evidence triangulation using TCGA-SKCM, DepMap/CCLE, GDSC2, ChEMBL and Open Targets was used for contextual interpretation of model-prioritized signals. HSP90AB1 emerged as the strongest exploratory cross-evidence signal, while FOS was retained as a moderate signal. Several strong model signals, including PABPC1, RPL15, RPL8, ACTG1 and RACK1, were downgraded because of housekeeping, ribosomal, pan-essential or broad-dependency concerns. The framework provides hypothesis-generating evidence to guide future experimental and raw-count single-cell follow-up.

## Keywords

Geneformer; melanoma; single-cell transcriptomics; in silico perturbation; malignant cell state; computational biology; evidence triangulation


## 1. Introduction

Melanoma malignant cells occupy diverse transcriptional states, making computational prioritization of candidate regulators difficult. Single-cell RNA sequencing has made these state programs measurable at cellular resolution, but marker-based descriptions alone do not indicate whether a gene is informative for a model representation of state transition or merely reflects broad expression, lineage, immune-context or housekeeping biology. A useful computational framework for this setting therefore requires both cell-state-aware modeling and explicit interpretation safeguards.

Foundation models trained on large transcriptomic corpora provide a new route for representing single-cell states and testing model-dependent perturbation hypotheses. Geneformer is particularly relevant because it uses ranked gene-token representations that can be adapted to supervised single-cell classification and in silico gene deletion. However, applying such models to tumor state analysis is challenging for three reasons: sample-level leakage can inflate apparent performance, perturbation scores can be dominated by frequent or essential genes, and external resources often describe bulk or cell-line contexts rather than single-cell malignant-state biology.

We therefore developed a staged, exploratory Geneformer-guided framework for adverse-like malignant cell-state analysis in melanoma. The framework first assigns public datasets to non-overlapping analytical roles, then constructs Geneformer-compatible inputs with preserved sample and cell metadata, fine-tunes a binary adverse-like versus melanocytic-like model, evaluates readiness with grouped and sensitivity analyses, performs expanded in silico deletion, and finally integrates artifact filters with external evidence resources. The goal is to generate prioritized hypotheses with explicit interpretation boundaries, rather than to establish clinical actionability or mechanism.

Using this framework, we generated a model-dependent perturbation ranking from GSE115978 adverse-like malignant cells and an integrated exploratory evidence matrix spanning TCGA-SKCM, DepMap/CCLE, GDSC2, ChEMBL and Open Targets. HSP90AB1 is highlighted as the most coherent exploratory cross-evidence signal after artifact-aware filtering, while FOS is retained as a moderate signal. In contrast, several strong model-level signals are downgraded because they overlap with housekeeping, ribosomal, translation, pan-essential or broad dependency features. The manuscript presents these findings as a computational biology framework for hypothesis generation in melanoma, with downstream experimental and independent raw-count single-cell follow-up required.



## 2. Materials and methods

### 2.1 Public data acquisition and dataset roles

Public melanoma single-cell and external molecular resources were curated with predefined, non-overlapping roles. GSE115978 was used as the primary single-cell dataset for supervised Geneformer modeling and expanded in silico deletion. GSE72056 was retained only as a processed-expression sensitivity dataset because the checked expression matrix was non-integer processed expression. GSE120575 was treated as an immune-response-related resource and was not used for malignant-cell discovery or perturbation ranking. TCGA-SKCM, DepMap/CCLE, GDSC2, ChEMBL and Open Targets were used as exploratory external evidence resources. DrugBank was not queried automatically and requires manual confirmation.

### 2.2 Single-cell preprocessing and AnnData construction

We used processed matrices or author-provided count/expression files; FASTQ and SRA-level raw sequencing files were not downloaded. AnnData objects were constructed in earlier phases for GSE72056, GSE115978 and GSE120575. Metadata checks covered cell annotations, patient or sample identifiers, malignant/non-malignant or cell-type information, treatment-group information where available, and matrix type. GSE115978 was prioritized for model input because it provided the primary malignant-cell context used in the supervised adverse-like task.

### 2.3 Gene identifier mapping and Geneformer tokenization

Gene symbol to Ensembl ID mapping was performed before tokenization. Unmapped and duplicated genes were recorded in dedicated tables and logs, and no untracked gene deletion or merging rule was introduced during manuscript preparation. Geneformer-compatible tokenized outputs preserved original cell identifiers, sample or tumor identifiers, treatment group metadata, malignant-state labels and split units to support sample-level evaluation.

### 2.4 Malignant-state label construction

The final supervised task was Binary A, contrasting melanocytic_like cells with adverse_like cells. The adverse_like class combined malignant states selected during earlier label-strategy refinement. Ambiguous cells were excluded from supervised modeling and perturbation input. This label strategy served as a modeling construct for exploratory perturbation ranking and was not treated as a universal melanoma state taxonomy.

### 2.5 Geneformer fine-tuning and model calibration

Geneformer-V2-104M_CLcancer was used as the main pretrained model. Supervised fine-tuning was performed on GSE115978 tokenized data using sample-level split units to avoid cell-level leakage. Model readiness was assessed using grouped held-out testing, repeated grouped retraining, threshold robustness and GSE72056 processed-expression sensitivity. The class-weighted calibrated model was selected for perturbation analysis. Primary perturbation reporting used continuous delta P(adverse_like); the threshold of 0.70 was used only for pilot hard-label summaries.

### 2.6 In silico deletion analysis

Expanded in silico deletion was performed for 143 eligible genes across 392 GSE115978 adverse_like cells. For each gene, deletion was evaluated only in cells where the gene token was present. The primary readout was mean delta P(adverse_like), calculated as post-deletion minus pre-deletion model-predicted adverse_like probability. Negative values therefore indicated a model-predicted decrease in adverse_like probability after token deletion. Batch inference, per-gene processing and resume logic were used to preserve partial results.

### 2.7 Bootstrap and sample-level stability analysis

Bootstrap analysis used 1000 iterations per gene to estimate confidence intervals for delta P(adverse_like). Sample-level summaries recorded the number of samples containing each gene, the fraction of samples with negative mean delta P(adverse_like), sample-dominance indicators and direction consistency. Genes with confidence intervals crossing zero, inconsistent sample-level direction, low support or GSE72056 opposite direction were flagged for downgraded interpretation.

### 2.8 Processed-expression sensitivity analysis

GSE72056 was used only as a processed-expression sensitivity analysis. It was not used for training, parameter selection, primary ranking or strong cross-dataset interpretation. Direction consistency was summarized relative to GSE115978 perturbation effects as same_direction, opposite_direction or not evaluable.

### 2.9 External evidence triangulation

TCGA-SKCM was used for bulk expression and exploratory survival association context. DepMap/CCLE provided melanoma-lineage and pan-cancer dependency context. GDSC2 log2AUC collapsed features were used for exploratory expression-drug sensitivity correlations. ChEMBL and Open Targets provided target class, tractability, drug/candidate and disease association context. These resources were interpreted as association or context layers and were not used to make clinical or mechanistic claims.

### 2.10 Artifact, housekeeping and pan-essentiality risk filtering

Interpretation filters were applied to distinguish model-prioritized perturbation signals from broad biological or technical risks. Risk tags included ribosomal or translation-related status, housekeeping-like or cytoskeletal biology, pan-essential risk, broad cancer dependency risk, detection-frequency-driven perturbation risk, immune context, stromal/ECM context, low support and GSE72056 opposite direction. These filters did not remove genes from the record; instead, they changed interpretation priority.

### 2.11 Statistical analysis

Model performance was summarized using accuracy, balanced accuracy, macro-F1 and class-specific recall. Perturbation effects were summarized using mean and median delta P(adverse_like), bootstrap confidence intervals, fraction of cells and samples with negative delta P(adverse_like), and sample-level warning flags. External evidence summaries used the statistical outputs produced in Phase 5C, including TCGA-SKCM exploratory association tables, DepMap dependency summaries and GDSC2 expression-drug sensitivity correlations.

### 2.12 Reproducibility and compute environment

The analysis was staged with explicit logs for data preparation, tokenization, runtime environment checks, model training, perturbation inference and evidence integration. Phase 4A/4A.1 confirmed the Geneformer runtime and GPU-enabled PyTorch environment. Phase 5D converted verified outputs into manuscript-level figure and table plans without adding new analyses.



## 3. Results

### 3.1 Overall workflow and dataset roles

We assembled a role-separated computational workflow for melanoma cell-state perturbation analysis (Figure 1; Table 1). GSE115978 served as the primary dataset for supervised Geneformer modeling and perturbation analysis. GSE72056 was retained only as processed-expression sensitivity because the checked matrix was non-integer processed expression. GSE120575 was documented as an immune-response-related dataset and was not used for malignant-cell perturbation ranking. TCGA-SKCM, DepMap/CCLE, GDSC2, ChEMBL and Open Targets were assigned as external context resources. This design kept training, sensitivity and external evidence layers analytically distinct.

### 3.2 Binary malignant-state model development and evaluation

Geneformer-V2-104M_CLcancer was fine-tuned for the Binary A task contrasting melanocytic_like and adverse_like malignant cells (Figure 2; Table 2). On the primary grouped held-out GSE115978 test set, the calibrated model had n = 147 cells, accuracy 0.714, balanced accuracy 0.733 and macro-F1 0.714 at the pilot threshold of 0.70. Class-specific recall was 0.636 for melanocytic_like and 0.831 for adverse_like. Repeated grouped retraining over 5 folds included n = 610 supervised cells and showed balanced accuracy 0.811 and macro-F1 0.792 in the summarized output, while also documenting sample-composition sensitivity. In GSE72056 processed-expression sensitivity, macro-F1 was 0.613 and adverse_like recall was 0.497. These results supported continuous delta P(adverse_like) as the perturbation readout, while arguing against strong hard-label interpretation across datasets.

### 3.3 Expanded in silico deletion and stability ranking

Expanded in silico deletion evaluated 143 eligible genes across 392 GSE115978 adverse_like cells (Figure 3; Table 3). The top model-dependent decreases in P(adverse_like) included PABPC1, FOS, RPL15, HLA-B and HSP90AB1. HSP90AB1 had mean delta P(adverse_like) -0.001284, while FOS had mean delta P(adverse_like) -0.002686. Bootstrap and sample-level summaries identified comparatively stable model signals including PABPC1, FOS, RPL15, HSP90AB1, RPL8, ACTG1 and RACK1. However, stability alone was insufficient for biological prioritization, because several highly ranked genes also carried housekeeping, ribosomal/translation, pan-essential or broad-dependency risk tags.

### 3.4 External evidence integration and candidate stratification

The integrated matrix contained 13 genes and combined Phase 5B perturbation output with artifact filters and external context from TCGA-SKCM, DepMap/CCLE, GDSC2, ChEMBL and Open Targets (Figure 4; Table 4). HSP90AB1 was categorized as exploratory_high and FOS as exploratory_moderate. PABPC1, RPL15, RPL8, ACTG1 and RACK1 were downgraded because their model signals overlapped with housekeeping, ribosomal/translation, pan-essential or broad dependency concerns. JUN, COL1A2, ATF3, FN1, TGFBI and HLA-B were downgraded because of direction inconsistency, low support, ECM/stromal or immune-context confounding, or stability limitations. These external resources provided structured context rather than confirmatory evidence.

### 3.5 HSP90AB1-focused exploratory cross-evidence signal

HSP90AB1 emerged as the most coherent exploratory cross-evidence signal after integrating model ranking and external context (Figure 5). In the integrated table, HSP90AB1 retained same-direction GSE72056 sensitivity, a stable model-dependent deletion signal, Open Targets drug-candidate count 10 and ChEMBL mechanism count 0. At the same time, it was flagged for broad cancer dependency, pan-essential risk and stress-response-related biology. HSP90AB1 should therefore be interpreted as a context-supported hypothesis for further investigation, not as an established treatment claim.



## 4. Discussion

### 4.1 Principal findings

This study presents an exploratory computational framework for prioritizing melanoma adverse-like malignant cell-state perturbation hypotheses with Geneformer. The main contribution is the staged integration of sample-level guarded model training, model-dependent in silico deletion, bootstrap and sample-level stability analysis, processed-expression sensitivity, and external evidence triangulation. The framework is intended to produce ranked hypotheses and interpretation boundaries rather than definitive biological conclusions.

### 4.2 Interpretation of Geneformer-derived perturbation ranking

The Geneformer-derived ranking should be interpreted as a model-dependent readout of how token deletion changes predicted adverse_like probability. This readout is useful because it provides a consistent way to compare perturbation hypotheses across genes within the same trained model. It is also limited because model probability shifts can reflect expression frequency, broad cellular programs or dependency biology rather than melanoma-specific regulation. For that reason, the study emphasizes continuous delta P(adverse_like), grouped evaluation and stability filters.

### 4.3 Why HSP90AB1 emerged after external evidence integration

HSP90AB1 emerged because it combined a stable perturbation signal with the strongest integrated exploratory priority in the Phase 5C evidence matrix. Its interpretation remains bounded by broad stress-response and dependency caveats. This balance is central to the study: model ranking alone would be insufficient, whereas external evidence triangulation helps distinguish a context-supported hypothesis from artifact-prone signals. HSP90AB1 should be discussed as an exploratory cross-evidence signal requiring future experimental follow-up.

### 4.4 Why strong model signals were downgraded

PABPC1, RPL15, RPL8, ACTG1 and RACK1 illustrate why highly ranked perturbation signals require artifact-aware review. These genes showed strong or stable model-level effects, but their biological annotations and external dependency context raised housekeeping, ribosomal/translation, pan-essential, cytoskeletal or broad-dependency concerns. This downgrading step is a strength of the framework because it reduces the risk of overinterpreting ranked model outputs.

### 4.5 Relationship to melanoma cell-state biology

The Binary A task operationalized an adverse_like state as a modeling construct for perturbation analysis. This framing is compatible with the broader concept that melanoma cells can shift between melanocytic and adverse transcriptional programs, but it should not be treated as a universal melanoma taxonomy. FOS and JUN-related results are consistent with stress-response and AP-1 context, while ECM and immune-associated genes such as FN1, TGFBI, COL1A2 and HLA-B require caution because single-cell malignant-state signals can be confounded by stromal, immune or processed-expression context.

### 4.6 Strengths of the framework

The framework has three practical strengths for computational biology. First, it separates dataset roles before modeling, reducing the chance that training, sensitivity and external evidence layers are conflated. Second, it preserves sample-level split units and metadata through tokenization and evaluation. Third, it applies artifact, essentiality and stability filters after perturbation ranking, making interpretation more conservative and transparent.

### 4.7 Limitations

Several limitations constrain interpretation. Geneformer fine-tuning was conditionally adequate for exploratory continuous delta P(adverse_like), but sample-composition sensitivity remained. GSE72056 was processed/non-integer expression and could only support processed-expression sensitivity analysis. External resources are heterogeneous: TCGA-SKCM is bulk tissue, DepMap and GDSC2 are cell-line resources, and ChEMBL/Open Targets are curated database contexts. DrugBank was not queried automatically and requires manual confirmation. No wet-lab perturbation experiment was performed in this phase.

### 4.8 Future directions

Future work should evaluate HSP90AB1 and selected comparator genes using orthogonal perturbation assays followed by single-cell readouts of malignant-state shifts. Additional raw-count melanoma single-cell cohorts with compatible malignant-state annotations are needed to reassess model calibration and state definitions. External evidence integration should also be extended only after drug and dependency annotations are manually reviewed and linked to experimentally interpretable endpoints.



## 5. Conclusions

This study provides an exploratory Geneformer-guided framework for prioritizing adverse malignant cell-state perturbation hypotheses in melanoma. Integrated evidence highlighted HSP90AB1 as a context-supported exploratory signal, while artifact filters downgraded several housekeeping, ribosomal and broad-dependency model signals. The findings require future wet-lab studies and independent raw-count single-cell follow-up before stronger biological or translational interpretation is warranted.



## Data availability

All primary analyses used public datasets and resources listed in Table 1 and the supplementary table index. Processed project outputs are organized in the project tables, logs and manuscript-preparation directories.

Repository link to be added before submission.

Final public repository deposition, data-linking requirements, release file names and access dates require author confirmation before journal upload.

## Code availability

Analysis scripts generated during the staged workflow are available in the project scripts directory. Repository link to be added before submission. The final public code release should be reviewed to remove local paths and to ensure that no restricted files, credentials or nonredistributable model-cache artifacts are included.

## CRediT author statement

Jiatian Qian: Conceptualization, Methodology, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review & editing, Visualization, and Project administration.

## Author contributions

Jiatian Qian conceived and designed the study, curated the data, performed the computational analyses, interpreted the results, drafted the manuscript, revised it critically for important intellectual content, and approved the final version.

## Declaration of competing interest

The author declares no competing interests.

## Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## Ethics statement

This study used public, de-identified datasets and did not generate new human participant data. Final institutional requirements for secondary use of public data should be confirmed before submission.

## Acknowledgements

Acknowledgements should be added after author review. This section should not be included on the title page.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the author used ChatGPT and Codex to assist with code organization, workflow documentation, manuscript structuring, language refinement, and submission-format checking. The author reviewed, edited, and verified all AI-assisted outputs and takes full responsibility for the content of the manuscript.

## References

[1] Theodoris CV, Xiao L, Chopra A, Chaffin MD, Al Sayed ZR, Hill MC, et al.. Transfer learning enables predictions in network biology. Nature. 2023;618(7965):616-624. doi:10.1038/s41586-023-06139-9.
[2] Tirosh I, Izar B, Prakadan SM, Wadsworth MH, Treacy D, Trombetta JJ, et al.. Dissecting the multicellular ecosystem of metastatic melanoma by single-cell RNA-seq. Science. 2016;352(6282):189-196. doi:10.1126/science.aad0501.
[3] Jerby-Arnon L, Shah P, Cuoco MS, Rodman C, Su MJ, Melms JC, et al.. A Cancer Cell Program Promotes T Cell Exclusion and Resistance to Checkpoint Blockade. Cell. 2018;175(4):984-997.e24. doi:10.1016/j.cell.2018.09.006.
[4] Sade-Feldman M, Yizhak K, Bjorgaard SL, Ray JP, de Boer CG, Jenkins RW, et al.. Defining T Cell States Associated with Response to Checkpoint Immunotherapy in Melanoma. Cell. 2018;175(4):998-1013.e20. doi:10.1016/j.cell.2018.10.038.
[5] Akbani R, Akdemir KC, Aksoy BA, Albert M, Ally A, Amin SB, et al.. Genomic Classification of Cutaneous Melanoma. Cell. 2015;161(7):1681-1696. doi:10.1016/j.cell.2015.05.044.
[6] Cerami E, Gao J, Dogrusoz U, Gross BE, Sumer SO, Aksoy BlA, et al.. The cBio Cancer Genomics Portal: An Open Platform for Exploring Multidimensional Cancer Genomics Data. Cancer Discovery. 2012;2(5):401-404. doi:10.1158/2159-8290.cd-12-0095.
[7] Gao J, Aksoy BlA, Dogrusoz U, Dresdner G, Gross B, Sumer SO, et al.. Integrative Analysis of Complex Cancer Genomics and Clinical Profiles Using the cBioPortal. Science Signaling. 2013;6(269). doi:10.1126/scisignal.2004088.
[8] Tsherniak A, Vazquez F, Montgomery PG, Weir BA, Kryukov G, Cowley GS, et al.. Defining a Cancer Dependency Map. Cell. 2017;170(3):564-576.e16. doi:10.1016/j.cell.2017.06.010.
[9] Ghandi M, Huang FW, Jané-Valbuena J, Kryukov GV, Lo CC, McDonald ER, et al.. Next-generation characterization of the Cancer Cell Line Encyclopedia. Nature. 2019;569(7757):503-508. doi:10.1038/s41586-019-1186-3.
[10] Yang W, Soares J, Greninger P, Edelman EJ, Lightfoot H, Forbes S, et al.. Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. Nucleic Acids Research. 2012;41(D1):D955-D961. doi:10.1093/nar/gks1111.
[11] Iorio F, Knijnenburg TA, Vis DJ, Bignell GR, Menden MP, Schubert M, et al.. A Landscape of Pharmacogenomic Interactions in Cancer. Cell. 2016;166(3):740-754. doi:10.1016/j.cell.2016.06.017.
[12] Mendez D, Gaulton A, Bento APc, Chambers J, De Veij M, Félix E, et al.. ChEMBL: towards direct deposition of bioassay data. Nucleic Acids Research. 2019;47(D1):D930-D940. doi:10.1093/nar/gky1075.
[13] Ochoa D, Hercules A, Carmona M, Suveges D, Gonzalez-Uriarte A, Malangone C, et al.. Open Targets Platform: supporting systematic drug–target identification and prioritisation. Nucleic Acids Research. 2021;49(D1):D1302-D1310. doi:10.1093/nar/gkaa1027.
[14] Rambow F, Rogiers A, Marin-Bejar O, Aibar S, Femel J, Dewaele M, et al.. Toward Minimal Residual Disease-Directed Therapy in Melanoma. Cell. 2018;174(4):843-855.e19. doi:10.1016/j.cell.2018.06.025.
[15] Tsoi J, Robert L, Paraiso K, Galvan C, Sheu KM, Lay J, et al.. Multi-stage Differentiation Defines Melanoma Subtypes with Differential Vulnerability to Drug-Induced Iron-Dependent Oxidative Stress. Cancer Cell. 2018;33(5):890-904.e5. doi:10.1016/j.ccell.2018.03.017.
[16] Wouters J, Kalender-Atak Z, Minnoye L, Spanier KI, De Waegeneer M, Bravo González-Blas C, et al.. Robust gene expression programs underlie recurrent cell states and phenotype switching in melanoma. Nature Cell Biology. 2020;22(8):986-998. doi:10.1038/s41556-020-0547-3.
[17] Whitesell L, Lindquist SL. HSP90 and the chaperoning of cancer. Nature Reviews Cancer. 2005;5(10):761-772. doi:10.1038/nrc1716.
[18] Trepel J, Mollapour M, Giaccone G, Neckers L. Targeting the dynamic HSP90 complex in cancer. Nature Reviews Cancer. 2010;10(8):537-549. doi:10.1038/nrc2887.
[19] Yang F, Wang W, Wang F, Fang Y, Tang D, Huang J, et al.. scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data. Nature Machine Intelligence. 2022;4(10):852-866. doi:10.1038/s42256-022-00534-z.
[20] Cui H, Wang C, Maan H, Pang K, Luo F, Duan N, et al.. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. Nature Methods. 2024;21(8):1470-1480. doi:10.1038/s41592-024-02201-0.

Resource accessions and database links are listed in references_CBAC_verified.md and require final access-date/version confirmation before submission. DrugBank was not queried automatically and requires manual confirmation.

## Supplementary information

Supplementary tables are listed in `tables_CBAC_ready/supplementary_table_index_CBAC.csv` and `tables_CBAC_ready/supplementary_table_index_CBAC.md`.


# Figure legends

The following legends are adapted for Computational Biology and Chemistry style. All abbreviations are defined within each legend, and all interpretations remain exploratory.

## Figure 1. Study design and analysis workflow.
Schematic overview of the staged analysis framework. Public melanoma single-cell and bulk/cell-line resources were assigned non-overlapping roles, with GSE115978 used for supervised Geneformer training and exploratory perturbation analysis, GSE72056 retained as a processed-expression sensitivity dataset, and The Cancer Genome Atlas Skin Cutaneous Melanoma (TCGA-SKCM), Cancer Dependency Map (DepMap), Genomics of Drug Sensitivity in Cancer 2 (GDSC2), Open Targets and ChEMBL used for external evidence context. The workflow emphasizes sample-level separation, model calibration, expanded in silico deletion, bootstrap and sample-level stability assessment, and downstream evidence integration. All downstream prioritization is exploratory and model-dependent.

## Figure 2. Geneformer model development and conditional readiness for perturbation analysis.
Summary of the binary adverse-like versus melanocytic-like Geneformer modeling strategy. Panels should show the grouped training and held-out evaluation design, primary held-out performance at the selected pilot threshold, repeated grouped retraining metrics, threshold robustness, and GSE72056 processed-expression sensitivity. GSE72056 was not used for training and is shown only to illustrate processed-expression domain sensitivity. The model was therefore used for continuous delta P(adverse_like)-based perturbation readouts rather than formal hard-label target discovery.

## Figure 3. Expanded in silico deletion ranking and stability analysis.
Expanded model-dependent in silico deletion was performed across 143 eligible genes in 392 GSE115978 adverse-like cells. Panels should display ranked mean delta P(adverse_like), bootstrap confidence intervals, fraction of cells and samples with negative delta P, sample-level stability, and GSE72056 sensitivity direction. Negative delta P(adverse_like) indicates a model-predicted decrease in adverse-like probability after gene deletion. Signals are interpreted as exploratory perturbation hypotheses, and genes with low support, confidence intervals crossing zero, sample-level inconsistency, or GSE72056 opposite direction are downgraded.

## Figure 4. Integrated exploratory evidence matrix for prioritized perturbation candidates.
Integrated evidence matrix combining Phase 5B model perturbation support with Phase 5C external context. Columns should summarize bootstrap/sample stability, GSE72056 sensitivity direction, artifact and essentiality risk, The Cancer Genome Atlas Skin Cutaneous Melanoma (TCGA-SKCM) exploratory clinical association, Cancer Dependency Map (DepMap) melanoma and pan-cancer dependency context, Genomics of Drug Sensitivity in Cancer 2 (GDSC2) drug sensitivity associations, and Open Targets/ChEMBL tractability information. The matrix highlights HSP90AB1 as the strongest exploratory cross-evidence signal and FOS as a moderate exploratory signal, while ribosomal, housekeeping, ECM/stromal, immune and direction-inconsistent genes are downgraded.

## Figure 5. HSP90AB1-focused exploratory evidence panel.
Focused summary of HSP90AB1 across model and external evidence layers. Panels should show the Phase 5B deletion effect and bootstrap interval, melanoma and pan-cancer Cancer Dependency Map (DepMap) dependency context, Genomics of Drug Sensitivity in Cancer 2 (GDSC2) associations involving melanoma-relevant drug features, Open Targets and ChEMBL tractability context, and The Cancer Genome Atlas Skin Cutaneous Melanoma (TCGA-SKCM) exploratory association summaries if included. HSP90AB1 is presented as an exploratory, context-supported hypothesis with broad stress-response and dependency caveats, not as a treatment claim.
