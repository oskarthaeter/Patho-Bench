# Datasets on /mnt/nfs03-R6 — Patho-Bench Coverage

This document maps the datasets mounted at `/mnt/nfs03-R6` to their corresponding
[Patho-Bench](https://github.com/mahmoodlab/Patho-Bench) task identifiers and benchmark tasks.

---

## Preprocessing Status & Minimal Path to Running Experiments

Patho-Bench requires slide-level pooled embeddings as `.h5` files (one per slide, key `features`,
stored under `<configured_path>/by_<sample_col>/`).  
Use [`convert_nfs_features.py`](convert_nfs_features.py) to prepare the two datasets that have
pre-computed patch features on the NFS share:

```bash
python convert_nfs_features.py --nfs_root /mnt/nfs03-R6 --output_root /path/to/pooled_embeddings
```

Then point `pooled_dirs_yaml` at a YAML like:

```yaml
bracs:
  uni:      /path/to/pooled_embeddings/bracs/uni
  gigapath: /path/to/pooled_embeddings/bracs/gigapath
imp:
  uni:      /path/to/pooled_embeddings/imp/uni
```

### What's already on disk (no GPU needed)

| Dataset | Model | Source on NFS | Format | Action needed |
|---------|-------|---------------|--------|---------------|
| `bracs` | UNI | `BRACS/feats/feats_UNI/pt_files/*.pth` | `[N_patches × 1024]` tensor | Mean-pool → `.h5` via `convert_nfs_features.py` |
| `bracs` | GigaPath | `BRACS/gigapath_embeddings/**/*.pt` | slide-level dict with `last_layer_embed [1×768]` | Extract key → `.h5` via `convert_nfs_features.py` |
| `imp` | UNI | `IMP_CRC/feats/feats_UNI/pt_files/*.pth` | `[N_patches × 1024]` tensor | Mean-pool → `.h5` via `convert_nfs_features.py` |

### What requires GPU feature extraction (run a foundation model first)

All CPTAC datasets (`CCRCC`, `GBM`, `HNSCC`, `LSCC`, `LUAD`, `PDA`) have only raw `.svs` slides
and CLAM patch coordinates under `CPTAC/<cohort>/clam_20/`. You must extract patch features with a
foundation model (e.g. UNI, GigaPath, CHIEF) and save as `.h5` files before running Patho-Bench.

---

## Directly Supported by Patho-Bench

### BRACS
**Mount:** `/mnt/nfs03-R6/BRACS/`  
**Patho-Bench key:** `bracs`  
**Category:** Morphological Subtyping  
**Pre-computed features:** UNI (544 slides), GigaPath (546 slides)  
**Tasks:**

| Task ID | Description |
|---------|-------------|
| `bracs--slidelevel_fine` | Fine-grained breast tissue classification |
| `bracs--slidelevel_coarse` | Coarse-grained breast tissue classification |

---

### CPTAC
**Mount:** `/mnt/nfs03-R6/CPTAC/`  
Contains subdirectories: `AML`, `CCRCC`, `CM`, `GBM`, `HNSCC`, `LSCC`, `LUAD`, `PDA`, `SAR`, `UCEC`  
**Pre-computed features:** None — raw `.svs` slides + CLAM patch coordinates only. Requires GPU feature extraction.

The following CPTAC sub-cohorts have Patho-Bench tasks defined:

#### CPTAC-CCRCC (Clear Cell Renal Cell Carcinoma)
**Patho-Bench key:** `cptac_ccrcc`  
**Category:** Mutation Status + Survival  

| Task ID | Type |
|---------|------|
| `cptac_ccrcc--BAP1_mutation` | Classification |
| `cptac_ccrcc--PBRM1_mutation` | Classification |
| `cptac_ccrcc--OS` | Survival (use `coxnet`) |

#### CPTAC-GBM (Glioblastoma)
**Patho-Bench key:** `cptac_gbm`  
**Category:** Mutation Status  

| Task ID | Type |
|---------|------|
| `cptac_gbm--EGFR_mutation` | Classification |
| `cptac_gbm--TP53_mutation` | Classification |

#### CPTAC-HNSC (Head & Neck Squamous Cell Carcinoma)
**Patho-Bench key:** `cptac_hnsc`  
**Category:** Mutation Status + Survival  

| Task ID | Type |
|---------|------|
| `cptac_hnsc--CASP8_mutation` | Classification |
| `cptac_hnsc--OS` | Survival (use `coxnet`) |

#### CPTAC-LSCC (Lung Squamous Cell Carcinoma)
**Patho-Bench key:** `cptac_lscc`  
**Category:** Mutation Status  

| Task ID | Type |
|---------|------|
| `cptac_lscc--KEAP1_mutation` | Classification |
| `cptac_lscc--ARID1A_mutation` | Classification |

#### CPTAC-LUAD (Lung Adenocarcinoma)
**Patho-Bench key:** `cptac_luad`  
**Category:** Mutation Status + Survival  

| Task ID | Type |
|---------|------|
| `cptac_luad--EGFR_mutation` | Classification |
| `cptac_luad--STK11_mutation` | Classification |
| `cptac_luad--TP53_mutation` | Classification |
| `cptac_luad--OS` | Survival (use `coxnet`) |

#### CPTAC-PDA (Pancreatic Ductal Adenocarcinoma)
**Patho-Bench key:** `cptac_pda`  
**Category:** Mutation Status + Survival  

| Task ID | Type |
|---------|------|
| `cptac_pda--SMAD4_mutation` | Classification |
| `cptac_pda--OS` | Survival (use `coxnet`) |

> **CPTAC sub-cohorts with no Patho-Bench tasks:** `AML`, `CM`, `SAR`, `UCEC`

---

### IMP_CRC (Invasive Micropapillary Colorectal Cancer)
**Mount:** `/mnt/nfs03-R6/IMP_CRC/`  
**Patho-Bench key:** `imp`  
**Category:** Tumor Grading  
**Pre-computed features:** UNI (2032 patches-per-slide files)  

| Task ID | Type |
|---------|------|
| `imp--grade` | Classification |

---

### TCGA
**Mount:** `/mnt/nfs03-R6/TCGA/`  
~10,700 raw WSI slides spanning multiple TCGA projects (BRCA, GBM, LGG, UCEC, KIRC, COAD, HNSC, …).

This is **raw slide data** (GDC manifest `2023-12-04`). It backs several Patho-Bench datasets that are not mounted as separate CPTAC cohorts:

| TCGA project | Likely backs Patho-Bench dataset | Key |
|---|---|---|
| `TCGA-BRCA` (1133 slides) | CPTAC-BRCA mutation tasks | `cptac_brca` |
| `TCGA-COAD` (459 slides) | CPTAC-COAD mutation tasks | `cptac_coad` |
| `TCGA-KIRC` (519 slides) | Mutation-Heterogeneity RCC | `mut-het-rcc` |

> Note: The TCGA slides still require patch extraction and embedding computation before use as Patho-Bench inputs. The `cptac_brca` and `cptac_coad` datasets are absent from `/mnt/nfs03-R6/CPTAC/`; the TCGA slides are the likely source material.

---

## Mounted but NOT Covered by Patho-Bench Tasks

These datasets are present on the NFS share but have no corresponding task defined in
[`advanced_usage/configs/tasks.yaml`](advanced_usage/configs/tasks.yaml):

| Mount path | Dataset | Notes |
|---|---|---|
| `/mnt/nfs03-R6/CAMELYON16/` | CAMELYON16 | Tumor detection benchmark; not in Patho-Bench task registry |
| `/mnt/nfs03-R6/CAMELYON17/` | CAMELYON17 | Multi-center tumor detection; not in Patho-Bench task registry |
| `/mnt/nfs03-R6/GTEx/` | GTEx | Normal tissue atlas; not in Patho-Bench task registry |
| `/mnt/nfs03-R6/HEST-1k/` | HEST-1k | Spatial transcriptomics + histology; not in Patho-Bench task registry |
| `/mnt/nfs03-R6/Yale_breast/` | Yale Breast (HER2 + trastuzumab response cohorts + TCGA-BRCA) | Not directly mapped to a Patho-Bench key |
| `/mnt/nfs03-R6/mhist/` | MHIST | Colorectal polyp classification; not in Patho-Bench task registry |

---

## Patho-Bench Datasets NOT Found on /mnt/nfs03-R6

These datasets are fully supported by Patho-Bench (splits on HuggingFace) but their raw slides
are not present in this NFS share:

| Patho-Bench key | Tasks |
|---|---|
| `bcnb` | `er`, `pr`, `her2` |
| `boehmk_` | `PFS` |
| `cptac_brca` | `PIK3CA_mutation`, `TP53_mutation` *(raw slides may be in /mnt/nfs03-R6/TCGA/)* |
| `cptac_coad` | `KRAS_mutation`, `TP53_mutation` *(raw slides may be in /mnt/nfs03-R6/TCGA/)* |
| `ebrains` | `diagnosis`, `diagnosis_group` |
| `mbc_` | `Recist`, `OS` |
| `mut-het-rcc` | `BAP1_mutation`, `PBRM1_mutation`, `SETD2_mutation` *(raw slides may be TCGA-KIRC)* |
| `nadt` | `response` |
| `natbrca` | `lymphovascular_invasion` |
| `ovarian` | `response` |
| `panda` | `isup_grade` |
| `sr386_` | `braf_mutant_binary`, `ras_mutant_binary`, `mmr_loss_binary`, `died_within_5_years`, `OS` |

---

## Quick Reference

```
/mnt/nfs03-R6/
├── BRACS/           → bracs (slidelevel_fine, slidelevel_coarse)
├── CAMELYON16/      → NOT in Patho-Bench
├── CAMELYON17/      → NOT in Patho-Bench
├── CPTAC/
│   ├── CCRCC/       → cptac_ccrcc (BAP1_mutation, PBRM1_mutation, OS)
│   ├── GBM/         → cptac_gbm   (EGFR_mutation, TP53_mutation)
│   ├── HNSCC/       → cptac_hnsc  (CASP8_mutation, OS)
│   ├── LSCC/        → cptac_lscc  (KEAP1_mutation, ARID1A_mutation)
│   ├── LUAD/        → cptac_luad  (EGFR_mutation, STK11_mutation, TP53_mutation, OS)
│   ├── PDA/         → cptac_pda   (SMAD4_mutation, OS)
│   ├── AML/         → NOT in Patho-Bench
│   ├── CM/          → NOT in Patho-Bench
│   ├── SAR/         → NOT in Patho-Bench
│   └── UCEC/        → NOT in Patho-Bench
├── GTEx/            → NOT in Patho-Bench
├── HEST-1k/         → NOT in Patho-Bench
├── IMP_CRC/         → imp (grade)
├── TCGA/            → raw slides; backs cptac_brca, cptac_coad, mut-het-rcc
├── Yale_breast/     → NOT directly in Patho-Bench
└── mhist/           → NOT in Patho-Bench
```
