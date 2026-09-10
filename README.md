<h1 align="center">
  AxonAD
</h1>

<div align="center">

# Predictable Query Dynamics for Time-Series Anomaly Detection

[![Springer](https://img.shields.io/badge/Springer-Published-6DB33F.svg?style=for-the-badge)](https://link.springer.com/chapter/10.1007/978-3-032-37685-5_13)
[![Best Paper Award](https://img.shields.io/badge/ECML--PKDD%202026-Best%20Paper%20Award-gold.svg?style=for-the-badge)](#)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)

[**Kadir-Kaan Özer**](#)<sup>1,2</sup> · [**René Ebeling**](#)<sup>1</sup> · [**Markus Enzweiler**](#)<sup>2</sup>

<sup>1</sup> **Mercedes-Benz AG, Germany** · <sup>2</sup> **Institute for Intelligent Systems, Esslingen University of Applied Sciences, Germany**

---

[![Springer](https://img.shields.io/badge/Springer-Published-6DB33F.svg?style=plastic)](https://link.springer.com/chapter/10.1007/978-3-032-37685-5_13)
[![arXiv](https://img.shields.io/badge/arXiv-2603.12916-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2603.12916)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/papers/2603.12916)
[![Best Paper](https://img.shields.io/badge/🏆%20Best%20Paper-Applied%20Data%20Science-gold.svg?style=plastic)](#)

**Published at ECML PKDD 2026, Applied Data Science Track · 🏆 Best Paper Award**

</div>

## 📢 Updates
- **[May 29, 2026]** Code released
- **[May 28, 2026]** 🎉 Paper accepted to **ECML-PKDD 2026 Applied Data Science Track**
- **[Coming Soon]** We will release the full training and evaluation scripts upon paper acceptance.

## Overview

This repository contains the code for **AxonAD**, introduced in **Predictable Query Dynamics for Time-Series Anomaly Detection**.

The codebase is built on top of **TSB-AD** and keeps its benchmark structure, datasets, and evaluation pipeline, while adding the **AxonAD** model, paper-specific experiment scripts, ablations, sensitivity studies, mechanistic analysis, and failure-mode analysis.

If your goal is to **reproduce the paper numbers**, the main entry point is the **root `main.py`** file. The other scripts are supporting utilities for tuning, ablations, post-processing, and deeper analysis.

## What is in this repository?

| Path | Purpose |
| --- | --- |
| `main.py` | Primary reproduction pipeline for benchmark CSV generation |
| `TSB_AD/models/AxonAD.py` | AxonAD model implementation |
| `TSB_AD/model_wrapper.py` | Benchmark wrappers and dispatch logic |
| `TSB_AD/HP_list.py` | Hyperparameter grids and selected benchmark defaults |
| `TSB_AD/main.py` | Single-series runner |
| `benchmark_exp/Run_Detector_M.py` | Multivariate benchmark runner |
| `benchmark_exp/Run_Detector_U.py` | Univariate benchmark runner |
| `benchmark_exp/HP_Tuning_M.py` | Multivariate tuning runner |
| `benchmark_exp/HP_Tuning_U.py` | Univariate tuning runner |
| `run_AxonAD_ablations.py` | AxonAD ablation study |
| `sweep_axonad_tuning_sensitivity.py` | AxonAD sensitivity sweep |
| `postprocess_axonad_results.py` | AxonAD post-processing and comparison tables |
| `aggregate_results.py` | Aggregates per-series benchmark CSVs |
| `run_mechanistic_auto.py` | Mechanistic analysis pipeline |
| `analyze_model_failures.py` | Failure-mode analysis |

## Installation

We recommend **Python 3.11**.

```bash
git clone <your-fork-url>
cd Eval_Scripts-main

conda create -n axonad python=3.11 -y
conda activate axonad

pip install -r requirements.txt
pip install -e .
```

If `torch` installation via `pip` is problematic on your machine, install it separately first and then continue with the remaining requirements.

## Dataset setup

This repository expects the official **TSB-AD** datasets and uses the split CSV files under `Datasets/File_List/`.

Download and unpack the datasets so the structure looks like:

```text
Datasets/
├── File_List/
│   ├── TSB-AD-M-Eva.csv
│   ├── TSB-AD-M-Tuning.csv
│   ├── TSB-AD-M-Telemetry.csv
│   ├── TSB-AD-U-Eva.csv
│   ├── TSB-AD-U-Eva-Full.csv
│   └── TSB-AD-U-Tuning.csv
├── TSB-AD-M/
└── TSB-AD-U/
```

For semisupervised runs, the train prefix is inferred from the `_tr_<index>_` token embedded in each filename.

## Reproducing the paper results

The **root `main.py`** is the intended reproduction script for the paper results. It discovers the datasets, applies the official split filters, runs the selected detectors, and writes the paper-style benchmark CSV outputs.

### 1. Configure `main.py`

Edit the configuration block near the top of `main.py`:

- `AD_NAMES`
- `BENCHMARK`
- `SPLIT_MODE`
- `DATA_ROOTS`

For an **AxonAD-only** run, the minimal setup is:

```python
AD_NAMES = ["AxonAD"]
BENCHMARK = "M"        # or "U"
SPLIT_MODE = "eval"    # or "tuning"
DATA_ROOTS = ["Datasets/TSB-AD-M"]   # or ["Datasets/TSB-AD-U"]
```

For a **comparison table** against other models, keep `AxonAD` in `AD_NAMES` and add the baselines you want to evaluate in the same run.

### 2. Run the benchmark pipeline

```bash
python main.py
```

This writes repository-level result files such as:

```text
tsb_ad_m_eval_results.csv
tsb_ad_u_tuning_results.csv
```

These CSVs are the main paper-reproduction artifacts.

### 3. Aggregate the results

```bash
python aggregate_results.py
```

### 4. Build AxonAD-centered comparison tables

```bash
python postprocess_axonad_results.py \
  --input tsb_ad_m_eval_results.csv \
  --benchmark M \
  --split_mode eval \
  --reference_model AxonAD \
  --out_dir analysis/axonad_post
```

## Quick sanity check

Before launching a full benchmark, you can run AxonAD on a single series:

```bash
python -m TSB_AD.main \
  --AD_Name AxonAD \
  --data_direc Datasets/TSB-AD-U/ \
  --filename 001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.csv
```

## Supporting experiment scripts

The scripts below are useful for the rest of the paper, but they are **not** the primary reproduction path for the final benchmark CSVs.

### Hyperparameter tuning

Multivariate:

```bash
python benchmark_exp/HP_Tuning_M.py \
  --AD_Name AxonAD \
  --dataset_dir Datasets/TSB-AD-M/ \
  --file_lsit Datasets/File_List/TSB-AD-M-Tuning.csv \
  --save_dir outputs/hp_tuning/multi
```

Univariate:

```bash
python benchmark_exp/HP_Tuning_U.py \
  --AD_Name AxonAD \
  --dataset_dir Datasets/TSB-AD-U/ \
  --file_lsit Datasets/File_List/TSB-AD-U-Tuning.csv \
  --save_dir outputs/hp_tuning/uni
```

### Direct benchmark runners

If you want to run only one benchmark split without using the root reproduction script:

```bash
python benchmark_exp/Run_Detector_M.py \
  --AD_Name AxonAD \
  --dataset_dir Datasets/TSB-AD-M/ \
  --file_lsit Datasets/File_List/TSB-AD-M-Eva.csv \
  --score_dir outputs/score/multi \
  --save_dir outputs/metrics/multi \
  --save True
```

```bash
python benchmark_exp/Run_Detector_U.py \
  --AD_Name AxonAD \
  --dataset_dir Datasets/TSB-AD-U/ \
  --file_lsit Datasets/File_List/TSB-AD-U-Eva.csv \
  --score_dir outputs/score/uni \
  --save_dir outputs/metrics/uni \
  --save True
```

### AxonAD ablations

```bash
python run_AxonAD_ablations.py \
  --benchmark M \
  --split_mode tuning
```

Optional quick debug run:

```bash
python run_AxonAD_ablations.py \
  --benchmark M \
  --split_mode tuning \
  --max_datasets 10
```

This writes a CSV such as:

```text
tsb_ad_m_tuning_axonad_ablations_results.csv
```

### Sensitivity sweep

```bash
python sweep_axonad_tuning_sensitivity.py \
  --dataset_dir Datasets/TSB-AD-M \
  --file_list Datasets/File_List/TSB-AD-M-Tuning.csv \
  --out_dir outputs_sweeps \
  --mode oat
```

### Mechanistic analysis

```bash
python run_mechanistic_auto.py
```

Configuration lives in the `CFG` block near the top of the file.

### Failure-mode analysis

```bash
python analyze_model_failures.py
```

Outputs are written under `Results/analysis_failure_modes/`.

## Metrics

The repository uses the TSB-AD evaluation pipeline and reports the benchmark metrics used in the paper, including:

- `AUC-PR`
- `AUC-ROC`
- `VUS-PR`
- `VUS-ROC`
- `Standard-F1`
- `PA-F1`
- `Event-based-F1`
- `R-based-F1`
- `Affiliation-F`

## Citation

If you use this repository, please cite:

```bibtex
@misc{ozer2026predictable,
  title={Predictable Query Dynamics for Time-Series Anomaly Detection},
  author={Kadir-Kaan Özer and René Ebeling and Markus Enzweiler},
  year={2026},
  eprint={2603.12916},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2603.12916}
}
```

## Acknowledgment

This repository builds on top of **TSB-AD**. We thank the TSB-AD authors for releasing the benchmark, datasets, and evaluation pipeline that make direct and reproducible comparisons possible.

Or describe it in Issues.

### 🎉 Acknowledgement
We appreciate the following github repos a lot for their valuable code base:
* https://github.com/thedatumorg/TSB-AD
* https://github.com/yzhao062/pyod
* https://github.com/TimeEval/TimeEval-algorithms
* https://github.com/thuml/Time-Series-Library/
* https://github.com/dawnvince/EasyTSAD
