# Rethinking Uncertainty Attribution for Sequential Recommendation: A Reproducibility and Fairness Study

This repository contains the code and experiments for reproducing and extending the study on uncertainty quantification (UQ) methods for sequential recommendation systems. It covers three UQ paradigms — **MC Dropout**, **Dirichlet-based methods** (Belief Matching & Evidential Deep Learning), and **Deep Ensembles** — all built on top of the SASRec backbone via RecBole.

---

## Project Structure

```
exua-repro/
├── scripts/
│   ├── baseline-sasrec-*.py          # SASRec baseline scripts for seeds 2020-2024
│   ├── uncertainty_quantification/   # UQ method implementations
│   │   ├── ut_methods.py             # Uncertainty from Targets (UT)
│   │   ├── ua_methods.py             # Uncertainty Attribution (UA)
│   │   ├── ue_methods.py             # Uncertainty from Embeddings (UE)
│   │   └── ablation_methods.py       # Ablation studies (α=1, α=0)
│   └── run_experiments.py
├── data/
│   └── raw/                          # Raw datasets
├── results/
│   ├── rq1.csv                       # RQ1 results
│   └── rq2.csv                       # RQ2 results
│
├── saved/                            # Checkpoints (Pre-trained .pth files)
│   
│
└── README.md
---

## Datasets

Three datasets are used across two preprocessing variants each:

| Dataset | Variant |
|---|---|---|
| **MovieLens-1M** | `default` | 
| **MovieLens-1M** | `filtered` |
| **Amazon Office** | `default` | 
| **Amazon Office** | `filtered` |
| **Amazon Beauty** | `default` | 
| **Amazon Beauty** | `filtered` |

---

## Pre-trained SASRec Checkpoints

All SASRec baselines are pre-trained and stored in `saved/`. Use the paths below when passing `--checkpoint` to any UQ script.

### MovieLens-1M — Default

| Seed | Checkpoint path |
|------|----------------|
| 2020 | `saved/SASRec-Feb-18-2026_10-53-03.pth` |
| 2021 | `saved/SASRec-Feb-17-2026_12-15-15.pth` |
| 2022 | `saved/SASRec-Mar-22-2026_13-57-46.pth` |
| 2023 | `saved/SASRec-Mar-23-2026_14-28-26.pth` |
| 2024 | `saved/SASRec-Mar-24-2026_07-23-07.pth` |

### MovieLens-1M — Filtered

| Seed | Checkpoint path |
|------|----------------|
| 2020 | `saved/SASRec-Mar-19-2026_09-59-32.pth` |
| 2021 | `saved/SASRec-Mar-18-2026_11-55-28.pth` |
| 2022 | `saved/SASRec-Mar-18-2026_11-55-54.pth` |
| 2023 | `saved/SASRec-Mar-18-2026_11-56-10.pth` |
| 2024 | `saved/SASRec-Mar-18-2026_11-56-30.pth` |

### Amazon Office — Default

| Seed | Checkpoint path |
|------|----------------|
| 2020 | `saved/SASRec-Mar-11-2026_13-03-57.pth` |
| 2021 | `saved/SASRec-Mar-11-2026_13-09-16.pth` |
| 2022 | `saved/SASRec-Mar-11-2026_13-09-48.pth` |
| 2023 | `saved/SASRec-Mar-11-2026_13-10-06.pth` |
| 2024 | `saved/SASRec-Mar-11-2026_13-10-16.pth` |

### Amazon Office — Filtered

| Seed | Checkpoint path |
|------|----------------|
| 2020 | `saved/SASRec-Mar-12-2026_10-56-38.pth` |
| 2021 | `saved/SASRec-Mar-13-2026_10-25-01.pth` |
| 2022 | `saved/SASRec-Mar-13-2026_10-27-28.pth` |
| 2023 | `saved/SASRec-Mar-13-2026_10-27-56.pth` |
| 2024 | `saved/SASRec-Mar-13-2026_10-28-19.pth` |

### Amazon Beauty — Default

| Seed | Checkpoint path |
|------|----------------|
| 2020 | `saved/SASRec-Mar-26-2026_20-47-33.pth` |
| 2021 | `saved/SASRec-Mar-11-2026_13-29-59.pth` |
| 2022 | `saved/SASRec-Mar-11-2026_13-30-10.pth` |
| 2023 | `saved/SASRec-Mar-11-2026_13-30-17.pth` |
| 2024 | `saved/SASRec-Mar-11-2026_13-30-24.pth` |

### Amazon Beauty — Filtered

| Seed | Checkpoint path |
|------|----------------|
| 2020 | `saved/SASRec-Mar-25-2026_15-17-11.pth` |
| 2021 | `saved/SASRec-Mar-25-2026_15-17-56.pth` |
| 2022 | `saved/SASRec-Mar-25-2026_15-18-09.pth` |
| 2023 | `saved/SASRec-Mar-25-2026_15-19-28.pth` |
| 2024 | `saved/SASRec-Mar-25-2026_15-19-37.pth` |

---

## Quick Start

### Requirements

Install recbole as in [their guideline](https://recbole.io/docs/get_started/install.html)

### Step 1 — (Optional) Train a SASRec Baseline from Scratch

If you prefer to train your own baseline instead of using the pre-trained checkpoints above:

```bash
cd scripts
python baseline-sasrec-2020.py   # replace with desired seed: 2020–2024
```

Baseline results are saved in `results/baselines/`.

### Step 2 — Run Uncertainty Quantification

Each UQ script supports two modes via `--mode`:
- `uncertainty` — evaluates the model and reports uncertainty stats
- `posthoc` — runs uncertainty-aware post-hoc fine-tuning and reports before/after metrics

---

## MC Dropout (MCD)

```bash
# Table 1 — uncertainty analysis only
python sasrec_mcd.py \
  --checkpoint saved/SASRec-Feb-18-2026_10-53-03.pth \
  --dataset ml-1m \
  --mode uncertainty

# Table 3 — post-hoc training
python saserc_mcd.py \
  --checkpoint saved/SASRec-Feb-18-2026_10-53-03.pth \
  --dataset ml-1m \
  --mode posthoc \
  --save_path saved/sasrec_mcd_output.pth
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `None` | Path to SASRec `.pth` file. Omit to train from scratch. |
| `--dataset` | `ml-1m` | Dataset name (`ml-1m`, `office`, `beauty`) |
| `--mode` | `uncertainty` | `uncertainty` or `posthoc` |
| `--save_path` | auto-timestamped | Output path for post-hoc model |
| `--seed` | `2020` | Random seed |

---

## Dirichlet UQ — Belief Matching & EDL

```bash
# Belief Matching, uncertainty only
python sasrec_dirichlet.py \
  --checkpoint saved/SASRec-Feb-18-2026_10-53-03.pth \
  --method bm \
  --dataset ml-1m \
  --mode uncertainty

# Table 3 — Evidential Deep Learning, post-hoc training
python sasrec_dirichlet.py \
  --checkpoint saved/SASRec-Feb-18-2026_10-53-03.pth \
  --method edl \
  --dataset ml-1m \
  --mode posthoc \
  --save_path saved/sasrec_dirichlet_edl_output.pth
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `None` | Path to SASRec `.pth` file. Omit to train from scratch. |
| `--method` | `bm` | `bm` (Belief Matching) or `edl` (Evidential Deep Learning) |
| `--dataset` | `ml-1m` | Dataset name |
| `--mode` | `uncertainty` | `uncertainty` or `posthoc` |
| `--warmup_epochs` | `10` | Epochs to train evidence head only before ExUA. Set to `0` if loading a checkpoint from `dirichlet_train_table1.py`. |
| `--max_epochs` | `30` | Maximum post-hoc training epochs |
| `--kl_weight` | (config) | KL divergence loss weight |

---

## Deep Ensembles (DE)

The ensemble script expects **N separate SASRec checkpoints**, one per ensemble member.

```bash
# Table 1 — uncertainty analysis
python sasrec_de.py \
  --checkpoints saved/SASRec-Mar-11-2026_13-03-57.pth \
                saved/SASRec-Mar-11-2026_13-09-16.pth \
                saved/SASRec-Mar-11-2026_13-09-48.pth \
  --dataset office \
  --mode uncertainty

# Table 3 — post-hoc training
python sasrec_de.py \
  --checkpoints saved/SASRec-Mar-11-2026_13-03-57.pth \
                saved/SASRec-Mar-11-2026_13-09-16.pth \
                saved/SASRec-Mar-11-2026_13-09-48.pth \
  --dataset office \
  --mode posthoc \
  --save_dir saved/ensemble/office/posthoc
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoints` | `None` | Space-separated paths to N `.pth` files. Omit to train from scratch. |
| `--n_models` | (config) | Number of ensemble members (auto-adjusted to match `--checkpoints`) |
| `--dataset` | `ml-1m` | Dataset name |
| `--mode` | `uncertainty` | `uncertainty` or `posthoc` |
| `--save_dir` | `saved/ensemble/...` | Directory to save post-hoc ensemble checkpoints |

---

## Evaluation Modes Explained

Each entry point runs in one of two modes, controlled by `--mode`:

**`uncertainty` (Table 1 — baseline comparison)**
1. Loads the SASRec checkpoint and wraps it with the chosen UQ method.
2. Reports uncertainty decomposition: Total (U_T), Aleatoric (U_A), Epistemic (U_E) on the first test batch.
3. Evaluates ranking and fairness metrics on the full test set.
4. Computes the **mURR** (mean Uncertainty-Reranking Ratio) score.

**`posthoc` (Table 3 — ExUA fine-tuning)**
1. Runs all of the above as a baseline snapshot.
2. Performs uncertainty-aware post-hoc fine-tuning (ExUA loss: BPR + α · ExUA term).
3. Re-evaluates and prints a side-by-side before/after comparison.

---

## Metrics

**Ranking:** Recall@{10,20}, MRR@{10,20}, NDCG@{10,20}, Hit@{10,20}, Precision@{10,20}

**Fairness:** ItemCoverage, ShannonEntropy, GiniIndex, AveragePopularity, TailPercentage

**Uncertainty:** mURR (mean Uncertainty-Reranking Ratio, K=5 or K=10 depending on method)

---

## Reproducibility

All experiments are reproducible across five seeds:

```
2020  2021  2022  2023  2024
```

Pass `--seed <value>` to any script to select the seed. The pre-trained checkpoints in `saved/` correspond to these exact seeds per dataset/variant (see checkpoint tables above).

---