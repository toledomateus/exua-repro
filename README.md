# Rethinking Uncertainty Attribution for Sequential Recommendation: A Reproducibility and Fairness Study

This repository contains the code and experiments for reproducing and extending the study on uncertainty quantification (UQ) methods for sequential recommendation systems. It covers three UQ paradigms — **MC Dropout**, **Dirichlet-based methods** (Belief Matching & Evidential Deep Learning), and **Deep Ensembles** — all built on top of the SASRec backbone via RecBole.

---

## Datasets

Three datasets are used across two preprocessing variants each — a 5-core filtered protocol and an unfiltered (0-core) protocol. The Avg/User column reflects the varying density regimes evaluated in RQ2.

| Dataset | Condition | Users | Items | Interactions | **Avg/User** | Sparsity |
|---|---|---:|---:|---:|---:|---:|
| MovieLens | Unfiltered | 6,041 | 3,884 | 1,000,209 | **165.60** | 95.70% |
| MovieLens | Filtered | 6,041 | 3,417 | 999,611 | **165.50** | 95.16% |
| Amazon Office | Unfiltered | 909,315 | 134,839 | 1,243,186 | **1.37** | 99.99% |
| Amazon Office | Filtered | 4,906 | 2,421 | 53,258 | **10.86** | 99.60% |
| Amazon Beauty | Unfiltered | 1,210,272 | 259,205 | 2,023,070 | **1.67** | 99.99% |
| Amazon Beauty | Filtered | 22,364 | 12,102 | 198,502 | **8.88** | 99.92% |

---

## Results

ALl the findings and statistical results are salved into results folder.

## Requirements

### Environment Setup (conda)

We recommend using conda to manage the environment. The steps below will create a clean environment and install RecBole along with its dependencies.

```bash
# Create and activate a new conda environment
conda create -n exua python=3.9 -y
conda activate exua

# Install PyTorch (adjust the CUDA version to match your setup)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install RecBole
pip install recbole
```

> If you prefer a different package manager (pip, poetry, or installing from source), refer to the [official RecBole installation guide](https://recbole.io/docs/get_started/install.html).

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

### Step 1 — (Optional) Train a SASRec Baseline from Scratch

If you prefer to train your own baseline instead of using the pre-trained checkpoints above:

```bash
cd scripts
python baseline-sasrec-2020.py   # replace with desired seed: 2020–2024
```

Baseline results are saved in `results/baselines/`.

### Step 2 — Run Uncertainty Quantification

Each UQ script supports two modes via `--mode`:
- `uncertainty` — evaluates the model, reports uncertainty stats, and computes the **mURR** metric (see [Metrics](#metrics) below).
- `posthoc` — runs uncertainty-aware post-hoc fine-tuning and reports the resulting metrics.

#### MC Dropout (MCD)

```bash
# Uncertainty analysis only
python scripts/sasrec_mcd.py \
  --checkpoint saved/SASRec-Feb-18-2026_10-53-03.pth \
  --dataset ml-1m \
  --mode uncertainty

# Post-hoc training
python scripts/sasrec_mcd.py \
  --checkpoint saved/SASRec-Feb-18-2026_10-53-03.pth \
  --dataset ml-1m \
  --mode posthoc \
  --save_path saved/sasrec_mcd_output.pth
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `None` | Path to SASRec `.pth` file. Omit to train from scratch. |
| `--dataset` | `ml-1m` | Dataset name (`ml-1m`, `amazon-office-products`, `amazon-beauty`) |
| `--mode` | `uncertainty` | `uncertainty` or `posthoc` |
| `--save_path` | auto-timestamped | Output path for post-hoc model |
| `--seed` | `2020` | Random seed |

#### Dirichlet UQ — Belief Matching & EDL

```bash
# Belief Matching, uncertainty only
python scripts/sasrec_dirichlet.py \
  --checkpoint saved/SASRec-Feb-18-2026_10-53-03.pth \
  --method bm \
  --dataset ml-1m \
  --mode uncertainty

# Evidential Deep Learning, post-hoc training
python scripts/sasrec_dirichlet.py \
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
| `--dataset` | `ml-1m` | Dataset name (`ml-1m`, `amazon-office-products`, `amazon-beauty`) |
| `--mode` | `uncertainty` | `uncertainty` or `posthoc` |
| `--warmup_epochs` | `10` | Epochs to train evidence head only before ExUA. Set to `0` if loading a checkpoint from `dirichlet_train_table1.py`. |
| `--max_epochs` | `30` | Maximum post-hoc training epochs |
| `--kl_weight` | (config) | KL divergence loss weight |

#### Deep Ensembles (DE)

The ensemble script expects **N separate SASRec checkpoints**, one per ensemble member.

```bash
# Uncertainty analysis
python scripts/sasrec_de.py \
  --checkpoints saved/SASRec-Mar-11-2026_13-03-57.pth \
                saved/SASRec-Mar-11-2026_13-09-16.pth \
                saved/SASRec-Mar-11-2026_13-09-48.pth \
  --dataset office \
  --mode uncertainty

# Post-hoc training
python scripts/sasrec_de.py \
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


> All scripts has their version using total Uncertainty (per default), Aleatoric Uncertainty```_ua.py```and Epistemic Uncertainty ```_ue.py```. Also their ablation versions ```_a1.py``` and ```_alpha0.py```.


### Step 3 — Compute Beyond-Accuracy Metrics

After running a UQ script, you can evaluate the popularity bias of any trained model using the scripts in `scripts/metrics/`. Each script corresponds to a UQ method (MCD, Dirichlet, DE) and computes the full set of beyond-accuracy metrics described below.

Open the relevant script and set the checkpoint path(s) to the model you want to evaluate, then run it:

```bash
# MC Dropout
python scripts/metrics/run_fairness_metrics_mcd.py

# Dirichlet (EDL / BM)
python scripts/metrics/run_fairness_metrics_dirichlet.py

# Deep Ensembles
python scripts/metrics/run_fairness_metrics_de.py
```

In each script, update the `checkpoint` (or `checkpoint_paths` for DE) variable to point to the model you want to evaluate. Results are printed to stdout.

---

## Metrics

### Ranking

NDCG@20, Hit@20, Recall@20, MRR@20, Precision@20

### Uncertainty

**mURR** (Maximum Uncertainty Reduction Rate) measures the maximum relative reduction in predictive uncertainty achieved by the post-hoc training. Obtained by running any UQ script with `--mode uncertainty`.

Total uncertainty U_T decomposes additively into aleatoric uncertainty U_A (irreducible data noise) and epistemic uncertainty U_E (model uncertainty reducible with more data):

- **U_T** — Shannon entropy of the predictive distribution: H[p(y | x, D)]
- **U_A** — expected entropy over the parameter posterior: E[H[p(y | x, θ)]]
- **U_E** — mutual information between prediction and parameters: U_T − U_A

For sampling-based methods (DE, MCD), these are estimated empirically across K forward passes. For Dirichlet-based methods (EDL, BM), all three components are computed analytically from the concentration parameters α in a single forward pass.

#### Beyond-Accuracy — Fairness

**PopREO — Popularity-based Ranking Equal Opportunity** measures exposure disparity between popular and long-tail items. Specifically, it compares the true-positive rate P(R@k | group, y=1) — the probability of being recommended given the user actually likes the item — across short-head and long-tail groups. Lower is better.

**PopRSP — Popularity-based Ranking Statistical Parity** measures success-rate disparity between popular and long-tail items. It compares P(R@k | group) — the probability of being recommended at all — across groups, regardless of user preference. Lower is better.

Items are split into **short-head** (top 20% by training interaction count) and **long-tail** (remaining 80%). Both metrics are implemented in `fairness_metrics.py`.

| Metric | Direction | Interpretation |
|--------|-----------|----------------|
| PopREO | ↓ lower is better | Equal true-positive rate across popularity groups |
| PopRSP | ↓ lower is better | Equal recommendation probability across popularity groups |

#### Beyond-Accuracy — Diversity

The following metrics are computed natively by RecBole's evaluator. See the [RecBole metrics documentation](https://recbole.io/docs/recbole/recbole.evaluator.metrics.html) for full details.

**GiniIndex** — measures inequality in the item recommendation frequency distribution across all users. A value of 0 means all items are recommended equally; a value of 1 means recommendations are entirely concentrated on a single item. Lower is better.

**ItemCoverage** — proportion of distinct catalogue items that appear in at least one top-K recommendation list across all users. Higher values indicate broader catalogue utilisation.

| Metric | Direction | Interpretation |
|--------|-----------|----------------|
| GiniIndex | ↓ lower is better | Equality of item recommendation frequency |
| ItemCoverage | ↑ higher is better | Catalogue breadth covered by recommendations |

#### Beyond-Accuracy — Popularity Bias

**ARP — Average Recommendation Popularity** (Abdollahpouri et al., 2017) — average training interaction count of recommended items, averaged over all users. Lower values indicate the recommender favours less-popular, long-tail items. Implemented in `fairness_metrics.py`.

**APLT — Average Percentage of Long-Tail items** (Abdollahpouri et al., 2017) — average fraction of long-tail items per recommendation list across users. Higher values indicate more long-tail exposure per user. Implemented in `fairness_metrics.py`.

**ACLT — Average Coverage of Long-Tail items** (Abdollahpouri et al., 2019) — total long-tail item appearances across all recommendation lists, averaged over users. Unlike APLT, this counts repeated appearances and reflects aggregate long-tail catalogue exposure. Implemented in `fairness_metrics.py`.

**AveragePopularity** — RecBole's native equivalent of ARP: mean training interaction count of recommended items. Enabled via the `metrics` field in the config.

**TailPercentage** — RecBole's native beyond-accuracy metric for long-tail exposure. Average fraction of long-tail items per list, where the long-tail threshold is controlled by the `tail_ratio` config parameter (set to 0.1 in our experiments).

| Metric | Direction | Interpretation |
|--------|-----------|----------------|
| ARP | ↓ lower is better | Avg. popularity of recommended items |
| APLT | ↑ higher is better | Fraction of long-tail items per list |
| ACLT | ↑ higher is better | Total long-tail coverage across all lists |
| AveragePopularity | ↓ lower is better | Avg. popularity of recommended items (RecBole) |
| TailPercentage | ↑ higher is better | Fraction of long-tail items per list (RecBole) |

---

## Reproducibility

All experiments are reproducible across five seeds:

```
2020  2021  2022  2023  2024
```

Pass `--seed <value>` to any script to select the seed. The pre-trained checkpoints in `saved/` correspond to these exact seeds per dataset/variant (see checkpoint tables above or the ones appointed in the rq1.csv or rq2.csv files).

---