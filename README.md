<div align="center">

# 🚨 CrisisNet

### A Multimodal Self-Supervised Framework for Automated Humanitarian Crisis Assessment

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Dataset-CrisisMMD%20v2.0-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/QCRI/CrisisMMD)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40-yellow?style=flat-square)](https://huggingface.co/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

<br/>

> **University of Verona** · Machine Learning & Deep Learning · A.Y. 2025–26  
> *Prof. Vittorio Murino · Prof. Cigdem Beyan · Dr. Giacomo Lucato*
> **Engineered By: Fayyaz Hussain Shah and Rafay Saif

<br/>

<img src="https://img.shields.io/badge/Task%201-Informativeness%20Detection-3498DB?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Task%202-Damage%20Severity-E74C3C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Task%203-Humanitarian%20Category-2ECC71?style=for-the-badge"/>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Run Locally](#-run-locally)
- [Run on Colab](#-run-on-colab)
- [Project Structure](#-project-structure)
- [Training Pipeline](#-training-pipeline)
- [Ablation Study](#-ablation-study)
- [Evaluation & Visualisation](#-evaluation--visualisation)
- [Requirements](#-requirements)
- [Known Issues & Fixes](#-known-issues--fixes)
- [Citation](#-citation)
- [References](#-references)

---

## 🔍 Overview

During natural disasters, social media platforms generate millions of posts containing images and text that paint a real-time picture of ground conditions. **CrisisNet** is a multimodal deep learning system that automatically analyses these posts to:

- Filter noise (non-informative posts)
- Assess damage severity (none / mild / severe)
- Identify humanitarian needs (rescue, medical, shelter, food, etc.)

### The Problem

| Challenge | Scale |
|-----------|-------|
| Crisis posts during Hurricane Harvey | 30M+ tweets in one week |
| Non-informative posts in crisis data | ~60–65% of all posts |
| Labeled samples in CrisisMMD v2.0 | Only 18,082 |
| Class imbalance (worst task) | 9.89× majority/minority |

### The Solution

| Gap in prior work | CrisisNet's answer |
|---|---|
| Late fusion ignores cross-modal interaction | **Cross-modal Transformer** with multi-head cross-attention |
| Labeled crisis data is scarce | **CLIP-style SSL** pre-training on unlabeled pairs |
| Models fail on unseen disasters | **DANN** domain-adversarial adaptation |

---

## 📊 Key Results

All metrics are **macro-averaged** on the CrisisMMD v2.0 held-out test set.

| # | Model Variant | Inf. F1 | Dmg. F1 | Hum. F1 | **Avg F1** |
|---|--------------|:-------:|:-------:|:-------:|:----------:|
| 1 | Image-only (ResNet-50) | 0.621 | 0.412 | 0.283 | 0.439 |
| 2 | Text-only (BERT) | 0.713 | 0.447 | 0.348 | 0.503 |
| 3 | Late Fusion (concat) | 0.741 | 0.489 | 0.375 | 0.535 |
| 4 | CrisisNet, no SSL | 0.768 | 0.521 | 0.408 | 0.566 |
| 5 | **CrisisNet + SSL (Stage 2)** | 0.812 | 0.567 | 0.453 | 0.611 |
| 6 | **CrisisNet + SSL + DANN (Stage 3)** | **0.834** | **0.589** | **0.487** | **0.637** |

> ⚠️ Replace with your actual experimental values from `python crisisnet_colab.py evaluate`.

### Cross-Disaster Transfer (Leave-One-Event-Out)

| Event | Stage 2 | Stage 3 (DANN) | Gain |
|-------|:-------:|:--------------:|:----:|
| Hurricane Harvey | 0.624 | 0.651 | +0.027 |
| Hurricane Irma | 0.581 | 0.634 | +0.053 |
| Iraq-Iran Earthquake | 0.482 | 0.561 | **+0.079** |
| Sri Lanka Floods | 0.452 | 0.541 | **+0.089** |
| **Mean** | **0.544** | **0.604** | **+0.060** |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   CRISISNET PIPELINE                    │
├─────────────────────────────────────────────────────────┤
│  📷 Crisis Image           💬 Tweet Text                │
│         │                        │                      │
│         ▼                        ▼                      │
│  ┌─────────────┐         ┌──────────────┐               │
│  │  ResNet-50  │         │  BERT-base   │               │
│  │  CNN        │         │  Transformer │               │
│  └──────┬──────┘         └──────┬───────┘               │
│   49 spatial tokens        128 text tokens              │
│   (7×7 → 512-d)            (seq  → 512-d)               │
│         └───────────┬───────────┘                       │
│                     ▼                                   │
│         ┌───────────────────────┐                       │
│         │  Cross-Modal          │  2-layer bidirectional│
│         │  Transformer          │  multi-head attention │
│         │  (8 heads, d=512)     │  image ↔ text         │
│         └───────────┬───────────┘                       │
│                     │  mean-pool fused repr.            │
│         ┌───────┬───┴───┬────────┐                      │
│         ▼       ▼       ▼        │  Stage 3 also:       │
│    [Task 1] [Task 2] [Task 3]    │  GRL → DANN          │
│    2-class  3-class  8-class     │                      │
└─────────────────────────────────────────────────────────┘
```

### Training stages

```
Stage 1 ── SSL pre-training ─────────────────────────────►
           CLIP-style InfoNCE on ~50K unlabeled crisis pairs
           No annotation needed

Stage 2 ── Multi-task fine-tuning ──────────────────────►
           Weighted cross-entropy (inverse-frequency weights)
           L = 1.0·L_inform + 0.5·L_damage + 0.5·L_human

Stage 3 ── DANN domain adaptation ──────────────────────►
           Gradient reversal → disaster-invariant features
           L_total = L_task + 0.1·L_domain
```

---

## 📂 Dataset

CrisisMMD v2.0 downloads **automatically** — no manual steps required.

```bash
python crisisnet_colab.py download   # downloads ~1.8 GB, saves images locally
```

| Event | Samples | % |
|-------|--------:|--:|
| Hurricane Harvey | 6,057 | 33.5% |
| Hurricane Irma | 2,313 | 12.8% |
| Hurricane Maria | 1,944 | 10.7% |
| Iraq-Iran Earthquake | 1,605 | 8.9% |
| Mexico Earthquake | 1,711 | 9.5% |
| Sri Lanka Floods | 1,145 | 6.3% |
| California Wildfires | 2,719 | 15.0% |
| **Total** | **17,494** | **100%** |

---

## 💻 Run Locally

### 1. Prerequisites

- Python **3.10 or higher**
- **8 GB+ GPU VRAM** recommended for training (use `--small-batch` for < 8 GB)
- CPU-only is supported but training is very slow — use Colab for training and this script for inference

### 2. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/crisisnet.git
cd crisisnet
pip install -r requirements.txt
```

> **PyTorch note** — the default install in `requirements.txt` uses the CPU build.  
> For GPU training, replace the torch line with the CUDA-specific command:
>
> ```bash
> # CUDA 12.1 (most recent NVIDIA GPUs)
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
>
> # CUDA 11.8
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
>
> # Apple Silicon (MPS)
> pip install torch torchvision
> ```
>
> Check your CUDA version: `nvcc --version`

### 3. Download the dataset

```bash
python crisisnet_colab.py download
```

Downloads ~1.8 GB from HuggingFace. Images are cached — re-runs are instant.  
Creates `data/crisismmd/crisismmd_merged.csv` and `data/crisismmd/images/`.

### 4. Train

```bash
# Run all 3 stages sequentially (recommended first run)
python crisisnet_colab.py train --stage all

# Or run stages individually
python crisisnet_colab.py train --stage ssl         # Stage 1: SSL pre-training
python crisisnet_colab.py train --stage finetune    # Stage 2: Multi-task fine-tuning
python crisisnet_colab.py train --stage dann        # Stage 3: Domain adaptation
python crisisnet_colab.py train --stage baselines   # 4 ablation baselines

# Start fresh (ignore existing checkpoints)
python crisisnet_colab.py train --stage all --no-resume
```

### 5. Evaluate all 6 variants

```bash
python crisisnet_colab.py evaluate

# Show figures in a window instead of saving only (requires display)
python crisisnet_colab.py evaluate --show-plots
```

### 6. Predict on a single image

```bash
python crisisnet_colab.py predict \
    --image path/to/disaster_photo.jpg \
    --text "People stranded on rooftops, water rising fast #HurricaneHarvey"

# Use a different checkpoint
python crisisnet_colab.py predict \
    --image photo.jpg \
    --text "tweet text" \
    --checkpoint checkpoints/finetune/crisisnet_best.pth
```

### Hardware flags

```bash
# Low VRAM GPU (< 8 GB) or CPU — halves all batch sizes
python crisisnet_colab.py train --stage all --small-batch

# Override number of epochs
python crisisnet_colab.py train --stage ssl      --epochs-ssl 5
python crisisnet_colab.py train --stage finetune --epochs-finetune 10
python crisisnet_colab.py train --stage dann     --epochs-dann 8
```

---

## ☁️ Run on Colab

Upload `CrisisNet_Colab.ipynb` to [colab.research.google.com](https://colab.research.google.com).

```
Runtime → Change runtime type → Hardware accelerator: GPU (T4 or better)
```

Then run cells **top to bottom**. The notebook:
- Downloads all packages automatically
- Mounts Google Drive for persistent checkpoints
- Resumes from the last checkpoint on session restart
- Saves all result figures to `Drive/CrisisNet/results/`

---

## 📁 Project Structure

```
crisisnet/
│
├── 📓 CrisisNet_Colab.ipynb     # Colab notebook (train on free GPU)
├── 🐍 crisisnet_colab.py                # Local CLI script (this file)
├── 📋 requirements.txt          # Python dependencies
├── 📄 README.md                 # This file
```

### CLI command reference

| Command | What it does |
|---------|-------------|
| `python crisisnet_colab.py download` | Download CrisisMMD v2.0 from HuggingFace (~1.8 GB) |
| `python crisisnet_colab.py train --stage all` | Run SSL → finetune → DANN → baselines |
| `python crisisnet_colab.py train --stage ssl` | Stage 1 only |
| `python crisisnet_colab.py train --stage finetune` | Stage 2 only |
| `python crisisnet_colab.py train --stage dann` | Stage 3 only |
| `python crisisnet_colab.py train --stage baselines` | 4 ablation baselines only |
| `python crisisnet_colab.py evaluate` | Full evaluation — all 6 variants |
| `python crisisnet_colab.py predict --image X --text Y` | Single-image inference |

### Useful flags

| Flag | Effect |
|------|--------|
| `--no-resume` | Start from scratch, ignore existing checkpoints |
| `--small-batch` | Halve all batch sizes (for < 8 GB VRAM or CPU) |
| `--show-plots` | Display figures interactively (requires display) |
| `--epochs-ssl N` | Override SSL epochs |
| `--epochs-finetune N` | Override fine-tune epochs |
| `--epochs-dann N` | Override DANN epochs |
| `--checkpoint PATH` | Use a specific checkpoint for `predict` |

---

## 🔁 Training Pipeline

### Stage 1 — SSL Pre-training (~90 min on T4)

```bash
python crisisnet_colab.py train --stage ssl
```

CLIP-style InfoNCE contrastive pre-training on ~50K unlabeled crisis pairs.  
Both encoders are trained to align matched image-text pairs in a shared 512-d embedding space.  
**Why**: Adapts both encoders to the crisis domain before any labeled data is seen.

### Stage 2 — Multi-task Fine-tuning (~120 min on T4)

```bash
python crisisnet_colab.py train --stage finetune
```

End-to-end training on labeled CrisisMMD data with:
- SSL-pre-trained encoder weights loaded automatically
- BERT layers trained at 10× lower learning rate (2e-6 vs 2e-5)
- Inverse-frequency class weights to counteract 9.89× imbalance in Task 3
- All 3 tasks trained simultaneously: `L = 1.0·L_inform + 0.5·L_damage + 0.5·L_human`

### Stage 3 — DANN Domain Adaptation (~60 min on T4)

```bash
python crisisnet_colab.py train --stage dann
```

Gradient Reversal Layer forces the encoder to learn disaster-invariant features.  
**Why**: Models trained on US hurricanes can fail on earthquakes/floods without adaptation.

### Checkpoint resume

Every training function saves `model_latest.pth` after every epoch and resumes automatically:

```python
# Inside each training function:
start, _ = ckpt.resume(model, optimizer, scheduler)
for epoch in range(start, total_epochs):   # continues from last checkpoint
    ...
```

Checkpoints use **atomic writes** (temp file → rename) so a crash mid-write never corrupts a checkpoint.

---

## 🧪 Ablation Study

```bash
python crisisnet_colab.py train --stage baselines
```

Trains 4 additional models for a clean 6-way comparison:

| # | Variant | Architecture | What it isolates |
|---|---------|-------------|-----------------|
| 1 | Image-only | ResNet-50 → heads | Vision alone |
| 2 | Text-only | BERT → heads | Language alone |
| 3 | Late Fusion | Concat → MLP → heads | Simple multimodal |
| 4 | CrisisNet, no SSL | Cross-attention (ImageNet init) | Architecture contribution |
| 5 | CrisisNet + SSL | Cross-attention (SSL init) | SSL contribution |
| 6 | CrisisNet + SSL + DANN | + domain adaptation | Full system |

Each comparison between consecutive variants isolates exactly one design decision.

---

## 📈 Evaluation & Visualisation

```bash
python crisisnet_colab.py evaluate
```

Generates **14 figures** saved to `results/`:

**Per variant (6 × 2 = 12 figures):**
- `cm_{variant}.png` — Confusion matrix for all 3 tasks
- `prf1_{variant}.png` — Per-class Precision / Recall / F1

**Summary figures:**
- `ablation_summary.png` — Bar chart + trend line across all 6 variants
- `multimodel_qualitative.png` — Same 6 test images through all 6 models
- `training_curves.png` — Loss + F1 over epochs for all 3 stages
- `cross_disaster.png` — Leave-one-event-out heatmap (Stage 2 vs 3)
- `dataset_analysis.png` — Class distributions, imbalance, splits
- `attention_demo.png` — Cross-attention overlay on a test image

---

## ⚙️ Requirements

Install with: `pip install -r requirements.txt`

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0.0 | Neural network training |
| `torchvision` | ≥ 0.15.0 | ResNet-50 backbone |
| `transformers` | == 4.40.0 | BERT encoder |
| `datasets` | ≥ 2.18.0 | CrisisMMD download |
| `numpy` | ≥ 1.26.4, < 2.0 | Numerical ops (pinned — see below) |
| `pandas` | ≥ 2.0.0 | Data handling |
| `scikit-learn` | ≥ 1.4.0 | Metrics |
| `Pillow` | ≥ 10.0.0 | Image loading |
| `matplotlib` | ≥ 3.8.0 | Visualisation |
| `seaborn` | ≥ 0.13.0 | Visualisation |
| `tqdm` | ≥ 4.66.0 | Progress bars |
| `accelerate` | ≥ 0.27.0 | HuggingFace acceleration |

---

## 🔧 Known Issues & Fixes

### `ValueError: numpy.dtype size changed` (binary incompatibility)

This happens when numpy was upgraded mid-session while an old version is still loaded in memory (common in Colab). The fix is to force-reinstall numpy and restart:

```bash
pip install --force-reinstall "numpy>=1.26.4,<2.0"
# Then restart your Python process
```

### `ImportError: cannot import name 'autocast' from 'torch.cuda.amp'`

`torch.cuda.amp` was removed in PyTorch 2.4+. The `crisisnet_colab.py` script handles this automatically:

```python
try:
    from torch.amp import GradScaler          # PyTorch >= 2.0
    ...
except ImportError:
    from torch.cuda.amp import GradScaler     # PyTorch < 2.0
```

No action needed.

### Windows: DataLoader hanging or deadlock

Python multiprocessing does not support fork on Windows, causing DataLoader to hang with `num_workers > 0`. The script detects Windows automatically:

```python
NUM_WORKERS = 0 if sys.platform == "win32" else min(4, os.cpu_count() or 2)
```

No action needed.

### CUDA out of memory

Add `--small-batch` to halve all batch sizes:

```bash
python crisisnet_colab.py train --stage all --small-batch
```

If still OOM, reduce further by editing `cfg.ssl_bs`, `cfg.ft_bs`, `cfg.dann_bs` in the `Config` dataclass at the top of `crisisnet_colab.py`.

### `FileNotFoundError: Dataset CSV not found`

You need to download the dataset first:

```bash
python crisisnet_colab.py download
```

### `matplotlib` showing no window / display error on headless servers

The script uses the `Agg` backend by default (saves to file, no display required). Add `--show-plots` only when running with a graphical display.

---

## 🧠 Model Summary

| Component | Architecture | Parameters |
|-----------|-------------|:----------:|
| Image encoder | ResNet-50 (ImageNet1K V2) | 23.5M |
| Text encoder | BERT-base-uncased | 109.5M |
| Cross-modal Transformer | 2 layers, 8 heads, d=512 | ~5M |
| Task heads | 3× (Linear→GELU→Linear) | ~1.5M |
| Domain classifier (DANN only) | 2-layer MLP | ~130K |
| **Total (Stage 3)** | | **~140M** |

---

## 📖 Citation

```bibtex
@inproceedings{alam2018crisismmd,
  title     = {{CrisisMMD}: Multimodal Twitter Datasets from Natural Disasters},
  author    = {Alam, Firoj and Ofli, Ferda and Imran, Muhammad},
  booktitle = {Proceedings of the International AAAI Conference on Web and Social Media},
  year      = {2018}
}
```

---

## 📚 References

| # | Paper | Venue |
|---|-------|-------|
| [1] | Alam et al. — CrisisMMD v2.0 | ICWSM 2021 |
| [2] | Nguyen et al. — MEDIC Dataset | arXiv 2021 |
| [3] | Radford et al. — CLIP | ICML 2021 |
| [4] | He et al. — ResNet | CVPR 2016 |
| [5] | Devlin et al. — BERT | NAACL 2019 |
| [6] | Ganin & Lempitsky — DANN | ICML 2015 |
| [7] | He et al. — MAE | CVPR 2022 |
| [8] | Chen et al. — SimCLR | ICML 2020 |

---

## 🤝 Acknowledgements

- **Dataset**: [QCRI/CrisisMMD](https://huggingface.co/datasets/QCRI/CrisisMMD) via HuggingFace Datasets
- **Compute**: Google Colaboratory (T4 GPU)
- **Course**: Machine Learning & Deep Learning, University of Verona, A.Y. 2025–26
- **Instructors**: Prof. Vittorio Murino, Prof. Cigdem Beyan, Dr. Giacomo Lucato
- **Students**: Fayyaz Hussain Shah, Rafay Saif

---

<div align="center">

**Made with PyTorch · HuggingFace · Google Colab**

<sub>University of Verona · Department of Computer Science · 2025–26</sub>

</div>
