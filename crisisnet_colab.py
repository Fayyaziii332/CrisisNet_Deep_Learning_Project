#!/usr/bin/env python3
# ruff: noqa: E501
"""
CrisisNet - Multimodal Self-Supervised Framework for Crisis Assessment
======================================================================
University of Verona · Machine Learning & Deep Learning · A.Y. 2025-26

QUICKSTART
----------
  python train.py download                        # 1. Download dataset (~1.8 GB)
  python train.py train --stage ssl               # 2. Stage 1: SSL pre-training
  python train.py train --stage finetune          # 3. Stage 2: Fine-tuning
  python train.py train --stage dann              # 4. Stage 3: Domain adaptation
  python train.py train --stage all               # Run all 3 stages sequentially
  python train.py train --stage baselines         # Train 4 ablation baselines
  python train.py evaluate                        # Evaluate all 6 variants
  python train.py predict --image img.jpg \\
                          --text "tweet text"     # Single-image inference

TIPS
----
  - All checkpoints save to ./checkpoints/ after every epoch and auto-resume.
  - On CPU or low-VRAM GPU, add --small-batch to halve all batch sizes.
  - Windows users: DataLoader workers are set to 0 automatically.
  - Add --show-plots to display figures interactively (requires a display).
"""

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# Non-interactive matplotlib backend — must be set before pyplot is imported.
# Switched to TkAgg/Qt5Agg if --show-plots is requested.
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Mixed-precision: compatible with PyTorch 1.x, 2.0, and 2.4+
try:
    from torch.amp import GradScaler as _GS

    def GradScaler(enabled=True):
        return _GS(device="cuda", enabled=enabled)

    def autocast(enabled=True):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.amp.autocast(device_type=dev,
                                  enabled=enabled and torch.cuda.is_available())
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[no-redef]

import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

# =============================================================================
# PATHS  (all relative to this file — move the folder, everything still works)
# =============================================================================
PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR    = PROJECT_DIR / "data" / "crisismmd"
IMAGE_DIR   = DATA_DIR   / "images"
CKPT_DIR    = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"

for _d in [DATA_DIR, IMAGE_DIR,
           CKPT_DIR / "ssl", CKPT_DIR / "finetune",
           CKPT_DIR / "dann", CKPT_DIR / "baselines",
           RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Platform-aware DataLoader settings.
# Windows does not support fork-based multiprocessing; use num_workers=0.
NUM_WORKERS = 0 if sys.platform == "win32" else min(4, os.cpu_count() or 2)
PIN_MEMORY  = torch.cuda.is_available()

# =============================================================================
# REPRODUCIBILITY & DEVICE
# =============================================================================
SEED = 42


def set_seed(s: int = 42) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global flag: whether to call plt.show() after saving figures.
SHOW_PLOTS: bool = False

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    bert_name:   str   = "bert-base-uncased"
    max_len:     int   = 128
    proj_dim:    int   = 512
    num_heads:   int   = 8
    num_layers:  int   = 2
    dropout:     float = 0.1
    img_size:    int   = 224

    # Class counts — updated automatically after dataset load
    n_inform:    int   = 2
    n_damage:    int   = 3   # HF CrisisMMD v2.0 removed 'dont_know'
    n_human:     int   = 8
    n_disasters: int   = 7

    # Stage 1 — SSL  (batch sizes smaller than Colab to fit local 8 GB GPUs)
    ssl_epochs:  int   = 10
    ssl_bs:      int   = 32
    ssl_lr:      float = 2e-4
    ssl_temp:    float = 0.07
    ssl_wd:      float = 1e-4

    # Stage 2 — Fine-tuning
    ft_epochs:   int   = 20
    ft_bs:       int   = 16
    ft_lr:       float = 2e-5
    ft_warmup:   float = 0.1
    ft_wd:       float = 0.01
    lam_info:    float = 1.0
    lam_dmg:     float = 0.5
    lam_hum:     float = 0.5

    # Stage 3 — DANN
    dann_epochs: int   = 15
    dann_bs:     int   = 16
    dann_lr:     float = 5e-6
    lam_domain:  float = 0.1

    grad_clip:   float = 1.0
    use_amp:     bool  = True   # disabled automatically on CPU

    def ckpt(self, stage: str) -> str:
        return str(CKPT_DIR / stage)


cfg = Config()

# AMP has no effect on CPU and wastes overhead
if not torch.cuda.is_available():
    cfg.use_amp = False

# =============================================================================
# LABEL MAPS
# =============================================================================
INFORM_MAP = {0: "not_informative", 1: "informative"}
DAMAGE_MAP = {0: "little_or_no_damage", 1: "mild_damage", 2: "severe_damage"}
HUMAN_MAP  = {
    0: "not_humanitarian",        1: "caution_and_advice",
    2: "missing_or_found_people", 3: "injured_or_dead_people",
    4: "affected_individuals",    5: "vehicle_damage",
    6: "infrastructure_damage",   7: "rescue_volunteering",
}
DISASTER_MAP = {
    "hurricane_harvey": 0,    "hurricane_irma": 1,     "hurricane_maria": 2,
    "iraq_iran_earthquake": 3, "mexico_earthquake": 4,
    "srilanka_floods": 5,     "california_wildfires": 6,
}
EVENTS       = list(DISASTER_MAP.keys())
INV_DISASTER = {v: k for k, v in DISASTER_MAP.items()}
TASK_MAPS    = {"informative": INFORM_MAP, "damage": DAMAGE_MAP, "humanitarian": HUMAN_MAP}
TASK_LABELS  = {t: sorted(m.keys()) for t, m in TASK_MAPS.items()}

# =============================================================================
# DATASET DOWNLOAD
# =============================================================================

def download_dataset() -> None:
    from datasets import load_dataset  # type: ignore

    print("Downloading CrisisMMD v2.0 from HuggingFace (~1.8 GB)...")
    tasks_hf = {
        "informative":  load_dataset("QCRI/CrisisMMD", "informative"),
        "damage":       load_dataset("QCRI/CrisisMMD", "damage"),
        "humanitarian": load_dataset("QCRI/CrisisMMD", "humanitarian"),
    }
    cfg.n_inform = tasks_hf["informative"]["train"].features["label"].num_classes
    cfg.n_damage = tasks_hf["damage"]["train"].features["label"].num_classes
    cfg.n_human  = tasks_hf["humanitarian"]["train"].features["label"].num_classes
    print(f"Classes: inform={cfg.n_inform}  damage={cfg.n_damage}  human={cfg.n_human}")

    def hf_to_df(task_name: str, split_name: str, hf_split) -> pd.DataFrame:
        rows: List[dict] = []
        for s in tqdm(hf_split, desc=f"  {task_name}/{split_name}", leave=False):
            img_id = s["image_id"]
            img_path = IMAGE_DIR / f"{img_id}.jpg"
            if not img_path.exists() and s.get("image") is not None:
                try:
                    s["image"].convert("RGB").save(str(img_path), "JPEG", quality=90)
                except Exception:
                    pass
            rows.append({
                "tweet_id":           s["tweet_id"],
                "image_id":           img_id,
                "tweet_text":         s["tweet_text"],
                "event_name":         s["event_name"],
                f"label_{task_name}": int(s["label"]),
            })
        return pd.DataFrame(rows)

    split_dfs: List[pd.DataFrame] = []
    for sn in ["train", "dev", "test"]:
        inf = hf_to_df("informative",  sn, tasks_hf["informative"][sn])
        dmg = hf_to_df("damage",       sn, tasks_hf["damage"][sn])
        hum = hf_to_df("humanitarian", sn, tasks_hf["humanitarian"][sn])
        merged = (
            inf
            .merge(dmg[["tweet_id", "label_damage"]],       on="tweet_id", how="left")
            .merge(hum[["tweet_id", "label_humanitarian"]], on="tweet_id", how="left")
        )
        merged["label_damage"]       = merged["label_damage"].fillna(0).astype(int)
        merged["label_humanitarian"] = merged["label_humanitarian"].fillna(0).astype(int)
        merged["domain_id"]          = merged["event_name"].map(DISASTER_MAP).fillna(0).astype(int)
        split_dfs.append(merged)

    full_df = (
        pd.concat(split_dfs, ignore_index=True)
        .rename(columns={"label_informative": "label_text"})
    )
    csv_path = DATA_DIR / "crisismmd_merged.csv"
    full_df.to_csv(str(csv_path), index=False)
    print(f"\n✅ {len(full_df):,} samples saved to {csv_path}")
    print(full_df["event_name"].value_counts().to_string())


def load_dataset_local() -> pd.DataFrame:
    csv_path = DATA_DIR / "crisismmd_merged.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {csv_path}\n"
            "Run first:  python train.py download"
        )
    df = pd.read_csv(str(csv_path))
    cfg.n_inform = int(df["label_text"].max()) + 1
    cfg.n_damage = int(df["label_damage"].max()) + 1
    cfg.n_human  = int(df["label_humanitarian"].max()) + 1
    print(f"Dataset: {len(df):,} samples | "
          f"n_inform={cfg.n_inform} n_damage={cfg.n_damage} n_human={cfg.n_human}")
    return df


# =============================================================================
# DATASET CLASSES
# =============================================================================
TRAIN_TF = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Tokenizer loaded lazily on first use to avoid loading BERT at import time
_tokenizer: Optional[BertTokenizer] = None


def get_tokenizer() -> BertTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained(cfg.bert_name)
    return _tokenizer


def _load_image_tensor(img_id: str, tf) -> torch.Tensor:
    for ext in (".jpg", ".png"):
        p = IMAGE_DIR / (img_id + ext)
        if p.exists():
            try:
                return tf(Image.open(str(p)).convert("RGB"))
            except Exception:
                pass
    return torch.zeros(3, cfg.img_size, cfg.img_size)


def _load_pil(img_id: str) -> Optional[Image.Image]:
    for ext in (".jpg", ".png"):
        p = IMAGE_DIR / (img_id + ext)
        if p.exists():
            try:
                return Image.open(str(p)).convert("RGB")
            except Exception:
                pass
    return None


class CrisisMMDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str = "train") -> None:
        self.df = df.reset_index(drop=True)
        self.tf = TRAIN_TF if split == "train" else EVAL_TF
        self.tok = get_tokenizer()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        r   = self.df.iloc[i]
        enc = self.tok(
            str(r.get("tweet_text", "")),
            max_length=cfg.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "image":          _load_image_tensor(str(r.get("image_id", "")), self.tf),
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "inform_label":   int(r.get("label_text", 0)),
            "damage_label":   int(r.get("label_damage", 0)),
            "human_label":    int(r.get("label_humanitarian", 0)),
            "domain_label":   int(r.get("domain_id", 0)),
        }


class UnlabeledPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df  = df.reset_index(drop=True)
        self.tok = get_tokenizer()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        r   = self.df.iloc[i]
        enc = self.tok(
            str(r.get("tweet_text", "")),
            max_length=cfg.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "image":          _load_image_tensor(str(r.get("image_id", "")), TRAIN_TF),
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def build_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr, va, te = [], [], []
    for ev in df["event_name"].unique():
        sub = df[df["event_name"] == ev].sample(frac=1, random_state=SEED)
        n = len(sub); t1, t2 = int(n * 0.70), int(n * 0.85)
        tr.append(sub.iloc[:t1])
        va.append(sub.iloc[t1:t2])
        te.append(sub.iloc[t2:])
    return pd.concat(tr), pd.concat(va), pd.concat(te)


def make_loader(ds: Dataset, bs: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        drop_last=shuffle,
    )


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================
class CheckpointManager:
    """Saves checkpoints after every epoch. Atomic writes prevent corruption."""

    def __init__(self, stage: str, model_name: str) -> None:
        self.dir  = CKPT_DIR / stage
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = model_name
        self.history: List[dict] = []
        self._best: Optional[float] = None

    def save(self, model, optim, epoch: int, metrics: dict,
             sched=None, higher_is_better: bool = True) -> bool:
        ckpt: dict = {
            "epoch": epoch, "model": model.state_dict(),
            "optim": optim.state_dict(), "metrics": metrics, "ts": time.time(),
        }
        if sched is not None:
            ckpt["sched"] = sched.state_dict()

        key = next(iter(metrics))
        val = metrics[key]
        is_best = (
            self._best is None
            or (higher_is_better and val > self._best)
            or (not higher_is_better and val < self._best)
        )
        if is_best:
            self._best = val

        self._atomic_save(ckpt, self.dir / f"{self.name}_latest.pth")
        if is_best:
            self._atomic_save(ckpt, self.dir / f"{self.name}_best.pth")

        self.history.append({"epoch": epoch, **metrics})
        with open(str(self.dir / "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        star = "  ⭐ BEST" if is_best else ""
        print(f"  💾 Saved epoch {epoch:03d}{star}")
        return is_best

    @staticmethod
    def _atomic_save(obj: dict, path: Path) -> None:
        tmp = str(path) + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, str(path))

    def resume(self, model, optim=None, sched=None, best: bool = False) -> Tuple[int, dict]:
        tag  = "best" if best else "latest"
        path = self.dir / f"{self.name}_{tag}.pth"
        if not path.exists():
            print(f"  No checkpoint at {path} — starting fresh.")
            return 0, {}
        return self._load(path, model, optim, sched)

    def _load(self, path: Path, model, optim, sched) -> Tuple[int, dict]:
        print(f"  📂 Loading {path}")
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        sd   = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(sd, strict=False)
        if optim is not None and "optim" in ckpt:
            try:
                optim.load_state_dict(ckpt["optim"])
            except Exception:
                pass
        if sched is not None and "sched" in ckpt:
            try:
                sched.load_state_dict(ckpt["sched"])
            except Exception:
                pass
        ep = ckpt.get("epoch", 0)
        m  = ckpt.get("metrics", {})
        print(f"     Resumed epoch {ep}  {m}")
        return ep, m

    def load_history(self) -> List[dict]:
        p = self.dir / "history.json"
        if p.exists():
            with open(str(p)) as f:
                return json.load(f)
        return []


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class ImageEncoder(nn.Module):
    def __init__(self, proj_dim: int = 512) -> None:
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.proj     = nn.Sequential(nn.Linear(2048, proj_dim), nn.LayerNorm(proj_dim))

    def forward(self, x: torch.Tensor, mode: str = "spatial") -> torch.Tensor:
        f = self.backbone(x)
        if mode == "global":
            return self.proj(self.pool(f).flatten(1))
        B, C, H, W = f.shape
        return self.proj(f.permute(0, 2, 3, 1).reshape(B, H * W, C))


class TextEncoder(nn.Module):
    def __init__(self, proj_dim: int = 512, bert_name: str = "bert-base-uncased") -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.proj = nn.Sequential(nn.Linear(768, proj_dim), nn.LayerNorm(proj_dim))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                mode: str = "sequence") -> torch.Tensor:
        h = self.proj(self.bert(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state)
        return h[:, 0, :] if mode == "cls" else h


class CrossModalLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n1   = nn.LayerNorm(dim)
        self.ffn  = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.n2   = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv, mask=None):
        a, w = self.attn(q, kv, kv, key_padding_mask=mask,
                         need_weights=True, average_attn_weights=True)
        q = self.n1(q + self.drop(a))
        q = self.n2(q + self.ffn(q))
        return q, w


class CrossModalTransformer(nn.Module):
    def __init__(self, dim: int, heads: int, n: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.i2t  = nn.ModuleList([CrossModalLayer(dim, heads, dropout) for _ in range(n)])
        self.t2i  = nn.ModuleList([CrossModalLayer(dim, heads, dropout) for _ in range(n)])
        self.isa  = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim*4, dropout,
                                       batch_first=True, norm_first=True)
            for _ in range(n)])
        self.tsa  = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim*4, dropout,
                                       batch_first=True, norm_first=True)
            for _ in range(n)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, img, txt, pad=None):
        w = None
        for i2t, t2i, isa, tsa in zip(self.i2t, self.t2i, self.isa, self.tsa):
            img, w = i2t(img, txt, pad)
            txt, _ = t2i(txt, img)
            img    = isa(img)
            txt    = tsa(txt, src_key_padding_mask=pad)
        return self.norm(torch.cat([img, txt], 1).mean(1)), w


class TaskHeads(nn.Module):
    def __init__(self, dim: int, ni: int, nd: int, nh: int, drop: float = 0.1) -> None:
        super().__init__()
        def _h(n):
            return nn.Sequential(
                nn.Dropout(drop), nn.Linear(dim, dim // 2),
                nn.GELU(), nn.Dropout(drop), nn.Linear(dim // 2, n),
            )
        self.info = _h(ni)
        self.dmg  = _h(nd)
        self.hum  = _h(nh)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"informative": self.info(x), "damage": self.dmg(x),
                "humanitarian": self.hum(x)}


class CrisisNet(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.img_enc = ImageEncoder(cfg.proj_dim)
        self.txt_enc = TextEncoder(cfg.proj_dim, cfg.bert_name)
        self.fusion  = CrossModalTransformer(cfg.proj_dim, cfg.num_heads,
                                              cfg.num_layers, cfg.dropout)
        self.heads   = TaskHeads(cfg.proj_dim, cfg.n_inform, cfg.n_damage,
                                  cfg.n_human, cfg.dropout)

    def forward(self, images, input_ids, attention_mask, return_attn=False, **kw):
        it = self.img_enc(images, "spatial")
        tt = self.txt_enc(input_ids, attention_mask, "sequence")
        f, w = self.fusion(it, tt, (attention_mask == 0))
        out = self.heads(f)
        return (out, w) if return_attn else out


class CLIPModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.img_enc   = ImageEncoder(cfg.proj_dim)
        self.txt_enc   = TextEncoder(cfg.proj_dim, cfg.bert_name)
        self.log_scale = nn.Parameter(torch.ones([]) * math.log(1 / cfg.ssl_temp))

    def forward(self, images, input_ids, attention_mask):
        img = F.normalize(self.img_enc(images, "global"), dim=-1)
        txt = F.normalize(self.txt_enc(input_ids, attention_mask, "cls"), dim=-1)
        return img, txt, self.log_scale.exp().clamp(max=100)


class _GRevFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.a = a
        return x

    @staticmethod
    def backward(ctx, g):
        return -ctx.a * g, None


class _GRev(nn.Module):
    def forward(self, x, a=1.0):
        return _GRevFn.apply(x, a)


class CrisisNetDANN(CrisisNet):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.grl = _GRev()
        self.dom = nn.Sequential(
            nn.Linear(cfg.proj_dim, 256), nn.ReLU(),
            nn.Dropout(cfg.dropout), nn.Linear(256, cfg.n_disasters),
        )

    def forward(self, images, input_ids, attention_mask,
                alpha=1.0, return_attn=False, **kw):
        it = self.img_enc(images, "spatial")
        tt = self.txt_enc(input_ids, attention_mask, "sequence")
        f, w = self.fusion(it, tt, (attention_mask == 0))
        tasks  = self.heads(f)
        domain = self.dom(self.grl(f, alpha))
        if return_attn:
            return tasks, domain, w
        return tasks, domain


# Ablation baselines
class ImageOnlyModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.heads    = TaskHeads(2048, cfg.n_inform, cfg.n_damage, cfg.n_human, cfg.dropout)

    def forward(self, images, input_ids=None, attention_mask=None, **kw):
        return self.heads(self.backbone(images).flatten(1))


class TextOnlyModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.bert  = BertModel.from_pretrained(cfg.bert_name)
        self.heads = TaskHeads(768, cfg.n_inform, cfg.n_damage, cfg.n_human, cfg.dropout)

    def forward(self, images=None, input_ids=None, attention_mask=None, **kw):
        cls = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.heads(cls)


class LateFusionModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.img_enc = ImageEncoder(cfg.proj_dim)
        self.txt_enc = TextEncoder(cfg.proj_dim, cfg.bert_name)
        self.fusion  = nn.Sequential(
            nn.Linear(cfg.proj_dim * 2, cfg.proj_dim),
            nn.GELU(), nn.Dropout(cfg.dropout),
        )
        self.heads = TaskHeads(cfg.proj_dim, cfg.n_inform, cfg.n_damage,
                               cfg.n_human, cfg.dropout)

    def forward(self, images, input_ids, attention_mask, **kw):
        return self.heads(self.fusion(torch.cat(
            [self.img_enc(images, "global"),
             self.txt_enc(input_ids, attention_mask, "cls")], dim=-1)))


# =============================================================================
# LOSS FUNCTIONS & EVALUATION
# =============================================================================

class InfoNCELoss(nn.Module):
    def forward(self, img, txt, scale):
        L   = scale * img @ txt.T
        lbl = torch.arange(len(img), device=img.device)
        return (F.cross_entropy(L, lbl) + F.cross_entropy(L.T, lbl)) / 2


def compute_class_weights(series, n_classes: int) -> torch.Tensor:
    c = pd.Series(series).value_counts().sort_index().reindex(range(n_classes), fill_value=1)
    w = 1.0 / c.values.astype(float)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


class MultiTaskLoss(nn.Module):
    def __init__(self, w_info, w_dmg, w_hum,
                 cw_info=None, cw_dmg=None, cw_hum=None) -> None:
        super().__init__()
        self.tw = {"informative": w_info, "damage": w_dmg, "humanitarian": w_hum}
        self.cw = {"informative": cw_info, "damage": cw_dmg, "humanitarian": cw_hum}

    def forward(self, logits, batch):
        tot = 0.0
        km  = {"informative": "inform_label", "damage": "damage_label",
               "humanitarian": "human_label"}
        for task, lk in km.items():
            if lk in batch:
                lbl = batch[lk].to(device)
                w   = self.cw[task].to(device) if self.cw[task] is not None else None
                tot += self.tw[task] * F.cross_entropy(logits[task], lbl, weight=w)
        return tot


def _make_loss(train_df: pd.DataFrame) -> MultiTaskLoss:
    return MultiTaskLoss(
        w_info=cfg.lam_info, w_dmg=cfg.lam_dmg, w_hum=cfg.lam_hum,
        cw_info=compute_class_weights(train_df["label_text"],          cfg.n_inform),
        cw_dmg =compute_class_weights(train_df["label_damage"],        cfg.n_damage),
        cw_hum =compute_class_weights(train_df["label_humanitarian"],  cfg.n_human),
    )


def _forward(model, images, ids, mask, stage: str):
    try:
        out = model(images, ids, mask, alpha=0.0)
        return out[0] if isinstance(out, tuple) else out
    except (TypeError, ValueError):
        return model(images, ids, mask)


@torch.no_grad()
def evaluate(model, loader, stage: str = "finetune") -> dict:
    model.eval()
    P: Dict[str, list] = {"informative": [], "damage": [], "humanitarian": []}
    T: Dict[str, list] = {"informative": [], "damage": [], "humanitarian": []}
    KM = {"informative": "inform_label", "damage": "damage_label",
          "humanitarian": "human_label"}
    for batch in tqdm(loader, desc="Eval", leave=False):
        imgs = batch["image"].to(device)
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lg   = _forward(model, imgs, ids, mask, stage)
        for task, lk in KM.items():
            if lk in batch:
                P[task].extend(lg[task].argmax(1).cpu().numpy())
                T[task].extend(batch[lk].numpy())
    metrics: dict = {}
    for task in P:
        if not P[task]:
            continue
        p, t = np.array(P[task]), np.array(T[task])
        metrics[f"{task}_f1"]  = f1_score(t, p, average="macro", zero_division=0)
        metrics[f"{task}_acc"] = accuracy_score(t, p)
    if metrics:
        metrics["avg_f1"] = np.mean([v for k, v in metrics.items() if "_f1" in k])
    model.train()
    return metrics


@torch.no_grad()
def full_predict(model, loader, stage: str = "finetune") -> dict:
    model.eval()
    P: Dict[str, list] = {"informative": [], "damage": [], "humanitarian": []}
    T: Dict[str, list] = {"informative": [], "damage": [], "humanitarian": []}
    KM = {"informative": "inform_label", "damage": "damage_label",
          "humanitarian": "human_label"}
    for batch in tqdm(loader, desc="Predict", leave=False):
        imgs = batch["image"].to(device)
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lg   = _forward(model, imgs, ids, mask, stage)
        for task, lk in KM.items():
            if lk in batch:
                P[task].extend(lg[task].argmax(1).cpu().numpy())
                T[task].extend(batch[lk].numpy())
    model.train()
    return {t: (np.array(P[t]), np.array(T[t])) for t in P if P[t]}


def _print_metrics(metrics: dict, epoch: int) -> None:
    print("  Epoch {:03d}".format(epoch)
          + "".join(f"  |  {k}: {v:.4f}" for k, v in metrics.items()))


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_ssl(ssl_ds, resume: bool = True):
    print("\n" + "=" * 60 + "\nSTAGE 1 — SSL Pre-training\n" + "=" * 60)
    model   = CLIPModel(cfg).to(device)
    loss_fn = InfoNCELoss()
    ckpt    = CheckpointManager("ssl", "clip")
    sl      = make_loader(ssl_ds, cfg.ssl_bs)
    ts      = len(sl) * cfg.ssl_epochs
    opt     = AdamW(model.parameters(), lr=cfg.ssl_lr, weight_decay=cfg.ssl_wd)
    sched   = get_linear_schedule_with_warmup(opt, max(1, int(0.05 * ts)), ts)
    scaler  = GradScaler(enabled=cfg.use_amp)
    start   = 0
    if resume:
        start, _ = ckpt.resume(model, opt, sched)
    best = float("inf")
    for ep in range(start, cfg.ssl_epochs):
        model.train()
        rl   = 0.0
        pbar = tqdm(sl, desc=f"SSL {ep+1:02d}/{cfg.ssl_epochs}", leave=False)
        for b in pbar:
            imgs = b["image"].to(device, non_blocking=True)
            ids  = b["input_ids"].to(device, non_blocking=True)
            mask = b["attention_mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                i, t, sc = model(imgs, ids, mask)
                loss = loss_fn(i, t, sc)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            rl += loss.item()
            pbar.set_postfix({"loss": f"{rl / (pbar.n + 1):.4f}"})
        avg  = rl / len(sl)
        is_b = avg < best
        if is_b:
            best = avg
        ckpt.save(model, opt, ep + 1, {"ssl_loss": avg}, sched,
                  higher_is_better=False)
        print(f"  Epoch {ep+1:02d}  loss={avg:.4f}{'  ⭐' if is_b else ''}")
    print("\n✅ Stage 1 complete.")
    return model


def train_finetune(train_ds, val_ds, train_df,
                   ssl_model_or_path=None, resume: bool = True):
    print("\n" + "=" * 60 + "\nSTAGE 2 — Multi-task Fine-tuning\n" + "=" * 60)
    model   = CrisisNet(cfg).to(device)
    loss_fn = _make_loss(train_df)
    ckpt    = CheckpointManager("finetune", "crisisnet")

    if ssl_model_or_path is not None:
        src = ssl_model_or_path
        if isinstance(src, CLIPModel):
            model.img_enc.load_state_dict(src.img_enc.state_dict(), strict=False)
            model.txt_enc.load_state_dict(src.txt_enc.state_dict(), strict=False)
            print("  ✅ SSL weights transferred.")
        elif isinstance(src, (str, Path)) and Path(src).exists():
            sm = CLIPModel(cfg).to(device)
            sm.load_state_dict(
                torch.load(str(src), map_location=device, weights_only=False)["model"])
            model.img_enc.load_state_dict(sm.img_enc.state_dict(), strict=False)
            model.txt_enc.load_state_dict(sm.txt_enc.state_dict(), strict=False)
            print("  ✅ SSL weights transferred from file.")
    else:
        print("  ⚠  No SSL weights — using ImageNet + random init.")

    bp  = list(model.txt_enc.bert.parameters())
    op  = [p for p in model.parameters() if not any(p is b for b in bp)]
    opt = AdamW([{"params": bp, "lr": cfg.ft_lr * 0.1},
                 {"params": op, "lr": cfg.ft_lr}], weight_decay=cfg.ft_wd)
    tl  = make_loader(train_ds, cfg.ft_bs)
    vl  = make_loader(val_ds, cfg.ft_bs, shuffle=False)
    ts  = len(tl) * cfg.ft_epochs
    sched  = get_linear_schedule_with_warmup(opt, max(1, int(cfg.ft_warmup * ts)), ts)
    scaler = GradScaler(enabled=cfg.use_amp)
    start  = 0
    if resume:
        start, _ = ckpt.resume(model, opt, sched)
    for ep in range(start, cfg.ft_epochs):
        model.train()
        rl = 0.0
        for b in tqdm(tl, desc=f"FT {ep+1:02d}/{cfg.ft_epochs}", leave=False):
            imgs = b["image"].to(device, non_blocking=True)
            ids  = b["input_ids"].to(device, non_blocking=True)
            mask = b["attention_mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                loss = loss_fn(model(imgs, ids, mask), b)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            rl += loss.item()
        vm = evaluate(model, vl)
        vm["train_loss"] = rl / len(tl)
        _print_metrics(vm, ep + 1)
        ckpt.save(model, opt, ep + 1, vm, sched)
    print("\n✅ Stage 2 complete.")
    return model


def _get_alpha(ep: int, total: int) -> float:
    p = ep / max(total, 1)
    return 2.0 / (1.0 + math.exp(-10 * p)) - 1.0


def train_dann(train_ds, val_ds, train_df,
               ft_model_or_path=None, resume: bool = True):
    print("\n" + "=" * 60 + "\nSTAGE 3 — DANN Domain Adaptation\n" + "=" * 60)
    model   = CrisisNetDANN(cfg).to(device)
    loss_fn = _make_loss(train_df)
    ckpt    = CheckpointManager("dann", "crisisnet_dann")

    ft_ckpt_path = CKPT_DIR / "finetune" / "crisisnet_best.pth"
    src = ft_model_or_path
    if isinstance(src, CrisisNet):
        model.load_state_dict(src.state_dict(), strict=False)
        print("  FT weights loaded from object.")
    elif isinstance(src, (str, Path)) and Path(src).exists():
        model.load_state_dict(
            torch.load(str(src), map_location=device, weights_only=False)["model"],
            strict=False)
        print(f"  FT weights loaded from {src}")
    elif ft_ckpt_path.exists():
        model.load_state_dict(
            torch.load(str(ft_ckpt_path), map_location=device, weights_only=False)["model"],
            strict=False)
        print(f"  Auto-loaded Stage 2 best checkpoint: {ft_ckpt_path}")
    else:
        print("  ⚠  No fine-tuned checkpoint found — starting from random weights.")

    opt    = AdamW(model.parameters(), lr=cfg.dann_lr, weight_decay=cfg.ft_wd)
    tl     = make_loader(train_ds, cfg.dann_bs)
    vl     = make_loader(val_ds, cfg.dann_bs, shuffle=False)
    scaler = GradScaler(enabled=cfg.use_amp)
    start  = 0
    if resume:
        start, _ = ckpt.resume(model, opt)
    for ep in range(start, cfg.dann_epochs):
        model.train()
        alpha = _get_alpha(ep, cfg.dann_epochs)
        rl    = 0.0
        for b in tqdm(tl,
                      desc=f"DANN {ep+1:02d}/{cfg.dann_epochs}  α={alpha:.3f}",
                      leave=False):
            imgs = b["image"].to(device, non_blocking=True)
            ids  = b["input_ids"].to(device, non_blocking=True)
            mask = b["attention_mask"].to(device, non_blocking=True)
            dom  = b["domain_label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                to, do = model(imgs, ids, mask, alpha=alpha)
                loss   = loss_fn(to, b) + cfg.lam_domain * F.cross_entropy(do, dom)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            rl += loss.item()
        vm = evaluate(model, vl, stage="dann")
        vm["train_loss"] = rl / len(tl)
        _print_metrics(vm, ep + 1)
        ckpt.save(model, opt, ep + 1, vm)
    print("\n✅ Stage 3 complete.")
    return model


def train_baseline(model, model_name: str, train_ds, val_ds,
                   train_df, n_epochs: int = 10, resume: bool = True):
    ckpt    = CheckpointManager("baselines", model_name)
    loss_fn = _make_loss(train_df)
    opt     = AdamW(model.parameters(), lr=cfg.ft_lr, weight_decay=cfg.ft_wd)
    tl      = make_loader(train_ds, cfg.ft_bs)
    vl      = make_loader(val_ds, cfg.ft_bs, shuffle=False)
    ts      = len(tl) * n_epochs
    sched   = get_linear_schedule_with_warmup(opt, int(0.1 * ts), ts)
    scaler  = GradScaler(enabled=cfg.use_amp)
    start, _ = ckpt.resume(model, opt, sched) if resume else (0, {})
    for ep in range(start, n_epochs):
        model.train()
        rl = 0.0
        for b in tqdm(tl, desc=f"{model_name} {ep+1}/{n_epochs}", leave=False):
            imgs = b["image"].to(device, non_blocking=True)
            ids  = b["input_ids"].to(device, non_blocking=True)
            mask = b["attention_mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                loss = loss_fn(model(imgs, ids, mask), b)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            rl += loss.item()
        vm = evaluate(model, vl)
        ckpt.save(model, opt, ep + 1, vm, sched)
        print(f"  {model_name:<22}  epoch {ep+1:02d}  avg_f1={vm.get('avg_f1', 0):.4f}")
    return model


# =============================================================================
# EVALUATION & VISUALISATION
# =============================================================================

def _show_or_close():
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_dataset_analysis(full_df, train_df, val_df, test_df) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CrisisMMD v2.0 — Dataset Analysis", fontsize=15, fontweight="bold")
    ev = full_df["event_name"].value_counts()
    axes[0, 0].pie(ev.values, labels=[e.replace("_", "\n") for e in ev.index],
                   autopct="%1.1f%%", startangle=90, textprops={"fontsize": 7},
                   colors=plt.cm.Set3(np.linspace(0, 1, len(ev))))
    axes[0, 0].set_title("Samples by Disaster Event", fontweight="bold")
    ic = full_df["label_text"].value_counts().sort_index()
    bars = axes[0, 1].bar([INFORM_MAP[i] for i in ic.index], ic.values,
                           color=["#E74C3C", "#2ECC71"])
    axes[0, 1].set_title("Task 1: Informative", fontweight="bold")
    axes[0, 1].set_ylabel("Count")
    for bar, v in zip(bars, ic.values):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                        f"{v:,}\n({v / len(full_df) * 100:.1f}%)", ha="center", fontsize=9)
    dc = full_df["label_damage"].value_counts().sort_index()
    axes[0, 2].bar([DAMAGE_MAP[i].replace("_", "\n") for i in dc.index], dc.values,
                   color=["#3498DB", "#F39C12", "#E74C3C"])
    axes[0, 2].set_title("Task 2: Damage Severity", fontweight="bold")
    axes[0, 2].set_ylabel("Count")
    hc = full_df["label_humanitarian"].value_counts().sort_index()
    axes[1, 0].bar([HUMAN_MAP[i].replace("_", "\n") for i in hc.index], hc.values,
                   color=plt.cm.Set2(np.linspace(0, 1, len(hc))))
    axes[1, 0].set_title("Task 3: Humanitarian", fontweight="bold")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].tick_params(axis="x", labelsize=6, rotation=30)
    x, w = np.arange(len(EVENTS)), 0.25
    for i, (sn, sd) in enumerate([("Train", train_df), ("Val", val_df), ("Test", test_df)]):
        cnt = sd["event_name"].value_counts()
        axes[1, 1].bar(x + i * w, [cnt.get(ev, 0) for ev in EVENTS], w, label=sn)
    axes[1, 1].set_xticks(x + w)
    axes[1, 1].set_xticklabels([e.replace("_", "\n") for e in EVENTS], fontsize=6)
    axes[1, 1].set_title("Samples per Event per Split", fontweight="bold")
    axes[1, 1].legend(fontsize=8)
    ts = {
        "Informative\n(T1)": full_df["label_text"],
        "Damage\n(T2)":      full_df["label_damage"],
        "Humanitarian\n(T3)":full_df["label_humanitarian"],
    }
    rats = [v.value_counts().max() / v.value_counts().min() for v in ts.values()]
    bars2 = axes[1, 2].bar(list(ts.keys()), rats, color=["#9B59B6", "#1ABC9C", "#E67E22"])
    axes[1, 2].set_title("Class Imbalance Ratio (max/min)", fontweight="bold")
    axes[1, 2].set_ylabel("Ratio")
    for bar, v in zip(bars2, rats):
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{v:.1f}x", ha="center", fontsize=12, fontweight="bold",
                        color="darkred")
    plt.tight_layout()
    p = RESULTS_DIR / "dataset_analysis.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    _show_or_close()
    print(f"✅ Saved: {p}")


def _save_confusion_matrices(res: dict, variant_name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Confusion Matrices — {variant_name}", fontsize=13, fontweight="bold")
    for ax, (task, (p, t)) in zip(axes, res.items()):
        lmap  = TASK_MAPS[task]
        lbls  = TASK_LABELS[task]
        names = [lmap[i].replace("_", "\n") for i in lbls]
        cm    = confusion_matrix(t, p, labels=lbls)
        cmn   = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-8)
        im    = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{cmn[i,j]:.2f}\n({cm[i,j]})",
                        ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
        f1 = f1_score(t, p, average="macro", zero_division=0)
        ax.set_title(f"{task.capitalize()} (F1={f1:.3f})", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    slug = variant_name.replace(" ", "_").replace("+", "plus").lower()[:30]
    p    = RESULTS_DIR / f"cm_{slug}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p.name}")


def _save_prf1(res: dict, variant_name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Per-class P/R/F1 — {variant_name}", fontsize=13, fontweight="bold")
    for ax, (task, (p, t)) in zip(axes, res.items()):
        lmap  = TASK_MAPS[task]
        lbls  = TASK_LABELS[task]
        names = [lmap[i].replace("_", " ") for i in lbls]
        pr, re, f1, _ = precision_recall_fscore_support(t, p, labels=lbls, zero_division=0)
        x, w = np.arange(len(names)), 0.25
        ax.bar(x - w, pr, w, label="Precision", color="#3498DB", alpha=0.85)
        ax.bar(x,     re, w, label="Recall",    color="#2ECC71", alpha=0.85)
        ax.bar(x + w, f1, w, label="F1",        color="#E74C3C", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7,
                           rotation=30, ha="right")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)
        ax.set_ylabel("Score")
        ax.set_title(task.capitalize(), fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax.text(len(names) - 0.5, 1.08, f"Macro F1={f1.mean():.3f}",
                ha="right", fontsize=9, color="darkred", fontweight="bold")
    plt.tight_layout()
    slug = variant_name.replace(" ", "_").replace("+", "plus").lower()[:30]
    p    = RESULTS_DIR / f"prf1_{slug}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p.name}")


def run_full_evaluation(img_model, txt_model, late_model, nossl_model,
                        ft_model, dann_model, test_ds, test_df) -> None:
    test_loader = make_loader(test_ds, cfg.ft_bs, shuffle=False)

    registry = OrderedDict([
        ("1. Image-only (ResNet-50)",      (img_model,   "finetune")),
        ("2. Text-only (BERT)",            (txt_model,   "finetune")),
        ("3. Late Fusion (concat)",        (late_model,  "finetune")),
        ("4. CrisisNet, no SSL",           (nossl_model, "finetune")),
        ("5. CrisisNet + SSL (Stage 2)",   (ft_model,    "finetune")),
        ("6. CrisisNet + SSL + DANN",      (dann_model,  "dann")),
    ])

    all_res: Dict[str, dict] = {}
    for name, (model, stage) in registry.items():
        print(f"\nPredicting: {name}")
        all_res[name] = full_predict(model, test_loader, stage)

    # Per-variant figures
    print("\nGenerating confusion matrices and P/R/F1 for all variants...")
    for name, res in all_res.items():
        _save_confusion_matrices(res, name)
        _save_prf1(res, name)

    # Summary ablation table
    rows = []
    for name, res in all_res.items():
        row: dict = {"Variant": name}
        all_f1 = []
        for task, (p, t) in res.items():
            lbls = TASK_LABELS[task]
            pr, re, f1, _ = precision_recall_fscore_support(
                t, p, labels=lbls, average="macro", zero_division=0)
            acc = accuracy_score(t, p)
            short = task[:4].title()
            row[f"{short}.Prec"] = pr
            row[f"{short}.Rec"]  = re
            row[f"{short}.F1"]   = f1
            row[f"{short}.Acc"]  = acc
            all_f1.append(f1)
        row["Avg F1"] = float(np.mean(all_f1))
        rows.append(row)

    abl = pd.DataFrame(rows)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 70)
    print("ABLATION TABLE — Test Set (Macro P/R/F1 per task)")
    print("=" * 70)
    print(abl.to_string(index=False))

    # Ablation bar chart
    f1_cols = [c for c in abl.columns if ".F1" in c]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Ablation Study — All 6 Variants (Test Set)", fontsize=13, fontweight="bold")
    x, w = np.arange(len(abl)), 0.22
    for i, (col, color) in enumerate(zip(f1_cols, ["#3498DB", "#E74C3C", "#2ECC71"])):
        axes[0].bar(x + i * w, abl[col].fillna(0), w, label=col, color=color, alpha=0.85)
    short_labels = [r.split(".")[-1].strip()[:18] for r in abl["Variant"]]
    axes[0].set_xticks(x + w)
    axes[0].set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Macro F1")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Per-task Macro F1", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].axhline(0.33, color="gray", linestyle="--", alpha=0.4)
    axes[0].grid(axis="y", alpha=0.3)
    av = abl["Avg F1"].fillna(0).values
    axes[1].plot(range(len(av)), av, "o-", color="#E74C3C", linewidth=2.5, markersize=9)
    axes[1].fill_between(range(len(av)), av, alpha=0.12, color="#E74C3C")
    for xi, v in enumerate(av):
        axes[1].annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
    bi = int(np.argmax(av))
    axes[1].scatter([bi], [av[bi]], s=200, color="gold", zorder=5,
                    edgecolors="black", linewidths=1.5)
    axes[1].set_xticks(range(len(av)))
    axes[1].set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Average Macro F1")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Average F1 Across Tasks", fontweight="bold")
    axes[1].axhline(0.33, color="gray", linestyle="--", alpha=0.4)
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = RESULTS_DIR / "ablation_summary.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    _show_or_close()
    print(f"✅ Ablation chart saved: {p}")


def plot_training_curves() -> None:
    stages = [
        ("ssl",      "clip",           "Stage 1 — SSL"),
        ("finetune", "crisisnet",      "Stage 2 — Fine-tuning"),
        ("dann",     "crisisnet_dann", "Stage 3 — DANN"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CrisisNet — Training Curves", fontsize=15, fontweight="bold")
    TC = {"informative_f1": "#E74C3C", "damage_f1": "#3498DB",
          "humanitarian_f1": "#2ECC71", "avg_f1": "black"}
    for col, (stage, name, title) in enumerate(stages):
        hist = CheckpointManager(stage, name).load_history()
        if not hist:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, f"No history\n({stage})",
                                    ha="center", va="center",
                                    transform=axes[row, col].transAxes,
                                    color="gray", fontsize=11)
                axes[row, col].set_title(title)
            continue
        df  = pd.DataFrame(hist)
        eps = df["epoch"].values if "epoch" in df.columns else np.arange(1, len(df) + 1)
        ax0 = axes[0, col]
        lc  = "ssl_loss" if "ssl_loss" in df.columns else "train_loss"
        if lc in df.columns:
            ax0.plot(eps, df[lc], "b-o", markersize=4, linewidth=1.8, label="Loss")
            ax0.fill_between(eps, df[lc], alpha=0.1, color="blue")
            mi = int(df[lc].idxmin())
            ax0.scatter([eps[mi]], [df[lc].iloc[mi]], s=120, color="gold",
                        zorder=5, edgecolors="black")
        ax0.set_title(f"{title}\nLoss", fontweight="bold", fontsize=10)
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Loss")
        ax0.legend(fontsize=8)
        ax0.grid(alpha=0.3)
        ax1 = axes[1, col]
        for fc in [c for c in df.columns if c.endswith("_f1")]:
            lw = 2.5 if fc == "avg_f1" else 1.5
            ls = "--" if fc == "avg_f1" else "-"
            label = fc.replace("_f1", "").replace("_", " ").title()
            ax1.plot(eps, df[fc], ls + "o",
                     color=TC.get(fc, "gray"), markersize=3, linewidth=lw, label=label)
        if "avg_f1" in df.columns:
            bi = int(df["avg_f1"].idxmax())
            ax1.axvline(x=eps[bi], color="gold", linestyle=":", alpha=0.8, linewidth=1.5)
        ax1.set_title(f"{title}\nMacro F1", fontweight="bold", fontsize=10)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Macro F1")
        ax1.set_ylim(0, 1)
        ax1.legend(fontsize=7)
        ax1.grid(alpha=0.3)
    plt.tight_layout()
    p = RESULTS_DIR / "training_curves.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    _show_or_close()
    print(f"✅ Saved: {p}")


@torch.no_grad()
def cross_disaster_heatmap(model_ft, model_dann, full_df) -> None:
    results: Dict[str, dict] = {"Stage 2\n(no adapt)": {}, "Stage 3\n(DANN)": {}}
    for ev in EVENTS:
        sub = full_df[full_df["event_name"] == ev]
        if len(sub) == 0:
            continue
        ds  = CrisisMMDDataset(sub, "val")
        ld  = make_loader(ds, cfg.dann_bs, shuffle=False)
        results["Stage 2\n(no adapt)"][ev] = evaluate(model_ft,  ld, "finetune").get("avg_f1", 0)
        results["Stage 3\n(DANN)"][ev]     = evaluate(model_dann, ld, "dann").get("avg_f1", 0)
        s2 = results["Stage 2\n(no adapt)"][ev]
        s3 = results["Stage 3\n(DANN)"][ev]
        print(f"  {ev:<35}  S2={s2:.3f}  S3={s3:.3f}  Δ={s3 - s2:+.3f}")
    ev_short = [e.replace("_", "\n") for e in EVENTS]
    v2 = [results["Stage 2\n(no adapt)"].get(ev, 0) for ev in EVENTS]
    v3 = [results["Stage 3\n(DANN)"].get(ev, 0) for ev in EVENTS]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Cross-Disaster Transfer (Leave-One-Event-Out)", fontsize=13, fontweight="bold")
    x, w = np.arange(len(EVENTS)), 0.35
    axes[0].bar(x - w / 2, v2, w, label="Stage 2 (no adapt)", color="#3498DB", alpha=0.85)
    axes[0].bar(x + w / 2, v3, w, label="Stage 3 (DANN)",     color="#E74C3C", alpha=0.85)
    for xi, (a, b) in enumerate(zip(v2, v3)):
        axes[0].annotate(f"{b - a:+.3f}", (xi, max(a, b) + 0.01),
                         ha="center", fontsize=7.5,
                         color="#27AE60" if b > a else "#C0392B",
                         fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ev_short, fontsize=8)
    axes[0].set_ylabel("Avg Macro F1")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Per-Disaster F1", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].axhline(0.33, color="gray", linestyle="--", alpha=0.4)
    hmap = np.array([v2, v3])
    im   = axes[1].imshow(hmap, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Avg F1")
    axes[1].set_xticks(range(len(EVENTS)))
    axes[1].set_xticklabels(ev_short, fontsize=8)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Stage 2\n(no adapt)", "Stage 3\n(DANN)"], fontsize=10)
    axes[1].set_title("F1 Heatmap by Disaster", fontweight="bold")
    for ri in range(2):
        for ci in range(len(EVENTS)):
            val = hmap[ri, ci]
            axes[1].text(ci, ri, f"{val:.3f}", ha="center", va="center",
                         fontsize=8.5, fontweight="bold",
                         color="black" if 0.25 < val < 0.75 else "white")
    plt.tight_layout()
    p = RESULTS_DIR / "cross_disaster.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    _show_or_close()
    sm = pd.DataFrame({"Event": EVENTS, "Stage2": v2, "Stage3": v3,
                       "Gain": [b - a for a, b in zip(v2, v3)]})
    sm["Trend"] = sm["Gain"].apply(lambda x: f"up {x:+.3f}" if x > 0 else f"dn {x:+.3f}")
    print(sm.to_string(index=False))
    print(f"  Mean DANN gain : {sm['Gain'].mean():+.4f}")
    print(f"✅ Saved: {p}")


@torch.no_grad()
def plot_multimodel_qualitative(registry: OrderedDict,
                                test_df: pd.DataFrame, n_samples: int = 6) -> None:
    samples = (
        test_df.groupby("event_name", group_keys=False)
        .apply(lambda g: g.sample(1, random_state=42))
        .reset_index(drop=True)
        .head(n_samples)
    )
    model_names = list(registry.keys())
    n_models    = len(model_names)
    fig = plt.figure(figsize=(3.0 + n_models * 2.4, len(samples) * 3.8))
    gs  = fig.add_gridspec(len(samples), 1 + n_models,
                           width_ratios=[2.8] + [1.8] * n_models,
                           hspace=0.45, wspace=0.12)
    border_clr = {3: "#27AE60", 2: "#F9BF3B", 1: "#E67E22", 0: "#C0392B"}
    bg_clr     = {3: "#d5f5e3", 2: "#fef9e7", 1: "#fdebd0", 0: "#fadbd8"}
    tok = get_tokenizer()
    for ri, (_, row) in enumerate(samples.iterrows()):
        gt = {
            "i": int(row.get("label_text", 0)),
            "d": int(row.get("label_damage", 0)),
            "h": int(row.get("label_humanitarian", 0)),
        }
        img_id  = str(row.get("image_id", ""))
        raw_pil = _load_pil(img_id)
        ax_img  = fig.add_subplot(gs[ri, 0])
        if raw_pil:
            ax_img.imshow(raw_pil)
        else:
            ax_img.set_facecolor("#ddd")
        ax_img.axis("off")
        ev  = str(row.get("event_name", "")).replace("_", " ")[:22]
        tw  = str(row.get("tweet_text", ""))[:55]
        gt_str = (f"[{ev}]\n\"{tw}...\"\n\n"
                  f"GT Inform: {INFORM_MAP.get(gt['i'], '?')[:9]}\n"
                  f"GT Damage: {DAMAGE_MAP.get(gt['d'], '?')[:11]}\n"
                  f"GT Human:  {HUMAN_MAP.get(gt['h'], '?')[:13]}")
        ax_img.set_title(gt_str, fontsize=5.5, loc="left", pad=2)
        if raw_pil:
            img_t = EVAL_TF(raw_pil).unsqueeze(0).to(device)
        else:
            img_t = torch.zeros(1, 3, cfg.img_size, cfg.img_size, device=device)
        enc  = tok(str(row.get("tweet_text", "")), max_length=cfg.max_len,
                   padding="max_length", truncation=True, return_tensors="pt")
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        for ci, (mname, (model, stage)) in enumerate(registry.items()):
            ax = fig.add_subplot(gs[ri, ci + 1])
            logits = _forward(model, img_t, ids, mask, stage)
            pi  = logits["informative"].argmax(1).item()
            pd_ = logits["damage"].argmax(1).item()
            ph  = logits["humanitarian"].argmax(1).item()
            ok_i = pi  == gt["i"]
            ok_d = pd_ == gt["d"]
            ok_h = ph  == gt["h"]
            n_ok = sum([ok_i, ok_d, ok_h])
            ax.set_facecolor(bg_clr[n_ok])
            for sp in ax.spines.values():
                sp.set_edgecolor(border_clr[n_ok])
                sp.set_linewidth(2.2)
            chk = lambda b: "+" if b else "-"
            txt = (f"{chk(ok_i)} {INFORM_MAP.get(pi,'?')[:10]}\n"
                   f"{chk(ok_d)} {DAMAGE_MAP.get(pd_,'?')[:12]}\n"
                   f"{chk(ok_h)} {HUMAN_MAP.get(ph,'?')[:14]}")
            ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=6.5,
                    transform=ax.transAxes, fontfamily="monospace")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
    handles = [
        mpatches.Patch(facecolor="#d5f5e3", edgecolor="#27AE60", lw=2, label="All 3 correct"),
        mpatches.Patch(facecolor="#fef9e7", edgecolor="#F9BF3B", lw=2, label="2 correct"),
        mpatches.Patch(facecolor="#fdebd0", edgecolor="#E67E22", lw=2, label="1 correct"),
        mpatches.Patch(facecolor="#fadbd8", edgecolor="#C0392B", lw=2, label="All wrong"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Multi-Model Qualitative Comparison\n"
                 "All 6 variants evaluated on the same test samples",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = RESULTS_DIR / "multimodel_qualitative.png"
    plt.savefig(str(p), dpi=130, bbox_inches="tight")
    _show_or_close()
    print(f"✅ Saved: {p}")


@torch.no_grad()
def visualise_attention(model, image_path: str, tweet_text: str,
                        save_path: Optional[str] = None) -> None:
    raw   = Image.open(image_path).convert("RGB").resize((224, 224))
    img_t = EVAL_TF(raw).unsqueeze(0).to(device)
    tok   = get_tokenizer()
    enc   = tok(tweet_text, max_length=cfg.max_len, padding="max_length",
                truncation=True, return_tensors="pt")
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    try:
        out    = model(img_t, ids, mask, return_attn=True)
        logits = out[0] if isinstance(out[0], dict) else out
        attn_w = out[-1] if isinstance(out, tuple) else None
        if attn_w is not None:
            attn = attn_w[0].mean(dim=-1).cpu().numpy().reshape(7, 7)
            attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        else:
            attn = np.ones((7, 7))
    except Exception:
        logits = _forward(model, img_t, ids, mask, "dann")
        attn   = np.ones((7, 7))
    pi  = logits["informative"].argmax(1).item()
    pd_ = logits["damage"].argmax(1).item()
    ph  = logits["humanitarian"].argmax(1).item()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(raw)
    axes[0].set_title("Input image", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(raw)
    axes[1].imshow(np.kron(attn, np.ones((32, 32))),
                   alpha=0.55, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Cross-attention (image to text)", fontsize=10)
    axes[1].axis("off")
    axes[2].axis("off")
    axes[2].text(
        0.05, 0.5,
        f"Tweet:\n{tweet_text[:120]}...\n\n"
        f"Informative : {INFORM_MAP.get(pi, '?')}\n"
        f"Damage      : {DAMAGE_MAP.get(pd_, '?')}\n"
        f"Humanitarian: {HUMAN_MAP.get(ph, '?')}",
        transform=axes[2].transAxes, fontsize=8, va="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
    )
    plt.suptitle("CrisisNet — Cross-Attention Visualisation",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    _show_or_close()


def predict(image_path_or_url: str, tweet_text: str, model=None,
            verbose: bool = True) -> dict:
    if model is None:
        best_ckpt = CKPT_DIR / "dann" / "crisisnet_dann_best.pth"
        model = CrisisNetDANN(cfg).to(device)
        if best_ckpt.exists():
            model.load_state_dict(
                torch.load(str(best_ckpt), map_location=device,
                           weights_only=False)["model"], strict=False)
        model.eval()

    if str(image_path_or_url).startswith("http"):
        import io
        import urllib.request
        raw = Image.open(
            io.BytesIO(urllib.request.urlopen(image_path_or_url).read())
        ).convert("RGB")
    else:
        raw = Image.open(image_path_or_url).convert("RGB")

    img_t = EVAL_TF(raw).unsqueeze(0).to(device)
    tok   = get_tokenizer()
    enc   = tok(tweet_text, max_length=cfg.max_len, padding="max_length",
                truncation=True, return_tensors="pt")
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = _forward(model, img_t, ids, mask,
                          "dann" if isinstance(model, CrisisNetDANN) else "finetune")

    def decode(lv, lmap):
        probs = F.softmax(lv, dim=-1).squeeze()
        pred  = int(probs.argmax().item())
        return (lmap.get(pred, "?"),
                {lmap.get(i, "?"): f"{p:.3f}" for i, p in enumerate(probs.tolist())})

    ri, ci = decode(logits["informative"],  INFORM_MAP)
    rd, cd = decode(logits["damage"],       DAMAGE_MAP)
    rh, ch = decode(logits["humanitarian"], HUMAN_MAP)

    if verbose:
        print(f"\nTweet       : {tweet_text[:100]}")
        print(f"Informative : {ri}  (conf={ci.get(ri, '?')})")
        print(f"Damage      : {rd}  (conf={cd.get(rd, '?')})")
        print(f"Humanitarian: {rh}  (conf={ch.get(rh, '?')})")

    return {
        "informative": ri, "informative_conf": ci,
        "damage": rd,      "damage_conf": cd,
        "humanitarian": rh, "humanitarian_conf": ch,
    }


# =============================================================================
# CLI — argument parsing and stage dispatch
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CrisisNet — Multimodal Crisis Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download CrisisMMD v2.0 from HuggingFace (~1.8 GB)")

    tr = sub.add_parser("train", help="Train one or more pipeline stages")
    tr.add_argument("--stage",
                    choices=["ssl", "finetune", "dann", "baselines", "all"],
                    default="all",
                    help="Stage to run (default: all)")
    tr.add_argument("--no-resume", action="store_true",
                    help="Start from scratch even if a checkpoint exists")
    tr.add_argument("--small-batch", action="store_true",
                    help="Halve all batch sizes (for GPUs < 8 GB or CPU)")
    tr.add_argument("--epochs-ssl",      type=int, help="Override SSL epochs")
    tr.add_argument("--epochs-finetune", type=int, help="Override fine-tune epochs")
    tr.add_argument("--epochs-dann",     type=int, help="Override DANN epochs")

    ev = sub.add_parser("evaluate", help="Run full evaluation for all 6 variants")
    ev.add_argument("--show-plots", action="store_true",
                    help="Display plots interactively (requires a display)")

    pr = sub.add_parser("predict", help="Inference on a single image + tweet")
    pr.add_argument("--image",      required=True, help="Path to image file or URL")
    pr.add_argument("--text",       required=True, help="Tweet text")
    pr.add_argument("--checkpoint", default=None,
                    help="Path to .pth checkpoint (default: Stage 3 best)")

    return parser.parse_args()


def _apply_overrides(args: argparse.Namespace) -> None:
    global SHOW_PLOTS
    if getattr(args, "show_plots", False):
        SHOW_PLOTS = True
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
    if getattr(args, "small_batch", False):
        cfg.ssl_bs  //= 2
        cfg.ft_bs   //= 2
        cfg.dann_bs //= 2
        print(f"Small-batch: ssl_bs={cfg.ssl_bs}  ft_bs={cfg.ft_bs}  dann_bs={cfg.dann_bs}")
    if getattr(args, "epochs_ssl", None):
        cfg.ssl_epochs = args.epochs_ssl
    if getattr(args, "epochs_finetune", None):
        cfg.ft_epochs = args.epochs_finetune
    if getattr(args, "epochs_dann", None):
        cfg.dann_epochs = args.epochs_dann


def _load_model_from_ckpt(cls, ckpt_name: str, stage: str):
    m = cls(cfg).to(device)
    p = CKPT_DIR / stage / f"{ckpt_name}_best.pth"
    if p.exists():
        m.load_state_dict(
            torch.load(str(p), map_location=device,
                       weights_only=False)["model"], strict=False)
        print(f"  Loaded: {p.name}")
    else:
        print(f"  ⚠  Not found: {p} — using random weights.")
    return m


def main() -> None:
    args = _parse_args()
    _apply_overrides(args)

    print(f"\nDevice  : {device}")
    if torch.cuda.is_available():
        p    = torch.cuda.get_device_properties(0)
        vram = p.total_memory / 1e9
        print(f"GPU     : {p.name}  ({vram:.1f} GB VRAM)")
        if vram < 8:
            print("⚠  Low VRAM — consider --small-batch if you get OOM errors.")
    else:
        print("⚠  No GPU. Training will be very slow on CPU.")
        print("   Consider Google Colab for training; use this script for inference only.")
    print(f"Workers : {NUM_WORKERS}  |  AMP: {cfg.use_amp}")

    # ── DOWNLOAD ──────────────────────────────────────────────────
    if args.command == "download":
        download_dataset()
        return

    # ── PREDICT ───────────────────────────────────────────────────
    if args.command == "predict":
        ckpt_path = args.checkpoint or str(CKPT_DIR / "dann" / "crisisnet_dann_best.pth")
        m = CrisisNetDANN(cfg).to(device)
        if Path(ckpt_path).exists():
            m.load_state_dict(
                torch.load(ckpt_path, map_location=device,
                           weights_only=False)["model"], strict=False)
            print(f"Loaded: {ckpt_path}")
        else:
            print(f"⚠  Checkpoint not found: {ckpt_path}")
        m.eval()
        predict(args.image, args.text, model=m)
        return

    # ── All commands that need the dataset ────────────────────────
    full_df = load_dataset_local()
    train_df, val_df, test_df = build_splits(full_df)
    print(f"Split  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    train_ds = CrisisMMDDataset(train_df, "train")
    val_ds   = CrisisMMDDataset(val_df,   "val")
    test_ds  = CrisisMMDDataset(test_df,  "test")
    ssl_ds   = UnlabeledPairDataset(full_df)
    resume   = not getattr(args, "no_resume", False)

    # ── TRAIN ─────────────────────────────────────────────────────
    if args.command == "train":
        stage      = args.stage
        ssl_model  = None
        ft_model   = None
        dann_model = None

        if stage in ("ssl", "all"):
            ssl_model = train_ssl(ssl_ds, resume=resume)

        if stage in ("finetune", "all"):
            ft_model = train_finetune(
                train_ds, val_ds, train_df,
                ssl_model_or_path=ssl_model, resume=resume)

        if stage in ("dann", "all"):
            dann_model = train_dann(
                train_ds, val_ds, train_df,
                ft_model_or_path=ft_model, resume=resume)

        if stage in ("baselines", "all"):
            train_baseline(ImageOnlyModel(cfg).to(device),
                           "image_only",      train_ds, val_ds, train_df, 10, resume)
            train_baseline(TextOnlyModel(cfg).to(device),
                           "text_only",       train_ds, val_ds, train_df, 10, resume)
            train_baseline(LateFusionModel(cfg).to(device),
                           "late_fusion",     train_ds, val_ds, train_df, 10, resume)
            train_baseline(CrisisNet(cfg).to(device),
                           "nossl_crisisnet", train_ds, val_ds, train_df, 15, resume)
        return

    # ── EVALUATE ──────────────────────────────────────────────────
    if args.command == "evaluate":
        img_model   = _load_model_from_ckpt(ImageOnlyModel,  "image_only",      "baselines")
        txt_model   = _load_model_from_ckpt(TextOnlyModel,   "text_only",       "baselines")
        late_model  = _load_model_from_ckpt(LateFusionModel, "late_fusion",     "baselines")
        nossl_model = _load_model_from_ckpt(CrisisNet,       "nossl_crisisnet", "baselines")
        ft_model    = _load_model_from_ckpt(CrisisNet,       "crisisnet",       "finetune")
        dann_model  = _load_model_from_ckpt(CrisisNetDANN,   "crisisnet_dann",  "dann")

        print("\nDataset analysis...")
        plot_dataset_analysis(full_df, train_df, val_df, test_df)

        print("\nFull evaluation — all 6 variants...")
        run_full_evaluation(img_model, txt_model, late_model, nossl_model,
                            ft_model, dann_model, test_ds, test_df)

        print("\nMulti-model qualitative comparison...")
        eval_registry = OrderedDict([
            ("1. Image-only",         (img_model,   "finetune")),
            ("2. Text-only",          (txt_model,   "finetune")),
            ("3. Late Fusion",        (late_model,  "finetune")),
            ("4. CrisisNet (no SSL)", (nossl_model, "finetune")),
            ("5. CrisisNet + SSL",    (ft_model,    "finetune")),
            ("6. CrisisNet+SSL+DANN", (dann_model,  "dann")),
        ])
        plot_multimodel_qualitative(eval_registry, test_df)

        print("\nTraining curves...")
        plot_training_curves()

        print("\nCross-disaster transfer...")
        cross_disaster_heatmap(ft_model, dann_model, full_df)

        # Attention demo on first test sample
        sample   = test_df.iloc[0]
        img_path = None
        for ext in (".jpg", ".png"):
            p = IMAGE_DIR / (str(sample.get("image_id", "")) + ext)
            if p.exists():
                img_path = str(p)
                break
        if img_path:
            visualise_attention(
                dann_model, img_path,
                str(sample.get("tweet_text", "")),
                save_path=str(RESULTS_DIR / "attention_demo.png"))

        print(f"\n✅ All results saved to: {RESULTS_DIR}")
        return


if __name__ == "__main__":
    main()
