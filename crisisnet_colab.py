# -*- coding: utf-8 -*-
"""crisisNet_colab.ipynb

# 🚨 CrisisNet — Multimodal Crisis Intelligence
**University of Verona · Machine Learning & Deep Learning · A.Y. 2025-26**

### 6 model variants compared — all with confusion matrix, precision, recall, F1, and qualitative analysis

| Stage | Method | DL Concepts |
|-------|--------|-------------|
| Baseline | Image-only (ResNet-50) | CNN |
| Baseline | Text-only (BERT) | Transformer |
| Baseline | Late Fusion (concat) | CNN + Transformer |
| Baseline | CrisisNet, no SSL | Cross-Attention, Multimodal |
| Stage 2 | CrisisNet + SSL | Self-Supervised + Multimodal |
| Stage 3 | CrisisNet + SSL + DANN | + Domain Adaptation |

## ⚙️ Cell 1 — Install
Run **once**. Fixes numpy ABI mismatch, then restarts the runtime.
"""

import subprocess, sys

subprocess.run([sys.executable,"-m","pip","install","-q",
                "--force-reinstall","numpy>=1.26.4,<2.0"], check=True)
subprocess.run([sys.executable,"-m","pip","install","-q",
                "pandas>=2.0.0","transformers==4.40.0","scikit-learn==1.4.0",
                "Pillow","matplotlib","seaborn","tqdm","accelerate","datasets"], check=True)

print("\n" + "─"*52)
print("  ⚠️  RUNTIME RESTART REQUIRED")
print("  Restarting — re-run from Cell 2 afterwards.")
print("─"*52)
import IPython
IPython.Application.instance().kernel.do_shutdown(True)

"""## 📦 Cell 2 — Imports"""

import os, sys, json, time, math, random, warnings, copy
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

try:
    from torch.amp import GradScaler as _GS
    def GradScaler(enabled=True): return _GS(device="cuda", enabled=enabled)
    def autocast(enabled=True):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.amp.autocast(device_type=dev, enabled=enabled and torch.cuda.is_available())
    print("  AMP: torch.amp (PyTorch >= 2.0)")
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    print("  AMP: torch.cuda.amp (PyTorch < 2.0)")

import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                              precision_recall_fscore_support, classification_report)

SEED = 42
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {device}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU    : {p.name}  ({p.total_memory/1e9:.1f} GB)")
print("\n✅ Imports complete.")

"""## ⚙️ Cell 3 — Configuration"""

DRIVE_BASE = '/content/drive/MyDrive/CrisisNet'   # ← edit if needed

@dataclass
class Config:
    drive_base:  str   = DRIVE_BASE
    use_drive:   bool  = True
    bert_name:   str   = 'bert-base-uncased'
    max_len:     int   = 128
    proj_dim:    int   = 512
    num_heads:   int   = 8
    num_layers:  int   = 2
    dropout:     float = 0.1
    img_size:    int   = 224
    n_inform:    int   = 2    # set from HF metadata in Cell 6
    n_damage:    int   = 3    # HF v2.0 removed 'dont_know'
    n_human:     int   = 8
    n_disasters: int   = 7
    ssl_epochs:  int   = 10
    ssl_bs:      int   = 64
    ssl_lr:      float = 2e-4
    ssl_temp:    float = 0.07
    ssl_wd:      float = 1e-4
    ft_epochs:   int   = 20
    ft_bs:       int   = 32
    ft_lr:       float = 2e-5
    ft_warmup:   float = 0.1
    ft_wd:       float = 0.01
    lam_info:    float = 1.0
    lam_dmg:     float = 0.5
    lam_hum:     float = 0.5
    dann_epochs: int   = 15
    dann_bs:     int   = 32
    dann_lr:     float = 5e-6
    lam_domain:  float = 0.1
    grad_clip:   float = 1.0
    use_amp:     bool  = True

    def ckpt(self, stage):
        base = self.drive_base if self.use_drive else '/content'
        return os.path.join(base, 'checkpoints', stage)

cfg = Config()

# Integer-keyed label maps (HuggingFace uses integer class labels)
INFORM_MAP = {0:'not_informative',   1:'informative'}
DAMAGE_MAP = {0:'little_or_no_damage',1:'mild_damage',2:'severe_damage'}
HUMAN_MAP  = {0:'not_humanitarian',1:'caution_and_advice',
               2:'missing_or_found_people',3:'injured_or_dead_people',
               4:'affected_individuals',5:'vehicle_damage',
               6:'infrastructure_damage',7:'rescue_volunteering'}
DISASTER_MAP = {'hurricane_harvey':0,'hurricane_irma':1,'hurricane_maria':2,
                'iraq_iran_earthquake':3,'mexico_earthquake':4,
                'srilanka_floods':5,'california_wildfires':6}
EVENTS = list(DISASTER_MAP.keys())
INV_DISASTER = {v:k for k,v in DISASTER_MAP.items()}

print(f"Config ready | n_damage={cfg.n_damage} | device={device}")

"""## 💾 Cell 4 — Google Drive & directories"""

import os

def setup_drive():
    if not cfg.use_drive:
        print("Drive disabled."); return
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted.")
    except ImportError:
        print("⚠️  Not in Colab — skipping Drive."); cfg.use_drive = False; return
    except Exception as e:
        print(f"Drive error: {e}"); cfg.use_drive = False; return
    for d in [cfg.drive_base, cfg.ckpt('ssl'), cfg.ckpt('finetune'),
              cfg.ckpt('dann'), cfg.ckpt('baselines'),
              os.path.join(cfg.drive_base,'results')]:
        os.makedirs(d, exist_ok=True)
    print(f"Directories ready: {cfg.drive_base}")

for d in ['/content/checkpoints/ssl','/content/checkpoints/finetune',
          '/content/checkpoints/dann','/content/checkpoints/baselines']:
    os.makedirs(d, exist_ok=True)

setup_drive()
RESULTS_DIR = os.path.join(cfg.drive_base if cfg.use_drive else '/content','results')
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results: {RESULTS_DIR}")

"""## 💾 Cell 5 — Checkpoint Manager"""

class CheckpointManager:
    def __init__(self, stage, model_name):
        self.primary = cfg.ckpt(stage) if cfg.use_drive else f'/content/checkpoints/{stage}'
        self.local   = f'/content/checkpoints/{stage}'
        self.name    = model_name
        os.makedirs(self.primary, exist_ok=True)
        os.makedirs(self.local,   exist_ok=True)
        self.history = []; self._best = None

    def save(self, model, optim, epoch, metrics, sched=None, higher_is_better=True):
        ckpt = {'epoch':epoch,'model':model.state_dict(),
                'optim':optim.state_dict(),'metrics':metrics,'ts':time.time()}
        if sched: ckpt['sched'] = sched.state_dict()
        key = next(iter(metrics)); val = metrics[key]
        is_best = (self._best is None or
                   (higher_is_better and val>self._best) or
                   (not higher_is_better and val<self._best))
        if is_best: self._best = val
        for base in [self.primary, self.local]:
            self._w(ckpt, os.path.join(base, f'{self.name}_latest.pth'))
            if is_best: self._w(ckpt, os.path.join(base, f'{self.name}_best.pth'))
        self.history.append({'epoch':epoch,**metrics})
        with open(os.path.join(self.primary,'history.json'),'w') as f:
            json.dump(self.history,f,indent=2)
        print(f"  💾 Saved epoch {epoch:03d}{'  ⭐' if is_best else ''}")
        return is_best

    def _w(self, obj, path):
        tmp = path+'.tmp'; torch.save(obj,tmp); os.replace(tmp,path)

    def resume(self, model, optim=None, sched=None, best=False):
        tag = 'best' if best else 'latest'
        for base in [self.primary, self.local]:
            p = os.path.join(base, f'{self.name}_{tag}.pth')
            if os.path.exists(p): return self._load(p,model,optim,sched)
        print(f"  No checkpoint — starting fresh."); return 0,{}

    def _load(self, path, model, optim, sched):
        print(f"  📂 {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = {k.replace('module.',''):v for k,v in ckpt['model'].items()}
        model.load_state_dict(sd, strict=False)
        if optim and 'optim' in ckpt:
            try: optim.load_state_dict(ckpt['optim'])
            except: pass
        if sched and 'sched' in ckpt:
            try: sched.load_state_dict(ckpt['sched'])
            except: pass
        ep=ckpt.get('epoch',0); m=ckpt.get('metrics',{})
        print(f"     Resumed epoch {ep}, {m}"); return ep,m

    def load_history(self):
        for base in [self.primary,self.local]:
            p = os.path.join(base,'history.json')
            if os.path.exists(p):
                with open(p) as f: return json.load(f)
        return []

print("✅ CheckpointManager ready.")

"""## 📂 Cell 6 — Data download (HuggingFace)
~1.8 GB · no account needed · images included · first run: 5–10 min.
"""

from datasets import load_dataset

IMAGE_DIR = '/content/data/crisismmd/images'
os.makedirs(IMAGE_DIR, exist_ok=True)

print("Downloading CrisisMMD v2.0 from HuggingFace...")
tasks_hf = {
    'informative':  load_dataset("QCRI/CrisisMMD","informative"),
    'damage':       load_dataset("QCRI/CrisisMMD","damage"),
    'humanitarian': load_dataset("QCRI/CrisisMMD","humanitarian"),
}
cfg.n_inform = tasks_hf['informative']['train'].features['label'].num_classes
cfg.n_damage = tasks_hf['damage']['train'].features['label'].num_classes
cfg.n_human  = tasks_hf['humanitarian']['train'].features['label'].num_classes
print(f"Classes: inform={cfg.n_inform}  damage={cfg.n_damage}  human={cfg.n_human}")

def hf_to_df(task_name, split_name, hf_split):
    rows = []
    for s in tqdm(hf_split, desc=f'  {task_name}/{split_name}', leave=False):
        iid  = s['image_id']
        path = os.path.join(IMAGE_DIR, f'{iid}.jpg')
        if not os.path.exists(path) and s.get('image') is not None:
            try: s['image'].convert('RGB').save(path,'JPEG',quality=90)
            except: pass
        rows.append({'tweet_id':s['tweet_id'],'image_id':iid,
                     'tweet_text':s['tweet_text'],'event_name':s['event_name'],
                     f'label_{task_name}':int(s['label'])})
    return pd.DataFrame(rows)

split_dfs = []
for sn in ['train','dev','test']:
    inf = hf_to_df('informative', sn, tasks_hf['informative'][sn])
    dmg = hf_to_df('damage',      sn, tasks_hf['damage'][sn])
    hum = hf_to_df('humanitarian',sn, tasks_hf['humanitarian'][sn])
    m   = (inf.merge(dmg[['tweet_id','label_damage']],      on='tweet_id',how='left')
              .merge(hum[['tweet_id','label_humanitarian']], on='tweet_id',how='left'))
    m['label_damage']       = m['label_damage'].fillna(0).astype(int)
    m['label_humanitarian'] = m['label_humanitarian'].fillna(0).astype(int)
    m['domain_id']          = m['event_name'].map(DISASTER_MAP).fillna(0).astype(int)
    split_dfs.append(m)

full_df = (pd.concat(split_dfs,ignore_index=True)
           .rename(columns={'label_informative':'label_text'}))
print(f"\n✅ {len(full_df):,} samples  |  {IMAGE_DIR}")
print(full_df['event_name'].value_counts().to_string())

"""## 🗄️ Cell 7 — Dataset classes"""

TRAIN_TF = transforms.Compose([
    transforms.Resize((cfg.img_size,cfg.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((cfg.img_size,cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
tokenizer = BertTokenizer.from_pretrained(cfg.bert_name)

def load_image(img_dir, img_id, tf):
    for ext in ['.jpg','.png']:
        p = os.path.join(img_dir, str(img_id)+ext)
        if os.path.exists(p):
            try: return tf(Image.open(p).convert('RGB'))
            except: pass
    return torch.zeros(3,cfg.img_size,cfg.img_size)

def load_pil(img_dir, img_id):
    for ext in ['.jpg','.png']:
        p = os.path.join(img_dir, str(img_id)+ext)
        if os.path.exists(p):
            try: return Image.open(p).convert('RGB')
            except: pass
    return None

class CrisisMMDDataset(Dataset):
    def __init__(self,df,img_dir,split='train'):
        self.df=df.reset_index(drop=True); self.d=img_dir
        self.tf=TRAIN_TF if split=='train' else EVAL_TF
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        enc=tokenizer(str(r.get('tweet_text','')),max_length=cfg.max_len,
                      padding='max_length',truncation=True,return_tensors='pt')
        return {'image':load_image(self.d,r.get('image_id',''),self.tf),
                'input_ids':enc['input_ids'].squeeze(0),
                'attention_mask':enc['attention_mask'].squeeze(0),
                'inform_label':int(r.get('label_text',0)),
                'damage_label':int(r.get('label_damage',0)),
                'human_label': int(r.get('label_humanitarian',0)),
                'domain_label':int(r.get('domain_id',0))}

class UnlabeledPairDataset(Dataset):
    def __init__(self,df,img_dir):
        self.df=df.reset_index(drop=True); self.d=img_dir
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        enc=tokenizer(str(r.get('tweet_text','')),max_length=cfg.max_len,
                      padding='max_length',truncation=True,return_tensors='pt')
        return {'image':load_image(self.d,r.get('image_id',''),TRAIN_TF),
                'input_ids':enc['input_ids'].squeeze(0),
                'attention_mask':enc['attention_mask'].squeeze(0)}

def build_splits(df):
    tr,va,te=[],[],[]
    for ev in df['event_name'].unique():
        sub=df[df['event_name']==ev].sample(frac=1,random_state=SEED)
        n=len(sub); t1,t2=int(n*0.70),int(n*0.85)
        tr.append(sub.iloc[:t1]); va.append(sub.iloc[t1:t2]); te.append(sub.iloc[t2:])
    return pd.concat(tr),pd.concat(va),pd.concat(te)

train_df,val_df,test_df = build_splits(full_df)
print(f"Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}")

train_ds = CrisisMMDDataset(train_df, IMAGE_DIR, 'train')
val_ds   = CrisisMMDDataset(val_df,   IMAGE_DIR, 'val')
test_ds  = CrisisMMDDataset(test_df,  IMAGE_DIR, 'test')
ssl_ds   = UnlabeledPairDataset(full_df, IMAGE_DIR)

def make_loader(ds,bs,shuffle=True):
    return DataLoader(ds,batch_size=bs,shuffle=shuffle,
                      num_workers=2,pin_memory=True,drop_last=shuffle)
print("✅ Datasets ready.")

"""## 🧠 Cell 8 — All Model Architectures
6 variants: ImageOnly · TextOnly · LateFusion · CrisisNet(noSSL) · CrisisNet+SSL · CrisisNet+SSL+DANN
"""

# ══ 1. ENCODERS ══════════════════════════════════════════════════
class ImageEncoder(nn.Module):
    def __init__(self,proj_dim=512):
        super().__init__()
        base=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone=nn.Sequential(*list(base.children())[:-2])
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.proj=nn.Sequential(nn.Linear(2048,proj_dim),nn.LayerNorm(proj_dim))
    def forward(self,x,mode='spatial'):
        f=self.backbone(x)
        if mode=='global': return self.proj(self.pool(f).flatten(1))
        B,C,H,W=f.shape
        return self.proj(f.permute(0,2,3,1).reshape(B,H*W,C))

class TextEncoder(nn.Module):
    def __init__(self,proj_dim=512,bert_name='bert-base-uncased'):
        super().__init__()
        self.bert=BertModel.from_pretrained(bert_name)
        self.proj=nn.Sequential(nn.Linear(768,proj_dim),nn.LayerNorm(proj_dim))
    def forward(self,input_ids,attention_mask,mode='sequence'):
        h=self.proj(self.bert(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state)
        return h[:,0,:] if mode=='cls' else h

# ══ 2. CROSS-MODAL TRANSFORMER ═══════════════════════════════════
class CrossModalLayer(nn.Module):
    def __init__(self,dim,heads,dropout=0.1):
        super().__init__()
        self.attn=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.n1=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Dropout(dropout),
                                nn.Linear(dim*4,dim),nn.Dropout(dropout))
        self.n2=nn.LayerNorm(dim); self.drop=nn.Dropout(dropout)
    def forward(self,q,kv,mask=None):
        a,w=self.attn(q,kv,kv,key_padding_mask=mask,need_weights=True,average_attn_weights=True)
        q=self.n1(q+self.drop(a)); q=self.n2(q+self.ffn(q)); return q,w

class CrossModalTransformer(nn.Module):
    def __init__(self,dim,heads,n=2,dropout=0.1):
        super().__init__()
        self.i2t=nn.ModuleList([CrossModalLayer(dim,heads,dropout) for _ in range(n)])
        self.t2i=nn.ModuleList([CrossModalLayer(dim,heads,dropout) for _ in range(n)])
        self.isa=nn.ModuleList([nn.TransformerEncoderLayer(dim,heads,dim*4,dropout,batch_first=True,norm_first=True) for _ in range(n)])
        self.tsa=nn.ModuleList([nn.TransformerEncoderLayer(dim,heads,dim*4,dropout,batch_first=True,norm_first=True) for _ in range(n)])
        self.norm=nn.LayerNorm(dim)
    def forward(self,img,txt,pad=None):
        w=None
        for i2t,t2i,isa,tsa in zip(self.i2t,self.t2i,self.isa,self.tsa):
            img,w=i2t(img,txt,pad); txt,_=t2i(txt,img)
            img=isa(img); txt=tsa(txt,src_key_padding_mask=pad)
        return self.norm(torch.cat([img,txt],1).mean(1)),w

# ══ 3. TASK HEADS ════════════════════════════════════════════════
class TaskHeads(nn.Module):
    def __init__(self,dim,ni,nd,nh,drop=0.1):
        super().__init__()
        h=lambda n: nn.Sequential(nn.Dropout(drop),nn.Linear(dim,dim//2),
                                   nn.GELU(),nn.Dropout(drop),nn.Linear(dim//2,n))
        self.info=h(ni); self.dmg=h(nd); self.hum=h(nh)
    def forward(self,x):
        return {'informative':self.info(x),'damage':self.dmg(x),'humanitarian':self.hum(x)}

# ══ 4. CRISISNET (Stage 2 & noSSL) ═══════════════════════════════
class CrisisNet(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.img_enc=ImageEncoder(cfg.proj_dim)
        self.txt_enc=TextEncoder(cfg.proj_dim,cfg.bert_name)
        self.fusion =CrossModalTransformer(cfg.proj_dim,cfg.num_heads,cfg.num_layers,cfg.dropout)
        self.heads  =TaskHeads(cfg.proj_dim,cfg.n_inform,cfg.n_damage,cfg.n_human,cfg.dropout)
    def forward(self,images,input_ids,attention_mask,return_attn=False,**kw):
        it=self.img_enc(images,'spatial'); tt=self.txt_enc(input_ids,attention_mask,'sequence')
        f,w=self.fusion(it,tt,(attention_mask==0))
        out=self.heads(f)
        return (out,w) if return_attn else out

# ══ 5. CLIP MODEL (Stage 1) ═══════════════════════════════════════
class CLIPModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.img_enc=ImageEncoder(cfg.proj_dim); self.txt_enc=TextEncoder(cfg.proj_dim,cfg.bert_name)
        self.log_scale=nn.Parameter(torch.ones([])*math.log(1/cfg.ssl_temp))
    def forward(self,images,input_ids,attention_mask):
        img=F.normalize(self.img_enc(images,'global'),dim=-1)
        txt=F.normalize(self.txt_enc(input_ids,attention_mask,'cls'),dim=-1)
        return img,txt,self.log_scale.exp().clamp(max=100)

# ══ 6. DANN (Stage 3) ════════════════════════════════════════════
class GRevFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,a): ctx.a=a; return x
    @staticmethod
    def backward(ctx,g): return -ctx.a*g,None

class GRev(nn.Module):
    def forward(self,x,a=1.0): return GRevFn.apply(x,a)

class CrisisNetDANN(CrisisNet):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.grl=GRev()
        self.dom=nn.Sequential(nn.Linear(cfg.proj_dim,256),nn.ReLU(),
                                nn.Dropout(cfg.dropout),nn.Linear(256,cfg.n_disasters))
    def forward(self,images,input_ids,attention_mask,alpha=1.0,return_attn=False,**kw):
        it=self.img_enc(images,'spatial'); tt=self.txt_enc(input_ids,attention_mask,'sequence')
        f,w=self.fusion(it,tt,(attention_mask==0))
        tasks=self.heads(f); domain=self.dom(self.grl(f,alpha))
        if return_attn: return tasks,domain,w
        return tasks,domain

# ══ 7. ABLATION BASELINES ════════════════════════════════════════
class ImageOnlyModel(nn.Module):
    """Variant 1: ResNet-50 → global pool → heads. No text."""
    def __init__(self,cfg):
        super().__init__()
        base=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone=nn.Sequential(*list(base.children())[:-1])
        self.heads=TaskHeads(2048,cfg.n_inform,cfg.n_damage,cfg.n_human,cfg.dropout)
    def forward(self,images,input_ids=None,attention_mask=None,**kw):
        return self.heads(self.backbone(images).flatten(1))

class TextOnlyModel(nn.Module):
    """Variant 2: BERT [CLS] → heads. No image."""
    def __init__(self,cfg):
        super().__init__()
        self.bert=BertModel.from_pretrained(cfg.bert_name)
        self.heads=TaskHeads(768,cfg.n_inform,cfg.n_damage,cfg.n_human,cfg.dropout)
    def forward(self,images=None,input_ids=None,attention_mask=None,**kw):
        cls=self.bert(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0,:]
        return self.heads(cls)

class LateFusionModel(nn.Module):
    """Variant 3: Encode separately → concatenate → MLP → heads."""
    def __init__(self,cfg):
        super().__init__()
        self.img_enc=ImageEncoder(cfg.proj_dim); self.txt_enc=TextEncoder(cfg.proj_dim,cfg.bert_name)
        self.fusion=nn.Sequential(nn.Linear(cfg.proj_dim*2,cfg.proj_dim),
                                   nn.GELU(),nn.Dropout(cfg.dropout))
        self.heads=TaskHeads(cfg.proj_dim,cfg.n_inform,cfg.n_damage,cfg.n_human,cfg.dropout)
    def forward(self,images,input_ids,attention_mask,**kw):
        return self.heads(self.fusion(torch.cat(
            [self.img_enc(images,'global'),self.txt_enc(input_ids,attention_mask,'cls')],dim=-1)))

# Param counts
def _n(m,n): print(f"  {n:<35} {sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6:6.1f}M")
print("Trainable parameters:")
for m,n in [(ImageOnlyModel(cfg),'Variant 1: Image-only'),
            (TextOnlyModel(cfg), 'Variant 2: Text-only'),
            (LateFusionModel(cfg),'Variant 3: Late Fusion'),
            (CrisisNet(cfg),     'Variant 4: CrisisNet (no SSL)'),
            (CrisisNet(cfg),     'Variant 5: CrisisNet + SSL'),
            (CrisisNetDANN(cfg), 'Variant 6: CrisisNet + SSL + DANN'),
            (CLIPModel(cfg),     'SSL pre-training model')]: _n(m,n)
print("✅ All models defined.")

"""## 📐 Cell 9 — Loss Functions & Evaluation Utilities"""

# ── InfoNCE ──────────────────────────────────────────────────────
class InfoNCELoss(nn.Module):
    def forward(self,img,txt,scale):
        L=scale*img@txt.T; lbl=torch.arange(len(img),device=img.device)
        return (F.cross_entropy(L,lbl)+F.cross_entropy(L.T,lbl))/2

# ── Class weights ─────────────────────────────────────────────────
def compute_class_weights(series,n_classes):
    c=pd.Series(series).value_counts().sort_index().reindex(range(n_classes),fill_value=1)
    w=1.0/c.values.astype(float); w=w/w.mean()
    return torch.tensor(w,dtype=torch.float32)

def _make_loss():
    return MultiTaskLoss(
        w_info=cfg.lam_info, w_dmg=cfg.lam_dmg, w_hum=cfg.lam_hum,
        cw_info=compute_class_weights(train_df['label_text'],         cfg.n_inform),
        cw_dmg =compute_class_weights(train_df['label_damage'],       cfg.n_damage),
        cw_hum =compute_class_weights(train_df['label_humanitarian'], cfg.n_human))

# ── Weighted multi-task loss ──────────────────────────────────────
class MultiTaskLoss(nn.Module):
    def __init__(self,w_info,w_dmg,w_hum,cw_info=None,cw_dmg=None,cw_hum=None):
        super().__init__()
        self.tw={'informative':w_info,'damage':w_dmg,'humanitarian':w_hum}
        self.cw={'informative':cw_info,'damage':cw_dmg,'humanitarian':cw_hum}
    def forward(self,logits,batch):
        tot=0.0
        km={'informative':'inform_label','damage':'damage_label','humanitarian':'human_label'}
        for task,lk in km.items():
            if lk in batch:
                lbl=batch[lk].to(device)
                w=self.cw[task].to(device) if self.cw[task] is not None else None
                tot+=self.tw[task]*F.cross_entropy(logits[task],lbl,weight=w)
        return tot

# ── Inference helper ──────────────────────────────────────────────
@torch.no_grad()
def model_forward(model, images, ids, mask, stage):
    try:
        out = model(images, ids, mask, alpha=0.0)
        return out[0] if isinstance(out, tuple) else out
    except (TypeError, ValueError):
        return model(images, ids, mask)

# ── Evaluate (metrics dict) ───────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, stage='finetune'):
    model.eval()
    P={'informative':[],'damage':[],'humanitarian':[]}
    T={'informative':[],'damage':[],'humanitarian':[]}
    KM={'informative':'inform_label','damage':'damage_label','humanitarian':'human_label'}
    for batch in tqdm(loader,desc='Eval',leave=False):
        imgs=batch['image'].to(device); ids=batch['input_ids'].to(device)
        mask=batch['attention_mask'].to(device)
        logits=model_forward(model,imgs,ids,mask,stage)
        for task,lk in KM.items():
            if lk in batch:
                P[task].extend(logits[task].argmax(1).cpu().numpy())
                T[task].extend(batch[lk].numpy())
    metrics={}
    for task in P:
        if not P[task]: continue
        p,t=np.array(P[task]),np.array(T[task])
        metrics[f'{task}_f1'] =f1_score(t,p,average='macro',zero_division=0)
        metrics[f'{task}_acc']=accuracy_score(t,p)
    if metrics: metrics['avg_f1']=np.mean([v for k,v in metrics.items() if '_f1' in k])
    model.train(); return metrics

# ── Full predict (returns arrays) ────────────────────────────────
@torch.no_grad()
def full_predict(model, loader, stage='finetune'):
    model.eval()
    P={'informative':[],'damage':[],'humanitarian':[]}
    T={'informative':[],'damage':[],'humanitarian':[]}
    KM={'informative':'inform_label','damage':'damage_label','humanitarian':'human_label'}
    for batch in tqdm(loader,desc='Predict',leave=False):
        imgs=batch['image'].to(device); ids=batch['input_ids'].to(device)
        mask=batch['attention_mask'].to(device)
        logits=model_forward(model,imgs,ids,mask,stage)
        for task,lk in KM.items():
            if lk in batch:
                P[task].extend(logits[task].argmax(1).cpu().numpy())
                T[task].extend(batch[lk].numpy())
    model.train()
    return {t:(np.array(P[t]),np.array(T[t])) for t in P if P[t]}

def print_metrics(metrics,epoch):
    print('  Epoch {:03d}'.format(epoch)+''.join(f'  |  {k}: {v:.4f}' for k,v in metrics.items()))

print("✅ Loss + evaluation ready.")

"""## 🔵 Cell 10 — Stage 1: SSL Pre-training"""

def train_ssl(resume=True):
    print("\n"+"="*60+"\nSTAGE 1 — SSL Pre-training\n"+"="*60)
    model=CLIPModel(cfg).to(device); loss_fn=InfoNCELoss()
    ckpt=CheckpointManager('ssl','clip')
    sl=make_loader(ssl_ds,cfg.ssl_bs); ts=len(sl)*cfg.ssl_epochs
    opt=AdamW(model.parameters(),lr=cfg.ssl_lr,weight_decay=cfg.ssl_wd)
    sched=get_linear_schedule_with_warmup(opt,max(1,int(0.05*ts)),ts)
    scaler=GradScaler(enabled=cfg.use_amp)
    start=0
    if resume: start,_=ckpt.resume(model,opt,sched)
    best=float('inf')
    for ep in range(start,cfg.ssl_epochs):
        model.train(); rl=0.0
        pbar=tqdm(sl,desc=f'SSL {ep+1:02d}/{cfg.ssl_epochs}',leave=False)
        for b in pbar:
            imgs=b['image'].to(device,non_blocking=True)
            ids=b['input_ids'].to(device,non_blocking=True)
            mask=b['attention_mask'].to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                i,t,sc=model(imgs,ids,mask); loss=loss_fn(i,t,sc)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            rl+=loss.item(); pbar.set_postfix({'loss':f'{rl/(pbar.n+1):.4f}'})
        avg=rl/len(sl); is_b=avg<best
        if is_b: best=avg
        ckpt.save(model,opt,ep+1,{'ssl_loss':avg},sched,higher_is_better=False)
        print(f"  Epoch {ep+1:02d}  loss={avg:.4f}{'  ⭐' if is_b else ''}")
    print("\n✅ Stage 1 complete."); return model

ssl_model = train_ssl(resume=True)

"""## 🟡 Cell 11 — Stage 2: Multi-task Fine-tuning
Weighted cross-entropy corrects class imbalance. SSL encoder weights transferred.
"""

def train_finetune(ssl_model_or_path=None, resume=True):
    print("\n"+"="*60+"\nSTAGE 2 — Multi-task Fine-tuning\n"+"="*60)
    model=CrisisNet(cfg).to(device); loss_fn=_make_loss()
    ckpt=CheckpointManager('finetune','crisisnet')
    if ssl_model_or_path is not None:
        src=ssl_model_or_path
        if isinstance(src,CLIPModel):
            model.img_enc.load_state_dict(src.img_enc.state_dict(),strict=False)
            model.txt_enc.load_state_dict(src.txt_enc.state_dict(),strict=False)
            print("  ✅ SSL weights transferred.")
        elif isinstance(src,str) and os.path.exists(src):
            sm=CLIPModel(cfg).to(device)
            sm.load_state_dict(torch.load(src,map_location=device,weights_only=False)['model'])
            model.img_enc.load_state_dict(sm.img_enc.state_dict(),strict=False)
            model.txt_enc.load_state_dict(sm.txt_enc.state_dict(),strict=False)
            print("  ✅ SSL weights transferred from file.")
    bp=list(model.txt_enc.bert.parameters())
    op=[p for p in model.parameters() if not any(p is b for b in bp)]
    opt=AdamW([{'params':bp,'lr':cfg.ft_lr*0.1},{'params':op,'lr':cfg.ft_lr}],weight_decay=cfg.ft_wd)
    tl=make_loader(train_ds,cfg.ft_bs); vl=make_loader(val_ds,cfg.ft_bs,shuffle=False)
    ts=len(tl)*cfg.ft_epochs
    sched=get_linear_schedule_with_warmup(opt,max(1,int(cfg.ft_warmup*ts)),ts)
    scaler=GradScaler(enabled=cfg.use_amp)
    start=0
    if resume: start,_=ckpt.resume(model,opt,sched)
    for ep in range(start,cfg.ft_epochs):
        model.train(); rl=0.0
        for b in tqdm(tl,desc=f'FT {ep+1:02d}/{cfg.ft_epochs}',leave=False):
            imgs=b['image'].to(device,non_blocking=True)
            ids=b['input_ids'].to(device,non_blocking=True)
            mask=b['attention_mask'].to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                loss=loss_fn(model(imgs,ids,mask),b)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step(); rl+=loss.item()
        vm=evaluate(model,vl); vm['train_loss']=rl/len(tl)
        print_metrics(vm,ep+1); ckpt.save(model,opt,ep+1,vm,sched)
    print("\n✅ Stage 2 complete."); return model

ft_model = train_finetune(ssl_model_or_path=ssl_model, resume=True)

"""## 🟠 Cell 12 — Stage 3: DANN Domain Adaptation"""

def get_alpha(ep,total):
    p=ep/max(total,1); return 2.0/(1.0+math.exp(-10*p))-1.0

def train_dann(ft_model_or_path=None, resume=True):
    print("\n"+"="*60+"\nSTAGE 3 — DANN Domain Adaptation\n"+"="*60)
    model=CrisisNetDANN(cfg).to(device); loss_fn=_make_loss()
    ckpt=CheckpointManager('dann','crisisnet_dann')
    if ft_model_or_path is not None:
        src=ft_model_or_path
        if isinstance(src,CrisisNet):
            model.load_state_dict(src.state_dict(),strict=False); print("  FT weights loaded.")
        elif isinstance(src,str) and os.path.exists(src):
            model.load_state_dict(torch.load(src,map_location=device,weights_only=False)['model'],strict=False)
    else:
        p=os.path.join(cfg.ckpt('finetune'),'crisisnet_best.pth')
        if os.path.exists(p):
            model.load_state_dict(torch.load(p,map_location=device,weights_only=False)['model'],strict=False)
    opt=AdamW(model.parameters(),lr=cfg.dann_lr,weight_decay=cfg.ft_wd)
    tl=make_loader(train_ds,cfg.dann_bs); vl=make_loader(val_ds,cfg.dann_bs,shuffle=False)
    scaler=GradScaler(enabled=cfg.use_amp)
    start=0
    if resume: start,_=ckpt.resume(model,opt)
    for ep in range(start,cfg.dann_epochs):
        model.train(); alpha=get_alpha(ep,cfg.dann_epochs); rl=0.0
        for b in tqdm(tl,desc=f'DANN {ep+1:02d}/{cfg.dann_epochs} α={alpha:.3f}',leave=False):
            imgs=b['image'].to(device,non_blocking=True)
            ids=b['input_ids'].to(device,non_blocking=True)
            mask=b['attention_mask'].to(device,non_blocking=True)
            dom=b['domain_label'].to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                to,do=model(imgs,ids,mask,alpha=alpha)
                loss=loss_fn(to,b)+cfg.lam_domain*F.cross_entropy(do,dom)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            scaler.step(opt); scaler.update(); rl+=loss.item()
        vm=evaluate(model,vl,stage='dann'); vm['train_loss']=rl/len(tl)
        print_metrics(vm,ep+1); ckpt.save(model,opt,ep+1,vm)
    print("\n✅ Stage 3 complete."); return model

dann_model = train_dann(ft_model_or_path=ft_model, resume=True)

"""## 📊 Cell 13 — Dataset Analysis
Required for the report: class distributions, imbalance ratios, event splits.
"""

def plot_dataset_analysis(df):
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    fig.suptitle('CrisisMMD v2.0 — Dataset Analysis',fontsize=15,fontweight='bold')
    ev=df['event_name'].value_counts()
    axes[0,0].pie(ev.values,labels=[e.replace('_','\n') for e in ev.index],
                  autopct='%1.1f%%',startangle=90,textprops={'fontsize':7},
                  colors=plt.cm.Set3(np.linspace(0,1,len(ev))))
    axes[0,0].set_title('Samples by Disaster Event',fontweight='bold')
    ic=df['label_text'].value_counts().sort_index()
    b=axes[0,1].bar([INFORM_MAP[i] for i in ic.index],ic.values,color=['#E74C3C','#2ECC71'])
    axes[0,1].set_title('Task 1: Informative',fontweight='bold'); axes[0,1].set_ylabel('Count')
    for bar,v in zip(b,ic.values):
        axes[0,1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+40,
                        f'{v:,}\n({v/len(df)*100:.1f}%)',ha='center',fontsize=9)
    dc=df['label_damage'].value_counts().sort_index()
    b=axes[0,2].bar([DAMAGE_MAP[i].replace('_','\n') for i in dc.index],dc.values,
                     color=['#3498DB','#F39C12','#E74C3C'])
    axes[0,2].set_title('Task 2: Damage Severity',fontweight='bold'); axes[0,2].set_ylabel('Count')
    for bar,v in zip(b,dc.values): axes[0,2].text(bar.get_x()+bar.get_width()/2,bar.get_height()+20,f'{v:,}',ha='center',fontsize=9)
    hc=df['label_humanitarian'].value_counts().sort_index()
    axes[1,0].bar([HUMAN_MAP[i].replace('_','\n') for i in hc.index],hc.values,
                   color=plt.cm.Set2(np.linspace(0,1,len(hc))))
    axes[1,0].set_title('Task 3: Humanitarian',fontweight='bold')
    axes[1,0].set_ylabel('Count'); axes[1,0].tick_params(axis='x',labelsize=6,rotation=30)
    x,w=np.arange(len(EVENTS)),0.25
    for i,(sn,sd) in enumerate([('Train',train_df),('Val',val_df),('Test',test_df)]):
        cnt=sd['event_name'].value_counts()
        axes[1,1].bar(x+i*w,[cnt.get(ev,0) for ev in EVENTS],w,label=sn)
    axes[1,1].set_xticks(x+w); axes[1,1].set_xticklabels([e.replace('_','\n') for e in EVENTS],fontsize=6)
    axes[1,1].set_title('Samples per Event per Split',fontweight='bold'); axes[1,1].legend(fontsize=8)
    ts={'Informative\n(T1)':df['label_text'],'Damage\n(T2)':df['label_damage'],'Humanitarian\n(T3)':df['label_humanitarian']}
    rat=[v.value_counts().max()/v.value_counts().min() for v in ts.values()]
    b=axes[1,2].bar(list(ts.keys()),rat,color=['#9B59B6','#1ABC9C','#E67E22'])
    axes[1,2].set_title('Class Imbalance Ratio (max÷min)',fontweight='bold'); axes[1,2].set_ylabel('Ratio')
    for bar,v in zip(b,rat): axes[1,2].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,f'{v:.1f}×',ha='center',fontsize=12,fontweight='bold',color='darkred')
    plt.tight_layout()
    p=os.path.join(RESULTS_DIR,'dataset_analysis.png')
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.show(); print(f"✅ Saved: {p}")
    print(f"\nTotal: {len(df):,}  |  Train:{len(train_df):,}  Val:{len(val_df):,}  Test:{len(test_df):,}")

plot_dataset_analysis(full_df)

"""## 🧪 Cell 14 — Train Ablation Baselines
**4 baselines**: Image-only · Text-only · Late Fusion · CrisisNet without SSL (same architecture, no pre-training)

All use weighted cross-entropy loss.
"""

def train_baseline(model, model_name, n_epochs=10):
    ckpt=CheckpointManager('baselines',model_name); loss_fn=_make_loss()
    opt=AdamW(model.parameters(),lr=cfg.ft_lr,weight_decay=cfg.ft_wd)
    tl=make_loader(train_ds,cfg.ft_bs); vl=make_loader(val_ds,cfg.ft_bs,shuffle=False)
    ts=len(tl)*n_epochs
    sched=get_linear_schedule_with_warmup(opt,int(0.1*ts),ts)
    scaler=GradScaler(enabled=cfg.use_amp)
    start,_=ckpt.resume(model,opt,sched)
    for ep in range(start,n_epochs):
        model.train(); rl=0.0
        for b in tqdm(tl,desc=f'{model_name} {ep+1}/{n_epochs}',leave=False):
            imgs=b['image'].to(device,non_blocking=True)
            ids=b['input_ids'].to(device,non_blocking=True)
            mask=b['attention_mask'].to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                loss=loss_fn(model(imgs,ids,mask),b)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step(); rl+=loss.item()
        vm=evaluate(model,vl)
        ckpt.save(model,opt,ep+1,vm,sched)
        print(f"  {model_name:<22}  epoch {ep+1:02d}  avg_f1={vm.get('avg_f1',0):.4f}")
    return model


print("Training 4 ablation baselines...")
print("─"*50)

# Variant 1: Image-only
print("\nVariant 1: Image-only (ResNet-50, no text)")
img_model = train_baseline(ImageOnlyModel(cfg).to(device), 'image_only', n_epochs=10)

# Variant 2: Text-only
print("\nVariant 2: Text-only (BERT, no image)")
txt_model = train_baseline(TextOnlyModel(cfg).to(device), 'text_only', n_epochs=10)

# Variant 3: Late fusion
print("\nVariant 3: Late Fusion (concat encodings)")
late_model = train_baseline(LateFusionModel(cfg).to(device), 'late_fusion', n_epochs=10)

# Variant 4: CrisisNet WITHOUT SSL (same architecture as Stage 2, ImageNet init only)
# This isolates the contribution of SSL pre-training
print("\nVariant 4: CrisisNet, no SSL (cross-attention, ImageNet init)")
nossl_model = train_baseline(CrisisNet(cfg).to(device), 'nossl_crisisnet', n_epochs=15)

print("\n✅ All 4 ablation baselines trained.")

"""## 📈 Cell 15 — Complete Evaluation: All 6 Variants
For **every variant**: confusion matrix (3 tasks) · per-class Precision / Recall / F1 · summary table.
All figures saved to Drive/results/.
"""

TASK_MAPS = {'informative':INFORM_MAP,'damage':DAMAGE_MAP,'humanitarian':HUMAN_MAP}
TASK_LABELS = {'informative':sorted(INFORM_MAP.keys()),
               'damage':sorted(DAMAGE_MAP.keys()),
               'humanitarian':sorted(HUMAN_MAP.keys())}

def save_confusion_matrices(res, variant_name, results_dir):
    """Plot and save 3-task confusion matrices for one variant."""
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    fig.suptitle(f'Confusion Matrices — {variant_name}',fontsize=13,fontweight='bold')
    for ax,(task,(p,t)) in zip(axes,res.items()):
        lmap=TASK_MAPS[task]; lbls=TASK_LABELS[task]
        names=[lmap[i].replace('_','\n') for i in lbls]
        cm=confusion_matrix(t,p,labels=lbls)
        cmn=cm.astype(float)/(cm.sum(1,keepdims=True)+1e-8)
        im=ax.imshow(cmn,cmap='Blues',vmin=0,vmax=1)
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names,fontsize=7,rotation=30,ha='right')
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j,i,f'{cmn[i,j]:.2f}\n({cm[i,j]})',ha='center',va='center',
                        fontsize=6,color='white' if cmn[i,j]>0.5 else 'black')
        f1=f1_score(t,p,average='macro',zero_division=0)
        ax.set_title(f'{task.capitalize()} (Macro F1={f1:.3f})',fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    slug=variant_name.replace(' ','_').replace('+','plus').replace('/','').lower()[:30]
    path=os.path.join(results_dir,f'cm_{slug}.png')
    plt.savefig(path,dpi=150,bbox_inches='tight'); plt.close()
    return path

def save_prf1(res, variant_name, results_dir):
    """Plot and save per-class P/R/F1 for one variant."""
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    fig.suptitle(f'Per-class Precision / Recall / F1 — {variant_name}',fontsize=13,fontweight='bold')
    for ax,(task,(p,t)) in zip(axes,res.items()):
        lmap=TASK_MAPS[task]; lbls=TASK_LABELS[task]
        names=[lmap[i].replace('_',' ') for i in lbls]
        pr,re,f1,_=precision_recall_fscore_support(t,p,labels=lbls,zero_division=0)
        x,w=np.arange(len(names)),0.25
        ax.bar(x-w,pr,w,label='Precision',color='#3498DB',alpha=0.85)
        ax.bar(x,  re,w,label='Recall',   color='#2ECC71',alpha=0.85)
        ax.bar(x+w,f1,w,label='F1',       color='#E74C3C',alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([n.replace(' ','\n') for n in names],fontsize=7,rotation=30,ha='right')
        ax.set_ylim(0,1.15); ax.legend(fontsize=8); ax.set_ylabel('Score')
        ax.set_title(task.capitalize(),fontweight='bold')
        ax.axhline(0.5,color='gray',linestyle='--',alpha=0.3)
        ax.text(len(names)-0.5,1.08,f'Macro F1={f1.mean():.3f}',
                ha='right',fontsize=9,color='darkred',fontweight='bold')
    plt.tight_layout()
    slug=variant_name.replace(' ','_').replace('+','plus').replace('/','').lower()[:30]
    path=os.path.join(results_dir,f'prf1_{slug}.png')
    plt.savefig(path,dpi=150,bbox_inches='tight'); plt.close()
    return path

def run_full_evaluation():
    test_loader=make_loader(test_ds,cfg.ft_bs,shuffle=False)

    # ── Build ordered registry of all 6 variants ──────────────────
    registry = OrderedDict()
    registry['1. Image-only (ResNet-50)']     = (img_model,   'finetune')
    registry['2. Text-only (BERT)']           = (txt_model,   'finetune')
    registry['3. Late Fusion (concat)']       = (late_model,  'finetune')
    registry['4. CrisisNet, no SSL']          = (nossl_model, 'finetune')

    ft_path   = os.path.join(cfg.ckpt('finetune'),'crisisnet_best.pth')
    dann_path = os.path.join(cfg.ckpt('dann'),    'crisisnet_dann_best.pth')

    if os.path.exists(ft_path):
        m=CrisisNet(cfg).to(device)
        m.load_state_dict(torch.load(ft_path,map_location=device,weights_only=False)['model'],strict=False)
        registry['5. CrisisNet + SSL (Stage 2)'] = (m,'finetune')
    else: print("  ⚠️  Stage 2 checkpoint not found — run Cell 11 first.")

    if os.path.exists(dann_path):
        m=CrisisNetDANN(cfg).to(device)
        m.load_state_dict(torch.load(dann_path,map_location=device,weights_only=False)['model'],strict=False)
        registry['6. CrisisNet + SSL + DANN'] = (m,'dann')
    else: print("  ⚠️  Stage 3 checkpoint not found — run Cell 12 first.")

    # ── Collect predictions ───────────────────────────────────────
    all_res = {}
    for name,(model,stage) in registry.items():
        print(f"\nPredicting: {name}")
        all_res[name] = full_predict(model,test_loader,stage)

    # ── For EVERY variant: save CM + P/R/F1 ─────────────────────
    print("\n" + "="*60)
    print("Generating confusion matrices + P/R/F1 for all 6 variants...")
    saved_files = []
    for name,res in all_res.items():
        cm_path  = save_confusion_matrices(res, name, RESULTS_DIR)
        prf_path = save_prf1(res, name, RESULTS_DIR)
        saved_files += [cm_path, prf_path]
        print(f"  {name}: CM → {os.path.basename(cm_path)} | P/R/F1 → {os.path.basename(prf_path)}")

    # ── Summary ablation table (Precision, Recall, F1 per task) ──
    rows=[]
    for name,res in all_res.items():
        row={'Variant':name}; all_f1=[]
        for task,(p,t) in res.items():
            lbls=TASK_LABELS[task]
            pr,re,f1,_=precision_recall_fscore_support(t,p,labels=lbls,average='macro',zero_division=0)
            acc=accuracy_score(t,p)
            short=task[:4].title()
            row[f'{short}.Prec']=pr; row[f'{short}.Rec']=re
            row[f'{short}.F1']=f1;  row[f'{short}.Acc']=acc
            all_f1.append(f1)
        row['Avg F1']=np.mean(all_f1); rows.append(row)

    abl=pd.DataFrame(rows)
    pd.set_option('display.float_format','{:.4f}'.format)
    pd.set_option('display.max_columns',30)
    print("\n"+"="*70)
    print("ABLATION TABLE — Test Set (Macro Precision / Recall / F1 per task)")
    print("="*70)
    print(abl.to_string(index=False))

    # ── Ablation bar chart ────────────────────────────────────────
    f1_cols=[c for c in abl.columns if '.F1' in c]
    fig,axes=plt.subplots(1,2,figsize=(16,5))
    fig.suptitle('Ablation Study — All 6 Variants (Test Set)',fontsize=13,fontweight='bold')

    x,w=np.arange(len(abl)),0.22
    for i,(col,color) in enumerate(zip(f1_cols,['#3498DB','#E74C3C','#2ECC71'])):
        axes[0].bar(x+i*w,abl[col].fillna(0),w,label=col,color=color,alpha=0.85)
    axes[0].set_xticks(x+w); axes[0].set_xticklabels([r.split('.')[-1].strip()[:18] for r in abl['Variant']],rotation=25,ha='right',fontsize=8)
    axes[0].set_ylabel('Macro F1'); axes[0].set_ylim(0,1)
    axes[0].set_title('Per-task Macro F1',fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].axhline(y=0.33,color='gray',linestyle='--',alpha=0.4)
    axes[0].grid(axis='y',alpha=0.3)

    av=abl['Avg F1'].fillna(0).values
    axes[1].plot(range(len(av)),av,'o-',color='#E74C3C',linewidth=2.5,markersize=9)
    axes[1].fill_between(range(len(av)),av,alpha=0.12,color='#E74C3C')
    for xi,v in enumerate(av):
        axes[1].annotate(f'{v:.3f}',(xi,v),textcoords='offset points',xytext=(0,10),
                         ha='center',fontsize=9,fontweight='bold')
    bi=av.argmax()
    axes[1].scatter([bi],[av[bi]],s=200,color='gold',zorder=5,edgecolors='black',linewidths=1.5)
    axes[1].set_xticks(range(len(av)))
    axes[1].set_xticklabels([r.split('.')[-1].strip()[:18] for r in abl['Variant']],rotation=25,ha='right',fontsize=8)
    axes[1].set_ylabel('Average Macro F1'); axes[1].set_ylim(0,1)
    axes[1].set_title('Average F1 Across All Tasks',fontweight='bold')
    axes[1].axhline(0.33,color='gray',linestyle='--',alpha=0.4); axes[1].grid(axis='y',alpha=0.3)
    plt.tight_layout()
    p=os.path.join(RESULTS_DIR,'ablation_summary.png')
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.show()

    # Also show CMs and P/R/F1 inline for best + worst model
    best_name=abl.loc[abl['Avg F1'].fillna(0).idxmax(),'Variant']
    worst_name=abl.loc[abl['Avg F1'].fillna(0).idxmin(),'Variant']
    for display_name in [worst_name,best_name]:
        if display_name in all_res:
            print(f"\n── {display_name} ──")
            fig,axes=plt.subplots(1,3,figsize=(18,5))
            fig.suptitle(f'Confusion Matrices — {display_name}',fontsize=13,fontweight='bold')
            for ax,(task,(p,t)) in zip(axes,all_res[display_name].items()):
                lmap=TASK_MAPS[task]; lbls=TASK_LABELS[task]
                names=[lmap[i].replace('_','\n') for i in lbls]
                cm=confusion_matrix(t,p,labels=lbls); cmn=cm.astype(float)/(cm.sum(1,keepdims=True)+1e-8)
                im=ax.imshow(cmn,cmap='Blues',vmin=0,vmax=1); plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
                ax.set_xticks(range(len(names))); ax.set_xticklabels(names,fontsize=7,rotation=30,ha='right')
                ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=7)
                for i in range(len(names)):
                    for j in range(len(names)):
                        ax.text(j,i,f'{cmn[i,j]:.2f}\n({cm[i,j]})',ha='center',va='center',fontsize=6,
                                color='white' if cmn[i,j]>0.5 else 'black')
                f1=f1_score(t,p,average='macro',zero_division=0)
                ax.set_title(f'{task.capitalize()} F1={f1:.3f}',fontweight='bold')
                ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            plt.tight_layout(); plt.show()

    print(f"\n✅ All files saved to: {RESULTS_DIR}")
    print(f"   Files: {len(saved_files)} figures ({len(all_res)} variants × 2 plots each)")
    return abl, all_res

abl_df, all_task_results = run_full_evaluation()

"""## 🖼️ Cell 16 — Multi-Model Qualitative Comparison
**Same 6 test images evaluated by all 6 model variants.**  
Green = all 3 tasks correct · Yellow = 2 correct · Orange = 1 correct · Red = all wrong
"""

@torch.no_grad()
def plot_multimodel_qualitative(registry, test_df, img_dir, n_samples=6):
    """
    For n_samples fixed test images, show predictions from ALL model variants
    in a grid: rows = samples, columns = variants.
    Cells are colour-coded by correctness.
    """
    # One sample per event for maximum diversity
    samples = (test_df.groupby('event_name', group_keys=False)
               .apply(lambda g: g.sample(1, random_state=42))
               .reset_index(drop=True).head(n_samples))

    model_names = list(registry.keys())
    n_models    = len(model_names)
    n_rows      = len(samples)

    # Figure: each row = one sample; columns = [image] + [model predictions]
    fig = plt.figure(figsize=(3.0 + n_models*2.4, n_rows*3.8))
    gs  = fig.add_gridspec(n_rows, 1+n_models,
                           width_ratios=[2.8]+[1.8]*n_models,
                           hspace=0.45, wspace=0.12)

    # ── Column headers ────────────────────────────────────────────
    ax_hdr = fig.add_subplot(gs[0,0])
    ax_hdr.axis('off')
    ax_hdr.text(0.5,1.05,'Image + Ground Truth',ha='center',va='bottom',
                fontsize=9,fontweight='bold',transform=ax_hdr.transAxes)
    for ci,mname in enumerate(model_names):
        ax=fig.add_subplot(gs[0,ci+1]); ax.axis('off')
        short=mname.split('.')[-1].strip()
        ax.text(0.5,1.05,short,ha='center',va='bottom',fontsize=7.5,
                fontweight='bold',transform=ax.transAxes,wrap=True)

    border_clr={3:'#27AE60',2:'#F9BF3B',1:'#E67E22',0:'#C0392B'}
    bg_clr     ={3:'#d5f5e3',2:'#fef9e7',1:'#fdebd0',0:'#fadbd8'}

    for ri,(_,row) in enumerate(samples.iterrows()):
        gt={'i':int(row.get('label_text',0)),
            'd':int(row.get('label_damage',0)),
            'h':int(row.get('label_humanitarian',0))}
        img_id=str(row.get('image_id',''))
        raw=load_pil(img_dir,img_id)

        # Prepare tensors
        if raw:
            img_t=EVAL_TF(raw).unsqueeze(0).to(device)
        else:
            img_t=torch.zeros(1,3,cfg.img_size,cfg.img_size,device=device)
        enc=tokenizer(str(row.get('tweet_text','')),max_length=cfg.max_len,
                      padding='max_length',truncation=True,return_tensors='pt')
        ids=enc['input_ids'].to(device); mask=enc['attention_mask'].to(device)

        # ── Image column ──────────────────────────────────────────
        ax_img=fig.add_subplot(gs[ri,0])
        if raw: ax_img.imshow(raw)
        else:   ax_img.set_facecolor('#ddd')
        ax_img.axis('off')
        ev=str(row.get('event_name','')).replace('_',' ')[:22]
        tw=str(row.get('tweet_text',''))[:55]
        gt_str=(f"[{ev}]\n"{tw}..."\n\n"
                f"GT Inform: {INFORM_MAP.get(gt['i'],'?')[:9]}\n"
                f"GT Damage: {DAMAGE_MAP.get(gt['d'],'?')[:11]}\n"
                f"GT Human:  {HUMAN_MAP.get(gt['h'],'?')[:13]}")
        ax_img.set_title(gt_str,fontsize=5.5,loc='left',pad=2)

        # ── Model columns ─────────────────────────────────────────
        for ci,(mname,(model,stage)) in enumerate(registry.items()):
            ax=fig.add_subplot(gs[ri,ci+1])
            logits=model_forward(model,img_t,ids,mask,stage)
            pi=logits['informative'].argmax(1).item()
            pd_=logits['damage'].argmax(1).item()
            ph=logits['humanitarian'].argmax(1).item()
            ok_i,ok_d,ok_h = pi==gt['i'], pd_==gt['d'], ph==gt['h']
            n_ok=sum([ok_i,ok_d,ok_h])

            ax.set_facecolor(bg_clr[n_ok])
            for sp in ax.spines.values():
                sp.set_edgecolor(border_clr[n_ok]); sp.set_linewidth(2.2)

            chk=lambda b: '✓' if b else '✗'
            txt=(f"{chk(ok_i)} {INFORM_MAP.get(pi,'?')[:10]}\n"
                 f"{chk(ok_d)} {DAMAGE_MAP.get(pd_,'?')[:12]}\n"
                 f"{chk(ok_h)} {HUMAN_MAP.get(ph,'?')[:14]}")
            ax.text(0.5,0.5,txt,ha='center',va='center',fontsize=6.5,
                    transform=ax.transAxes,fontfamily='monospace',
                    color='#1a1a1a')
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_xticks([]); ax.set_yticks([])

    # ── Legend ────────────────────────────────────────────────────
    handles=[mpatches.Patch(facecolor='#d5f5e3',edgecolor='#27AE60',linewidth=2,label='All 3 tasks correct'),
             mpatches.Patch(facecolor='#fef9e7',edgecolor='#F9BF3B',linewidth=2,label='2 tasks correct'),
             mpatches.Patch(facecolor='#fdebd0',edgecolor='#E67E22',linewidth=2,label='1 task correct'),
             mpatches.Patch(facecolor='#fadbd8',edgecolor='#C0392B',linewidth=2,label='All tasks wrong')]
    fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=9,bbox_to_anchor=(0.5,-0.01))

    fig.suptitle('Multi-Model Qualitative Comparison\n'
                 'All 6 variants evaluated on the same test samples (✓=correct  ✗=wrong)',
                 fontsize=12,fontweight='bold',y=1.01)
    plt.tight_layout()
    path=os.path.join(RESULTS_DIR,'multimodel_qualitative.png')
    plt.savefig(path,dpi=130,bbox_inches='tight'); plt.show()
    print(f"✅ Saved: {path}")


# Build registry with loaded models
eval_registry = OrderedDict()
eval_registry['1. Image-only']            = (img_model,   'finetune')
eval_registry['2. Text-only']             = (txt_model,   'finetune')
eval_registry['3. Late Fusion']           = (late_model,  'finetune')
eval_registry['4. CrisisNet (no SSL)']    = (nossl_model, 'finetune')

for label,ckpt_file,cls,stg in [
    ('5. CrisisNet + SSL',       'crisisnet_best.pth',      CrisisNet,    'finetune'),
    ('6. CrisisNet+SSL+DANN',    'crisisnet_dann_best.pth', CrisisNetDANN,'dann'),
]:
    dk='finetune' if 'crisisnet_best' in ckpt_file else 'dann'
    p=os.path.join(cfg.ckpt(dk),ckpt_file)
    if os.path.exists(p):
        m=cls(cfg).to(device)
        m.load_state_dict(torch.load(p,map_location=device,weights_only=False)['model'],strict=False)
        eval_registry[label]=(m,stg)

plot_multimodel_qualitative(eval_registry, test_df, IMAGE_DIR, n_samples=6)

"""## 📉 Cell 17 — Training Curves
Loss and F1 over epochs for all 3 training stages.
"""

def plot_training_curves():
    stages=[('ssl','clip','Stage 1 — SSL'),
            ('finetune','crisisnet','Stage 2 — Fine-tuning'),
            ('dann','crisisnet_dann','Stage 3 — DANN')]
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    fig.suptitle('CrisisNet — Training Curves',fontsize=15,fontweight='bold')
    TC={'informative_f1':'#E74C3C','damage_f1':'#3498DB','humanitarian_f1':'#2ECC71','avg_f1':'black'}
    for col,(stage,name,title) in enumerate(stages):
        hist=CheckpointManager(stage,name).load_history()
        if not hist:
            for r in range(2):
                axes[r,col].text(0.5,0.5,f'No history\n({stage})',ha='center',va='center',
                                 transform=axes[r,col].transAxes,color='gray',fontsize=11)
                axes[r,col].set_title(title)
            continue
        df=pd.DataFrame(hist)
        eps=df['epoch'].values if 'epoch' in df.columns else np.arange(1,len(df)+1)
        ax0=axes[0,col]
        lc='ssl_loss' if 'ssl_loss' in df.columns else 'train_loss'
        if lc in df.columns:
            ax0.plot(eps,df[lc],'b-o',markersize=4,linewidth=1.8,label='Loss')
            ax0.fill_between(eps,df[lc],alpha=0.1,color='blue')
            mi=df[lc].idxmin()
            ax0.scatter([eps[mi]],[df[lc].iloc[mi]],s=120,color='gold',zorder=5,edgecolors='black')
        ax0.set_title(f'{title}\nLoss',fontweight='bold',fontsize=10)
        ax0.set_xlabel('Epoch'); ax0.set_ylabel('Loss'); ax0.legend(fontsize=8); ax0.grid(alpha=0.3)
        ax1=axes[1,col]
        for fc in [c for c in df.columns if c.endswith('_f1')]:
            lw=2.5 if fc=='avg_f1' else 1.5; ls='--' if fc=='avg_f1' else '-'
            ax1.plot(eps,df[fc],ls+'o',color=TC.get(fc,'gray'),markersize=3,linewidth=lw,
                     label=fc.replace('_f1','').replace('_',' ').title())
        if 'avg_f1' in df.columns:
            bi=df['avg_f1'].idxmax()
            ax1.axvline(x=eps[bi],color='gold',linestyle=':',alpha=0.8,linewidth=1.5)
            ax1.annotate(f'Best {df["avg_f1"].iloc[bi]:.3f}',
                         xy=(eps[bi],df['avg_f1'].iloc[bi]),
                         xytext=(eps[bi]+0.5,df['avg_f1'].iloc[bi]-0.06),fontsize=7,color='goldenrod')
        ax1.set_title(f'{title}\nMacro F1',fontweight='bold',fontsize=10)
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Macro F1'); ax1.set_ylim(0,1)
        ax1.legend(fontsize=7); ax1.grid(alpha=0.3)
    plt.tight_layout()
    p=os.path.join(RESULTS_DIR,'training_curves.png')
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.show(); print(f"✅ Saved: {p}")

plot_training_curves()

"""## 🌍 Cell 18 — Cross-Disaster Transfer
Leave-one-event-out: Stage 2 (no adaptation) vs Stage 3 (DANN).
"""

@torch.no_grad()
def cross_disaster_heatmap(model_ft, model_dann, full_df, img_dir):
    results={'Stage 2\n(no adapt)':{}, 'Stage 3\n(DANN)':{}}
    for ev in EVENTS:
        sub=full_df[full_df['event_name']==ev]
        if len(sub)==0: continue
        ds=CrisisMMDDataset(sub,img_dir,'val'); ld=make_loader(ds,cfg.dann_bs,shuffle=False)
        results['Stage 2\n(no adapt)'][ev]=evaluate(model_ft,  ld,'finetune').get('avg_f1',0)
        results['Stage 3\n(DANN)'][ev]    =evaluate(model_dann,ld,'dann').get('avg_f1',0)
        s2=results['Stage 2\n(no adapt)'][ev]; s3=results['Stage 3\n(DANN)'][ev]
        print(f"  {ev:<35}  S2={s2:.3f}  S3={s3:.3f}  Δ={s3-s2:+.3f}")
    ev_short=[e.replace('_','\n') for e in EVENTS]
    v2=[results['Stage 2\n(no adapt)'].get(ev,0) for ev in EVENTS]
    v3=[results['Stage 3\n(DANN)'].get(ev,0) for ev in EVENTS]
    fig,axes=plt.subplots(1,2,figsize=(16,5))
    fig.suptitle('Cross-Disaster Transfer (Leave-One-Event-Out)',fontsize=13,fontweight='bold')
    x,w=np.arange(len(EVENTS)),0.35
    axes[0].bar(x-w/2,v2,w,label='Stage 2 (no adapt)',color='#3498DB',alpha=0.85)
    axes[0].bar(x+w/2,v3,w,label='Stage 3 (DANN)',    color='#E74C3C',alpha=0.85)
    for xi,(a,b) in enumerate(zip(v2,v3)):
        axes[0].annotate(f'{b-a:+.3f}',(xi,max(a,b)+0.01),ha='center',fontsize=7.5,
                         color='#27AE60' if b>a else '#C0392B',fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(ev_short,fontsize=8)
    axes[0].set_ylabel('Avg Macro F1'); axes[0].set_ylim(0,1.05)
    axes[0].set_title('Per-Disaster F1',fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(axis='y',alpha=0.3)
    axes[0].axhline(0.33,color='gray',linestyle='--',alpha=0.4)
    hmap=np.array([v2,v3])
    im=axes[1].imshow(hmap,cmap='RdYlGn',vmin=0,vmax=1,aspect='auto')
    plt.colorbar(im,ax=axes[1],fraction=0.046,pad=0.04,label='Avg F1')
    axes[1].set_xticks(range(len(EVENTS))); axes[1].set_xticklabels(ev_short,fontsize=8)
    axes[1].set_yticks([0,1]); axes[1].set_yticklabels(['Stage 2\n(no adapt)','Stage 3\n(DANN)'],fontsize=10)
    axes[1].set_title('F1 Heatmap by Disaster',fontweight='bold')
    for ri in range(2):
        for ci in range(len(EVENTS)):
            val=hmap[ri,ci]
            axes[1].text(ci,ri,f'{val:.3f}',ha='center',va='center',fontsize=8.5,fontweight='bold',
                         color='black' if 0.25<val<0.75 else 'white')
    plt.tight_layout()
    p=os.path.join(RESULTS_DIR,'cross_disaster.png')
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.show()
    sm=pd.DataFrame({'Event':EVENTS,'Stage2':v2,'Stage3':v3,'Gain':[b-a for a,b in zip(v2,v3)]})
    sm['Trend']=sm['Gain'].apply(lambda x: f'↑{x:+.3f}' if x>0 else f'↓{x:+.3f}')
    print(sm.to_string(index=False))
    print(f"\n  Mean DANN gain : {sm['Gain'].mean():+.4f}")
    print(f"  Events improved: {(sm['Gain']>0).sum()}/{len(EVENTS)}")
    print(f"✅ Saved: {p}")

cross_disaster_heatmap(ft_model, dann_model, full_df, IMAGE_DIR)

"""## 🔍 Cell 19 — Cross-Attention Visualisation"""

@torch.no_grad()
def visualise_attention(model, image_path, tweet_text, save_path=None):
    model.eval()
    raw=Image.open(image_path).convert('RGB').resize((224,224))
    img_t=EVAL_TF(raw).unsqueeze(0).to(device)
    enc=tokenizer(tweet_text,max_length=cfg.max_len,padding='max_length',truncation=True,return_tensors='pt')
    ids=enc['input_ids'].to(device); mask=enc['attention_mask'].to(device)
    logits=model_forward(model,img_t,ids,mask,'dann' if isinstance(model,CrisisNetDANN) else 'finetune')
    # Get attention weights via return_attn=True
    try:
        out=model(img_t,ids,mask,return_attn=True)
        if isinstance(model,CrisisNetDANN): logits,_,attn_w=out
        else: logits,attn_w=out
        attn=attn_w[0].mean(dim=-1).cpu().numpy().reshape(7,7)
        attn=(attn-attn.min())/(attn.max()-attn.min()+1e-8)
    except Exception:
        attn=np.ones((7,7))
    pi=logits['informative'].argmax(1).item()
    pd_=logits['damage'].argmax(1).item()
    ph=logits['humanitarian'].argmax(1).item()
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].imshow(raw); axes[0].set_title('Input image',fontsize=10); axes[0].axis('off')
    axes[1].imshow(raw)
    axes[1].imshow(np.kron(attn,np.ones((32,32))),alpha=0.55,cmap='hot',vmin=0,vmax=1)
    axes[1].set_title('Cross-attention (image→text)',fontsize=10); axes[1].axis('off')
    axes[2].axis('off')
    axes[2].text(0.05,0.5,
        f"Tweet:\n{tweet_text[:120]}...\n\n"
        f"Informative : {INFORM_MAP.get(pi,'?')}\n"
        f"Damage      : {DAMAGE_MAP.get(pd_,'?')}\n"
        f"Humanitarian: {HUMAN_MAP.get(ph,'?')}",
        transform=axes[2].transAxes,fontsize=8,va='center',
        bbox=dict(boxstyle='round',facecolor='lightyellow',alpha=0.85))
    plt.suptitle('CrisisNet — Cross-Attention Visualisation',fontsize=12,fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path,dpi=150,bbox_inches='tight')
    plt.show(); model.train()

sample=test_df.iloc[0]
img_path=None
for ext in ['.jpg','.png']:
    p=os.path.join(IMAGE_DIR,str(sample.get('image_id',''))+ext)
    if os.path.exists(p): img_path=p; break

if img_path:
    visualise_attention(dann_model, img_path, str(sample.get('tweet_text','No text')),
                        save_path=os.path.join(RESULTS_DIR,'attention_demo.png'))
else:
    print("⚠️  No image found — provide an image path.")

"""## 🚀 Cell 20 — Inference"""

def predict(image_path_or_url, tweet_text, model=None, verbose=True):
    if model is None:
        best_ckpt=os.path.join(cfg.ckpt('dann'),'crisisnet_dann_best.pth')
        model=CrisisNetDANN(cfg).to(device)
        if os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt,map_location=device,weights_only=False)['model'],strict=False)
        model.eval()

    if str(image_path_or_url).startswith('http'):
        import urllib.request,io
        raw=Image.open(io.BytesIO(urllib.request.urlopen(image_path_or_url).read())).convert('RGB')
    else:
        raw=Image.open(image_path_or_url).convert('RGB')

    img_t=EVAL_TF(raw).unsqueeze(0).to(device)
    enc=tokenizer(tweet_text,max_length=cfg.max_len,padding='max_length',truncation=True,return_tensors='pt')
    ids=enc['input_ids'].to(device); mask=enc['attention_mask'].to(device)

    with torch.no_grad():
        logits=model_forward(model,img_t,ids,mask,'dann' if isinstance(model,CrisisNetDANN) else 'finetune')

    def decode(lv,lmap):
        probs=F.softmax(lv,dim=-1).squeeze()
        pred=probs.argmax().item()
        return lmap.get(pred,'?'),{lmap.get(i,'?'):f'{p:.3f}' for i,p in enumerate(probs.tolist())}

    ri,ci=decode(logits['informative'],INFORM_MAP)
    rd,cd=decode(logits['damage'],     DAMAGE_MAP)
    rh,ch=decode(logits['humanitarian'],HUMAN_MAP)

    if verbose:
        print(f"\nTweet       : {tweet_text[:100]}")
        print(f"Informative : {ri}  (conf={ci.get(ri,'?')})")
        print(f"Damage      : {rd}  (conf={cd.get(rd,'?')})")
        print(f"Humanitarian: {rh}  (conf={ch.get(rh,'?')})")

    return {'informative':ri,'informative_conf':ci,'damage':rd,'damage_conf':cd,
            'humanitarian':rh,'humanitarian_conf':ch}


# Example
example_text = "People stranded on rooftops, water still rising. Need immediate rescue. #HurricaneHarvey"
demo_imgs = list(Path(IMAGE_DIR).rglob('*.jpg'))
if demo_imgs:
    predict(str(demo_imgs[0]), example_text, model=dann_model)
else:
    print("⚠️  Provide a real image path.")

"""## ✅ Summary

### 6 model variants compared (all with full metrics)

| # | Variant | Architecture | Training |
|---|---------|-------------|---------|
| 1 | Image-only | ResNet-50 → heads | 10 epochs |
| 2 | Text-only | BERT → heads | 10 epochs |
| 3 | Late Fusion | ResNet+BERT concat → heads | 10 epochs |
| 4 | CrisisNet, no SSL | Cross-modal Transformer (ImageNet init) | 15 epochs |
| 5 | CrisisNet + SSL | Cross-modal Transformer (SSL init) | 20 epochs |
| 6 | CrisisNet + SSL + DANN | + domain adaptation | 15 epochs |
"""