# Dynamic-DeepHit-inspired — Source paths and fixed run settings
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 2 ---
from pathlib import Path
from functools import partial
import hashlib
import json
import math
import os
import time
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

SOURCE = Path(os.getenv("EHR_MAIN_RUN_DIR", "runs/CRESCENT_Final")).expanduser().resolve()
RUN_DIR = Path(os.getenv("EHR_DDH_OUTPUT", "runs/dynamic_deephit")).expanduser().resolve()

WORKFLOW = os.getenv("EHR_DDH_WORKFLOW","train_and_evaluate")
RESUME = os.getenv("EHR_DDH_RESUME","").strip()
DEVICE = torch.device(os.getenv("EHR_DEVICE","cuda:0" if torch.cuda.is_available() else "cpu").replace("auto","cuda:0" if torch.cuda.is_available() else "cpu"))
MAX_UPDATES, BATCH_SIZE = 50_000, 128
WARMUP, LOSS_INTERVAL, REPORT_INTERVAL = 1_000, 1_000, 5_000
LR, LR_FLOOR, WEIGHT_DECAY = 5e-5, 5e-6, 0.01
HORIZONS = (12,24,60,120)
DENSE = tuple(range(12,121,6))
POINT = (24,60)
WIDTH, BINS, ENTRY = 6, 22, 6
ENTRY_EPS, TIME_EPS, G_FLOOR = 1e-6, 1e-8, .01
SEED = int(os.getenv("EHR_SEED","42"))
BOOTSTRAPS, BOOT_SEED = 200, 101
BEST = RUN_DIR / "best_model_by_val_loss.pt"
LAST = RUN_DIR / "last_checkpoint.pt"
if SOURCE == RUN_DIR or SOURCE in RUN_DIR.parents or RUN_DIR in SOURCE.parents:
    raise ValueError("Use separate source and output directories.")
if WORKFLOW not in {"train_and_evaluate","evaluation_only"}:
    raise ValueError("Unknown workflow.")
if RESUME and (Path(RESUME).expanduser().resolve()!=LAST or not LAST.is_file() or not BEST.is_file() or WORKFLOW!="train_and_evaluate"):
    raise ValueError("Resume requires this run's last_checkpoint.pt and best_model_by_val_loss.pt.")
if WORKFLOW=="evaluation_only" and not BEST.is_file():
    raise FileNotFoundError(BEST)
if WORKFLOW=="train_and_evaluate" and not RESUME and RUN_DIR.exists() and any(RUN_DIR.iterdir()):
    raise FileExistsError(f"Use a fresh output directory or resume {LAST}.")
RUN_DIR.mkdir(parents=True,exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type=="cuda":
    torch.cuda.set_device(DEVICE)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def save_csv(frame,path):
    tmp=path.with_name(path.name+".tmp")
    frame.to_csv(tmp,index=False)
    tmp.replace(path)

def save_pt(payload,path):
    tmp=path.with_suffix(".pt.tmp")
    torch.save(payload,tmp)
    tmp.replace(path)

def log(message):
    print(message,flush=True)
    with (RUN_DIR/"training.log").open("a",encoding="utf-8") as f:
        print(message,file=f)

console_path = RUN_DIR / "cell6_full_training_console.log"
if not console_path.exists() and not console_path.is_symlink():
    console_path.symlink_to("training.log")
src_cfg=read_json(SOURCE/"config.json")
settings=src_cfg["config"]
entry_contract=src_cfg["delayed_entry_contract"]
if (settings["ENTRY_DELAY_MONTHS"]!=ENTRY or tuple(settings["EVAL_HORIZONS_MONTHS"])!=HORIZONS
    or entry_contract["training_likelihood"]!="exact delayed-entry likelihood over post-entry exposure only"
    or src_cfg["hazard_binning"]["N_BINS"]!=BINS or src_cfg["hazard_binning"]["BIN_WIDTH_MONTHS"]!=WIDTH
    or settings["LOSS_ALPHA_NLL"]!=1 or settings["LOSS_GAMMA_DEATH"]!=.1
    or src_cfg["fixed_index_contract"]["calendar_months_from_recruitment"]!=1):
    raise ValueError("Source does not match the CRESCENT six-month delayed-entry study.")
hashes=read_json(SOURCE/"model_input_manifest.json")["input_hashes"]
vpath=SOURCE/"cell3_vocab_scalers_deathsafe.json"
if sha256(vpath)!=hashes["vocabulary_and_scalers_sha256"]:
    raise ValueError("Vocabulary/scaler hash mismatch.")
vocab=read_json(vpath)
meta=read_json(SOURCE/"canonical_target_metadata.json")
names=list(map(str,meta["condition_names"]))
K=len(names)
if len(set(names))!=K or any(int(vocab["condition_vocab"]["code_to_id"].get(n,-1))!=k for k,n in enumerate(names)):
    raise ValueError("Inconsistent target identity or ordering.")
ev,ty,st=vocab["event_vocab"],vocab["event_type_vocab"],vocab["static_categories"]
PAD,CLS,INDEX=(int(ev[k]) for k in ["EV_PAD_ID","EV_CLS_ID","EV_NEAR_RECRUITMENT_INDEX_ID"])
TPAD,TCLS=int(ty["TYPE_PAD_ID"]),int(ty["TYPE_CLS_ID"])
CRP_TYPE=int({str(k).lower():v for k,v in ty["TYPE_TO_ID"].items()}["crp"])
CRP_IDS=np.asarray([v for k,v in ev["EV_TOKEN_TO_ID"].items() if k.upper()=="CRP_MEASURED" or k.upper().startswith("CRP_D")],int)
def load_frame(split):
    p=SOURCE/f"canonical_{split}_samples.pkl"
    if sha256(p)!=hashes[f"canonical_{split}_samples_sha256"]:
        raise ValueError(f"{split} hash differs from CRESCENT.")
    f=pd.read_pickle(p).reset_index(drop=True)
    if f.empty or f.eid.astype(str).duplicated().any():
        raise ValueError(f"Invalid {split} participant axis.")
    return f

train_frame=load_frame("train")
val_frame=load_frame("val")
dev_ids=set(train_frame.eid.astype(str))
if dev_ids.intersection(val_frame.eid.astype(str)):
    raise ValueError("Train/validation overlap.")
dev_ids.update(val_frame.eid.astype(str))
n_cont=len(train_frame.iloc[0].continuous_static)
contract={"model_variant":"DynamicDeepHit_task_adapted_50k_v1","model_input_hashes":hashes,
          "target_metadata_sha256":sha256(SOURCE/"canonical_target_metadata.json"),
          "condition_names":names,"delayed_entry_contract":entry_contract,"hazard_binning":src_cfg["hazard_binning"],
          "config":{"ENTRY_DELAY_MONTHS":ENTRY,"MAX_ITERATIONS":MAX_UPDATES,"BATCH_SIZE":BATCH_SIZE,
                    "LR":LR,"MIN_LR":LR_FLOOR,"WEIGHT_DECAY":WEIGHT_DECAY,"WARMUP_ITERATIONS":WARMUP,
                    "VALIDATION_INTERVAL":LOSS_INTERVAL,"PERFORMANCE_INTERVAL":REPORT_INTERVAL,"SEED":SEED,
                    "dropout":0.10,"event_dim":256,"static_dim":64,"gru_hidden_dim":512,"gru_layers":1,
                    "condition_chunk_size":128,"prior_pseudomonths":120,"precision":"float32","EMA":False,
                    "death_nll_weight":0.1,"parameter_initialization":"from_scratch"}}
config_path=RUN_DIR/"config.json"
if config_path.exists() and read_json(config_path)!=contract:
    raise ValueError("Incompatible output run or checkpoint configuration.")
config_path.write_text(json.dumps(contract,indent=2),encoding="utf-8")
log(f"{len(train_frame):,} training; {len(val_frame):,} validation; {K} outcomes; device {DEVICE}")
