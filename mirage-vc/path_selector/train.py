"""
Listwise selector training with group-balanced sampling,
excluding all-negative groups entirely (pos-only).
"""

from __future__ import annotations
import json, pickle, random, hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import ndcg_score

@dataclass
class CFG:
    prompt_jsonl : str = "../graph_retrieval_data/context/prompts_3hop_groups.jsonl"
    pred_jsonl   : str = "../graph_retrieval_data/context/predictions_3hop_groups.jsonl"
    emb_pkl      : str = "../graph_retrieval_data/context/prompt_emb.pkl"
    out_selector : str = "selector.pt"

    train_ratio  : float = 0.7
    val_ratio    : float = 0.15   

    groups_per_batch: int = 256   
    hidden       : int   = 768
    epochs       : int   = 30
    lr           : float = 5e-4

    tau          : float = 0.5    
    alpha_list   : float = 1.0    
    beta_pair    : float = 0.1    
    eps_gain     : float = 2e-2   

    seed         : int   = 0

CFG = CFG()

def set_seed(sd:int):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)

def md5(s:str)->str: return hashlib.md5(s.encode("utf-8")).hexdigest()

print("[1] loading prompts jsonl …")
group_prompts : Dict[Tuple[str,int,str],Tuple[str,Dict[str,str]]] = {}
with open(CFG.prompt_jsonl) as f:
    for ln in f:
        obj=json.loads(ln)
        gkey=(obj["target_id"], obj["hop"], obj["parent_label"])
        base   = obj["base_prompt"]
        exts   = {e["candidate_id"]: e["prompt"] for e in obj["extended"]}
        group_prompts[gkey]=(base, exts)
print(f"  groups described: {len(group_prompts)}")

print("\n[2] loading embeddings pkl …")
with open(CFG.emb_pkl,'rb') as f:
    emb_dict: Dict[str,np.ndarray] = pickle.load(f)
dim = next(iter(emb_dict.values())).shape[0]
print(f"  vectors loaded: {len(emb_dict)}  | dim={dim}")

class Sample:
    def __init__(self, x:torch.Tensor, delta:float, gid:str):
        self.x      = x
        self.delta  = float(delta)
        self.gid    = gid

samples: List[Sample]=[]
group2idxs=defaultdict(list)

print("\n[3] assembling features …")
missing=0
with open(CFG.pred_jsonl) as f:
    for ln in tqdm(f):
        obj=json.loads(ln)
        gkey=(obj["target_id"], obj["hop"], obj["parent_label"])
        gid  = "|".join(map(str,gkey))

        base_prompt = group_prompts[gkey][0]
        h_base = md5(base_prompt)
        if h_base not in emb_dict:
            missing+=1; continue
        v_base = torch.from_numpy(emb_dict[h_base]).float()

        for ext in obj["extended"]:
            delt = ext["delta"]
            if delt is None:
                continue
            ext_prompt = group_prompts[gkey][1][ext["candidate_id"]]
            h_ext = md5(ext_prompt)
            if h_ext not in emb_dict:
                missing+=1; continue
            v_ext = torch.from_numpy(emb_dict[h_ext]).float()

            feat = torch.cat([v_base, v_ext, v_ext-v_base])
            idx = len(samples)
            samples.append(Sample(feat, float(delt), gid))
            group2idxs[gid].append(idx)

print(f"  usable samples: {len(samples)}  | raw groups: {len(group2idxs)}  | missing_vec={missing}")

class Group:
    def __init__(self, gid:str, X:torch.Tensor, deltas:torch.Tensor):
        self.gid    = gid
        self.X      = X              # (k, 3*dim)
        self.deltas = deltas         # (k,)
        self.k      = X.size(0)
        self.pos    = int((deltas>0).sum().item())
        self.maxabs = float(deltas.abs().max().item())

def make_group(gid:str)->Group:
    idxs = group2idxs[gid]
    X = torch.stack([samples[i].x for i in idxs], dim=0)
    d = torch.tensor([samples[i].delta for i in idxs], dtype=torch.float32)
    return Group(gid, X, d)

all_groups_raw: List[Group] = []
for gid,lst in group2idxs.items():
    if len(lst) >= 2:
        all_groups_raw.append(make_group(gid))

pos_groups = [g for g in all_groups_raw if g.pos > 0]
negonly_groups = [g for g in all_groups_raw if g.pos == 0]
print(f"  groups (k>=2): {len(all_groups_raw)}")
print(f"  kept pos-groups: {len(pos_groups)} | dropped all-negative: {len(negonly_groups)}")

set_seed(CFG.seed)
random.shuffle(pos_groups)
n = len(pos_groups)
n_train = int(n*CFG.train_ratio)
n_val   = int(n*CFG.val_ratio)

train_groups = pos_groups[:n_train]
val_groups   = pos_groups[n_train:n_train+n_val]
test_groups  = pos_groups[n_train+n_val:]

print(f"  split groups (pos-only) → train:{len(train_groups)}  val:{len(val_groups)}  test:{len(test_groups)}")

print("  bucket stats (train):")
from collections import Counter
cnts = Counter(g.pos for g in train_groups)
for p in sorted(cnts): print(f"    pos={p}: {cnts[p]}")

class GroupDataset(Dataset):
    def __init__(self, groups:List[Group]):
        self.groups = groups
        self.buckets = defaultdict(list)
        for i,g in enumerate(groups):
            self.buckets[g.pos].append(i)
        self.bucket_keys = sorted(self.buckets.keys())
        self.bucket_sizes= {k:len(self.buckets[k]) for k in self.bucket_keys}
        self.min_bucket  = min(self.bucket_sizes.values())

    def __len__(self):
        return self.min_bucket * len(self.bucket_keys)

    def __getitem__(self, idx):
        b = self.bucket_keys[idx % len(self.bucket_keys)]
        offset = (idx // len(self.bucket_keys)) % self.min_bucket
        gidx = self.buckets[b][offset]
        return self.groups[gidx]

train_ds = GroupDataset(train_groups)

def collate_groups(batch:List[Group]):
    return batch

train_loader = DataLoader(
    train_ds, batch_size=CFG.groups_per_batch,
    shuffle=True, collate_fn=collate_groups, drop_last=False
)

class Selector(nn.Module):
    def __init__(self,in_dim:int):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_dim,CFG.hidden), nn.ReLU(),
            nn.Linear(CFG.hidden,1))
    def forward(self,x):
        return self.mlp(x).squeeze(-1)

selector = Selector(dim*3).to("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE   = next(selector.parameters()).device
opt=AdamW(selector.parameters(), lr=CFG.lr)

def group_listwise_loss(group:Group)->torch.Tensor:
    X = group.X.to(DEVICE)           # (k, in_dim)
    d = group.deltas.to(DEVICE)      # (k,)
    s = selector(X)                  # (k,)

    r = torch.clamp(d, min=0.0)
    q = torch.softmax(r / CFG.tau, dim=0)
    p = torch.softmax(s / CFG.tau, dim=0)
    l_list = F.kl_div(p.log(), q, reduction='batchmean')

    l_pair = torch.tensor(0.0, device=DEVICE)
    if CFG.beta_pair > 0 and group.k >= 2:
        for i in range(group.k):
            for j in range(group.k):
                if d[i] > d[j]:
                    l_pair = l_pair + F.margin_ranking_loss(
                        s[i].unsqueeze(0), s[j].unsqueeze(0),
                        torch.ones(1, device=DEVICE), margin=0.2
                    )

    span = float((d.max() - d.min()).abs().item())
    w = min(1.0, span / CFG.eps_gain)

    return w * (CFG.alpha_list*l_list + CFG.beta_pair*l_pair)

def evaluate_groupwise(groups: List[Group]):
    selector.eval()

    all_samples: List[Sample] = []
    for g in groups:
        for i in range(g.k):
            all_samples.append(Sample(g.X[i].cpu(), float(g.deltas[i].item()), g.gid))

    with torch.no_grad():
        pos = [s for s in all_samples if s.delta > 0]
        neg = [s for s in all_samples if s.delta <= 0]
        if not pos or not neg:
            pair_acc = 0.0
        else:
            pa = torch.stack([s.x for s in pos]).to(DEVICE)
            na = torch.stack([s.x for s in neg]).to(DEVICE)
            sc_pos = selector(pa).cpu()
            sc_neg = selector(na).cpu()
            cmp = sc_pos.unsqueeze(1) > sc_neg.unsqueeze(0)
            pair_acc = cmp.float().mean().item()

    ndcgs = []
    hits  = 0
    total = 0
    with torch.no_grad():
        for g in groups:
            if g.k < 2: continue
            X = g.X.to(DEVICE)
            scores = selector(X).detach().cpu().numpy()
            deltas = g.deltas.cpu().numpy()

            shift  = -min(0.0, deltas.min())
            gains  = (deltas + shift).reshape(1, -1)
            ndcgs.append(ndcg_score(gains, scores.reshape(1, -1), k=1))

            pred_idx  = int(scores.argmax())
            max_gain  = deltas.max()
            max_set   = np.where(deltas == max_gain)[0]
            if pred_idx in max_set:
                hits += 1
            total += 1

    ndcg1 = float(np.mean(ndcgs)) if ndcgs else 0.0
    top1_hit = (hits / total) if total > 0 else 0.0
    return pair_acc, ndcg1, top1_hit

print("\n[5] training…")
best_val= -1.0

for ep in range(1, CFG.epochs+1):
    selector.train()
    run_loss, gcount = 0.0, 0
    for batch in train_loader:
        loss = torch.tensor(0.0, device=DEVICE)
        for g in batch:
            loss = loss + group_listwise_loss(g)
        loss = loss / len(batch)

        opt.zero_grad(); loss.backward(); opt.step()
        run_loss += loss.item() * len(batch); gcount += len(batch)

    train_loss = run_loss / gcount
    val_pair, val_ndcg, val_top1 = evaluate_groupwise(val_groups)
    print(f"Ep{ep:02d} | loss {train_loss:.4f} | "
          f"val PairAcc {val_pair:.3f}  NDCG@1 {val_ndcg:.3f}  Top1 {val_top1:.3f}")

    if val_ndcg > best_val + 1e-4:
        best_val = val_ndcg
        torch.save(selector.state_dict(), CFG.out_selector)

selector.load_state_dict(torch.load(CFG.out_selector, map_location=DEVICE))
pair_acc, ndcg1, top1 = evaluate_groupwise(test_groups)
print(f"\n[Test] Pair-Acc {pair_acc:.3f} | NDCG@1 {ndcg1:.3f} | Top-1 {top1:.3f}")
print(f"[DONE] selector saved → {CFG.out_selector}")