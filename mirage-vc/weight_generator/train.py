#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle, csv, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import torch, numpy as np, random, os
from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_recall_curve, classification_report,
    confusion_matrix, accuracy_score
)


EMB_NPZ      = Path('../text_embed.npz')  
LBL_NPY      = Path('../labels.npy')      
TIME_CSV     = Path('../qualified_new_company.csv')
LABEL_PKL    = Path('../label_dict.pkl')
ATTR_NPY     = Path('../company_attr_target.npy')
ATTR_ID_ORDER= Path('../company_id_order.npy')

WB          = 50.0
BATCH_SIZE  = 128
EPOCHS      = 200
LR          = 3e-5
PATIENCE    = 7
DEVICE      = 'cuda:1' if torch.cuda.is_available() else 'cpu'
MIN_EPOCHS  = 5
delta_auc   = 5e-5
delta_loss  = 5e-4

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
os.environ["PYTHONHASHSEED"] = str(SEED)

npz = np.load(EMB_NPZ, allow_pickle=True)
company_ids_raw = npz['company_ids']                       
V_text = npz['vectors_text'].astype(np.float32)            
bool_v = np.load(LBL_NPY).astype(np.float32)[..., None]    

N, n_views, D_text = V_text.shape
assert n_views == 3, f"Expect 3 views from input, got {n_views}"
view_dim = D_text + 1 

attr_mat = np.load(ATTR_NPY)        
attr_ids = np.load(ATTR_ID_ORDER)   
id2attr = {cid: i for i, cid in enumerate(attr_ids)}

time_map = {}
with TIME_CSV.open() as f:
    for row in csv.DictReader(f):
        time_map[row['CompanyID']] = int(row['time'])

with LABEL_PKL.open('rb') as f:
    label_map = pickle.load(f)  

times, y_true = [], []
valid_idx, valid_attr_idx = [], []
for i, cid in enumerate(company_ids_raw):
    if cid in time_map and cid in label_map and cid in id2attr:
        valid_idx.append(i)
        valid_attr_idx.append(id2attr[cid])
        times.append(time_map[cid])
        y_true.append(label_map[cid])

times  = np.array(times,  dtype=np.int16)
y_true = np.array(y_true, dtype=np.int8)
V_text = V_text[valid_idx]        
bool_v = bool_v[valid_idx]         
C_all  = attr_mat[valid_attr_idx] 
company_ids = company_ids_raw[valid_idx]

print(f"Views: {n_views}, Text dim per view: {D_text}, Company feature dim: {C_all.shape[1]}")

train_mask = (times>=50)  & (times<=150)
val_mask   = (times>=151) & (times<=170)
test_mask  = (times>=171) & (times<=190)
idx_train = np.where(train_mask)[0]
idx_val   = np.where(val_mask)[0]
idx_test  = np.where(test_mask)[0]
print(f"train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")

class ViewDataset(Dataset):
    def __init__(self, v_idx):
        self.Vt = torch.from_numpy(V_text[v_idx]).float()   
        self.Bb = torch.from_numpy(bool_v[v_idx]).float()   
        self.C  = torch.from_numpy(C_all[v_idx]).float()    
        self.y  = torch.from_numpy(y_true[v_idx]).float()
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        return self.Vt[i], self.Bb[i], self.C[i], self.y[i]

dls = {
    split: DataLoader(ViewDataset(idx), batch_size=BATCH_SIZE, shuffle=(split=='train'))
    for split, idx in [('train',idx_train), ('val',idx_val), ('test',idx_test)]
}

comp_dim = C_all.shape[1]

class MoEGate(nn.Module):
    def __init__(self, text_dim=384, comp_dim=1, n_views=3):
        super().__init__()
        self.n_views = n_views
        self.log_wb = nn.Parameter(torch.tensor(1.1))  

        view_dim = text_dim + 1         

        self.gate = nn.Sequential(
            nn.Linear(n_views*view_dim + comp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_views)
        )
        self.clf = nn.Sequential(
            nn.Linear(view_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, V_text, B_bool, C):
        wb = torch.exp(self.log_wb)
        V = torch.cat([V_text, wb * B_bool], dim=-1)  

        B = V.size(0)
        flat = torch.cat([V.view(B, -1), C], dim=-1)  
        alpha = torch.softmax(self.gate(flat), dim=-1) 
        v_agg = (alpha.unsqueeze(-1) * V).sum(1)        
        logit = self.clf(v_agg).squeeze(-1)             
        return logit, alpha

model = MoEGate(text_dim=D_text, comp_dim=comp_dim, n_views=n_views).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR)

pos_cnt = y_true.sum()
neg_cnt = len(y_true) - pos_cnt
pos_weight = torch.tensor([neg_cnt / max(pos_cnt, 1)], device=DEVICE)  
print(f"Pos {pos_cnt}  Neg {neg_cnt}  pos_weight={pos_weight.item():.2f}")

bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_val_auc, best_val_loss = 0.0, 1e9
pat_left = PATIENCE

def evaluate(split):
    model.eval()
    ys, ps, losses = [], [], []
    with torch.no_grad():
        for Vb, Bb, Cb, yb in dls[split]:
            Vb, Bb, Cb, yb = [t.to(DEVICE) for t in (Vb, Bb, Cb, yb)]
            logits, _ = model(Vb, Bb, Cb)
            loss  = bce(logits, yb)
            losses.append(loss.item())
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(yb.cpu().numpy())
    p = np.concatenate(ps); y = np.concatenate(ys)
    return roc_auc_score(y,p), f1_score(y,(p>0.5).astype(int)), np.mean(losses)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for Vt,Bb,Cb,yb in dls['train']:
        Vt,Bb,Cb,yb = Vt.to(DEVICE), Bb.to(DEVICE), Cb.to(DEVICE), yb.to(DEVICE)
        logits,_ = model(Vt,Bb,Cb)
        loss = bce(logits, yb)
        optim.zero_grad(); loss.backward(); optim.step()
        train_losses.append(loss.item())

    tr_loss = np.mean(train_losses)
    val_auc, val_f1, val_loss = evaluate('val')

    print(f"Epoch {epoch:02d} │ "
          f"TrainLoss {tr_loss:.4f} │ "
          f"ValLoss {val_loss:.4f} │ "
          f"ValAUC {val_auc:.4f} │ ValF1 {val_f1:.4f}")

    improved = False
    if val_auc > best_val_auc + delta_auc:
        best_val_auc, improved = val_auc, True
    if val_loss < best_val_loss - delta_loss:
        best_val_loss, improved = val_loss, True

    if improved:
        torch.save(model.state_dict(), 'best_gate.pt')
        pat_left = PATIENCE
    elif epoch >= MIN_EPOCHS:
        pat_left -= 1
        if pat_left == 0:
            print("Early-Stopping triggered")
            break

def collect(split):
    model.eval(); ps, ys = [], []
    loader = dls[split]
    with torch.no_grad():
        for Vb, Bb, Cb, yb in loader:
            Vb, Bb, Cb = [t.to(DEVICE) for t in (Vb, Bb, Cb)]
            prob = torch.sigmoid(model(Vb, Bb, Cb)[0]).cpu().numpy()
            ps.append(prob); ys.append(yb.numpy())
    return np.concatenate(ps), np.concatenate(ys)

model.load_state_dict(torch.load('best_gate.pt', map_location=DEVICE))

p_val, y_val = collect('val')
prec, rec, thr = precision_recall_curve(y_val, p_val)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.argmax(f1s)
tau_star = thr[best_idx]
print(f"\n best τ={tau_star:.3f}  (Val F1={f1s[best_idx]:.4f}, "
      f"Prec={prec[best_idx]:.4f}, Rec={rec[best_idx]:.4f})")

p_test, y_test = collect('test')
y_pred = (p_test >= tau_star).astype(int)

test_auc = roc_auc_score(y_test, p_test)
test_f1  = f1_score(y_test, y_pred)

print(f"\n==  Final Test  ==  AUC {test_auc:.4f}   F1@τ* {test_f1:.4f}")
print("\n=== Classification Report (Test) ===")
print(classification_report(
    y_test, y_pred, digits=4, target_names=['negative','positive'])
)

print("Confusion Matrix [TN FP; FN TP]:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")