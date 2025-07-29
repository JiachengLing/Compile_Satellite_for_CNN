# -*- coding: utf-8 -*-

"""
CNN-Based Species Distribution Modeling (SDM) for Shorebirds

This program implements a Convolutional Neural Network (CNN) based approach using ResNet-18,
a widely used deep learning architecture originally developed for image recognition tasks.

Unlike traditional SDMs that rely heavily on environmental variables such as temperature or precipitation,
this method leverages **remote sensing imagery** to directly extract features from habitat structure.
This is particularly useful for modeling **shorebirds**, whose distribution depends more on specific habitat
patterns than on stable climatic zones, due to their highly migratory nature.

The CNN model automatically learns and extracts important visual features from satellite images (e.g., RGB bands
and other spectral bands), and uses them as predictors for presence/absence modeling:

    Remote Sensing Imagery  →  Image Features (learned by CNN)  →  SDM Prediction

---

**Preprocessing & Setup**

Before running this script, we recommend executing the following preparation scripts:
- `SELF_TEST_B4_RUN.py`: For system and environment checking.
- `IMAGE_CLIPPING.py`: For interactive data preparation via UI (e.g., extracting image patches).

---

**Expected Directory Structure**

Your project folder should contain the following:
- GeoTIFF files: `*.tif`, named as `<SiteID>.tif` or specified via 'path' in `tile_index.csv`
- `tile_index.csv`: Must include at least `SiteID` (or `path`), `presence`, `minx`, `miny`, `maxx`, `maxy`
- *(Optional)* `fishnet_with_SiteID.gpkg`: For spatial cross-validation. If not provided, CSV boundaries are used.

---

**Model Modes**

You can run the program in two different modes:

1. **Benchmark Mode** *(takes longer)*:
   - Trains the model across multiple background sampling ratios.
   - Produces `predictions.csv` and a summary report `summary.pdf`.

2. **Single Mode** *(fast & simple)*:
   - Trains the model with one ratio and one set of hyperparameters.
   - Outputs:
     - `predictions_single_r{ratio}.csv`
     - `single_r{ratio}_metrics.json`
     - `summary_single_r{ratio}.pdf`

---

Running the Program

Configuration for the environment, in conda for example:

conda create -n cnn_sdm python=3.10 -y
conda activate cnn_sdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (pip install torch torchvision torchaudio    if CPU only)
pip install numpy pandas rasterio scikit-learn matplotlib geopandas

If you wish to use --compile for acceleration ,please check:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
to see whether PyTorch version >= 2.0 and CUDA is available.


It is recommended to run this script from the command line:

    Press `Win+R` → type `cmd` → navigate to the script folder and execute:

    python [path to the file]/cnn_sdm.py --mode single --ratio 10 --boots 5

    python [path to the file]/cnn_sdm.py --mode benchmark --folder /path/to/data --ratios 10,50,100 --boots 5


"""



import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------- defaults ---------------------
BG_RATIOS   = [1]     # number of background points
N_BOOT      = 10      # number of bootstraps to evaluate model robustness
IMG_SIZE    = 224     # input image size (all images will be transformed to 224 * 224 for example)
BATCH_SIZE  = 32      # Number of images processed in each iteration (the larger is the number, the faster is the training process, but cost more RAMs)
MAX_EPOCHS  = 12      # maximum epochs
PATIENCE    = 3       # number of epochs without performance improvement before earlystopping
NUM_WORKERS = 4       # number of threads used by Dataloader.
SEED        = 42
USE_BANDS   = None         # e.g., [3,2,1]; None = all bands
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Optional accelerators (set via CLI)
USE_MIXED   = False        # --mixed
USE_COMPILE = False        # --compile (Pytorch 2.0, torch.compile() for acceleration)
CHANNELS_LAST = False      # --channels-last

# Small hyperparameter grid   learning rate, weight decay, dropout ratio, augmentation
GRID_HPARAMS = [
    {"lr": 1e-3, "wd": 1e-4, "drop": 0.0, "aug": False},
    {"lr": 3e-4, "wd": 1e-4, "drop": 0.2, "aug": True},
    {"lr": 1e-4, "wd": 5e-5, "drop": 0.3, "aug": True},
]

rng = np.random.RandomState(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

# --------------------- DATA ASSEMBLY ---------------------
def load_from_folder(folder):
    """
    Read tile_index.csv and assemble:
      - df: DataFrame indexed by SiteID with path_resolved, presence_norm, minx..maxy
      - pos_ids: SiteIDs with presence==1
      - bg_pool: SiteIDs with presence==NaN (unlabeled background pool)
      - has_vector: whether fishnet_with_SiteID.gpkg exists
    """
    csv_path = os.path.join(folder, "tile_index.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"tile_index.csv not found in: {folder}")

    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "presence" not in cols:
        raise ValueError("tile_index.csv must include a 'presence' column.")

    pres_col = cols["presence"]
    pres_raw = df[pres_col]
    def _to_pres(v):
        try:
            return 1 if int(v) == 1 else np.nan
        except Exception:
            return np.nan
    df["presence_norm"] = [ _to_pres(v) for v in pres_raw ]

    # resolve tile path
    if "path" in cols:
        df["path_resolved"] = df[cols["path"]].astype(str)
        df.loc[~df["path_resolved"].str.contains(r":\\|^/", regex=True), "path_resolved"] = \
            df.loc[~df["path_resolved"].str.contains(r":\\|^/", regex=True), "path_resolved"].apply(
                lambda p: os.path.join(folder, p))
    else:
        if "siteid" not in cols:
            raise ValueError("tile_index.csv must include 'SiteID' or 'path'.")
        df["path_resolved"] = df[cols["siteid"]].astype(str).apply(
            lambda sid: os.path.join(folder, f"{sid}.tif"))

    df["exists"] = df["path_resolved"].apply(os.path.isfile)
    df = df[df["exists"]].copy()
    if df.empty:
        raise RuntimeError("No existing .tif found according to tile_index.csv.")

    needed = ["minx","miny","maxx","maxy"]
    if not all(n in cols for n in needed):
        for n in needed:
            if n not in df.columns:
                df[n] = np.nan

    if "siteid" in cols:
        sid_col = cols["siteid"]
        df["SiteID"] = df[sid_col].astype(str)
    else:
        df["SiteID"] = df["path_resolved"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])

    df = df.set_index("SiteID", drop=False)
    pos_ids = df.index[df["presence_norm"] == 1].tolist()
    bg_pool = df.index[df["presence_norm"].isna()].tolist()
    has_vector = os.path.isfile(os.path.join(folder, "fishnet_with_SiteID.gpkg"))
    return df, pos_ids, bg_pool, folder, has_vector
def centroid_blocks(df, folder, nx=5, ny=5, use_vector=True):
    """
    Make spatial group IDs for CV:
      1) centroids from gpkg if available
      2) else centroids from CSV bounds
      3) else random groups
    Returns dict {SiteID -> 'x_y'}.
    """
    gpkg = os.path.join(folder, "fishnet_with_SiteID.gpkg")
    cx, cy = None, None
    if use_vector and os.path.isfile(gpkg):
        try:
            import geopandas as gpd
            gdf = gpd.read_file(gpkg)
            gdf["SiteID"] = gdf["SiteID"].astype(str)
            cent = gdf.geometry.centroid
            cx = pd.Series(cent.x.values, index=gdf["SiteID"])
            cy = pd.Series(cent.y.values, index=gdf["SiteID"])
        except Exception:
            cx = cy = None

    if cx is None or cy is None:
        if df[["minx","miny","maxx","maxy"]].isna().any(axis=None):
            sids = df["SiteID"].tolist()
            rng.shuffle(sids)
            groups = {}
            kx, ky = nx, ny
            for i, sid in enumerate(sids):
                groups[sid] = f"{i % kx}_{(i // kx) % ky}"
            return groups
        else:
            cx = ((df["minx"].values + df["maxx"].values) / 2.0)
            cy = ((df["miny"].values + df["maxy"].values) / 2.0)
            ids = df["SiteID"].values
            cx = pd.Series(cx, index=ids)
            cy = pd.Series(cy, index=ids)

    xbin = pd.qcut(cx, nx, labels=False, duplicates="drop")
    ybin = pd.qcut(cy, ny, labels=False, duplicates="drop")
    gid = (xbin.astype(int).astype(str) + "_" + ybin.astype(int).astype(str))
    groups = {sid: str(g) for sid, g in zip(cx.index, gid.values)}
    return groups
# -------------DATASET & MODEL  ---------------------
def read_tile(path, bands=None):
    with rasterio.open(path) as ds:
        arr = ds.read()  # (C,H,W)
        if bands is not None:
            arr = arr[np.array(bands)]
        arr = arr.astype(np.float32)
        vmax = np.percentile(arr, 99.5)
        if vmax > 0: arr = np.clip(arr / vmax, 0, 1)
        return arr
class TileDataset(Dataset):
    def __init__(self, site_ids, labels, df, img_size=224, aug=False):
        self.ids = list(site_ids)
        self.labels = None if labels is None else np.array(labels, dtype=np.float32)
        self.df = df
        if aug:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        first_path = df.loc[self.ids[0], "path_resolved"]
        with rasterio.open(first_path) as ds:
            self.in_ch = ds.count if USE_BANDS is None else len(USE_BANDS)

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        path = self.df.loc[sid, "path_resolved"]
        x = read_tile(path, USE_BANDS)          # (C,H,W)
        x = np.moveaxis(x, 0, 2)                # HWC
        x = self.tf(x)                          # CHW
        if self.labels is None:
            return x, sid
        return x, float(self.labels[i])
class SmallResNet(nn.Module):
    def __init__(self, in_ch=3, drop=0.0, pretrained=True):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained and in_ch==3 else None)
        if in_ch != 3:
            w = base.conv1.weight.data
            base.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(base.conv1.weight, nonlinearity='relu')
            if w.shape[1]==3:
                base.conv1.weight.data = w.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
        in_f = base.fc.in_features
        base.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(in_f, 1))
        self.net = base
    def forward(self, x): return self.net(x)

# ----------------  BOYCE INDEX ---------------------

"""
Boyce index is a rank-based metric for evaluating SDM performance:
It is calculated by:
STEP 1     Dividing prediction scores in to bins
STEP 2     Calculating the ratio of observed presences to background frequencies in each bin,
STEP 3     Correlating these ratios with prediction score bins using Spearman's rank correlation.


Possible cases where NaN is displayed:

1. if the number of presence in the fold of CV is fewer than 5
2. if the number of background points in the fold of CV is fewer than 50.
Boyce index will not be calculated in case of 1 or 2, make sure you have enough presence points and background.

"""
# bin number is set 20 in default, the lowest 1% and highest 1% are excluded to avoid extremes.

def _rank(a):
    temp = a.argsort()
    ranks = np.empty_like(temp, dtype=np.float64)
    ranks[temp] = np.arange(len(a))
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    avg = sums / counts
    return avg[inv]
def spearman_rho(x, y):
    rx, ry = _rank(np.asarray(x)), _rank(np.asarray(y))
    xz, yz = (rx - rx.mean()), (ry - ry.mean())
    denom = np.sqrt((xz**2).sum() * (yz**2).sum())
    return float((xz*yz).sum() / denom) if denom>0 else np.nan
def boyce_cbi(pres_scores, back_scores, n_bins=20):
    pres = np.asarray(pres_scores); back = np.asarray(back_scores)
    pres = pres[np.isfinite(pres)]; back = back[np.isfinite(back)]
    if len(pres) < 5 or len(back) < 50: return np.nan
    smin, smax = np.percentile(back, 1), np.percentile(back, 99)
    edges = np.linspace(smin, smax, n_bins+1)
    mids  = (edges[:-1]+edges[1:])/2
    pe, x = [], []
    eps=1e-9
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]+eps
        n_p = ((pres>=lo)&(pres<hi)).sum()
        n_b = ((back>=lo)&(back<hi)).sum()
        if n_b==0: continue
        pe.append( (n_p/max(1,len(pres))) / (n_b/max(1,len(back))) )
        x.append(mids[i])
    if len(pe)<3: return np.nan
    return spearman_rho(x, pe)

# --------------------- training & eval helpers ---------------------
def make_dataloader(ds, shuffle, num_workers):
    kwargs = dict(batch_size=BATCH_SIZE,
                  shuffle=shuffle,
                  num_workers=num_workers,
                  pin_memory=True if DEVICE=='cuda' else False)
    if num_workers > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
    return DataLoader(ds, **kwargs)
def build_model(in_ch, drop, pretrained):
    model = SmallResNet(in_ch=in_ch, drop=drop, pretrained=pretrained)
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(DEVICE)
    if USE_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, fullgraph=False)
        except Exception:
            pass
    return model
def train_one_split(site_tr, y_tr, site_va, y_va, hparams, in_ch, df, num_workers=NUM_WORKERS):
    ds_tr = TileDataset(site_tr, y_tr, df, img_size=IMG_SIZE, aug=hparams["aug"])
    ds_va = TileDataset(site_va, y_va, df, img_size=IMG_SIZE, aug=False)
    dl_tr = make_dataloader(ds_tr, shuffle=True,  num_workers=num_workers)
    dl_va = make_dataloader(ds_va, shuffle=False, num_workers=num_workers)

    model = build_model(in_ch=in_ch, drop=hparams["drop"], pretrained=(in_ch==3))
    pos_rate = float(np.mean(y_tr)) if len(y_tr)>0 else 0.5
    pos_weight = None
    if 0 < pos_rate < 0.5:
        pos_weight = torch.tensor([(1-pos_rate)/pos_rate], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=hparams["wd"])

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_MIXED and DEVICE=='cuda'))

    best_auc, best_state, wait = -1, None, 0
    for ep in range(1, MAX_EPOCHS+1):
        model.train()
        for xb, yb in dl_tr:
            if CHANNELS_LAST:
                xb = xb.to(memory_format=torch.channels_last)
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE).view(-1,1).float()
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_MIXED and DEVICE=='cuda')):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        # validate
        model.eval(); probs=[]; ytrue=[]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_MIXED and DEVICE=='cuda')):
            for xb, yb in dl_va:
                if CHANNELS_LAST:
                    xb = xb.to(memory_format=torch.channels_last)
                p = torch.sigmoid(model(xb.to(DEVICE, non_blocking=True))).cpu().numpy().ravel()
                probs.extend(p.tolist()); ytrue.extend(yb.numpy().ravel().tolist())
        try:
            auc = roc_auc_score(ytrue, probs)
        except Exception:
            auc = np.nan

        if (not np.isnan(auc)) and auc > best_auc:
            best_auc, best_state, wait = auc, {k:v.cpu() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model, best_auc
def eval_on_loader(model, dl):
    model.eval()
    probs, ytrue = [], []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_MIXED and DEVICE=='cuda')):
        for xb, yb in dl:
            if CHANNELS_LAST:
                xb = xb.to(memory_format=torch.channels_last)
            p = torch.sigmoid(model(xb.to(DEVICE, non_blocking=True))).cpu().numpy().ravel()
            probs.extend(p.tolist()); ytrue.extend(yb.numpy().ravel().tolist())
    ytrue = np.array(ytrue); probs = np.array(probs)
    try:
        auc = roc_auc_score(ytrue, probs)
    except Exception:
        auc = np.nan
    cbi = boyce_cbi(probs[ytrue==1], probs[ytrue==0])
    return auc, cbi, probs, ytrue



# ------------TWO MODES : BENCHMARK AND SINGLE RUN ---------------------
def run_benchmark(folder, ratios=None, n_boot=N_BOOT, num_workers=NUM_WORKERS):
    folder = os.path.abspath(folder)
    out_dir = os.path.join(folder, "cnn_sdm_out")
    os.makedirs(out_dir, exist_ok=True)

    df, pos_ids, bg_pool, tiles_dir, has_vector = load_from_folder(folder)
    assert len(pos_ids) >= 10, "Too few presences with tiles."

    with rasterio.open(df.loc[pos_ids[0], "path_resolved"]) as ds_any:
        in_ch = (len(USE_BANDS) if USE_BANDS is not None else ds_any.count)

    groups_map = centroid_blocks(df, folder, nx=5, ny=5, use_vector=has_vector)

    bg_ratios = ratios if ratios is not None else BG_RATIOS
    results = []
    for ratio in bg_ratios:
        for b in range(n_boot):
            n_pos = len(pos_ids)
            n_bg  = max(int(ratio) * n_pos, 1)
            replace = n_bg > len(bg_pool)
            bg_ids = rng.choice(bg_pool, size=n_bg, replace=replace).tolist()
            sids = pos_ids + bg_ids
            y    = np.array([1]*len(pos_ids) + [0]*len(bg_ids), dtype=np.float32)
            groups = np.array([groups_map.get(sid,"0_0") for sid in sids])

            # tune per bootstrap
            best_cfg, best_auc = None, -1
            for cfg in GRID_HPARAMS:
                aucs, cbis = [], []
                gkf = GroupKFold(n_splits=5)
                for tr, va in gkf.split(sids, groups=groups):
                    site_tr = [sids[i] for i in tr]; y_tr = y[tr]
                    site_va = [sids[i] for i in va]; y_va = y[va]
                    model, auc = train_one_split(site_tr, y_tr, site_va, y_va, cfg, in_ch, df, num_workers=num_workers)
                    ds_va = TileDataset(site_va, y_va, df, img_size=IMG_SIZE, aug=False)
                    dl_va = make_dataloader(ds_va, shuffle=False, num_workers=num_workers)
                    a, c, _, _ = eval_on_loader(model, dl_va)
                    aucs.append(a); cbis.append(c)
                mean_auc, mean_cbi = float(np.nanmean(aucs)), float(np.nanmean(cbis))
                if mean_auc > best_auc: best_auc, best_cfg = mean_auc, cfg

            # record with best cfg
            aucs, cbis = [], []
            gkf = GroupKFold(n_splits=5)
            for tr, va in gkf.split(sids, groups=groups):
                site_tr = [sids[i] for i in tr]; y_tr = y[tr]
                site_va = [sids[i] for i in va]; y_va = y[va]
                model, auc = train_one_split(site_tr, y_tr, site_va, y_va, best_cfg, in_ch, df, num_workers=num_workers)
                ds_va = TileDataset(site_va, y_va, df, img_size=IMG_SIZE, aug=False)
                dl_va = make_dataloader(ds_va, shuffle=False, num_workers=num_workers)
                a, c, _, _ = eval_on_loader(model, dl_va)
                aucs.append(a); cbis.append(c)
            results.append({"ratio":int(ratio),"boot":int(b),"cfg":best_cfg,
                            "AUCs":aucs,"CBIs":cbis,
                            "AUC_mean":float(np.nanmean(aucs)),
                            "CBI_mean":float(np.nanmean(cbis))})
            print(f"[ratio {ratio} | boot {b}] AUC={np.nanmean(aucs):.3f}  CBI={np.nanmean(cbis):.3f}  cfg={best_cfg}")

    # choose best ratio by CBI mean (AUC tie-break)
    df_res = pd.DataFrame([{k:v for k,v in r.items() if k not in ("AUCs","CBIs","cfg")} for r in results])
    grp = df_res.groupby("ratio").agg({"CBI_mean":"mean","AUC_mean":"mean"}).reset_index()
    grp = grp.sort_values(["CBI_mean","AUC_mean"], ascending=False)
    best_ratio = int(grp.iloc[0]["ratio"])
    # majority vote cfg under best ratio
    votes = {}
    for r in results:
        if r["ratio"]==best_ratio:
            key = json.dumps(r["cfg"], sort_keys=True)
            votes[key] = votes.get(key,0)+1
    best_cfg = json.loads(max(votes.items(), key=lambda kv: kv[1])[0])
    print(f"[BEST] ratio={best_ratio}  cfg={best_cfg}")

    # retrain (best ratio & cfg) and predict all
    n_pos = len(pos_ids); n_bg = max(best_ratio*n_pos,1)
    replace = n_bg > len(bg_pool)
    bg_ids = rng.choice(bg_pool, size=n_bg, replace=replace).tolist()
    sids = pos_ids + bg_ids
    y    = np.array([1]*len(pos_ids) + [0]*len(bg_ids), dtype=np.float32)
    groups = np.array([groups_map.get(sid,"0_0") for sid in sids])
    gkf = GroupKFold(n_splits=5)
    best_model, best_auc = None, -1
    for tr, va in gkf.split(sids, groups=groups):
        m, auc = train_one_split([sids[i] for i in tr], y[tr],
                                 [sids[i] for i in va], y[va],
                                 best_cfg, in_ch, df, num_workers=num_workers)
        if auc>best_auc: best_auc, best_model = auc, m

    all_ids = df["SiteID"].tolist()
    ds_all = TileDataset(all_ids, None, df, img_size=IMG_SIZE, aug=False)
    dl_all = make_dataloader(ds_all, shuffle=False, num_workers=num_workers)
    preds=[]
    best_model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_MIXED and DEVICE=='cuda')):
        for xb, sids in dl_all:
            if CHANNELS_LAST:
                xb = xb.to(memory_format=torch.channels_last)
            p = torch.sigmoid(best_model(xb.to(DEVICE, non_blocking=True))).cpu().numpy().ravel().tolist()
            preds.extend(list(zip(list(sids), p)))
    out_csv = os.path.join(out_dir, "predictions.csv")
    pd.DataFrame(preds, columns=["SiteID","p_hat"]).to_csv(out_csv, index=False)
    print("Predictions ->", out_csv)

    # PDF summary
    pdf_path = os.path.join(out_dir, "summary.pdf")
    with PdfPages(pdf_path) as pdf:
        # page 1
        plt.figure(figsize=(8.5,11)); plt.axis('off')
        txt = f"""CNN SDM summary (benchmark)
Folder: {folder}
tile_index.csv used for labels (presence); vector present: {has_vector}
Tiles:  {len(all_ids)}  | Presences: {len(pos_ids)}
Ratios: {bg_ratios}   Bootstrap: {n_boot}
Best ratio: {best_ratio}   Best cfg: {best_cfg}
Metrics: AUC (discrimination), Boyce (CBI)
"""
        plt.text(0.05,0.95,txt,va='top',family='monospace'); pdf.savefig(); plt.close()

        # page 2: AUC boxplot
        plt.figure(figsize=(10,5))
        data = [[auc for r in results if r["ratio"]==ratio for auc in r["AUCs"]] for ratio in sorted(set(bg_ratios))]
        plt.boxplot(data, labels=[str(r) for r in sorted(set(bg_ratios))], showmeans=True)
        plt.title("AUC by background ratio (5-fold × bootstraps)")
        plt.xlabel("Background : Presence"); plt.ylabel("AUC"); pdf.savefig(); plt.close()

        # page 3: Boyce boxplot
        plt.figure(figsize=(10,5))
        data = [[c for r in results if r["ratio"]==ratio for c in r["CBIs"]] for ratio in sorted(set(bg_ratios))]
        plt.boxplot(data, labels=[str(r) for r in sorted(set(bg_ratios))], showmeans=True)
        plt.title("Boyce index (CBI) by background ratio")
        plt.xlabel("Background : Presence"); plt.ylabel("CBI (Boyce)"); pdf.savefig(); plt.close()

        # page 4: means table
        plt.figure(figsize=(8.5,6)); plt.axis('off')
        tbl = pd.DataFrame({
            "ratio": sorted(set(bg_ratios)),
            "AUC_mean": [np.nanmean([auc for r in results if r["ratio"]==ratio for auc in r["AUCs"]]) for ratio in sorted(set(bg_ratios))],
            "CBI_mean": [np.nanmean([c   for r in results if r["ratio"]==ratio for c   in r["CBIs"]]) for ratio in sorted(set(bg_ratios))],
        })
        plt.table(cellText=np.round(tbl.values,3), colLabels=tbl.columns, loc='center')
        plt.title("Mean metrics across bootstraps & folds")
        pdf.savefig(); plt.close()
    print("Summary PDF ->", pdf_path)
def run_single(folder, ratio=10, n_splits=5, num_workers=NUM_WORKERS):
    """
    Single-run mode:
      1) sample background at a given ratio
      2) small hyperparameter search via GroupKFold(n_splits)
      3) train with best hyperparameters; pick the best fold model
      4) evaluate once (AUC/CBI) and predict over all tiles
    """
    folder = os.path.abspath(folder)
    out_dir = os.path.join(folder, "cnn_sdm_out")
    os.makedirs(out_dir, exist_ok=True)

    df, pos_ids, bg_pool, tiles_dir, has_vector = load_from_folder(folder)
    assert len(pos_ids) >= 10, "Too few presences with tiles."

    with rasterio.open(df.loc[pos_ids[0], "path_resolved"]) as ds_any:
        in_ch = (len(USE_BANDS) if USE_BANDS is not None else ds_any.count)

    groups_map = centroid_blocks(df, folder, nx=5, ny=5, use_vector=has_vector)

    # one sampling
    n_pos = len(pos_ids)
    n_bg  = max(int(ratio) * n_pos, 1)
    replace = n_bg > len(bg_pool)
    bg_ids = rng.choice(bg_pool, size=n_bg, replace=replace).tolist()
    sids = pos_ids + bg_ids
    y    = np.array([1]*len(pos_ids) + [0]*len(bg_ids), dtype=np.float32)
    groups = np.array([groups_map.get(sid,"0_0") for sid in sids])

    # hyperparameter search via CV
    best_cfg, best_auc_mean, best_cbi_mean = None, -1, -1
    for cfg in GRID_HPARAMS:
        aucs, cbis = [], []
        gkf = GroupKFold(n_splits=n_splits)
        for tr, va in gkf.split(sids, groups=groups):
            site_tr = [sids[i] for i in tr]; y_tr = y[tr]
            site_va = [sids[i] for i in va]; y_va = y[va]
            model, _ = train_one_split(site_tr, y_tr, site_va, y_va, cfg, in_ch, df, num_workers=num_workers)
            ds_va = TileDataset(site_va, y_va, df, img_size=IMG_SIZE, aug=False)
            dl_va = make_dataloader(ds_va, shuffle=False, num_workers=num_workers)
            a, c, _, _ = eval_on_loader(model, dl_va)
            aucs.append(a); cbis.append(c)
        mean_auc, mean_cbi = float(np.nanmean(aucs)), float(np.nanmean(cbis))
        print(f"[single][cfg={cfg}] AUC_mean={mean_auc:.3f}  CBI_mean={mean_cbi:.3f}")
        # AUC primary; CBI tie-break
        if (mean_auc > best_auc_mean) or (np.isclose(mean_auc, best_auc_mean) and mean_cbi > best_cbi_mean):
            best_auc_mean, best_cbi_mean, best_cfg = mean_auc, mean_cbi, cfg

    print(f"[single][BEST CFG] {best_cfg}  AUC_mean={best_auc_mean:.3f}  CBI_mean={best_cbi_mean:.3f}")

    # train n_splits folds with best cfg; pick the best fold model
    gkf = GroupKFold(n_splits=n_splits)
    best_model, best_auc = None, -1
    fold_metrics = []
    for k, (tr, va) in enumerate(gkf.split(sids, groups=groups), start=1):
        site_tr = [sids[i] for i in tr]; y_tr = y[tr]
        site_va = [sids[i] for i in va]; y_va = y[va]
        model, _ = train_one_split(site_tr, y_tr, site_va, y_va, best_cfg, in_ch, df, num_workers=num_workers)
        ds_va = TileDataset(site_va, y_va, df, img_size=IMG_SIZE, aug=False)
        dl_va = make_dataloader(ds_va, shuffle=False, num_workers=num_workers)
        a, c, _, _ = eval_on_loader(model, dl_va)
        fold_metrics.append({"fold": k, "AUC": float(a), "CBI": float(c)})
        if (not np.isnan(a)) and a > best_auc:
            best_auc, best_model = a, model

    # predict all tiles with the best fold model
    all_ids = df["SiteID"].tolist()
    ds_all = TileDataset(all_ids, None, df, img_size=IMG_SIZE, aug=False)
    dl_all = make_dataloader(ds_all, shuffle=False, num_workers=num_workers)
    preds=[]
    best_model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_MIXED and DEVICE=='cuda')):
        for xb, sids in dl_all:
            if CHANNELS_LAST:
                xb = xb.to(memory_format=torch.channels_last)
            p = torch.sigmoid(best_model(xb.to(DEVICE, non_blocking=True))).cpu().numpy().ravel().tolist()
            preds.extend(list(zip(list(sids), p)))

    out_dir = os.path.join(folder, "cnn_sdm_out")
    out_csv = os.path.join(out_dir, f"predictions_single_r{ratio}.csv")
    pd.DataFrame(preds, columns=["SiteID","p_hat"]).to_csv(out_csv, index=False)
    print("Predictions ->", out_csv)

    # save single-run metrics
    metrics = {
        "mode": "single",
        "ratio": int(ratio),
        "best_cfg": best_cfg,
        "cv_mean_auc": float(best_auc_mean),
        "cv_mean_cbi": float(best_cbi_mean),
        "fold_metrics": fold_metrics,
        "tiles": int(len(all_ids)),
        "presences": int(len(pos_ids)),
    }
    metrics_path = os.path.join(out_dir, f"single_r{ratio}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("Metrics ->", metrics_path)

    # short PDF
    pdf_path = os.path.join(out_dir, f"summary_single_r{ratio}.pdf")
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(8.5,11)); plt.axis('off')
        txt = f"""CNN SDM summary (single)
Folder: {folder}
vector present: {has_vector}
Tiles: {len(all_ids)} | Presences: {len(pos_ids)}
Ratio: {ratio}
Best cfg: {best_cfg}
CV mean AUC: {best_auc_mean:.3f} | CV mean CBI: {best_cbi_mean:.3f}
"""
        plt.text(0.05,0.95,txt,va='top',family='monospace'); pdf.savefig(); plt.close()

        # per-fold bars
        if len(fold_metrics) > 0:
            plt.figure(figsize=(10,5))
            plt.bar([m["fold"] for m in fold_metrics],
                    [m["AUC"] for m in fold_metrics])
            plt.title("Per-fold AUC (single mode)")
            plt.xlabel("Fold"); plt.ylabel("AUC"); pdf.savefig(); plt.close()

            plt.figure(figsize=(10,5))
            plt.bar([m["fold"] for m in fold_metrics],
                    [m["CBI"] for m in fold_metrics])
            plt.title("Per-fold CBI (single mode)")
            plt.xlabel("Fold"); plt.ylabel("CBI"); pdf.savefig(); plt.close()

    print("Summary PDF ->", pdf_path)



# --------------------- main ---------------------
def main():
    global USE_MIXED, USE_COMPILE, CHANNELS_LAST, NUM_WORKERS

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["benchmark","single"], default="benchmark",
                    help="benchmark: original bootstrap benchmark; single: one-shot tuning/eval/predict")
    ap.add_argument("--folder", required=True, help="Directory containing *.tif and tile_index.csv")
    ap.add_argument("--ratios", default=None, help="(benchmark) Comma-separated, e.g. '1,5,10,50,100'")
    ap.add_argument("--boots", type=int, default=N_BOOT, help="(benchmark) Number of bootstraps per ratio")
    ap.add_argument("--ratio", type=int, default=10, help="(single) background:presence ratio, e.g., 10 means 1:10")
    ap.add_argument("--splits", type=int, default=5, help="GroupKFold folds (default 5)")
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader worker processes")
    # speed switches
    ap.add_argument("--mixed", action="store_true", help="Enable AMP mixed precision (CUDA)")
    ap.add_argument("--compile", dest="compile_model", action="store_true", help="Enable torch.compile (PyTorch>=2.0)")
    ap.add_argument("--channels-last", action="store_true", help="Use channels_last memory format")
    args = ap.parse_args()


    USE_MIXED = bool(args.mixed)
    USE_COMPILE = bool(args.compile_model)
    CHANNELS_LAST = bool(args.channels_last)
    NUM_WORKERS = int(args.num_workers)

    ratios = None
    if args.ratios is not None:
        ratios = [int(x) for x in str(args.ratios).split(",") if str(x).strip()!=""]

    if args.mode == "single":
        run_single(args.folder, ratio=args.ratio, n_splits=args.splits, num_workers=NUM_WORKERS)
    else:
        run_benchmark(args.folder, ratios=ratios, n_boot=args.boots, num_workers=NUM_WORKERS)

if __name__ == "__main__":
    main()

# run this in Win+R command line:
# python [change to the folder]\CNN_based_SDM.py --folder "[folder path of data]" --ratios "1,5,10,50,100" --boots 10
