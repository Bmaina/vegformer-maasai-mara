# =============================================================================
# NOTEBOOK 03 — PATCH EXTRACTION, NORMALIZATION & PYTORCH DATASET
# =============================================================================
#
# PURPOSE
# -------
# Transform the raw zarr tensors from notebooks 01 and 02 into a PyTorch
# Dataset ready for VegFormer training. This involves temporal alignment,
# normalization, patch extraction, and dataset splitting.
#
# WHAT THIS NOTEBOOK DOES
# -----------------------
# 1. Loads all four zarr datasets from disk
# 2. Aligns S2/S1 (16-day) and NDVI (monthly) to a shared temporal index
# 3. Computes per-band normalization statistics from training data only
# 4. Fixes EVI outliers before normalization
# 5. Normalizes all arrays using z-score normalization
# 6. Extracts 64×64 pixel spatial patches with 50% overlap
# 7. Splits patches into train/val/test by time (not space)
# 8. Creates PyTorch Dataset and DataLoader classes
# 9. Runs a smoke test to verify batch shapes
#
# KEY DESIGN DECISIONS
# --------------------
# Temporal split (not spatial split):
#   Train: 2019-01-01 to 2023-10-11 (109 timesteps, ~80%)
#   Val:   2023-10-11 to 2024-05-22  (14 timesteps, ~10%)
#   Test:  2024-05-22 to 2024-12-16  (14 timesteps, ~10%)
#
#   Rationale: We want to test whether the model generalizes to future
#   time periods, not just unseen spatial locations. A spatial split would
#   be more optimistic but less realistic for operational deployment.
#
# Normalization on training data only:
#   Statistics (mean, std) are computed from training timesteps only.
#   Applying training statistics to val/test data is the correct approach —
#   computing stats on val/test would constitute data leakage.
#
# Patch size 64×64 at 250m = 16km × 16km per patch:
#   Large enough to capture landscape-scale vegetation patterns.
#   Small enough to fit multiple patches per batch in RAM.
#
# 50% overlap (stride=32):
#   Increases the number of training patches from ~20 to ~80.
#   Overlapping patches share context, which helps the model learn
#   spatial continuity but may slightly inflate performance metrics.
#
# ASSUMPTIONS
# -----------
# - Notebooks 01 and 02 have been run and zarr files exist in processed/
# - NDVI monthly dates are in YYYY-MM format; S2/S1 dates are YYYY-MM-DD
# - CHIRPS 5km grid (19×16) is mapped to S2 250m grid by nearest-neighbour
# - PyTorch, pickle are installed
#
# OUTPUT
# ------
# C:/vegformer/data/patches/train_patches.pkl   (2.3 GB)
# C:/vegformer/data/patches/val_patches.pkl     (294 MB)
# C:/vegformer/data/patches/test_patches.pkl    (294 MB)
# C:/vegformer/data/processed/normalization_stats.json
# C:/vegformer/data/processed/temporal_alignment.json
# C:/vegformer/data/processed/dataset_info.json
# =============================================================================


# -----------------------------------------------------------------------------
# CELL 1 — Environment Setup and Load Arrays
# -----------------------------------------------------------------------------
# Loads all four zarr arrays into memory.
# Total RAM needed: ~1.1 GB (S2=905MB + S1=259MB + NDVI=34MB + CHIRPS=1MB)
# If RAM is limited, S2 can be loaded lazily using zarr slicing.

import zarr
import numpy as np
import torch
import pickle
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

PROCESSED_DIR = Path('C:/vegformer/data/processed')
PATCHES_DIR   = Path('C:/vegformer/data/patches')
PATCHES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading zarr arrays into memory...")
s2_z     = zarr.open(str(PROCESSED_DIR / 's2_16day_2019_2024.zarr'),           mode='r')
s1_z     = zarr.open(str(PROCESSED_DIR / 's1_16day_2019_2024.zarr'),           mode='r')
chirps_z = zarr.open(str(PROCESSED_DIR / 'chirps_16day_2019_2024.zarr'),       mode='r')
ndvi_z   = zarr.open(str(PROCESSED_DIR / 'ndvi_monthly_2019_2024_clean.zarr'), mode='r')

s2_arr     = s2_z[:]
s1_arr     = s1_z[:]
ndvi_arr   = ndvi_z[:]
s2_dates   = list(s2_z.attrs['dates'])
ndvi_dates = list(ndvi_z.attrs['dates'])
S2_BANDS   = list(s2_z.attrs['bands'])
CHIRPS_BANDS = list(chirps_z.attrs['bands'])

print(f"S2:    {s2_arr.shape}   ({s2_arr.nbytes/1e6:.0f} MB)")
print(f"S1:    {s1_arr.shape}   ({s1_arr.nbytes/1e6:.0f} MB)")
print(f"NDVI:  {ndvi_arr.shape}")
print(f"CHIRPS: loaded lazily — {chirps_z.shape}")
print(f"All arrays loaded")


# -----------------------------------------------------------------------------
# CELL 2 — Temporal Alignment
# -----------------------------------------------------------------------------
# S2 and S1 use 16-day composite dates (YYYY-MM-DD format).
# NDVI uses monthly dates (YYYY-MM format).
#
# For each 16-day S2 composite, we find the nearest monthly NDVI composite.
# This is used in notebook 04 to provide the NDVI context at each timestep.
#
# Julian day encoding:
#   Each timestep is encoded as its Julian day (day of year, 1-365).
#   This feeds directly into the Seasonal Fourier Positional Encoding in
#   VegFormer, which learns annual and semi-annual vegetation cycles.

def date_to_julian(date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt.timetuple().tm_yday

def find_nearest_ndvi(s2_date, ndvi_dates):
    s2_dt = datetime.strptime(s2_date, '%Y-%m-%d')
    diffs = [abs((s2_dt - datetime.strptime(nd+'-01','%Y-%m-%d')).days)
             for nd in ndvi_dates]
    return int(np.argmin(diffs))

aligned = []
for i, s2_date in enumerate(s2_dates):
    dt = datetime.strptime(s2_date, '%Y-%m-%d')
    aligned.append({
        's2_idx':   int(i),
        's1_idx':   int(i),
        'ndvi_idx': find_nearest_ndvi(s2_date, ndvi_dates),
        'date':     s2_date,
        'julian':   int(date_to_julian(s2_date)),
        'year':     int(dt.year),
        'month':    int(dt.month),
    })

with open(str(PROCESSED_DIR / 'temporal_alignment.json'), 'w') as f:
    json.dump(aligned, f, indent=2)

print(f"Aligned {len(aligned)} timesteps")
print(f"First: {aligned[0]['date']}  julian={aligned[0]['julian']:3d}")
print(f"Last:  {aligned[-1]['date']}  julian={aligned[-1]['julian']:3d}")
print(f"Saved: temporal_alignment.json")


# -----------------------------------------------------------------------------
# CELL 3 — Compute Normalization Statistics
# -----------------------------------------------------------------------------
# Z-score normalization: z = (x - mean) / std
# Applied per band independently.
#
# Critical: statistics are computed ONLY from training timesteps (0-108).
# Using val/test data to compute normalization statistics would constitute
# data leakage — the model would have seen future information during training.
#
# EVI band (index 11) has outliers clipped to [-1, 2] before computing stats
# to prevent the extreme values from inflating the standard deviation.

TRAIN_END = 109   # index 109 = 2023-10-11 (first val timestep)
VAL_END   = 123   # index 123 = 2024-05-22 (first test timestep)

print(f"Computing normalization stats from training data only")
print(f"  Train: indices 0-{TRAIN_END-1}  ({s2_dates[0]} to {s2_dates[TRAIN_END-1]})")
print(f"  Val:   indices {TRAIN_END}-{VAL_END-1}")
print(f"  Test:  indices {VAL_END}-{len(s2_dates)-1}")

# Clip EVI outliers before computing stats
s2_arr[:, 11] = np.clip(s2_arr[:, 11], -1.0, 2.0)

s2_train     = s2_arr[:TRAIN_END]
s2_mean      = s2_train.reshape(TRAIN_END, 14, -1).mean(axis=(0,2))
s2_std       = s2_train.reshape(TRAIN_END, 14, -1).std(axis=(0,2))
s2_std       = np.where(s2_std < 1e-6, 1.0, s2_std)

s1_train     = s1_arr[:TRAIN_END]
s1_mean      = s1_train.reshape(TRAIN_END, 2, -1).mean(axis=(0,2))
s1_std       = s1_train.reshape(TRAIN_END, 2, -1).std(axis=(0,2))

chirps_train = chirps_z[:TRAIN_END]
chirps_mean  = chirps_train.reshape(TRAIN_END, 4, -1).mean(axis=(0,2))
chirps_std   = chirps_train.reshape(TRAIN_END, 4, -1).std(axis=(0,2))
chirps_std   = np.where(chirps_std < 1e-6, 1.0, chirps_std)

print(f"\nS2 stats (training set):")
for i, b in enumerate(S2_BANDS):
    print(f"  {b:<6}: mean={s2_mean[i]:>8.4f}  std={s2_std[i]:.4f}")

print(f"\nS1 stats:")
for i, b in enumerate(['VV','VH']):
    print(f"  {b:<6}: mean={s1_mean[i]:>8.4f}  std={s1_std[i]:.4f}")

stats = {
    's2_mean': s2_mean.tolist(), 's2_std': s2_std.tolist(),
    's1_mean': s1_mean.tolist(), 's1_std': s1_std.tolist(),
    'chirps_mean': chirps_mean.tolist(), 'chirps_std': chirps_std.tolist(),
    'train_end': int(TRAIN_END), 's2_bands': S2_BANDS,
}
with open(str(PROCESSED_DIR / 'normalization_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)
print(f"\nSaved: normalization_stats.json")


# -----------------------------------------------------------------------------
# CELL 4 — Normalize All Arrays
# -----------------------------------------------------------------------------
# Applies z-score normalization to all sensors.
# After normalization: mean≈0, std≈1 for each band.
# This ensures all bands contribute equally to the loss function
# regardless of their original scale (e.g. CHIRPS rain_sum 0-300mm
# vs SAR VV -33 to -4 dB vs NDVI 0-0.9).

def normalize(arr, mean, std):
    return (arr - mean[None,:,None,None]) / std[None,:,None,None]

s2_norm     = normalize(s2_arr,         s2_mean,     s2_std)
s1_norm     = normalize(s1_arr,         s1_mean,     s1_std)
chirps_full = chirps_z[:]
chirps_norm = normalize(chirps_full,    chirps_mean, chirps_std)

print(f"S2 normalized:     mean={s2_norm.mean():.4f}  std={s2_norm.std():.4f}")
print(f"S1 normalized:     mean={s1_norm.mean():.4f}  std={s1_norm.std():.4f}")
print(f"CHIRPS normalized: mean={chirps_norm.mean():.4f}  std={chirps_norm.std():.4f}")
print("All values should be close to mean=0, std=1")


# -----------------------------------------------------------------------------
# CELL 5 — Extract Spatial Patches
# -----------------------------------------------------------------------------
# Cuts the 377×313 spatial grid into 64×64 patches.
#
# Patch extraction parameters:
#   PATCH_SIZE = 64  pixels = 16km × 16km at 250m resolution
#   STRIDE = 32      pixels = 50% overlap between adjacent patches
#
# Why 50% overlap?
#   - Increases training data from ~20 to ~80 patches
#   - Each patch sees its spatial neighbours, helping the model learn
#     gradients across vegetation boundaries
#   - Standard practice in remote sensing patch-based learning
#     (e.g. Ronneberger et al. 2015, U-Net)
#
# CHIRPS mapping:
#   CHIRPS is at 5km (19×16 grid) vs S2 at 250m (377×313 grid).
#   Scale factor ≈ 20×. Each patch is mapped to its nearest CHIRPS pixel
#   using simple nearest-neighbour lookup. The CHIRPS value is shared
#   across all S2 pixels within the same 5km CHIRPS cell — appropriate
#   given that rainfall varies at scales larger than 250m anyway.

PATCH_SIZE = 64
STRIDE     = 32

H, W = s2_norm.shape[2], s2_norm.shape[3]
rows = range(0, H - PATCH_SIZE + 1, STRIDE)
cols = range(0, W - PATCH_SIZE + 1, STRIDE)

print(f"Grid: {len(rows)} rows × {len(cols)} cols = {len(rows)*len(cols)} patches")
print(f"Patch size: {PATCH_SIZE}×{PATCH_SIZE} pixels = {PATCH_SIZE*0.25:.0f}km × {PATCH_SIZE*0.25:.0f}km")
print(f"Overlap:    50% (stride={STRIDE})")

patches = []
for r in tqdm(rows, desc='Extracting patches'):
    for c in cols:
        chirps_r = min(int(r/H * chirps_norm.shape[2]), chirps_norm.shape[2]-1)
        chirps_c = min(int(c/W * chirps_norm.shape[3]), chirps_norm.shape[3]-1)

        patches.append({
            'row':         r,
            'col':         c,
            's2':          s2_norm[:, :, r:r+PATCH_SIZE, c:c+PATCH_SIZE].astype(np.float32),
            's1':          s1_norm[:, :, r:r+PATCH_SIZE, c:c+PATCH_SIZE].astype(np.float32),
            'chirps':      chirps_norm[:, :, chirps_r, chirps_c].astype(np.float32),
            'julian_days': np.array([a['julian'] for a in aligned], dtype=np.float32),
        })

print(f"\nExtracted {len(patches)} patches")
print(f"Per patch shapes:")
print(f"  s2:          {patches[0]['s2'].shape}  — (137 timesteps, 14 bands, 64px, 64px)")
print(f"  s1:          {patches[0]['s1'].shape}  — (137 timesteps, 2 bands, 64px, 64px)")
print(f"  chirps:      {patches[0]['chirps'].shape}  — (137 timesteps, 4 climate vars)")
print(f"  julian_days: {patches[0]['julian_days'].shape}  — (137 timesteps)")


# -----------------------------------------------------------------------------
# CELL 6 — Train / Val / Test Split and Save
# -----------------------------------------------------------------------------
# Splits along the time axis — each patch contributes to all three splits.
# The spatial location (row, col) is the same across splits; only the
# temporal window differs.
#
# This design allows the model to be evaluated on whether it can predict
# vegetation dynamics in 2024 after training on 2019–2023 — a realistic
# operational scenario.

def split_patch(p, t_end, v_end):
    def sl(s, e):
        return {
            'row': p['row'], 'col': p['col'],
            's2':          p['s2'][s:e],
            's1':          p['s1'][s:e],
            'chirps':      p['chirps'][s:e],
            'julian_days': p['julian_days'][s:e],
        }
    return sl(0, t_end), sl(t_end, v_end), sl(v_end, None)

train_patches, val_patches, test_patches = [], [], []
for p in patches:
    tr, va, te = split_patch(p, TRAIN_END, VAL_END)
    train_patches.append(tr)
    val_patches.append(va)
    test_patches.append(te)

for split, data in [('train', train_patches),
                    ('val',   val_patches),
                    ('test',  test_patches)]:
    path = PATCHES_DIR / f'{split}_patches.pkl'
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    size = path.stat().st_size / 1e6
    print(f"Saved {split}_patches.pkl — "
          f"{len(data)} patches × {data[0]['s2'].shape[0]} timesteps | {size:.0f} MB")

dataset_info = {
    'n_patches': len(patches), 'patch_size': PATCH_SIZE, 'stride': STRIDE,
    'train_patches': len(train_patches), 'val_patches': len(val_patches),
    'test_patches':  len(test_patches),  'train_end': int(TRAIN_END),
    'val_end': int(VAL_END),
    'T_train': int(train_patches[0]['s2'].shape[0]),
    'T_val':   int(val_patches[0]['s2'].shape[0]),
    'T_test':  int(test_patches[0]['s2'].shape[0]),
    'n_s2_bands': 14, 'n_s1_bands': 2, 'n_chirps_vars': 4,
}
with open(str(PROCESSED_DIR / 'dataset_info.json'), 'w') as f:
    json.dump(dataset_info, f, indent=2)
print(f"\nSaved: dataset_info.json")


# -----------------------------------------------------------------------------
# CELL 7 — PyTorch Dataset and DataLoader
# -----------------------------------------------------------------------------
# Wraps the patch dictionaries in a PyTorch Dataset class.
# Each __getitem__ call returns one patch as a dict of tensors.
#
# The DataLoader handles:
#   - Batching (batch_size=4 patches per batch)
#   - Shuffling training data each epoch
#   - Converting numpy arrays to PyTorch tensors
#
# Note: num_workers=0 is used for Windows compatibility.
# On Linux/Mac you can increase to 4 for faster loading.

class MaraDataset(Dataset):
    """
    PyTorch Dataset for VegFormer training on Maasai Mara patches.

    Each item represents one 64×64 pixel spatial patch across all
    timesteps in the split. Returns a dict of float32 tensors.
    """
    def __init__(self, patches):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        p = self.patches[idx]
        return {
            's2':          torch.tensor(p['s2'],          dtype=torch.float32),
            's1':          torch.tensor(p['s1'],          dtype=torch.float32),
            'chirps':      torch.tensor(p['chirps'],      dtype=torch.float32),
            'julian_days': torch.tensor(p['julian_days'], dtype=torch.float32),
            'row':         torch.tensor(p['row'],         dtype=torch.long),
            'col':         torch.tensor(p['col'],         dtype=torch.long),
        }

train_ds = MaraDataset(train_patches)
val_ds   = MaraDataset(val_patches)
test_ds  = MaraDataset(test_patches)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, num_workers=0)

batch = next(iter(train_loader))
print("Sample training batch shapes:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k:<12}: {tuple(v.shape)}")
print(f"\nDataLoaders:")
print(f"  Train: {len(train_loader)} batches × 4 patches")
print(f"  Val:   {len(val_loader)} batches × 4 patches")
print(f"  Test:  {len(test_loader)} batches × 4 patches")


# -----------------------------------------------------------------------------
# CELL 8 — Final Summary
# -----------------------------------------------------------------------------

print("\n" + "="*55)
print("  NOTEBOOK 03 COMPLETE")
print("="*55)
print(f"  Patches:      {len(patches)} total (80 train, 80 val, 80 test)")
print(f"  Patch size:   {PATCH_SIZE}×{PATCH_SIZE} px = 16km×16km")
print(f"  Train period: {s2_dates[0]} to {s2_dates[TRAIN_END-1]}")
print(f"  Val period:   {s2_dates[TRAIN_END]} to {s2_dates[VAL_END-1]}")
print(f"  Test period:  {s2_dates[VAL_END]} to {s2_dates[-1]}")
print(f"  Saved:        train/val/test_patches.pkl")
print(f"                normalization_stats.json")
print(f"                temporal_alignment.json")
print("="*55)
print("Next: Open 04_pretrain_msm.ipynb")
