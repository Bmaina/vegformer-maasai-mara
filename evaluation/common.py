"""Shared, torch-free helpers for VegFormer validation (numpy / scikit-learn only)."""
import pickle
import numpy as np

NDVI_IDX = 10  # optical stack = 10 S2 bands + [NDVI, EVI, NBR, NDWI]


def load_patches(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def patch_series(p):
    """Spatial-mean series per patch, exactly as the pre-training notebook feeds the model."""
    return {
        "sar": np.asarray(p["s1"], dtype=np.float32).mean(axis=(-2, -1)),      # (T, 2)
        "optical": np.asarray(p["s2"], dtype=np.float32).mean(axis=(-2, -1)),  # (T, n_bands)
        "climate": np.asarray(p["chirps"], dtype=np.float32),                  # (T, 4)
        "dates": np.asarray(p["julian_days"], dtype=np.float32),               # (T,)
    }


def concat_by_position(splits):
    """Join train/val/test patches that share (row, col) into one continuous series.

    The README split is temporal: the same ~80 positions appear in every split. Returns
    {(row, col): series dict over the full period} and the index where each split starts.
    """
    out, starts, t0 = {}, [], 0
    for patches in splits:
        starts.append(t0)
        t0 += len(patches[0]["julian_days"])
        for p in patches:
            key = (int(p["row"]), int(p["col"]))
            s = patch_series(p)
            if key not in out:
                out[key] = {k: [v] for k, v in s.items()}
            else:
                for k, v in s.items():
                    out[key][k].append(v)
    full = {}
    for key, d in out.items():
        if len(d["dates"]) == len(splits):   # present in every split
            full[key] = {k: np.concatenate(v, axis=0) for k, v in d.items()}
    return full, starts


# ---- masking ---------------------------------------------------------------------------
def make_mask(T, window, ratio, rng, run_length=1):
    """Boolean (T,) mask over `window=(lo,hi)`; masks runs of `run_length` steps (gap-filling)."""
    lo, hi = window
    n_runs = max(1, int(round((hi - lo) * ratio / run_length)))
    m = np.zeros(T, dtype=bool)
    starts = rng.choice(np.arange(lo, hi - run_length + 1), size=n_runs, replace=False)
    for s in starts:
        m[s:s + run_length] = True
    return m


# ---- non-learned baselines for masked-value reconstruction ------------------------------
def interp_baseline(x, mask):
    """Linear interpolation in time from visible steps, per channel. x:(T,C) mask:(T,)"""
    t = np.arange(len(x)); vis = ~mask
    return np.stack([np.interp(t, t[vis], x[vis, c]) for c in range(x.shape[1])], axis=1)


def climatology_baseline(train_x, train_dates, query_dates, period=16):
    """Mean over training years of each channel by 16-day composite-of-year."""
    def slot(d): return (np.asarray(d).astype(int) - 1) // period
    s_tr, s_q = slot(train_dates), slot(query_dates)
    out = np.zeros((len(query_dates), train_x.shape[1]), dtype=np.float32)
    for s in np.unique(s_q):
        src = train_x[s_tr == s]
        out[s_q == s] = src.mean(axis=0) if len(src) else train_x.mean(axis=0)
    return out


# ---- spatial cross-validation ------------------------------------------------------------
def spatial_blocks(rows, cols, block=192):
    """Group id per sample = coarse spatial block. block > patch size so that overlapping
    (50 % stride) patches never straddle train/test."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    r, c = rows // block, cols // block
    _, gid = np.unique(r * 10_000 + c, return_inverse=True)
    return gid


def bootstrap_ci(y_true, y_pred, metric, n=1000, seed=0):
    rng = np.random.default_rng(seed); N = len(y_true); vals = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        vals.append(metric(y_true[i], y_pred[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---- hand-crafted phenology features (the baseline a learned embedding must beat) --------
def handcrafted_features(sar, optical, climate, dates):
    """(N,T,.) arrays -> (N,F). NDVI summary + annual/semi-annual harmonic fit + SAR/rain stats."""
    ndvi = optical[..., NDVI_IDX]
    w = 2 * np.pi * (dates / 365.0)
    X = np.stack([np.ones_like(w), np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)], axis=-1)  # (N,T,5)
    coef = np.stack([np.linalg.lstsq(X[i], ndvi[i], rcond=None)[0] for i in range(len(ndvi))])
    f = [ndvi.mean(1), ndvi.std(1), ndvi.min(1), ndvi.max(1), ndvi.max(1) - ndvi.min(1), coef[:, 1:],
         sar.mean(1), sar.std(1), climate.mean(1), optical.mean(1), optical.std(1)]
    return np.column_stack([a if a.ndim > 1 else a[:, None] for a in f]).astype(np.float32)
