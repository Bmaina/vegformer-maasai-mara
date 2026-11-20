"""Linear-probe evaluation of the VegFormer representation against an independent reference map.

Question answered: do the pre-trained embeddings carry land-cover information beyond (a) a randomly
initialised network of the same architecture and (b) simple hand-crafted phenology features?
Spatial block cross-validation prevents neighbouring / overlapping patches leaking across folds.

Reference labels: an int array on the same 377x313 grid as the patches (-1 = no data), e.g. ESA
WorldCover / Dynamic World remapped to a few coarse classes, or digitised Braun-Blanquet plots.
Each patch gets the modal label of its pixels (ignoring -1); patches with <50 % labelled pixels are dropped.

Usage
  python evaluation/linear_probe.py --patches train_patches.pkl val_patches.pkl test_patches.pkl \
      --labels data/reference_labels.npy --ckpt checkpoints/msm_best.pt
"""
import argparse, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.common import (load_patches, concat_by_position, spatial_blocks, bootstrap_ci,
                               handcrafted_features)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.dummy import DummyClassifier


def modal_label(lab, row, col, size=64):
    w = lab[row:row + size, col:col + size].ravel(); v = w[w >= 0]
    if len(w) == 0 or len(v) < 0.5 * len(w): return -1
    return int(np.bincount(v).argmax())


def embed(series, ckpt, n_bands, d_model, pretrained):
    import torch
    from models.vegformer import build_vegformer
    T = series["dates"].shape[1]
    torch.manual_seed(0)
    m = build_vegformer({"n_bands": n_bands, "d_model": d_model, "max_seq_len": T}); m.eval()
    if pretrained: m.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state"])
    with torch.no_grad():
        b = {k: torch.tensor(series[k]) for k in ("sar", "optical", "climate", "dates")}
        return m(b)["embeddings"].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", nargs=3, required=True, help="train val test pkl")
    ap.add_argument("--labels", required=True); ap.add_argument("--ckpt")
    ap.add_argument("--n_bands", type=int, default=14); ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--block", type=int, default=192); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min_class", type=int, default=5, help="drop classes with fewer patches")
    ap.add_argument("--skip_model", action="store_true"); ap.add_argument("--out", default="evaluation/results")
    a = ap.parse_args()

    full, _ = concat_by_position([load_patches(p) for p in a.patches])
    lab = np.load(a.labels)
    keys = sorted(full)
    y = np.array([modal_label(lab, r, c) for r, c in keys]); keep = y >= 0
    cls, cnt = np.unique(y[keep], return_counts=True)
    keep &= np.isin(y, cls[cnt >= a.min_class])
    keys = [k for k, k_ in zip(keys, keep) if k_]; y = y[keep]
    print(f"{len(keys)} labelled positions, classes={dict(zip(*np.unique(y, return_counts=True)))}")
    if len(np.unique(y)) < 2: sys.exit("Need >=2 classes with enough positions.")
    ser = {k: np.stack([full[key][k] for key in keys]) for k in ("sar", "optical", "climate")}
    ser["dates"] = np.stack([full[key]["dates"] for key in keys])

    feats = {"hand_crafted_phenology": handcrafted_features(ser["sar"], ser["optical"], ser["climate"], ser["dates"])}
    if not a.skip_model:
        feats["random_init_encoder"] = embed(ser, a.ckpt, a.n_bands, a.d_model, False)
        feats["pretrained_encoder"] = embed(ser, a.ckpt, a.n_bands, a.d_model, True)
    groups = spatial_blocks([k[0] for k in keys], [k[1] for k in keys], a.block)
    n_splits = min(a.folds, len(np.unique(groups)))
    print(f"{len(np.unique(groups))} spatial blocks -> {n_splits}-fold block CV")

    rows = []
    models = {"majority_class": DummyClassifier(strategy="most_frequent")}
    for name in feats: models[name] = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.1, max_iter=2000, class_weight="balanced"))
    for name, clf in models.items():
        X = feats.get(name, np.zeros((len(y), 1)))
        pred = np.full(len(y), -1)
        for tr, te in GroupKFold(n_splits).split(X, y, groups):
            if len(np.unique(y[tr])) < 2 and name != "majority_class": pred[te] = y[tr][0]; continue
            pred[te] = clone_fit(clf, X[tr], y[tr]).predict(X[te])
        f1 = f1_score(y, pred, average="macro"); ba = balanced_accuracy_score(y, pred)
        lo, hi = bootstrap_ci(y, pred, lambda t, p: f1_score(t, p, average="macro"))
        rows.append(dict(features=name, macro_f1=f1, f1_lo=lo, f1_hi=hi, balanced_acc=ba))
        print(f"{name:<26} macro-F1 {f1:.3f} [{lo:.3f},{hi:.3f}]  balanced-acc {ba:.3f}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(Path(a.out) / "linear_probe.json", "w"), indent=1)
    print("\nNote: n is the number of labelled positions; with ~80 patches the CIs will be wide. "
          "Report them as is.")


def clone_fit(clf, X, y):
    from sklearn.base import clone
    return clone(clf).fit(X, y)


if __name__ == "__main__":
    main()
