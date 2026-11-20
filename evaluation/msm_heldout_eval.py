"""Held-out masked-sensor reconstruction: does MSM pre-training beat trivial baselines?

Design
  * Series for each position are joined across the train/val/test splits (same positions, disjoint
    time). Masks are placed ONLY in the test window, so every target value was never seen in training.
  * Baselines: mean-fill (0 in z-space), linear interpolation, train-period climatology, and a
    randomly initialised copy of the same architecture (isolates what pre-training adds).
  * Cross-modal test: optical reconstruction with SAR visible vs SAR hidden. If cross-attention
    does what the README claims, hiding SAR should hurt optical reconstruction.
  * Metric: RMSE in z-score units on masked entries, per sensor, mean + bootstrap 95 % CI over
    positions x repeats. Seeds are fixed, so results are repeatable.

Usage
  python evaluation/msm_heldout_eval.py --train train_patches.pkl --val val_patches.pkl \
      --test test_patches.pkl --ckpt checkpoints/msm_best.pt --n_bands 14 --d_model 384
"""
import argparse, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.common import (load_patches, concat_by_position, make_mask, interp_baseline,
                               climatology_baseline, bootstrap_ci)

SENSORS = {"sar": 2, "optical": None, "climate": 4}


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train"); ap.add_argument("--val"); ap.add_argument("--test"); ap.add_argument("--ckpt")
    ap.add_argument("--n_bands", type=int, default=14); ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--mask_ratio", type=float, default=0.2)
    ap.add_argument("--run_length", type=int, default=1, help="consecutive masked steps (3 ~ 48-day gap)")
    ap.add_argument("--repeats", type=int, default=10); ap.add_argument("--out", default="evaluation/results")
    ap.add_argument("--skip_model", action="store_true", help="baselines only (no torch needed)")
    a = ap.parse_args()

    splits = [load_patches(a.train), load_patches(a.val), load_patches(a.test)]
    full, starts = concat_by_position(splits)
    if not full: sys.exit("No positions found in all three splits; cannot build held-out windows.")
    T = len(next(iter(full.values()))["dates"]); window = (starts[2], T)
    n_opt = a.n_bands
    chan = {"sar": slice(0, 2), "optical": slice(2, 2 + n_opt), "climate": slice(2 + n_opt, 2 + n_opt + 4)}
    print(f"{len(full)} positions, T={T}, held-out window={window}")

    model = msm = rand_model = rand_msm = None
    if not a.skip_model:
        import torch
        from models.vegformer import build_vegformer, MaskedSensorModeling
        cfg = {"n_bands": n_opt, "d_model": a.d_model, "max_seq_len": T}
        def build():
            m = build_vegformer(cfg); return m, MaskedSensorModeling(m, a.mask_ratio)
        model, msm = build()
        ck = torch.load(a.ckpt, map_location="cpu")
        model.load_state_dict(ck["model_state"])
        if "msm_state" in ck: msm.load_state_dict(ck["msm_state"])
        else: print("WARNING: no msm_state in checkpoint; mask tokens are untrained -> numbers are pessimistic")
        torch.manual_seed(0); rand_model, rand_msm = build()
        for m_ in (model, rand_model): m_.eval()

    def run_model(mdl, mt, series, masks, hide_sar=False):
        import torch
        b = {k: torch.tensor(series[k])[None] for k in ("sar", "optical", "climate", "dates")}
        out = {}
        for name, tok in (("sar", mt.sar_mask), ("optical", mt.optical_mask), ("climate", mt.climate_mask)):
            x = b[name].clone(); m = torch.tensor(masks[name])
            if hide_sar and name == "sar": m = torch.ones_like(m)
            x[0, m] = tok.detach().expand(1, x.shape[1], x.shape[2])[0, m]; b[name] = x
        with torch.no_grad():
            rec = mdl.msm_decoder(mdl(b)["token_feats"])[0].numpy()
        return rec

    methods = ["mean_fill", "interp", "climatology"] + ([] if a.skip_model else ["random_init", "pretrained"])
    res = {(s, m): [] for s in SENSORS for m in methods}
    xmod = {"optical_sar_visible": [], "optical_sar_hidden": []}
    train_end = starts[1]
    keys = sorted(full)
    for rep in range(a.repeats):
        for ki, key in enumerate(keys):
            ser = full[key]; rng = np.random.default_rng(1000 * rep + ki)
            masks = {s: make_mask(T, window, a.mask_ratio, rng, a.run_length) for s in SENSORS}
            target = np.concatenate([ser["sar"], ser["optical"], ser["climate"]], axis=1)
            preds = {}
            preds["mean_fill"] = np.zeros_like(target)
            interp = np.zeros_like(target); clim = np.zeros_like(target)
            for s, sl in chan.items():
                interp[:, sl] = interp_baseline(target[:, sl], masks[s])
                clim[:, sl] = climatology_baseline(target[:train_end, sl], ser["dates"][:train_end], ser["dates"])
            preds["interp"], preds["climatology"] = interp, clim
            if not a.skip_model:
                preds["random_init"] = run_model(rand_model, rand_msm, ser, masks)
                preds["pretrained"] = run_model(model, msm, ser, masks)
                xmod["optical_sar_visible"].append(rmse(preds["pretrained"][masks["optical"]][:, chan["optical"]],
                                                        target[masks["optical"]][:, chan["optical"]]))
                rec_h = run_model(model, msm, ser, masks, hide_sar=True)
                xmod["optical_sar_hidden"].append(rmse(rec_h[masks["optical"]][:, chan["optical"]],
                                                       target[masks["optical"]][:, chan["optical"]]))
            for s, sl in chan.items():
                m = masks[s]
                for meth in methods:
                    res[(s, meth)].append(rmse(preds[meth][m][:, sl], target[m][:, sl]))

    rows = []
    print(f"\nHeld-out RMSE (z-score units), mean [95% CI], {a.repeats} repeats x {len(keys)} positions, "
          f"mask_ratio={a.mask_ratio}, run_length={a.run_length}")
    print(f"{'sensor':<9}" + "".join(f"{m:>26}" for m in methods))
    for s in SENSORS:
        line = f"{s:<9}"
        for meth in methods:
            v = np.array(res[(s, meth)]); lo, hi = bootstrap_ci(v, v, lambda a_, b_: a_.mean())
            line += f"{v.mean():>12.3f} [{lo:.3f},{hi:.3f}]"; rows.append(dict(sensor=s, method=meth, rmse=v.mean(), lo=lo, hi=hi))
        print(line)
    if not a.skip_model:
        v, h = np.mean(xmod["optical_sar_visible"]), np.mean(xmod["optical_sar_hidden"])
        print(f"\nCross-modal check (optical RMSE): SAR visible {v:.3f} vs SAR hidden {h:.3f}  (delta {h - v:+.3f})")
        rows.append(dict(sensor="optical", method="pretrained_sar_hidden", rmse=h, lo=None, hi=None))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(Path(a.out) / "msm_heldout.json", "w"), indent=1)


if __name__ == "__main__":
    main()
