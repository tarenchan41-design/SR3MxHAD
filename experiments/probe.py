"""Route-balanced linear probes on frozen down-view CNN features, not projections."""
import argparse
from pathlib import Path
import numpy as np
from experiments.protocol import PROTOCOL, TERRAIN, PreparedSplit, read_features, write_json

ALPHAS = [1e-4, 1e-3, 1e-2, .1, 1., 10., 100.]


def observations(ds):
    ids, weights, layers = [], [], []
    routes = [r for r in ds.routes if r["category"] in TERRAIN]
    for r in routes:
        ids.extend(range(r["offset"], r["offset"]+r["count"]))
        weights.extend([1/(len(routes)*r["count"])]*r["count"])
        layers.extend([r["height_layer"]]*r["count"])
    ids = np.asarray(ids)
    targets = np.column_stack((ds.arrays["agl"][ids], ds.arrays["slope"][ids])).astype(float)
    if not np.isfinite(targets).all(): raise ValueError("Missing terrain probe targets")
    return ids, np.asarray(weights), np.asarray(layers), targets


def scores(target, prediction, baseline, weight):
    weight = weight/weight.sum()
    center = np.sum(weight*target)
    denom = np.sum(weight*(target-center)**2)
    return {"mae": float(np.sum(weight*np.abs(target-prediction))),
            "rmse": float(np.sqrt(np.sum(weight*(target-prediction)**2))),
            "r2": float(1-np.sum(weight*(target-prediction)**2)/denom) if denom > 1e-8 else None,
            "train_mean_mae": float(np.sum(weight*np.abs(target-baseline)))}


def run(args):
    datasets = {s: PreparedSplit(args.prepared, s) for s in ("train", "val_seen", "val_unseen")}
    features = {s: read_features(args.cache, s, ds)[0]["down"] for s, ds in datasets.items()}
    rows = {s: observations(ds) for s, ds in datasets.items()}
    ids, weight, _, target = rows["train"]
    dim = features["train"].shape[1]
    xmean = np.zeros(dim); xsecond = np.zeros(dim)
    for start in range(0, len(ids), 1024):
        sl = slice(start, start+1024); x = np.asarray(features["train"][ids[sl]], dtype=float); w = weight[sl, None]
        xmean += (w*x).sum(0); xsecond += (w*x*x).sum(0)
    xscale = np.sqrt(np.maximum(xsecond-xmean*xmean, 1e-8))
    ymean = (target*weight[:, None]).sum(0)
    gram = np.zeros((dim, dim)); rhs = np.zeros((dim, 2))
    for start in range(0, len(ids), 1024):
        sl = slice(start, start+1024)
        x = (np.asarray(features["train"][ids[sl]], dtype=float)-xmean)/xscale
        rootw = np.sqrt(weight[sl, None]); x *= rootw
        gram += x.T@x; rhs += x.T@((target[sl]-ymean)*rootw)
    eigen, vectors = np.linalg.eigh(gram); eigen = np.maximum(eigen, 0)
    transformed = vectors.T@rhs
    val_ids, val_weight, _, val_target = rows["val_seen"]
    xv = (np.asarray(features["val_seen"][val_ids], dtype=float)-xmean)/xscale
    candidates = []
    for alpha in ALPHAS:
        beta = vectors@(transformed/(eigen[:, None]+alpha))
        error = (np.abs(val_target-(xv@beta+ymean))*val_weight[:, None]).sum(0)
        candidates.append((alpha, beta, error))
    selected = [min(candidates, key=lambda c: (c[2][j], -c[0])) for j in range(2)]
    beta = np.column_stack([selected[j][1][:, j] for j in range(2)])
    report = {"prepared": datasets["train"].protocol, "input": f"frozen_down_{PROTOCOL['backbone']}_{PROTOCOL['feature_dim']}_only", "fit_split": "train", "selection_split": "val_seen",
              "weighting": "equal routes, then equal observations within route", "alpha_grid": ALPHAS,
              "selected_alpha": {name: selected[j][0] for j, name in enumerate(("agl_m", "slope_deg"))},
              "interpretation": "Failure indicates poor linear readability, not absence of information", "splits": {}}
    for split, (index, w, layers, truth) in rows.items():
        prediction = np.empty_like(truth)
        for start in range(0, len(index), 1024):
            sl = slice(start, start+1024)
            prediction[sl] = ((np.asarray(features[split][index[sl]], dtype=float)-xmean)/xscale)@beta+ymean
        summary = {}
        for j, name in enumerate(("agl_m", "slope_deg")):
            summary[name] = scores(truth[:, j], prediction[:, j], ymean[j], w)
            summary[name]["by_agl"] = {layer: scores(truth[layers == layer, j], prediction[layers == layer, j], ymean[j], w[layers == layer])
                                       for layer in ("low", "medium", "high") if np.any(layers == layer)}
        report["splits"][split] = summary
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "linear_probe.npz", feature_mean=xmean, feature_scale=xscale, target_mean=ymean, beta=beta)
    write_json(out / "probe.json", report)
    print(report["splits"]["val_unseen"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True); p.add_argument("--cache", required=True); p.add_argument("--out", required=True)
    run(p.parse_args())

