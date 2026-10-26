"""Compare frozen runs on a development fold's held-out maps (experiments/fold.py `val_unseen`)."""
import argparse
from pathlib import Path

import numpy as np
import torch
from experiments.protocol import TERRAIN, PreparedSplit, read_features, read_json, write_json
from experiments.metrics import evaluate, follow_slope_edges
from experiments.train import predict, token_rows
from had.model import TerrainHAD
from had.tokenizer import Tokenizer


def route_beta(ds, prediction, follow):
    """Slope of predicted dz on next-step ground change over terrain routes of one cohort."""
    a = ds.arrays; num = den = 0.
    for r in ds.routes:
        if r["category"] not in TERRAIN or r["terrain_following"] != follow: continue
        s = slice(r["offset"], r["offset"] + r["count"]); valid = a["action_valid"][s]
        ground = a["position"][s][:, 2] - a["agl"][s]; g = np.r_[np.diff(ground), 0.][valid]
        num += float((prediction[s][valid, 2] * g).sum()); den += float((g * g).sum())
    return num / den if den else None


def run(args):
    torch.set_num_threads(4)
    ds = PreparedSplit(args.prepared, "val_unseen")
    if "fold" not in ds.protocol:
        raise ValueError("fold_report only reads a development fold, never the published val_unseen")
    features, _ = read_features(args.cache, "val_unseen", ds)
    statistics = read_json(Path(args.prepared) / "statistics.json"); edges = follow_slope_edges(PreparedSplit(args.prepared, "train"))
    tokenizer = Tokenizer.load(Path(args.prepared) / "tokenizer.json"); tokens = token_rows(ds, tokenizer)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    zero = evaluate(ds, np.broadcast_to([2., 0., 0., 0.], (ds.n, 4)), statistics, follow_slope_edges=edges)
    write_json(out / "fixed_forward.json", zero); rows = []
    for run_dir in map(Path, args.runs):
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True); identity = checkpoint["identity"]
        if identity["prepared"] != ds.protocol or identity["tokenizer"] != tokenizer.to_dict() or identity["statistics"] != statistics:
            raise ValueError(f"{run_dir} was not trained on this fold")
        selection = read_json(run_dir / "selection.json")
        model = TerrainHAD(identity["variant"], tokenizer, cached_only=True).to(args.device); model.load_head_state(checkpoint["head"])
        pred, probability = predict(model, ds, features, tokens, args.device)
        report = evaluate(ds, pred, statistics, probability, selection["stop"]["threshold"], follow_slope_edges=edges)
        report.update(variant=identity["variant"], seed=identity["seed"], epoch=checkpoint["epoch"],
                      beta={"follow": route_beta(ds, pred, True), "control": route_beta(ds, pred, False)})
        write_json(out / f"{run_dir.name}.json", report); rows.append((run_dir.name, report))
    print(f"held-out maps {ds.protocol['fold']['heldout_maps']}: {len(ds.routes)} routes; dz=0 follow dz "
          f"{100*zero['terrain']['follow/all']['dz_mae_m']:.2f}cm")
    for name, r in rows:
        t, c = r["terrain"], r["terrain_by_category"]
        cells = [f"follow dz {100*t['follow/all']['dz_mae_m']:.2f}", f"control dz {100*t['control/all']['dz_mae_m']:.2f}",
                 *(f"{k} {100*c[k]['follow/all']['dz_mae_m']:.2f}/{100*c[k]['control/all']['dz_mae_m']:.2f}" for k in TERRAIN),
                 f"beta {r['beta']['follow']:.2f}/{r['beta']['control']:.2f}", f"translation {100*r['overall']['translation_m']:.2f}",
                 f"turn yaw {r['overall']['turn_yaw_mae_deg']:.2f}", f"stop within20 {r['stop']['first_stop_within_20m']}/{r['stop']['routes']}"]
        print(f"{name} (epoch {r['epoch']}): " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True); p.add_argument("--cache", required=True); p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True); p.add_argument("--device", default="cuda")
    run(p.parse_args())
