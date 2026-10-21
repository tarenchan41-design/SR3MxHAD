"""Frozen-checkpoint reporting; val_unseen is opened only after all nine selections."""
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.protocol import PROTOCOL, PreparedSplit, read_features, read_json, write_json
from had.model import TerrainHAD
from had.tokenizer import Tokenizer
from experiments.metrics import evaluate, follow_slope_edges
from experiments.train import predict, token_rows


def evaluate_runs(args):
    torch.set_num_threads(4)
    run_paths = [Path(args.runs) / f"{v}_seed{s}" for s in PROTOCOL["seeds"] for v in PROTOCOL["variants"]]
    for run in run_paths:
        if not (run / "completion.json").exists() or read_json(run / "selection.json")["split"] != "val_seen":
            raise ValueError("All nine runs must be completed and frozen on val_seen before reporting")
    ds = PreparedSplit(args.prepared, args.split)
    features, _ = read_features(args.cache, args.split, ds)
    statistics = read_json(Path(args.prepared) / "statistics.json")
    edges = follow_slope_edges(PreparedSplit(args.prepared, "train"))
    tokenizer = Tokenizer.load(Path(args.prepared) / "tokenizer.json"); tokens = token_rows(ds, tokenizer)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name, action in (("fixed_forward", [2., 0., 0., 0.]), ("train_mean", statistics["mean"])):
        report = evaluate(ds, np.broadcast_to(action, (ds.n, 4)), statistics, follow_slope_edges=edges)
        write_json(out / f"{name}.json", dict(report, policy=name))
    for run in run_paths:
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        identity = checkpoint["identity"]
        if identity["prepared"] != ds.protocol or identity["statistics"] != statistics or identity["tokenizer"] != tokenizer.to_dict():
            raise ValueError("Checkpoint data/vocabulary identity mismatch")
        selection = read_json(run / "selection.json")
        model = TerrainHAD(identity["variant"], tokenizer, cached_only=True).to(args.device)
        model.load_head_state(checkpoint["head"])
        pred, probability = predict(model, ds, features, tokens, args.device)
        report = evaluate(ds, pred, statistics, probability, selection["stop"]["threshold"], follow_slope_edges=edges)
        report.update(variant=identity["variant"], seed=identity["seed"], epoch=checkpoint["epoch"], selection=selection)
        write_json(out / f"{run.name}.json", report)
        np.savez(out / f"{run.name}_predictions.npz", action=pred, stop_probability=probability)
        print(f"reported {run.name} on {args.split}", flush=True)
        del model, checkpoint
        if args.device.startswith("cuda"): torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True); p.add_argument("--cache", required=True); p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True); p.add_argument("--split", choices=("val_seen", "val_unseen"), default="val_unseen")
    p.add_argument("--device", default="cuda"); evaluate_runs(p.parse_args())
