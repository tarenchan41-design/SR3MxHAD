"""Derive a leave-one-map-out development fold from an existing preparation and feature cache.

The held-out maps leave train/val_seen; their train + val_seen routes become the fold's
`val_unseen` (a development split, never the published val_unseen). Vocabulary, action
statistics, slope bins and depth standardization are refit on the fold's train rows.
CNN features are copied from the source cache by row, so no image is reread.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from had.tokenizer import Tokenizer
from experiments.protocol import PROTOCOL, TERRAIN, action_statistics, code_version, read_json, write_json

SOURCE_SPLITS = ("train", "val_seen")


def map_of(route):
    return route["map_package"].rsplit("/", 1)[-1]


def gather(parts):
    """parts: [(split, routes, arrays)] -> routes with new offsets, arrays, (split, source row) per row."""
    routes, arrays, rows, offset = [], {}, [], 0
    for split, rr, aa in parts:
        for r in rr:
            sl = slice(r["offset"], r["offset"] + r["count"])
            for k, v in aa.items():
                arrays.setdefault(k, []).append(v[sl])
            rows.extend((split, i) for i in range(sl.start, sl.stop))
            routes.append(dict(r, offset=offset)); offset += r["count"]
    return routes, {k: np.concatenate(v) for k, v in arrays.items()}, rows


def build(args):
    source, holdout = Path(args.prepared), set(args.holdout)
    out = Path(args.out_prepared)
    if out.exists() or Path(args.out_cache).exists():
        raise ValueError("Fold outputs already exist; use new directories")
    info = read_json(source / "protocol.json")
    data = {}
    for s in SOURCE_SPLITS:
        with np.load(source / f"{s}.npz", allow_pickle=False) as z:
            data[s] = (read_json(source / f"{s}.json"), {k: z[k] for k in z.files})
    maps = {map_of(r) for s in SOURCE_SPLITS for r in data[s][0]}
    if not holdout <= maps or not {h.split("_")[1] for h in holdout} >= {"MTN", "DES"}:
        raise ValueError(f"Hold out one mountain and one desert map from {sorted(maps)}")
    keep = lambda rr, held: [r for r in rr if (map_of(r) in holdout) == held]
    splits = {"train": [("train", keep(data["train"][0], False), data["train"][1])],
              "val_seen": [("val_seen", keep(data["val_seen"][0], False), data["val_seen"][1])],
              "val_unseen": [(s, keep(data[s][0], True), data[s][1]) for s in SOURCE_SPLITS]}
    built = {k: gather(v) for k, v in splits.items()}
    if sum(len(b[0]) for b in built.values()) != sum(len(data[s][0]) for s in SOURCE_SPLITS):
        raise ValueError("Fold does not partition the source routes")

    train_routes, train_arrays, _ = built["train"]
    stats = action_statistics(train_arrays["action"], train_arrays["action_valid"])
    slope = train_arrays["slope"]; stats["slope_bins_deg"] = np.quantile(slope[np.isfinite(slope)], [1/3, 2/3]).tolist()
    tokenizer = Tokenizer.fit([r["instruction"] for r in train_routes], max_length=256)
    fold = {"heldout_maps": sorted(holdout), "val_unseen": "train + val_seen routes of the held-out maps (development only)",
            "source_prepared": str(source.resolve()),
            "source_protocol_sha256": hashlib.sha256((source / "protocol.json").read_bytes()).hexdigest()}
    protocol = {"protocol": PROTOCOL, "source_split": info["source_split"], "terrain_bindings": info["terrain_bindings"],
                "code": code_version(), "routes": {k: len(b[0]) for k, b in built.items()}, "prepared_version": 1, "fold": fold}
    out.mkdir(parents=True)
    for k, (routes, arrays, _) in built.items():
        write_json(out / f"{k}.json", routes); np.savez(out / f"{k}.npz", **arrays)
    tokenizer.save(out / "tokenizer.json"); write_json(out / "statistics.json", stats); write_json(out / "protocol.json", protocol)
    write_json(out / "preparation.json", {k: {"routes": len(b[0]), "observations": len(b[1]["state"]),
        "maps": dict(Counter(map_of(r) for r in b[0])),
        "terrain_cohorts": dict(Counter(f"{r['category']}/{'follow' if r['terrain_following'] else 'control'}"
                                        for r in b[0] if r["category"] in TERRAIN))} for k, b in built.items()})

    cnn = {s: {v: np.load(Path(args.cache) / s / f"{v}.npy", mmap_mode="r") for v in ("front", "down")} for s in SOURCE_SPLITS}
    depth = {s: {v: np.load(Path(args.depth) / s / f"{v}_depth.npy", mmap_mode="r") for v in ("front", "down")} for s in SOURCE_SPLITS}
    for s in SOURCE_SPLITS:
        meta = read_json(Path(args.depth) / s / "depth.json")
        if meta["rows"] != len(data[s][1]["state"]) or any(meta[k] != PROTOCOL["depth"][k] for k in ("model", "input_size", "grid", "output")):
            raise ValueError(f"Depth features do not match {s} or the PROTOCOL depth spec")
        if read_json(Path(args.cache) / s / "cache.json")["completed_rows"] != len(data[s][1]["state"]):
            raise ValueError(f"Incomplete source cache: {s}")
    def take(arrays, rows, view):
        out = np.empty((len(rows), arrays[SOURCE_SPLITS[0]][view].shape[1]), np.float32)
        for s in SOURCE_SPLITS:
            pos = [j for j, (split, _) in enumerate(rows) if split == s]
            if pos: out[pos] = arrays[s][view][[i for split, i in rows if split == s]]
        return out
    train_rows = built["train"][2]
    norm = {}
    for v in ("front", "down"):
        x = take(depth, train_rows, v); norm[v] = (x.mean(0), x.std(0) + 1e-6)
    cache = Path(args.out_cache)
    for k, (_, arrays, rows) in built.items():
        d = cache / k; d.mkdir(parents=True)
        for v in ("front", "down"):
            np.save(d / f"{v}.npy", take(cnn, rows, v))
            np.save(d / f"{v}_depth.npy", ((take(depth, rows, v) - norm[v][0]) / norm[v][1]).astype(np.float32))
        write_json(d / "cache.json", {"prepared": protocol, "backbone": PROTOCOL["weights"], "image_size": PROTOCOL["image_size"],
            "feature_stage": "cnn_before_trainable_projection", "dtype": "float32", "rows": len(rows), "completed_rows": len(rows),
            "derived_from": str(Path(args.cache).resolve()),
            "depth": {"spec": PROTOCOL["depth"], "source": str(Path(args.depth).resolve()),
                      "train_mean": {v: norm[v][0].tolist() for v in norm}, "train_std": {v: norm[v][1].tolist() for v in norm}}})
    print(json.dumps(read_json(out / "preparation.json"), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True); p.add_argument("--cache", required=True); p.add_argument("--depth", required=True)
    p.add_argument("--holdout", nargs=2, required=True, metavar="MAP"); p.add_argument("--out-prepared", required=True)
    p.add_argument("--out-cache", required=True)
    build(p.parse_args())
