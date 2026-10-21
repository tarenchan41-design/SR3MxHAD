"""Prepare the non-desert experiment from immutable published capture indices."""
import argparse
from collections import Counter
import gc
import importlib.util
import math
from pathlib import Path
import re

import numpy as np
from sr3m.data import CaptureRoutes, INDEX_DIR, split_hashes
from sr3m.geometry import adjacent_action
from sr3m.sim import EXECUTOR
from had.tokenizer import Tokenizer
from experiments.protocol import PROTOCOL, EXPECTED, action_statistics, code_version, machine_paths, read_json, stop_suffix, terrain_following, write_json


def load_split(root, split):
    rows = [__import__("json").loads(line) for line in (root / INDEX_DIR / f"{split}.jsonl").read_text().splitlines()]
    selected = {r["episode_id"] for r in rows if r["category"] != "desert"}
    if len(selected) != EXPECTED[split]:
        raise ValueError(f"Unexpected filtered {split} route count: {len(selected)}")
    return CaptureRoutes(root, split, selected)


def derive_terrain(datasets, terrain_dir):
    spec = importlib.util.spec_from_file_location("sr3m_ground", terrain_dir / "scripts/terrain_model.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    labels, bindings = {}, {}
    for mapid in ("L_MTN_A", "L_MTN_B", "L_MTN_C", "L_MTN_D"):
        persisted = read_json(terrain_dir / "evidence" / f"persisted-{mapid}.json")
        cert = read_json(terrain_dir / "evidence" / f"continuous-agl-{mapid}.json")
        design = read_json(terrain_dir / "design" / f"{mapid}.final.json")
        if cert["status"] != "pass" or cert["design_sha256"] != persisted["design_sha256"]:
            raise ValueError(f"Unbound/failed terrain certificate: {mapid}")
        designs = {r["route_id"]: r for r in design["routes"]}
        certificates = {r["route_id"]: r for r in cert["routes"]}
        terrain = module.Terrain(terrain_dir, mapid)
        if terrain.source_sha256 != cert["terrain"]["source_sha256"]:
            raise ValueError(f"Terrain export changed: {mapid}")
        bindings[persisted["map_package"]] = {"map_sha256": persisted["sha256_after"],
            "terrain_source_sha256": terrain.source_sha256, "design_sha256": cert["design_sha256"],
            "ground": "highest supplied nonvertical terrain triangle, excluding roofs"}
        for ds in datasets.values():
            for route in ds.routes:
                record = route["record"]
                if route["map_package"] != persisted["map_package"]:
                    continue
                if route["scene_fingerprint"].get(route["map_package"]) != persisted["sha256_after"]:
                    raise ValueError(f"Capture/map version mismatch: {record['episode_id']}")
                planned = designs[record["route_id"]]
                if planned["instruction_zh"] != record["instruction_zh"] or certificates[record["route_id"]]["status"] != "pass":
                    raise ValueError("Route/certificate identity mismatch")
                p = route["positions"]
                ground = terrain.sample(p[:, :2], return_details=True)
                agl = p[:, 2] - ground["z_m"]
                slope = np.degrees(np.arctan(np.linalg.norm(terrain.planes[ground["triangle_ids"], :2], axis=1)))
                layer = planned["height_layer"]
                if not np.isfinite(agl).all() or (agl < 0).any():
                    raise ValueError("Invalid AGL")
                certified_lo,certified_hi=certificates[record["route_id"]]["certified_agl_range_m"]
                if agl.min()<certified_lo-.02 or agl.max()>certified_hi+.02:
                    raise ValueError("Captured AGL is outside the source route's continuous certificate")
                limits = {"low": (0., 30.), "medium": (30., 80.), "high": (80., math.inf)}[layer]
                if agl.min() < limits[0] - .02 or agl.max() > limits[1] + .02:
                    raise ValueError("Captured AGL disagrees with certified height layer")
                labels[record["episode_id"]] = (layer, agl.astype("float32"), slope.astype("float32"))
        print(f"terrain {mapid}: bound and queried", flush=True)
        del terrain; gc.collect()
    if len(labels) != 201:
        raise ValueError(f"Expected all 201 mountain routes, got {len(labels)}")
    return labels, bindings


def compact(ds, terrain):
    routes, collected = [], {k: [] for k in ("state", "action", "position", "yaw", "stop", "endpoint", "action_valid", "distance_goal", "agl", "slope")}
    offset = 0
    for r in ds.routes:
        record = r["record"]; p, yaw = r["positions"], r["yaw"]; n = len(p)
        if re.search(r"[\u3400-\u9fff]", record["instruction"]) or not record["instruction"].strip():
            raise ValueError("Main experiment requires existing English instruction")
        stop, distance = stop_suffix(p, PROTOCOL["stop_radius_m"])
        endpoint = np.arange(n) == n - 1
        actions = np.full((n, 4), np.nan)
        actions[:-1] = [adjacent_action(p[t], yaw[t], p[t+1], yaw[t+1]) for t in range(n-1)]
        c, s = np.cos(yaw[0]), np.sin(yaw[0]); delta = p - p[0]
        state = np.column_stack(((c*delta[:, 0]+s*delta[:, 1])/100,
                                 (-s*delta[:, 0]+c*delta[:, 1])/100, delta[:, 2]/100,
                                 np.sin(yaw-yaw[0]), np.cos(yaw-yaw[0])))
        layer, agl, slope = terrain.get(record["episode_id"], (None, np.full(n, np.nan), np.full(n, np.nan)))
        routes.append({"episode_id": record["episode_id"], "route_id": record["route_id"],
                       "route_group_id": record["route_group_id"], "map_package": r["map_package"],
                       "category": record["category"], "instruction": record["instruction"],
                       "instruction_zh": record["instruction_zh"], "height_layer": layer,
                       "terrain_following": terrain_following(record) if record["category"] == "mountain" else False,
                       "directory": r["directory"].relative_to(ds.root).as_posix(),
                       "images": {v: [img[v] for img in r["images"]] for v in ("front", "down")},
                       "offset": offset, "count": n,
                       "length_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                       "start_goal_m": float(distance[0]), "total_turn_rad": float(np.abs(actions[:-1, 3]).sum())})
        values = (state, actions, p, yaw, stop, endpoint, ~endpoint & ~stop, distance, agl, slope)
        for key, value in zip(collected, values):
            collected[key].append(value)
        offset += n
    arrays = {k: np.concatenate(v).astype(bool if k in ("stop", "endpoint", "action_valid") else np.float32)
              for k, v in collected.items()}
    return routes, arrays


def choose_closed_loop(routes):
    selected = []
    for layer in ("low", "medium"):
        for follow in (False, True):
            candidates = sorted((r for r in routes if r["category"] == "mountain" and r["height_layer"] == layer
                                 and r["terrain_following"] == follow), key=lambda r: r["episode_id"])
            if len(candidates) < 3:
                raise ValueError("Insufficient mountain closed-loop cohort")
            selected.extend(candidates[:3])
    selected.extend(sorted((r for r in routes if r["category"] == "mountain" and r["height_layer"] == "high"
                            and r["terrain_following"]), key=lambda r: r["episode_id"])[:2])
    other = [r for r in routes if r["category"] != "mountain"]
    straight = sorted((r for r in other if r["start_goal_m"] > 20), key=lambda r: (r["total_turn_rad"], r["episode_id"]))
    returns = sorted((r for r in other if r["start_goal_m"] <= 20 and r["length_m"] >= 80), key=lambda r: r["episode_id"])
    if not straight or not returns:
        raise ValueError("Missing non-mountain straight/return route")
    selected.extend([straight[0], returns[0]])
    if len(selected) != 16 or len({r["episode_id"] for r in selected}) != 16:
        raise ValueError("Closed-loop quota not met")
    keys = ("episode_id", "map_package", "category", "terrain_following", "height_layer", "length_m", "start_goal_m")
    return {"split": "val_seen", "selection": "geometry_only_before_training", "routes": [{k: r[k] for k in keys} for r in selected],
            "expert_gate": {"completed": 16, "collisions": 0, "timeouts": 0, **EXECUTOR}}


def prepare(args):
    out = Path(args.out).resolve(); root = Path(args.root).resolve()
    if out.is_relative_to(root) or out.exists():
        raise ValueError("Choose a new preparation directory outside raw data")
    datasets = {}
    for split in ("train", "val_seen", "val_unseen"):
        datasets[split] = load_split(root, split)
        print(f"loaded {split}: {len(datasets[split].routes)} routes", flush=True)
    labels, bindings = derive_terrain(datasets, Path(args.terrain))
    out.mkdir(parents=True)
    all_routes, arrays, diagnostics = {}, {}, {}
    for split, ds in datasets.items():
        all_routes[split], arrays[split] = compact(ds, labels)
        write_json(out / f"{split}.json", all_routes[split])
        np.savez(out / f"{split}.npz", **arrays[split])
        a = arrays[split]
        counts = Counter(f"{'follow' if r['terrain_following'] else 'control'}/{r['height_layer']}"
                         for r in all_routes[split] if r["category"] == "mountain")
        diagnostics[split] = {"routes": len(ds.routes), "observations": len(a["state"]),
            "action_valid": int(a["action_valid"].sum()), "stop_positive": int(a["stop"].sum()),
            "mountain_cohorts": dict(counts), "height_missing_nonmountain": int(np.isnan(a["agl"]).sum())}
    stats = action_statistics(arrays["train"]["action"], arrays["train"]["action_valid"])
    slope = arrays["train"]["slope"]; stats["slope_bins_deg"] = np.quantile(slope[np.isfinite(slope)], [1/3, 2/3]).tolist()
    tokenizer = Tokenizer.fit([r["instruction"] for r in all_routes["train"]], max_length=256)
    tokenizer.save(out / "tokenizer.json")
    write_json(out / "statistics.json", stats)
    write_json(out / "closed_loop_routes.json", choose_closed_loop(all_routes["val_seen"]))
    info = {"protocol": PROTOCOL, "source_split": split_hashes(root / INDEX_DIR),
            "terrain_bindings": bindings, "code": code_version(), "routes": {s: EXPECTED[s] for s in datasets},
            "prepared_version": 1}
    write_json(out / "protocol.json", info)
    diagnostics["tokenizer"] = {s: tokenizer.describe([r["instruction"] for r in rr]) for s, rr in all_routes.items()}
    diagnostics["vocab_size"] = tokenizer.vocab_size
    write_json(out / "preparation.json", diagnostics)
    from experiments.protocol import PreparedSplit
    from experiments.metrics import evaluate
    seen=PreparedSplit(out,"val_seen")
    baselines={name: evaluate(seen,np.broadcast_to(action,(seen.n,4)),stats)
               for name,action in (("fixed_forward",[2.,0.,0.,0.]),("train_mean",stats["mean"]))}
    write_json(out/"baselines_val_seen.json",baselines)
    print(__import__("json").dumps(diagnostics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    paths = machine_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=paths["data_root"])
    parser.add_argument("--terrain", type=Path, default=paths["terrain_dir"])
    parser.add_argument("--out", required=True)
    prepare(parser.parse_args())

