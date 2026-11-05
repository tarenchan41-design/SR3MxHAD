"""Dataset v3: rendered routes -> the compact prepared layout train.py and cache.py read (protocol sections 14 and 17).

SR3MXHAD_PROTOCOL=v3_pilot (the pilot, section 14) or v3 (stage 2, section 17). Splits: train and val_seen (the 1-in-10 val routes
of the training maps) for fitting and selection; dev only for the closed-loop gates; stage 2 adds test (the C maps), prepared with
everything else but read by nothing until the configuration is frozen (section 17.9). The vocabulary and action statistics come
from train only. Labels are the pursuit actions stored with each route; where no offset was injected they must equal the pose
difference, which is checked. Stage 2 records also carry the target class, the unseen-class flag and the name form (full / short).
"""
import argparse
from pathlib import Path

import numpy as np

from experiments.protocol import PROTOCOL, action_statistics, code_version, read_json, stop_suffix, terrain_following, write_json
from experiments.v3_leak import Ground
from had.tokenizer import Tokenizer
from sr3m.geometry import adjacent_action, wrap

SPLITS = {"train": "train", "val": "val_seen", "dev": "dev", "test": "test"}
CATEGORY = {"L_MTN": "mountain", "L_MTR": "mountain", "L_DES": "desert"}
SOURCES = {"sr3m_had_v3_pilot": "/mnt/f/WSL_ColdStorage/datasets/sr3m_v3_pilot", "sr3m_had_v3": "/mnt/f/WSL_ColdStorage/datasets/sr3m_v3"}
MAX_TOKENS = 64


def slope_deg(ground, xy, step=2.):
    gx = (ground(xy + [step, 0]) - ground(xy - [step, 0])) / (2 * step)
    gy = (ground(xy + [0, step]) - ground(xy - [0, step])) / (2 * step)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def compact(route, ground, offset):
    P, Y, A = np.array(route["positions"], dtype=float), np.array(route["yaws"], dtype=float), np.array(route["actions"], dtype=float)
    n = len(P)
    if len(A) != n - 1 or route["rendered_steps"] != list(range(n)):
        raise ValueError(f"{route['id']}: poses, actions and rendered steps disagree")
    kicked = {k["step"] for k in route["kicks"]}
    for t in range(n - 1):
        if t + 1 not in kicked:
            moved = adjacent_action(P[t], Y[t], P[t + 1], Y[t + 1])
            if np.abs(np.subtract(moved[:3], A[t, :3])).max() > 1e-4 or abs(wrap(moved[3] - A[t, 3])) > 1e-6:
                raise ValueError(f"{route['id']}: label differs from the pose difference at step {t} without a kick")
    stop, distance = stop_suffix(P, PROTOCOL["stop_radius_m"])
    endpoint = np.arange(n) == n - 1
    actions = np.full((n, 4), np.nan)
    actions[:-1] = A
    c, s = np.cos(Y[0]), np.sin(Y[0])
    delta = P - P[0]
    state = np.column_stack(((c * delta[:, 0] + s * delta[:, 1]) / 100, (-s * delta[:, 0] + c * delta[:, 1]) / 100, delta[:, 2] / 100, np.sin(Y - Y[0]), np.cos(Y - Y[0])))
    agl = P[:, 2] - ground(P[:, :2])
    slope = slope_deg(ground, P[:, :2])
    map_id = route["map"]
    record = {"episode_id": f"{PROTOCOL['index']}/{map_id}/{route['id']}", "route_id": route["id"], "route_group_id": f"{map_id}/{route['object']}", "map_package": route["map_package"],
              "category": CATEGORY[map_id[:5]], "instruction": route["instruction"], "instruction_zh": route["instruction_zh"], "height_layer": "low" if route["agl"] <= 30 else "medium",
              "terrain_following": terrain_following(route), "directory": f"routes/{map_id}/{route['id']}", "images": route["images"], "offset": offset, "count": n,
              "length_m": float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), "start_goal_m": float(distance[0]), "total_turn_rad": float(np.abs(A[:, 3]).sum()),
              "target": route["target"], "top_z": route["top_z"], "object": route["object"], "cls": route["cls"], "word": route["word"], "mode": route["mode"], "kick_steps": sorted(kicked)}
    if "name_form" in route:  # stage 2
        record.update(unseen_class=route["unseen_class"], name_form=route["name_form"], split=route["split"])
    if record["terrain_following"] != (route["mode"] == "follow"):
        raise ValueError(f"{route['id']}: instruction cohort and route mode disagree")
    arrays = {"state": state, "action": actions, "position": P, "yaw": Y, "stop": stop, "endpoint": endpoint, "action_valid": ~endpoint & ~stop, "distance_goal": distance, "agl": agl, "slope": slope}
    return record, arrays


def prepare(args):
    if PROTOCOL["name"] not in SOURCES:
        raise ValueError("Run with SR3MXHAD_PROTOCOL=v3_pilot or v3")
    root, out = Path(args.root or SOURCES[PROTOCOL["name"]]), Path(args.out)
    if out.exists():
        raise ValueError(f"{out} exists; choose a new preparation directory")
    plan = read_json(root / "plan.json")
    routes = {s: [] for s in SPLITS.values()}
    for map_id in plan["maps"]:
        folder = root / "routes" / map_id
        log = read_json(folder / "checks.json") if (folder / "checks.json").exists() else {}
        accepted = [k for k, v in log.items() if v == "accepted"]
        if len(accepted) != plan["maps"][map_id]["quota"] and not args.partial:
            raise ValueError(f"{map_id}: {len(accepted)} rendered, quota {plan['maps'][map_id]['quota']}")
        for route_id in accepted:
            route = read_json(folder / route_id / "route.json")
            routes[SPLITS[route["split"]]].append(route)
    grounds = {m: Ground(m, args.cache) for m in plan["maps"]}
    out.mkdir(parents=True)
    counts, diagnostics = {}, {}
    routes = {s: items for s, items in routes.items() if items}  # the pilot has no test maps; a partial dry run may lack more
    for split, items in routes.items():
        records, collected, offset = [], {}, 0
        for route in items:
            record, arrays = compact(route, grounds[route["map"]], offset)
            records.append(record)
            for k, v in arrays.items():
                collected.setdefault(k, []).append(v)
            offset += record["count"]
        arrays = {k: np.concatenate(v).astype(bool if k in ("stop", "endpoint", "action_valid") else np.float32) for k, v in collected.items()}
        write_json(out / f"{split}.json", records)
        np.savez(out / f"{split}.npz", **arrays)
        counts[split] = len(records)
        diagnostics[split] = {"routes": len(records), "observations": int(len(arrays["state"])), "stop_positive": int(arrays["stop"].sum()), "action_valid": int(arrays["action_valid"].sum()),
                              "follow": sum(r["terrain_following"] for r in records), "by_map": {m: sum(r["route_id"].startswith(m) for r in records) for m in plan["maps"]}}
    train = np.load(out / "train.npz")
    stats = action_statistics(train["action"], train["action_valid"])
    slope = train["slope"]
    stats["slope_bins_deg"] = np.quantile(slope[np.isfinite(slope)], [1 / 3, 2 / 3]).tolist()
    tokenizer = Tokenizer.fit([r["instruction"] for r in routes["train"]], max_length=MAX_TOKENS)
    if any(tokenizer.encode(r["instruction"])["truncated_count"] for split in routes.values() for r in split):
        raise ValueError("An instruction is longer than the tokenizer keeps")
    tokenizer.save(out / "tokenizer.json")
    write_json(out / "statistics.json", stats)
    dev = read_json(out / "dev.json") if "dev" in routes else []
    write_json(out / "closed_loop_routes.json", {"split": "dev", "selection": "all dev routes, fixed by the plan before training",
                                                 "routes": [{k: r[k] for k in ("episode_id", "route_id", "map_package", "category", "terrain_following", "height_layer", "length_m", "start_goal_m", "cls", "word")} for r in dev]})
    if "test" in routes:
        test = read_json(out / "test.json")
        write_json(out / "test_routes.json", {"split": "test", "selection": "all test routes, fixed by the plan; not read until the configuration is frozen (section 17.9)",
                                              "routes": [{k: r[k] for k in ("episode_id", "route_id", "map_package", "category", "terrain_following", "height_layer", "cls", "word", "unseen_class", "name_form")} for r in test]})
    write_json(out / "protocol.json", {"protocol": PROTOCOL, "routes": counts, "source": {"root": str(root), "plan_seed": plan["seed"], "design": plan["design"]}, "code": code_version(), "prepared_version": 1,
                                       **({"partial": True} if args.partial else {})})
    diagnostics["tokenizer"] = {s: tokenizer.describe([r["instruction"] for r in rr]) for s, rr in routes.items()}
    diagnostics["vocab_size"] = tokenizer.vocab_size
    write_json(out / "preparation.json", diagnostics)
    print({s: diagnostics[s] for s in counts}, "vocab", tokenizer.vocab_size, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=None, help="default: the dataset of the chosen PROTOCOL"); p.add_argument("--out", required=True)
    p.add_argument("--partial", action="store_true", help="dry run on a render still in progress; marked partial in protocol.json, never for training")
    p.add_argument("--cache", default=str(Path.home() / "sr3mxhad_runs/v3/ground"))
    prepare(p.parse_args())
