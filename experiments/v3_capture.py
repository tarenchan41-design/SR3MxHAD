"""Dataset v3 pilot (protocol section 14): fix the route plan, then check and render it with K.

plan:   generates every candidate route in a fixed order, before any rendering; no simulator.
render: per map, checks each candidate in UE (collision sweep, ground clearance, line of sight to the target) and renders
        the first ones that pass, until the map's quota is met. Resumes per route. --preview renders only the first and
        last pose of a few routes, into a separate folder, to look at before the bulk run.
Close the GUI editor first. Writes only under --out.
"""
import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from experiments.closed_loop_k import capture_settings
from experiments.protocol import code_version, read_json, write_json
from experiments.v3_leak import DESIGNS, Ground

SEED = 20261103
DESIGN = "kick_start2_relief"
QUOTA = {"L_MTN_A": 100, "L_MTN_D": 100, "L_DES_A": 100, "L_DES_D": 100, "L_MTN_B": 30, "L_DES_B": 30}
DEV_MAPS = ("L_MTN_B", "L_DES_B")
CANDIDATES = 1.6  # candidates per quota slot, so UE rejections cannot run a map dry
VAL_EVERY = 10
MIN_AGL_M = 2.
SIGHT_MARGIN_M = 3.


def plain(route):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in route.items()}


def split_of(map_id, accepted_rank):
    if map_id in DEV_MAPS:
        return "dev"
    return "val" if accepted_rank % VAL_EVERY == VAL_EVERY - 1 else "train"


def sight_end(start, target, top_z, width):
    """A point short of the target top, so the sweep does not count the target itself as the obstacle."""
    end = np.array([target[0], target[1], top_z], dtype=float)
    direction = end - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(direction))
    return end - direction / length * min(length - 1., width / 2 + SIGHT_MARGIN_M)


def plan(args):
    from sr3m.routes_v3 import generate
    objects = json.load(open(args.objects, encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    design = DESIGNS[DESIGN]
    out = {"seed": args.seed, "design": DESIGN, "design_fields": dataclasses.asdict(design), "objects_file": args.objects, "maps": {}}
    for map_id, quota in QUOTA.items():
        ground = Ground(map_id, args.cache)
        routes, rejected = generate([o for o in objects if o["map"] == map_id], ground, int(quota * CANDIDATES), design, rng)
        for i, r in enumerate(routes):
            r["id"] = f"{map_id}_{i:03d}"
        out["maps"][map_id] = {"quota": quota, "rejected_by_design": rejected, "candidates": [plain(r) for r in routes]}
        print(map_id, len(routes), "candidates", flush=True)
    path = Path(args.out) / "plan.json"
    if path.exists():
        raise ValueError(f"{path} exists; the plan is fixed once written")
    write_json(path, out)


def episode_for(prepared, map_id):
    """A v2 route on the same map, for its capture settings, project profile and map package."""
    from experiments.protocol import PreparedSplit
    for split in ("train", "val_seen"):
        for r in PreparedSplit(prepared, split).routes:
            if r["map_package"].endswith("/" + map_id):
                return r["episode_id"]
    raise ValueError(f"No v2 route on {map_id}")


def check(backend, route):
    """UE checks before rendering; the reason, or None."""
    P = np.array(route["positions"])
    start = P[0]
    if backend.sweep(start, sight_end(start, route["target"], route["top_z"], route["width_m"]))["hit"]:
        return "target hidden at the start"
    for a, b in zip(P[:-1], P[1:]):
        hit = backend.sweep(a, b)
        if hit["hit"]:
            return f"collision: {hit.get('actor')}"
    for p in P:
        ground = backend.ground_z_m(p)
        if ground is not None and p[2] - ground < MIN_AGL_M:
            return "too close to the ground in UE"
    return None


def render_route(backend, route, folder, steps):
    folder.mkdir(parents=True, exist_ok=True)
    P, Y = np.array(route["positions"]), np.array(route["yaws"])
    images = {"front": [], "down": []}
    for t in steps:
        for view, image in backend.render(P[t], float(Y[t])).items():
            name = f"step_{t:06d}/{view}_rgb.png"
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            image.save(folder / name)
            images[view].append(name)
    return images


def render(args):
    from sim_k.backend import UEBackend
    root = Path(args.out)
    code = code_version()
    planned = read_json(root / "plan.json")
    for map_id in args.maps:
        entry = planned["maps"][map_id]
        quota = args.preview or entry["quota"]
        target = root / ("preview" if args.preview else "routes") / map_id
        target.mkdir(parents=True, exist_ok=True)
        log_path = target / "checks.json"
        log = read_json(log_path) if log_path.exists() else {}
        accepted = [k for k, v in log.items() if v == "accepted"]
        if len(accepted) >= quota:
            print(map_id, "complete", flush=True)
            continue
        _, capture, profile, map_package = capture_settings(args.data_root, episode_for(args.prepared, map_id))
        with UEBackend(target / "session", profile, map_package, capture, restart_every=args.restart_every) as backend:
            for route in entry["candidates"]:
                if len(accepted) >= quota:
                    break
                if route["id"] in log and log[route["id"]] != "accepted":
                    continue
                folder = target / route["id"]
                if route["id"] not in log:
                    reason = check(backend, route)
                    if reason:
                        log[route["id"]] = reason
                        write_json(log_path, log)
                        print(route["id"], "skipped:", reason, flush=True)
                        continue
                if not (folder / "route.json").exists():
                    n = len(route["positions"])
                    steps = [0, n - 1] if args.preview else list(range(n))
                    images = render_route(backend, route, folder, steps)
                    rank = len(accepted)
                    write_json(folder / "route.json", {**route, "split": split_of(map_id, rank), "rank": rank, "rendered_steps": steps, "images": images,
                                                       "map_package": map_package, "profile": profile, "capture": capture, "plan_seed": planned["seed"], "design": planned["design"], "code": code})
                log[route["id"]] = "accepted"
                write_json(log_path, log)
                accepted.append(route["id"])
                print(route["id"], "rendered", len(accepted), "/", quota, flush=True)
        print(map_id, "accepted", len(accepted), "renders", backend.renders, "unexpected exits", backend.unexpected_exits, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "render"))
    p.add_argument("--out", default="/mnt/f/WSL_ColdStorage/datasets/sr3m_v3_pilot")
    p.add_argument("--objects", default="configs/v3_objects.json"); p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--cache", default=str(Path.home() / "sr3mxhad_runs/v3/ground"))
    p.add_argument("--prepared", default="runs/terrain_v2_dino/prepared")
    p.add_argument("--data-root", default="/mnt/f/WSL_ColdStorage/datasets/DatasetCaptureBatches/workbench")
    p.add_argument("--maps", nargs="+", default=list(QUOTA)); p.add_argument("--preview", type=int, default=0)
    p.add_argument("--restart-every", type=int, default=32)
    args = p.parse_args()
    plan(args) if args.stage == "plan" else render(args)
