"""Short closed-loop rollouts on K: from recorded poses, the policy flies a few steps by itself.

Long flights leave the route early, so they cannot measure height tracking over the intended terrain. This does not
replace long-range SR20. Every window starts back on the expert route and is measured against the expert at the same step.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.closed_loop_k import capture_settings, load_policy, policy_state, route_info
from experiments.protocol import read_json, write_json
from sr3m.geometry import integrate, wrap

HORIZON = 10
EVERY = 20
RESAMPLES = 10000


def window_starts(count, horizon=HORIZON, every=EVERY):
    """Start indices whose window ends on a recorded pose; the last recorded pose has no action."""
    return list(range(0, count - horizon, every))


def rollout(backend, positions, yaws, policy, start, horizon=HORIZON):
    """STOP is recorded, not obeyed: this measures control, and a window must have a fixed length to be compared."""
    position, yaw = positions[start].copy(), float(yaws[start])
    steps = []
    for _ in range(horizon):
        images = backend.render(position, yaw) if getattr(policy, "needs_images", True) else {}
        decision = policy(images, policy_state(position, yaw, positions[0], float(yaws[0])))
        target, next_yaw = integrate(position, yaw, np.asarray(decision["action"], dtype=float))
        target, next_yaw = np.asarray(target, dtype=float), float(next_yaw)
        ground = backend.ground_z_m(target)
        steps.append({"position_m": target.tolist(), "yaw_rad": next_yaw, "action": [float(a) for a in decision["action"]], "stop_probability": decision["stop_probability"],
                      "hit": bool(backend.sweep(position, target)["hit"]), "agl_m": None if ground is None else float(target[2] - ground)})
        position, yaw = target, next_yaw
    return {"start": start, "steps": steps}


def run(args):
    from sim_k.backend import UEBackend
    routes = read_json(Path(args.prepared) / "closed_loop_routes.json")["routes"]
    for item in routes:
        episode = item["episode_id"]
        if args.only and args.only not in episode:
            continue
        out = Path(args.out) / (episode.split("--")[-1] + ".json")
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        _, route, positions, yaws = route_info(args.prepared, "val_seen", episode)
        _, capture, profile, map_package = capture_settings(args.data_root, episode)
        policy, identity = load_policy(args.checkpoint, route, args.device)
        with UEBackend(out.with_suffix(".session"), profile, map_package, capture, restart_every=args.restart_every) as backend:
            windows = [rollout(backend, positions, yaws, policy, s) for s in window_starts(len(positions))]
        write_json(out, {"episode_id": episode, "terrain_following": item["terrain_following"], "category": item["category"], "height_layer": item["height_layer"], "horizon": HORIZON, "every": EVERY,
                         "windows": windows, "unexpected_exits": backend.unexpected_exits, "renders": backend.renders, "starts": backend.starts, "identity": {k: identity[k] for k in identity if k != "tokenizer"}})
        print(out.stem[:40], len(windows), "windows", flush=True)


def window_errors(record, positions, yaws, at=HORIZON):
    """Per window at step `at`: |height error|, horizontal distance, |yaw error|, model and expert climb."""
    rows = []
    for w in record["windows"]:
        start, end = w["start"], w["start"] + at
        reached = np.array(w["steps"][at - 1]["position_m"])
        rows.append({"z": abs(reached[2] - positions[end, 2]), "h": float(np.linalg.norm(reached[:2] - positions[end, :2])), "yaw": abs(np.degrees(wrap(w["steps"][at - 1]["yaw_rad"] - yaws[end]))),
                     "climb": reached[2] - positions[start, 2], "expert_climb": positions[end, 2] - positions[start, 2], "hits": sum(s["hit"] for s in w["steps"][:at])})
    return rows


def summarize(root, prepared, at=HORIZON):
    """Mean over windows per route, then over seeds: {variant: {route: {metric: value}}} plus per-window climbs."""
    table, climbs, flags = {}, {}, {}
    for directory in sorted(Path(root).glob("*_seed*")):
        variant = directory.name.split("_seed")[0]
        for file in sorted(directory.glob("*.json")):
            record = read_json(file)
            _, _, positions, yaws = route_info(prepared, "val_seen", record["episode_id"])
            rows = window_errors(record, positions, yaws, at)
            flags[file.stem] = record["terrain_following"]
            entry = table.setdefault(variant, {}).setdefault(file.stem, {"z": [], "h": [], "yaw": [], "hits": []})
            for key in entry:
                entry[key].append(float(np.mean([r[key] for r in rows])))
            climbs.setdefault(variant, {}).setdefault(record["terrain_following"], []).extend((r["climb"], r["expert_climb"]) for r in rows)
    means = {v: {r: {k: float(np.mean(x)) for k, x in m.items()} for r, m in routes.items()} for v, routes in table.items()}
    return means, climbs, flags


def bootstrap(values, statistic, seed=0):
    rng = np.random.default_rng(seed)
    draws = [statistic(rng.integers(0, len(values), len(values))) for _ in range(RESAMPLES)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def report(args):
    print(f"(measured against the expert at the same step; bootstrap over routes; val_seen development)")
    for at in (5, HORIZON):
        means, climbs, flags = summarize(args.out, args.prepared, at)
        print(f"\n== step {at} ({at * 2} m)")
        for following in (True, False):
            names = sorted(n for n, f in flags.items() if f == following)
            print(f"{'terrain-following' if following else 'hold height'} routes n={len(names)}")
            for v in sorted(means):
                present = [n for n in names if n in means[v]]
                if present:
                    row = {k: np.mean([means[v][n][k] for n in present]) for k in ("z", "h", "yaw", "hits")}
                    x = np.array([c for c in climbs[v].get(following, [])])
                    beta = np.polyfit(x[:, 1], x[:, 0], 1)[0] if len(x) > 2 and x[:, 1].std() > 1e-6 else float("nan")
                    print(f"  {v:2s} routes {len(present):2d} |z| {row['z']:.3f} m, horizontal {row['h']:.3f} m, |yaw| {row['yaw']:.2f} deg, hits per window {row['hits']:.3f}, climb beta {beta:.2f}")
            if {"A", "B"} <= set(means):
                common = [n for n in names if n in means["A"] and n in means["B"]]
                if len(common) >= 2:
                    a, b = [np.array([means[v][n]["z"] for n in common]) for v in ("A", "B")]
                    if following:
                        low, high = bootstrap(a, lambda i: 100 * (a[i].mean() - b[i].mean()) / a[i].mean())
                        print(f"  B relative to A |z|: {100 * (a.mean() - b.mean()) / a.mean():.1f}% CI [{low:.1f}, {high:.1f}]")
                    else:
                        low, high = bootstrap(a, lambda i: 100 * (b[i].mean() - a[i].mean()))
                        print(f"  B-A |z| on hold routes: {100 * (b.mean() - a.mean()):+.1f} cm CI [{low:.1f}, {high:.1f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="runs/terrain_v2_dino/prepared")
    p.add_argument("--data-root", default="/mnt/f/WSL_ColdStorage/datasets/DatasetCaptureBatches/workbench")
    p.add_argument("--out", required=True, help="Output folder; with --report, the folder holding <variant>_seed<seed>")
    p.add_argument("--checkpoint"); p.add_argument("--device", default="cuda"); p.add_argument("--restart-every", type=int, default=32)
    p.add_argument("--only", help="Run only routes whose id contains this text"); p.add_argument("--report", action="store_true")
    args = p.parse_args()
    report(args) if args.report else run(args)
