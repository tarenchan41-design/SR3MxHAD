"""Dataset v3 pilot, closed loop on the dev routes (protocol section 14). Run with SR3MXHAD_PROTOCOL=v3_pilot; close the GUI editor.

fly:    one policy over all dev routes. A model checkpoint, or a reference: oracle (the pursuit expert without injected offsets)
        and straight (keep the start heading, stop after the expert's step count).
probe:  the recovery probe of runs/closed_loop/k/recovery_spec.md on the dev routes, for the gate on heading response.
report: success at STOP by horizontal distance to the target centre, the gates, and descriptions.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.closed_loop_k import MIN_AGL_M, load_policy, policy_state
from experiments.protocol import PreparedSplit, read_json, write_json
from experiments.recovery_probe import OFFSETS, close_back, offset_pose, required, turn_back
from experiments.v3_leak import Ground
from sr3m.geometry import integrate
from sr3m.routes_v3 import Design, follow_height, pursuit_yaw

RADII = (20., 10., 5.)
GATE_SR20 = .5
GATE_MARGIN = .2
PROBE_FRACTIONS = (.25, .5, .75)
RESAMPLES = 10000


def dev_routes(prepared):
    ds = PreparedSplit(prepared, "dev")
    out = []
    for r in ds.routes:
        window = slice(r["offset"], r["offset"] + r["count"])
        out.append((r, ds.arrays["position"][window].astype(float), ds.arrays["yaw"][window].astype(float)))
    return out


def pose_from_state(state, start, start_yaw):
    c, s = math.cos(start_yaw), math.sin(start_yaw)
    x, y, z = state[0] * 100, state[1] * 100, state[2] * 100
    return np.array([start[0] + c * x - s * y, start[1] + s * x + c * y, start[2] + z]), start_yaw + math.atan2(state[3], state[4])


def reference_policy(name, route, positions, yaws, ground):
    """Policies that need no model; they read the pose back from the policy state."""
    design = Design()
    start, start_yaw = positions[0], float(yaws[0])
    height = follow_height(ground, positions[0, 2] - float(ground(positions[:1, :2])[0]), design) if route["mode"] == "follow" else (lambda xy, previous, heading: positions[0, 2])
    target = route["target"]
    steps = {"n": 0}

    def policy(images, state):
        position, yaw = pose_from_state(state, start, start_yaw)
        steps["n"] += 1
        if name == "straight":
            return {"stop": steps["n"] > len(positions) - 1, "stop_probability": 0., "action": np.array([2., 0., 0., 0.])}
        distance = math.hypot(target[0] - position[0], target[1] - position[1])
        if distance <= 1.:
            return {"stop": True, "stop_probability": 1., "action": np.zeros(4)}
        forward = min(2., distance)
        moved = position + forward * np.array([math.cos(yaw), math.sin(yaw), 0.])
        z = height(moved[:2], position[2], math.atan2(target[1] - moved[1], target[0] - moved[0]))
        return {"stop": False, "stop_probability": 0., "action": np.array([forward, 0., z - position[2], pursuit_yaw(position, yaw, target, design)])}
    policy.needs_images = False
    return policy


def fly_route(backend, route, positions, yaws, policy, ground):
    position, yaw = positions[0].copy(), float(yaws[0])
    budget = 2 * (len(positions) - 1) + 20
    history, failure, stopped = [], None, False
    for t in range(budget):
        images = backend.render(position, yaw) if getattr(policy, "needs_images", True) else {}
        decision = policy(images, policy_state(position, yaw, positions[0], float(yaws[0])))
        if decision["stop"]:
            stopped = True
            history.append({"step": t, "position_m": position.tolist(), "yaw_rad": yaw, "stop_probability": decision["stop_probability"]})
            break
        action = np.asarray(decision["action"], dtype=float)
        target, next_yaw = integrate(position, yaw, action)
        target = np.asarray(target, dtype=float)
        hit = backend.sweep(position, target)
        below = backend.ground_z_m(target)
        agl = None if below is None else float(target[2] - below)
        terrain_agl = float(target[2] - ground(target[None, :2])[0])  # NaN once the drone leaves the cached terrain grid
        history.append({"step": t, "position_m": position.tolist(), "yaw_rad": yaw, "action": action.tolist(), "stop_probability": decision["stop_probability"], "agl_m": agl,
                        "terrain_agl_m": terrain_agl if math.isfinite(terrain_agl) else None})
        if hit["hit"]:
            failure = f"collision: {hit.get('actor')}"
            break
        if agl is not None and agl < MIN_AGL_M:
            failure = f"ground: AGL {agl:.2f} m"
            break
        position, yaw = target, float(next_yaw)
    horizontal = math.hypot(position[0] - route["target"][0], position[1] - route["target"][1])
    return {"route_id": route["route_id"], "steps": len(history), "expert_steps": len(positions) - 1, "stopped": stopped, "failure": failure, "horizontal_m": horizontal,
            "above_top_m": float(position[2] - route["top_z"]), "final_m": position.tolist(), "history": history,
            **{f"SR{int(r)}": bool(stopped and failure is None and horizontal <= r) for r in RADII}}


def finite_or_none(obj, path="", found=None):
    """JSON has no NaN: replace non-finite floats with None and list where they were (11/03: one B0 flight crashed the writer)."""
    found = [] if found is None else found
    if isinstance(obj, dict):
        return {k: finite_or_none(v, f"{path}.{k}", found)[0] for k, v in obj.items()}, found
    if isinstance(obj, list):
        return [finite_or_none(v, f"{path}[{i}]", found)[0] for i, v in enumerate(obj)], found
    if isinstance(obj, float) and not math.isfinite(obj):
        found.append(path)
        return None, found
    return obj, found


def fly(args):
    from sim_k.backend import UEBackend
    routes = dev_routes(args.prepared)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    grounds = {}
    for map_id in sorted({r["route_id"][:7] for r, _, _ in routes}):
        todo = [(r, p, y) for r, p, y in routes if r["route_id"].startswith(map_id) and not (out / f"{r['route_id']}.json").exists()]
        if not todo:
            continue
        ground = grounds.setdefault(map_id, Ground(map_id, args.cache))
        capture = read_json(Path(args.root) / "routes" / map_id / todo[0][0]["route_id"] / "route.json")
        with UEBackend(out / f"session_{map_id}", capture["profile"], capture["map_package"], capture["capture"], restart_every=args.restart_every) as backend:
            for route, positions, yaws in todo:
                if args.checkpoint:
                    policy, identity = load_policy(args.checkpoint, route, args.device)
                else:
                    policy, identity = reference_policy(args.name, route, positions, yaws, ground), {"variant": args.name}
                result = fly_route(backend, route, positions, yaws, policy, ground)
                result["identity"] = {k: identity[k] for k in identity if k != "tokenizer"}
                result, nonfinite = finite_or_none(result)
                if nonfinite:
                    result["nonfinite_fields"] = nonfinite
                    print("non-finite values set to null:", nonfinite[:5], len(nonfinite), flush=True)
                write_json(out / f"{route['route_id']}.json", result)
                print(args.name, route["route_id"], {k: result[k] for k in ("SR20", "SR10", "stopped", "failure", "steps")}, round(result["horizontal_m"], 1), flush=True)


def probe(args):
    """Same pairs as recovery_probe: offsets at 25/50/75 % of each dev route; the turn back toward the route, per model."""
    from sim_k.backend import UEBackend
    routes = dev_routes(args.prepared)
    out = Path(args.out) / args.probe_dir
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = dict(item.split("=") for item in args.models)
    for map_id in sorted({r["route_id"][:7] for r, _, _ in routes}):
        todo = [(r, p, y) for r, p, y in routes if r["route_id"].startswith(map_id) and not (out / f"{r['route_id']}.json").exists()]
        if not todo:
            continue
        capture = read_json(Path(args.root) / "routes" / map_id / todo[0][0]["route_id"] / "route.json")
        with UEBackend(out / f"session_{map_id}", capture["profile"], capture["map_package"], capture["capture"], restart_every=args.restart_every) as backend:
            for route, positions, yaws in todo:
                models = {name: load_policy(path, route, args.device)[0] for name, path in checkpoints.items()}
                points = []
                for fraction in PROBE_FRACTIONS:
                    step = round(fraction * (len(positions) - 1))
                    pos, yaw = positions[step], float(yaws[step])
                    on_images = backend.render(pos, yaw)
                    offsets = []
                    for kind, magnitude in OFFSETS:
                        for sign in (1, -1):
                            where, heading = offset_pose(pos, yaw, kind, sign * magnitude)
                            images = backend.render(where, heading)
                            state = policy_state(where, heading, positions[0], float(yaws[0]))
                            offsets.append({"kind": kind, "magnitude": magnitude, "sign": sign, **{name: [float(a) for a in m(images if m.needs_images else {}, state)["action"]] for name, m in models.items()}})
                    points.append({"step": step, "offsets": offsets})
                write_json(out / f"{route['route_id']}.json", {"route_id": route["route_id"], "points": points, "models": checkpoints})
                print("probe", route["route_id"], flush=True)


def bootstrap_difference(a, b, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    draws = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(RESAMPLES)]
    return float(d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def report(args):
    flights = Path(args.out)
    routes = {r["route_id"]: r for r, _, _ in dev_routes(args.prepared)}
    results = {}
    for folder in sorted(p for p in flights.iterdir() if p.is_dir() and not p.name.startswith("probe")):
        rows = [read_json(f) for f in sorted(folder.glob("*.json"))]
        if rows:
            results[folder.name] = {r["route_id"]: r for r in rows}
    summary = {}
    print(f"{'policy':10s} {'n':>3s} {'SR20':>5s} {'SR10':>5s} {'SR5':>5s} {'stopped':>7s} {'fail':>4s} {'NE med':>6s} | by map SR20 | follow |AGL err| med")
    for name, rows in results.items():
        values = list(rows.values())
        by_map = {m: np.mean([v["SR20"] for v in values if v["route_id"].startswith(m)]) for m in sorted({v["route_id"][:7] for v in values})}
        tracking = []  # Follow routes: how far the height above the terrain drifts from where it started.
        for v in values:
            agl = [h["terrain_agl_m"] for h in v["history"] if h.get("terrain_agl_m") is not None]
            if routes[v["route_id"]]["mode"] == "follow" and agl:
                tracking.append(float(np.median(np.abs(np.array(agl) - agl[0]))))
        summary[name] = {"n": len(values), **{f"SR{int(r)}": float(np.mean([v[f'SR{int(r)}'] for v in values])) for r in RADII}, "stopped": float(np.mean([v["stopped"] for v in values])),
                         "failures": sum(bool(v["failure"]) for v in values), "ne_median_m": float(np.median([v["horizontal_m"] for v in values])), "by_map_SR20": by_map,
                         "follow_agl_drift_median_m": float(np.median(tracking)) if tracking else None}
        s = summary[name]
        print(f"{name:10s} {s['n']:3d} {s['SR20']:5.2f} {s['SR10']:5.2f} {s['SR5']:5.2f} {s['stopped']:7.2f} {s['failures']:4d} {s['ne_median_m']:6.1f} | {' '.join(f'{m[2:]} {x:.2f}' for m, x in by_map.items())} | {s['follow_agl_drift_median_m']}")
    gates, g = {}, args.gated  # the gated model: B (section 14) or B4 (section 16)
    if g in results:
        gates[f"1 {g} SR20 >= 0.5"] = summary[g]["SR20"] >= GATE_SR20
        if "N" in results:
            common = sorted(set(results[g]) & set(results["N"]))
            diff = bootstrap_difference([results[g][k]["SR20"] for k in common], [results["N"][k]["SR20"] for k in common])
            summary[f"{g}_minus_N_SR20"] = diff
            gates[f"2 {g} - N SR20 >= 0.2"] = diff[0] >= GATE_MARGIN
            print(f"{g} - N SR20 {diff[0]:+.2f} CI [{diff[1]:+.2f}, {diff[2]:+.2f}] over {len(common)} routes")
    probe_dir = flights / args.probe_dir
    if probe_dir.exists():
        rows = [read_json(f) for f in sorted(probe_dir.glob("*.json"))]
        names = list(rows[0]["models"]) if rows else []
        summary["probe"] = {}
        for name in names:
            by = {}
            for row in rows:
                for point in row["points"]:
                    for kind, magnitude in OFFSETS:
                        found = {o["sign"]: o for o in point["offsets"] if o["kind"] == kind and o["magnitude"] == magnitude}
                        by.setdefault((kind, magnitude), []).append(turn_back(found[1][name], found[-1][name]))
            summary["probe"][name] = {f"{k}_{m:.3g}": {"median_deg": float(np.degrees(np.median(v))), "share_back": float((np.array(v) > 0).mean()), "needed_deg": float(np.degrees(required(k, m)))} for (k, m), v in by.items()}
            print(f"probe {name}: " + "; ".join(f"{k} {math.degrees(m) if k == 'heading' else m:.0f}: {np.degrees(np.median(v)):+.2f} deg/step, back {np.mean(np.array(v) > 0):.2f}" for (k, m), v in by.items()))
        if g in summary["probe"]:
            gates[f"3 {g} heading response > 0"] = all(summary["probe"][g][f"heading_{m:.3g}"]["median_deg"] > 0 for k, m in OFFSETS if k == "heading")
    summary["gates"] = gates
    print("gates:", {k: "pass" if v else "FAIL" for k, v in gates.items()})
    write_json(flights / args.summary, summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fly", "probe", "report"))
    p.add_argument("--prepared", default="runs/v3_pilot/prepared"); p.add_argument("--root", default="/mnt/f/WSL_ColdStorage/datasets/sr3m_v3_pilot")
    p.add_argument("--out", default="runs/v3_pilot/flights"); p.add_argument("--cache", default=str(Path.home() / "sr3mxhad_runs/v3/ground"))
    p.add_argument("--name"); p.add_argument("--checkpoint"); p.add_argument("--models", nargs="+", default=[], help="probe: name=checkpoint ...")
    p.add_argument("--device", default="cuda"); p.add_argument("--restart-every", type=int, default=32)
    p.add_argument("--probe-dir", default="probe"); p.add_argument("--gated", default="B", help="report: the model the gates are read on")
    p.add_argument("--summary", default="summary.json")
    args = p.parse_args()
    {"fly": fly, "probe": probe, "report": report}[args.stage](args)
