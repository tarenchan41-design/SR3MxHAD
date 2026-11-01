"""Recovery probe on K: offset the pose from a recorded expert point and read the action. No flying, no training.

Asks whether the policy steers back to the route once it has left it (runs/closed_loop/k/recovery_spec.md).
Offsets come in +/- pairs, so the turn of the route itself cancels.
"""
import argparse
import math
from pathlib import Path

import numpy as np

from experiments.closed_loop_k import MIN_AGL_M, capture_settings, load_policy, policy_state, route_info
from experiments.protocol import read_json, write_json
from sr3m.geometry import integrate, wrap

FRACTIONS = (.25, .5, .75)
OFFSETS = (("heading", math.radians(10)), ("heading", math.radians(20)), ("lateral", 2.), ("lateral", 5.))
PURSUIT_LOOKAHEAD_M = 20.
PURSUIT_STEPS = 10
STEP_M = 2.
MIN_SHARE = .75
MIN_RIGHT_FRACTION = .05
CONDITIONS = ("B", "B_image", "B_state", "N")
RESAMPLES = 10000


def probe_steps(count):
    """The last recorded pose has no action, so fractions are of the expert's count - 1 steps."""
    return [round(f * (count - 1)) for f in FRACTIONS]


def offset_pose(position, yaw, kind, value):
    """Positive heading turns right and positive lateral moves right, the same sense as a positive dyaw."""
    if kind == "heading":
        return np.asarray(position, dtype=float), wrap(yaw + value)
    target, _ = integrate(tuple(position), yaw, (0., value, 0., 0.))
    return np.asarray(target, dtype=float), yaw


def turn_back(plus, minus):
    """Radians per step toward the route, positive when the pair steers back; the route's own turn cancels."""
    return -(plus[3] - minus[3]) / 2


def close_back(plus, minus):
    """Metres per step toward the route from the right component alone."""
    return -(plus[1] - minus[1]) / 2


def required(kind, magnitude):
    """Half of a naive pursuit: aim at the route LOOKAHEAD ahead and turn there over PURSUIT_STEPS."""
    angle = magnitude if kind == "heading" else math.atan(magnitude / PURSUIT_LOOKAHEAD_M)
    return angle / PURSUIT_STEPS / 2


def spectral_radius(heading_gain, lateral_gain, right_gain=0., step_m=STEP_M):
    """Linearised error (lateral, heading): e' = (1 - right) e + step * psi, psi' = -lateral e + (1 - heading) psi."""
    matrix = np.array([[1 - right_gain, step_m], [-lateral_gain, 1 - heading_gain]])
    return float(np.abs(np.linalg.eigvals(matrix)).max())


def probe_point(backend, models, positions, yaws, step):
    position, yaw = positions[step], float(yaws[step])
    start, start_yaw = positions[0], float(yaws[0])
    on_state = policy_state(position, yaw, start, start_yaw)
    on_images = backend.render(position, yaw)
    answer = {"step": step, "base": {"B": models["B"](on_images, on_state), "N": models["N"]({}, on_state)}, "offsets": []}
    for kind, magnitude in OFFSETS:
        for sign in (1, -1):
            where, heading = offset_pose(position, yaw, kind, sign * magnitude)
            ground = backend.ground_z_m(where)
            hit = kind == "lateral" and bool(backend.sweep(position, where)["hit"])
            images = backend.render(where, heading)
            state = policy_state(where, heading, start, start_yaw)
            answer["offsets"].append({"kind": kind, "magnitude": magnitude, "sign": sign, "hit": hit, "agl_m": None if ground is None else float(where[2] - ground),
                                      "B": models["B"](images, state), "B_image": models["B"](images, on_state), "B_state": models["B"](on_images, state), "N": models["N"]({}, state)})
    return answer


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value.item() if isinstance(value, np.generic) else value


def run(args):
    from sim_k.backend import UEBackend
    for item in read_json(args.routes)["routes"]:
        episode = item["episode_id"]
        out = Path(args.out) / (episode.split("--")[-1] + ".json")
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        _, route, positions, yaws = route_info(args.prepared, args.split, episode)
        _, capture, profile, map_package = capture_settings(args.data_root, episode)
        models = {"B": load_policy(args.checkpoint_b, route, args.device)[0], "N": load_policy(args.checkpoint_n, route, args.device)[0]}
        with UEBackend(out.with_suffix(".session"), profile, map_package, capture, restart_every=args.restart_every) as backend:
            points = [probe_point(backend, models, positions, yaws, s) for s in probe_steps(len(positions))]
        write_json(out, plain({"episode_id": episode, "points": points, "unexpected_exits": backend.unexpected_exits, "renders": backend.renders, "starts": backend.starts}))
        print(out.stem[:40], backend.renders, "renders", flush=True)


def pairs(record):
    """One row per +/- pair that stayed clear of obstacles and ground, per point."""
    rows, dropped = [], 0
    for point in record["points"]:
        for kind, magnitude in OFFSETS:
            found = {o["sign"]: o for o in point["offsets"] if o["kind"] == kind and o["magnitude"] == magnitude}
            plus, minus = found[1], found[-1]
            if any(o["hit"] or (o["agl_m"] is not None and o["agl_m"] < MIN_AGL_M) for o in (plus, minus)):
                dropped += 1
                continue
            rows.append({"step": point["step"], "kind": kind, "magnitude": magnitude, "turn": {c: turn_back(plus[c]["action"], minus[c]["action"]) for c in CONDITIONS},
                         "right": {c: close_back(plus[c]["action"], minus[c]["action"]) for c in CONDITIONS},
                         "stop": {c: (plus[c]["stop_probability"] + minus[c]["stop_probability"]) / 2 for c in CONDITIONS}})
    return rows, dropped


def group_passes(values_by_magnitude, needed_by_magnitude, share=MIN_SHARE):
    every = np.concatenate(list(values_by_magnitude.values()))
    medians = {m: float(np.median(v)) for m, v in values_by_magnitude.items()}
    return {"share": float((every > 0).mean()), "medians": medians, "passed": bool((every > 0).mean() >= share and all(medians[m] >= needed_by_magnitude[m] for m in medians))}


def median_ci(by_route, seed=0):
    """Bootstrap over routes of the pooled median."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(RESAMPLES):
        chosen = rng.integers(0, len(by_route), len(by_route))
        draws.append(np.median(np.concatenate([by_route[i] for i in chosen])))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def report(args):
    records = [(f.stem, read_json(f)) for f in sorted(Path(args.out).glob("*.json")) if f.name != "summary.json"]
    table, dropped = {}, 0
    for name, record in records:
        rows, lost = pairs(record)
        dropped += lost
        for row in rows:
            for c in CONDITIONS:
                entry = table.setdefault((row["kind"], row["magnitude"], c), {})
                entry.setdefault(name, {"turn": [], "right": []})
                entry[name]["turn"].append(row["turn"][c])
                entry[name]["right"].append(row["right"][c])
    print(f"{len(records)} routes; {dropped} pairs dropped for a hit or AGL below {MIN_AGL_M} m; unexpected exits {sum(r['unexpected_exits'] for _, r in records)}")
    summary = {"dropped": dropped}
    print(f"\n{'offset':>10s} {'needed':>7s} {'cond':>8s} {'n':>4s} {'back':>6s} {'median':>7s} {'CI':>16s} {'right m':>8s} {'back':>6s}")
    for kind, magnitude in OFFSETS:
        label = f"{math.degrees(magnitude):.0f} deg" if kind == "heading" else f"{magnitude:.0f} m"
        needed = math.degrees(required(kind, magnitude))
        for c in CONDITIONS:
            entry = table[(kind, magnitude, c)]
            by_route = [np.degrees(entry[n]["turn"]) for n in entry]
            turns = np.concatenate(by_route)
            rights = np.concatenate([entry[n]["right"] for n in entry])
            low, high = median_ci(by_route)
            print(f"{label:>10s} {needed:7.2f} {c:>8s} {len(turns):4d} {(turns > 0).mean():6.2f} {np.median(turns):7.2f} [{low:6.2f},{high:6.2f}] {np.median(rights):8.3f} {(rights > 0).mean():6.2f}")
            summary[f"{kind}_{label}_{c}"] = {"needed_deg": needed, "share": float((turns > 0).mean()), "median_deg": float(np.median(turns)), "ci_deg": [low, high], "median_right_m": float(np.median(rights)), "share_right": float((rights > 0).mean())}
    def collect(kind, field, c="B"):
        return {m: np.concatenate([table[(kind, m, c)][n][field] for n in table[(kind, m, c)]]) for k, m in OFFSETS if k == kind}
    heading = group_passes(collect("heading", "turn"), {m: required("heading", m) for k, m in OFFSETS if k == "heading"})
    lateral_turn = group_passes(collect("lateral", "turn"), {m: required("lateral", m) for k, m in OFFSETS if k == "lateral"})
    lateral_right = group_passes(collect("lateral", "right"), {m: MIN_RIGHT_FRACTION * m for k, m in OFFSETS if k == "lateral"})
    recovers = heading["passed"] and (lateral_turn["passed"] or lateral_right["passed"])
    print("\n== verdict on B (actual image and state)")
    print(f"heading: share back {heading['share']:.2f}, passed {heading['passed']}")
    print(f"lateral by turning: share back {lateral_turn['share']:.2f}, passed {lateral_turn['passed']}")
    print(f"lateral by right: share back {lateral_right['share']:.2f}, passed {lateral_right['passed']}")
    print("VERDICT: recovers" if recovers else "VERDICT: does not reliably recover")
    summary.update(heading=heading, lateral_turn=lateral_turn, lateral_right=lateral_right, recovers=recovers)
    print("\n== linearised error system from median gains (small offset; radius below 1 is stable)")
    for c in CONDITIONS:
        k_heading = float(np.median(np.concatenate([table[("heading", OFFSETS[0][1], c)][n]["turn"] for n in table[("heading", OFFSETS[0][1], c)]]))) / OFFSETS[0][1]
        k_lateral = float(np.median(np.concatenate([table[("lateral", OFFSETS[2][1], c)][n]["turn"] for n in table[("lateral", OFFSETS[2][1], c)]]))) / OFFSETS[2][1]
        k_right = float(np.median(np.concatenate([table[("lateral", OFFSETS[2][1], c)][n]["right"] for n in table[("lateral", OFFSETS[2][1], c)]]))) / OFFSETS[2][1]
        radius = spectral_radius(k_heading, k_lateral, k_right)
        print(f"{c:>8s} heading gain {k_heading:7.4f} /step, lateral {math.degrees(k_lateral):7.4f} deg/step/m, right {k_right:7.4f} /step, radius {radius:.4f}")
        summary[f"linear_{c}"] = {"heading": k_heading, "lateral": k_lateral, "right": k_right, "radius": radius}
    if args.predictions and Path(args.predictions).exists():
        offline = np.load(args.predictions)
        gaps, stops_on, stops_off = [], [], []
        for _, record in records:
            _, route, _, _ = route_info(args.prepared, args.split, record["episode_id"])
            for point in record["points"]:
                gaps.append(abs(point["base"]["B"]["action"][3] - float(offline["action"][route["offset"] + point["step"], 3])))
                stops_on.append(point["base"]["B"]["stop_probability"])
                stops_off.extend(o["B"]["stop_probability"] for o in point["offsets"])
        print(f"\non-path B yaw vs offline: median {np.degrees(np.median(gaps)):.4f} deg, max {np.degrees(np.max(gaps)):.4f} deg (chain check)")
        print(f"STOP probability B: on path median {np.median(stops_on):.5f}, offset median {np.median(stops_off):.5f}")
    write_json(Path(args.out) / "summary.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="runs/terrain_v2_dino/prepared"); p.add_argument("--split", default="train")
    p.add_argument("--routes", default="runs/closed_loop/k/train_routes.json")
    p.add_argument("--data-root", default="/mnt/f/WSL_ColdStorage/datasets/DatasetCaptureBatches/workbench")
    p.add_argument("--out", default="runs/closed_loop/k/recovery")
    p.add_argument("--checkpoint-b", default="runs/terrain_v2_dino/models/B_seed42/best.pt")
    p.add_argument("--checkpoint-n", default="runs/terrain_v2_dino/models_n/N_seed42/best.pt")
    p.add_argument("--predictions", default="runs/closed_loop/k/flights_train/train_predictions.npz")
    p.add_argument("--device", default="cuda"); p.add_argument("--restart-every", type=int, default=32); p.add_argument("--report", action="store_true")
    args = p.parse_args()
    report(args) if args.report else run(args)
