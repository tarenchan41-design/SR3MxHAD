"""Dataset v3 stage 2, K closed loop on the development maps (protocol sections 17.7 and 17.9). SR3MXHAD_PROTOCOL=v3; close the GUI editor.

fly:    one policy over the 80 dev routes (L_MTR_B, L_DES_B_v3). A checkpoint, or a reference without a model: oracle (the
        pursuit expert without injected offsets) or straight (keep the start heading, stop after the expert's step count).
report: SR20/SR10/SR5, E on follow routes, the down-view grid rule (17.9) and the dev gate (17.9), descriptions by map,
        by unseen class and by name form.

The flight itself is v3_fly.fly_route of stage 1. Differences from v3_fly: the map comes from route_group_id (stage 2
names such as L_DES_B_v3 are longer than seven characters); the renderer runs the stage 2 console (STAGE2_CONSOLE, the same
as capture) and the memory restart policy of the bulk render; the start AGL is read from UE so E follows section 17.7.

E (17.7) = per-step |AGL_t - AGL_0| averaged over the route, AGL from UE (ground under the drone after each step, as the
flight's ground check reads it); AGL_0 the UE AGL at the start pose. Reported over follow routes as the mean of route values.
"""
import argparse
from pathlib import Path

import numpy as np

from experiments.closed_loop_k import load_policy
from experiments.protocol import PreparedSplit, read_json, write_json
from experiments.v3_fly import RADII, bootstrap_difference, finite_or_none, fly_route, reference_policy
from experiments.v3_leak import Ground
from experiments.v3_ue_check import STAGE2_CONSOLE

GATE_SR20 = .5
GRID_E_TOLERANCE = 1.1
MEMORY_CHECK_EVERY = 16


def map_of(route):
    return route["route_group_id"].split("/")[0]


def dev_routes(prepared):
    ds = PreparedSplit(prepared, "dev")
    out = []
    for r in ds.routes:
        window = slice(r["offset"], r["offset"] + r["count"])
        out.append((r, ds.arrays["position"][window].astype(float), ds.arrays["yaw"][window].astype(float)))
    return out


def route_e(result):
    """Section 17.7: mean over steps of |AGL_t - AGL_0|; None without a start AGL or any step AGL."""
    agl0 = result.get("start_agl_m")
    agl = [h["agl_m"] for h in result["history"] if h.get("agl_m") is not None]
    if agl0 is None or not agl:
        return None
    return float(np.mean(np.abs(np.array(agl) - agl0)))


def fly(args):
    from sim_k.backend import MemoryPolicy, UEBackend
    routes = dev_routes(args.prepared)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    for map_id in sorted({map_of(r) for r, _, _ in routes}):
        todo = [(r, p, y) for r, p, y in routes if map_of(r) == map_id and not (out / f"{r['route_id']}.json").exists()]
        if not todo:
            continue
        ground = Ground(map_id, args.cache)
        capture = read_json(Path(args.root) / "routes" / map_id / todo[0][0]["route_id"] / "route.json")
        if capture["console"] != STAGE2_CONSOLE:
            raise RuntimeError("The data were rendered with another console; flights must render as capture did")
        with UEBackend(out / f"session_{map_id}", capture["profile"], capture["map_package"], capture["capture"],
                       policy=MemoryPolicy(check_every=MEMORY_CHECK_EVERY), console=STAGE2_CONSOLE) as backend:
            for route, positions, yaws in todo:
                if args.checkpoint:
                    policy, identity = load_policy(args.checkpoint, route, args.device)
                else:
                    policy, identity = reference_policy(args.name, route, positions, yaws, ground), {"variant": args.name}
                below = backend.ground_z_m(positions[0])
                result = fly_route(backend, route, positions, yaws, policy, ground)
                result["start_agl_m"] = None if below is None else float(positions[0][2] - below)
                result["e_m"] = route_e(result)
                result["identity"] = {k: identity[k] for k in identity if k != "tokenizer"}
                result["checkpoint"] = args.checkpoint
                result, nonfinite = finite_or_none(result)
                if nonfinite:
                    result["nonfinite_fields"] = nonfinite
                write_json(out / f"{route['route_id']}.json", result)
                print(args.name, route["route_id"], {k: result[k] for k in ("SR20", "SR10", "stopped", "failure", "steps")},
                      round(result["horizontal_m"], 1), "E", None if result["e_m"] is None else round(result["e_m"], 2), flush=True)


def summarize(rows, routes):
    values = list(rows.values())
    follow = [v["e_m"] for v in values if routes[v["route_id"]]["mode"] == "follow" and v.get("e_m") is not None]

    def sr20_by(key):
        groups = sorted({str(key(routes[v["route_id"]])) for v in values})
        return {g: float(np.mean([v["SR20"] for v in values if str(key(routes[v["route_id"]])) == g])) for g in groups}
    return {"n": len(values), **{f"SR{int(r)}": float(np.mean([v[f'SR{int(r)}'] for v in values])) for r in RADII},
            "stopped": float(np.mean([v["stopped"] for v in values])), "failures": sum(bool(v["failure"]) for v in values),
            "ne_median_m": float(np.median([v["horizontal_m"] for v in values])),
            "E_follow_m": float(np.mean(follow)) if follow else None, "E_follow_n": len(follow),
            "SR20_by_map": sr20_by(map_of), "SR20_by_unseen_class": sr20_by(lambda r: r["unseen_class"]),
            "SR20_by_name_form": sr20_by(lambda r: r["name_form"]), "SR20_by_mode": sr20_by(lambda r: r["mode"])}


def report(args):
    flights = Path(args.out)
    routes = {r["route_id"]: r for r, _, _ in dev_routes(args.prepared)}
    results = {}
    for folder in sorted(p for p in flights.iterdir() if p.is_dir()):
        rows = [read_json(f) for f in sorted(folder.glob("*.json"))]
        if rows:
            results[folder.name] = {r["route_id"]: r for r in rows}
    summary = {name: summarize(rows, routes) for name, rows in results.items()}
    for name, s in summary.items():
        print(f"{name:8s} n {s['n']:3d} SR20 {s['SR20']:.3f} SR10 {s['SR10']:.3f} SR5 {s['SR5']:.3f} stopped {s['stopped']:.2f} fail {s['failures']:2d} "
              f"NE {s['ne_median_m']:.1f} m | E follow {s['E_follow_m']} (n {s['E_follow_n']}) | {s['SR20_by_map']}")
    decisions = {}
    if {"B", "B4"} <= set(summary):
        b, b4 = summary["B"], summary["B4"]
        common = sorted(set(results["B"]) & set(results["B4"]))
        summary["B4_minus_B_SR20"] = bootstrap_difference([results["B4"][k]["SR20"] for k in common], [results["B"][k]["SR20"] for k in common])
        four = b4["SR20"] > b["SR20"] and b4["E_follow_m"] <= GRID_E_TOLERANCE * b["E_follow_m"]
        decisions["grid"] = {"choice": "4x4" if four else "2x2", "B4_SR20_higher": b4["SR20"] > b["SR20"],
                             "E_ratio_B4_over_B": b4["E_follow_m"] / b["E_follow_m"], "rule": "protocol 17.9 (0047e17)"}
    gated = "B4" if decisions.get("grid", {}).get("choice") == "4x4" else "B"
    if gated in summary:
        gate = {"model": gated, "SR20 >= 0.5": summary[gated]["SR20"] >= GATE_SR20}
        if "A" in summary:
            gate["E(model) < E(A)"] = summary[gated]["E_follow_m"] < summary["A"]["E_follow_m"]
        decisions["dev_gate"] = gate
    summary["decisions"] = decisions
    print("decisions:", decisions)
    write_json(flights / args.summary, summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fly", "report"))
    p.add_argument("--prepared", default="runs/v3/stage2/prepared"); p.add_argument("--root", default="/mnt/f/WSL_ColdStorage/datasets/sr3m_v3")
    p.add_argument("--out", default="runs/v3/stage2/dev_look1"); p.add_argument("--cache", default=str(Path.home() / "sr3mxhad_runs/v3/ground"))
    p.add_argument("--name"); p.add_argument("--checkpoint"); p.add_argument("--device", default="cuda")
    p.add_argument("--summary", default="summary.json")
    args = p.parse_args()
    {"fly": fly, "report": report}[args.stage](args)
