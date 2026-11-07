"""Dataset v3 stage 2: the one opening of the test maps (protocol 17.10). SR3MXHAD_PROTOCOL=v3; close the GUI editor.

check:   every frozen checkpoint is complete, selected on val_seen with a ready STOP; the tree is clean.
offline: teacher-forced predictions on the test split from cached features (offline H1 and descriptions).
fly:     the K closed loop (experiments.v3s2_fly.fly with --split test) for every policy in POLICIES, in that order; resumes.
report:  T1, T2, T3, the secondary criteria and the descriptions of 17.10, judged as 17.8 says, to summary.json.

Per-route quantities (17.7), all from the flight record; z_0 is the start pose, ground_t = z_t - AGL_t with AGL from UE:
  E       mean_t |AGL_t - AGL_0| (follow routes; v3s2_fly.route_e).
  E_hold  mean_t |ground_0 - ground_t|: the E of holding z_0 along the policy's own horizontal path.
  beta    follow gain: least squares through the origin of z_t - z_0 on ground_t - ground_0 (1 follows, 0 holds; 11/07).
  hold_dz mean_t |z_t - z_0| (hold routes).
  reached the drone came within 20 m horizontally of the target at some step.
Comparisons (17.8): the three seeds are averaged inside each route first, then routes are resampled (paired bootstrap,
2000 draws, 95% percentile CI); 3-seed criteria also need every seed's point estimate to have the required sign.
"""
import argparse
from argparse import Namespace
import math
from pathlib import Path

import numpy as np

from experiments.protocol import PreparedSplit, code_version, read_features, read_json, write_json
from experiments.v3s2_fly import dev_routes, map_of

OUT = "runs/v3/stage2/test"
MODELS = "runs/v3/stage2/models"
SEEDS = (42, 43, 44)
RESAMPLES = 2000
SUCCESS_M = 20.
T3_MARGIN_M = .5
MAIN = ("B4", "A", "C4", "N", "AV", "DG")      # three seeds each; the paper's B, C are B4, C4 (17.6)
FEATURES = {"A": "features", "N": "features", "B4": "features_grid4", "C4": "features_grid4", "B40": "features_grid4",
            "DG": "features_grid4", "AV": "features_mosaic"}


def policies():
    """(flight folder, checkpoint) in run order: the main criteria first, descriptions after."""
    out = [("oracle", None)]
    for variant in MAIN:
        out += [(f"{variant}_seed{s}", f"{MODELS}/{variant}_seed{s}/best.pt") for s in SEEDS]
    out += [("B40_seed42", f"{MODELS}/B40_seed42/best.pt"),
            ("B4_frac25", "runs/v3/stage2/models_frac25/B4_seed42/best.pt"),
            ("B4_frac50", "runs/v3/stage2/models_frac50/B4_seed42/best.pt")]
    return out


def variant_of(name):
    return "B4" if name.startswith("B4_frac") else name.split("_seed")[0]


# ---------------------------------------------------------------- per-route quantities

def route_quantities(result, route, start):
    """17.7 quantities of one flight; start is the expert start position (x, y, z)."""
    z0 = float(start[2])
    steps = [h for h in result["history"] if "position_m" in h]
    out = {k: result[k] for k in ("SR20", "SR10", "SR5", "stopped", "horizontal_m")}
    out["failure"] = bool(result["failure"])
    tx, ty = route["target"][0], route["target"][1]
    out["reached"] = bool(min([math.hypot(h["position_m"][0] - tx, h["position_m"][1] - ty) for h in steps]
                              + [math.hypot(start[0] - tx, start[1] - ty)]) <= SUCCESS_M)
    if route["mode"] == "hold":
        out["hold_dz"] = float(np.mean([abs(h["position_m"][2] - z0) for h in steps])) if steps else None
        return out
    out["E"] = result.get("e_m")
    agl0 = result.get("start_agl_m")
    known = [h for h in steps if h.get("agl_m") is not None]
    if agl0 is None or not known:
        out["E_hold"] = out["beta"] = None
        return out
    ground0 = z0 - agl0
    dg = np.array([h["position_m"][2] - h["agl_m"] - ground0 for h in known])
    dz = np.array([h["position_m"][2] - z0 for h in known])
    out["E_hold"] = float(np.mean(np.abs(dg)))
    out["beta"] = float(dz @ dg / (dg @ dg)) if dg @ dg > 1e-6 else None
    return out


def route_relief(ds, route):
    """Expert-path relief for H2: max - min of the ground under the expert positions (z - AGL of the prepared rows)."""
    sl = slice(route["offset"], route["offset"] + route["count"])
    ground = ds.arrays["position"][sl, 2].astype(float) - ds.arrays["agl"][sl].astype(float)
    return float(ground.max() - ground.min())


def relief_edges(prepared):
    """H2 tertile edges of route relief, fitted on train follow routes only (as section 6 fits its slope bins on train)."""
    train = PreparedSplit(prepared, "train")
    return np.quantile([route_relief(train, r) for r in train.routes if r["mode"] == "follow"], [1/3, 2/3]).tolist()


# ---------------------------------------------------------------- statistics

def bootstrap(statistic, *columns, seed=0):
    """Point estimate and 95% percentile CI of statistic(*columns) over paired route resamples."""
    columns = [np.asarray(c, dtype=float) for c in columns]
    n = len(columns[0])
    rng = np.random.default_rng(seed)
    draws = [statistic(*[c[i] for c in columns]) for i in (rng.integers(0, n, n) for _ in range(RESAMPLES))]
    return {"estimate": float(statistic(*columns)), "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], "routes": n}


def mean_difference(a, b):
    return np.mean(a) - np.mean(b)


def relative_improvement(b, a):
    """R_cl = 1 - E(B)/E(A) (route means)."""
    return 1 - np.mean(b) / np.mean(a)


def table(flights, names, key, routes):
    """route -> values of key for each name, keeping routes where every name has a value."""
    out = {}
    for rid in routes:
        values = [flights.get(n, {}).get(rid, {}).get(key) for n in names]
        if all(v is not None for v in values):
            out[rid] = [float(v) for v in values]
    return out


def compare(flights, x, y, key, routes, statistic=mean_difference, seeds=SEEDS):
    """x vs y on key: seeds averaged inside each route, then a paired route bootstrap; per-seed point estimates too."""
    names = [f"{x}_seed{s}" for s in seeds] + [f"{y}_seed{s}" for s in seeds]
    rows = table(flights, names, key, routes)
    k = len(seeds)
    values = np.array(list(rows.values())).reshape(-1, 2 * k)
    out = bootstrap(statistic, values[:, :k].mean(1), values[:, k:].mean(1))
    out["per_seed"] = {}
    for i, s in enumerate(seeds):
        pair = table(flights, [f"{x}_seed{s}", f"{y}_seed{s}"], key, routes)
        p = np.array(list(pair.values()))
        out["per_seed"][str(s)] = float(statistic(p[:, 0], p[:, 1])) if len(p) else None
    return out


def compare_single(flights, x, y, key, routes, statistic=mean_difference):
    rows = np.array(list(table(flights, [x, y], key, routes).values()))
    return bootstrap(statistic, rows[:, 0], rows[:, 1])


def positive(result):
    """Point estimate and CI lower bound > 0, and every seed > 0."""
    return result["estimate"] > 0 and result["ci"][0] > 0 and all(v is not None and v > 0 for v in result["per_seed"].values())


# ---------------------------------------------------------------- stages

def check(args, require_clean=True):
    version = code_version()
    if require_clean and version["dirty"]:
        raise RuntimeError("The test opening runs only from a clean, committed tree")
    for name, checkpoint in policies():
        if checkpoint is None:
            continue
        folder = Path(checkpoint).parent
        selection = read_json(folder / "selection.json")
        if not (folder / "completion.json").exists() or selection["split"] != "val_seen" or not selection["stop"]["ready"]:
            raise RuntimeError(f"{name}: not completed, not selected on val_seen, or STOP not ready")
        if selection.get("val_unseen_read") or selection.get("test_read"):
            raise RuntimeError(f"{name}: selection read held-out data")
    print("check passed:", len(policies()), "policies, commit", version["commit"][:7], flush=True)
    return version


def offline(args):
    import torch
    from experiments.metrics import evaluate
    from experiments.train import predict, token_rows
    from had.model import GRIDS, VARIANTS, TerrainHAD, feature_dim
    from had.tokenizer import Tokenizer
    check(args)
    torch.set_num_threads(4)
    ds = PreparedSplit(args.prepared, "test")
    statistics = read_json(Path(args.prepared) / "statistics.json")
    tokenizer = Tokenizer.load(Path(args.prepared) / "tokenizer.json"); tokens = token_rows(ds, tokenizer)
    out = Path(args.out) / "offline"; out.mkdir(parents=True, exist_ok=True)
    for name, checkpoint in policies():
        if checkpoint is None or (out / f"{name}.json").exists():
            continue
        variant = variant_of(name)
        widths = {v: feature_dim(GRIDS.get(variant, {}).get(v, 2)) for v in VARIANTS[variant]} or None
        features, _ = read_features(Path(args.features) / FEATURES[variant], "test", ds, widths)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        identity = saved["identity"]
        if identity["prepared"] != ds.protocol or identity["statistics"] != statistics or identity["tokenizer"] != tokenizer.to_dict():
            raise ValueError(f"{name}: checkpoint data/vocabulary identity mismatch")
        model = TerrainHAD(identity["variant"], tokenizer, cached_only=True).to(args.device)
        model.load_head_state(saved["head"])
        selection = read_json(Path(checkpoint).parent / "selection.json")
        pred, probability = predict(model, ds, features, tokens, args.device)
        result = evaluate(ds, pred, statistics, probability, selection["stop"]["threshold"])
        result.update(variant=identity["variant"], seed=identity["seed"], checkpoint=checkpoint)
        write_json(out / f"{name}.json", result)
        np.savez(out / f"{name}_predictions.npz", action=pred, stop_probability=probability)
        print("offline", name, "dz MAE follow", result["terrain"]["follow/all"].get("dz_mae_m"), flush=True)
        del model, saved
        torch.cuda.empty_cache()


def fly(args):
    from experiments.v3s2_fly import fly as fly_policy
    check(args)
    for name, checkpoint in policies():
        print("== test flights:", name, flush=True)
        fly_policy(Namespace(split="test", name=name, checkpoint=checkpoint, out=str(Path(args.out) / "flights"),
                             prepared=args.prepared, root=args.root, cache=args.cache, device=args.device))


def load_flights(args, routes):
    flights = {}
    for folder in sorted(p for p in (Path(args.out) / "flights").iterdir() if p.is_dir()):
        rows = {}
        for f in sorted(folder.glob("*.json")):
            result = read_json(f)
            route, start = routes[result["route_id"]]
            rows[result["route_id"]] = route_quantities(result, route, start)
        if rows:
            flights[folder.name] = rows
    return flights


def describe(rows, routes):
    def mean(key, ids=None):
        values = [v[key] for rid, v in rows.items() if v.get(key) is not None and (ids is None or rid in ids)]
        return float(np.mean(values)) if values else None

    def by(field):
        groups = {}
        for rid in rows:
            groups.setdefault(str(field(routes[rid][0])), set()).add(rid)
        return {g: mean("SR20", ids) for g, ids in sorted(groups.items())}
    betas = [v["beta"] for v in rows.values() if v.get("beta") is not None]
    return {"n": len(rows), **{k: mean(k) for k in ("SR20", "SR10", "SR5", "stopped", "reached", "failure", "E", "E_hold", "hold_dz")},
            "ne_median_m": float(np.median([v["horizontal_m"] for v in rows.values()])),
            "beta_median": float(np.median(betas)) if betas else None, "beta_mean": float(np.mean(betas)) if betas else None,
            "SR20_by_map": by(map_of), "SR20_by_mode": by(lambda r: r["mode"]),
            "SR20_by_unseen_class": by(lambda r: r["unseen_class"]), "SR20_by_name_form": by(lambda r: r["name_form"])}


def offline_h1(args, follow, seeds=SEEDS):
    """Offline H1 (17.7, section 6 definition): 1 - dzMAE(B)/dzMAE(A) on follow routes, route means, teacher forcing."""
    folder = Path(args.out) / "offline"
    if not all((folder / f"{v}_seed{s}.json").exists() for v in ("B4", "A") for s in seeds):
        return None
    by_route = {f"{v}_seed{s}": read_json(folder / f"{v}_seed{s}.json")["by_route"] for v in ("B4", "A", "C4") for s in seeds
                if (folder / f"{v}_seed{s}.json").exists()}
    episodes = {r["route_id"]: r["episode_id"] for r, _ in follow.values()}
    flights = {name: {rid: {"dz": rows[episodes[rid]].get("dz_mae_m")} for rid in follow} for name, rows in by_route.items()}
    out = {"H1": compare(flights, "B4", "A", "dz", list(follow), relative_improvement)}
    out["H1"]["pass"] = positive(out["H1"])
    if all(f"C4_seed{s}" in flights for s in seeds):
        out["S2_offline"] = compare(flights, "C4", "A", "dz", list(follow), relative_improvement)
    return out


def report(args):
    split = getattr(args, "split", "test")  # dev only to exercise the report on dev flights
    routes = {r["route_id"]: (r, p[0]) for r, p, _ in dev_routes(args.prepared, split)}
    flights = load_flights(args, routes)
    every = sorted(routes)
    follow = {rid: v for rid, v in routes.items() if v[0]["mode"] == "follow"}
    hold = [rid for rid, v in routes.items() if v[0]["mode"] == "hold"]
    missing = {name: len(routes) - len(flights.get(name, {})) for name, _ in policies() if len(flights.get(name, {})) != len(routes)}
    summary = {"code": code_version(), "missing_routes": missing, "policies": {n: describe(rows, routes) for n, rows in flights.items()}}
    for v in MAIN:
        names = [f"{v}_seed{s}" for s in SEEDS]
        if all(n in flights for n in names):
            summary["policies"][f"{v}_mean_of_seeds"] = {k: float(np.mean([summary["policies"][n][k] for n in names]))
                                                         for k in ("SR20", "SR10", "SR5", "reached", "E", "E_hold", "hold_dz")}
    if missing:
        print("incomplete flights; criteria need every route of every policy:", missing)
        write_json(Path(args.out) / "summary.json", summary)
        return
    t1 = compare(flights, "B4", "A", "E", list(follow), relative_improvement)
    t2 = compare(flights, "B4", "A", "SR20", every)
    t3 = compare(flights, "B4", "A", "hold_dz", hold)
    t1["pass"], t2["pass"] = positive(t1), positive(t2)
    t3["pass"] = t3["ci"][1] < T3_MARGIN_M
    conclusion = {(True, True): "down view helps reaching and terrain following: RA-L + IROS",
                  (True, False): "claim limited to terrain following",
                  (False, True): "down view helps reaching; terrain following not supported, E reported as a benchmark finding",
                  (False, False): "benchmark paper per the research plan; target Drones"}[(t1["pass"], t2["pass"])]
    summary["main"] = {"T1_R_cl": t1, "T2_SR20_B_minus_A": t2, "T3_hold_dz_B_minus_A": t3, "conclusion": conclusion,
                       "harm": not t3["pass"]}
    rows = table(flights, [f"B4_seed{s}" for s in SEEDS], "E", list(follow))
    rows_hold = table(flights, [f"B4_seed{s}" for s in SEEDS], "E_hold", list(follow))
    common = sorted(set(rows) & set(rows_hold))
    hold_gap = bootstrap(mean_difference, [np.mean(rows[r]) for r in common], [np.mean(rows_hold[r]) for r in common])
    hold_gap["better_than_hold"] = hold_gap["ci"][1] < 0
    summary["secondary"] = {
        "B_E_minus_E_hold": hold_gap,
        "S2_R_cl_C_vs_A": compare(flights, "C4", "A", "E", list(follow), relative_improvement),
        "over_N": {f"{v}_minus_N": {"SR20": compare(flights, v, "N", "SR20", every), "E": compare(flights, v, "N", "E", list(follow))}
                   for v in ("B4", "A", "C4")},
        "offline": offline_h1(args, follow)}
    summary["secondary"]["S2_R_cl_C_vs_A"]["pass"] = positive(summary["secondary"]["S2_R_cl_C_vs_A"])
    edges = relief_edges(args.prepared)
    test = PreparedSplit(args.prepared, split)
    relief = {r["route_id"]: route_relief(test, r) for r in test.routes if r["mode"] == "follow"}
    bins = {name: [rid for rid in follow if lo < relief[rid] <= hi]
            for name, lo, hi in zip(("low", "medium", "high"), [-math.inf, *edges], [*edges, math.inf])}
    summary["descriptions"] = {
        "expert": summary["policies"].get("oracle"),
        "H2_relief": {"edges_m_fit_on_train": edges, "routes": {k: len(v) for k, v in bins.items()},
                      "R_cl_by_bin": {k: compare(flights, "B4", "A", "E", v, relative_improvement) for k, v in bins.items() if len(v) > 1}},
        "data_curve": {name: {k: summary["policies"][name][k] for k in ("SR20", "SR10", "E", "hold_dz")}
                       for name in ("B4_frac25", "B4_frac50", "B4_seed42")},
        "B0": {"SR20_B0_minus_B": compare_single(flights, "B40_seed42", "B4_seed42", "SR20", every),
               "E_B0_minus_B": compare_single(flights, "B40_seed42", "B4_seed42", "E", list(follow))},
        "D4": {f"B_minus_{v}": {"SR20": compare(flights, "B4", v, "SR20", every), "E": compare(flights, "B4", v, "E", list(follow))}
               for v in ("AV", "DG")}}
    write_json(Path(args.out) / "summary.json", summary)
    m = summary["main"]
    print(f"T1 R_cl {t1['estimate']:.3f} CI {t1['ci']} seeds {t1['per_seed']} pass {t1['pass']}")
    print(f"T2 SR20 B-A {t2['estimate']:.3f} CI {t2['ci']} seeds {t2['per_seed']} pass {t2['pass']}")
    print(f"T3 hold |z-z0| B-A {t3['estimate']:.3f} CI {t3['ci']} pass {t3['pass']}")
    print("conclusion:", m["conclusion"], "| harm" if m["harm"] else "")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "offline", "fly", "report"))
    p.add_argument("--prepared", default="runs/v3/stage2/prepared"); p.add_argument("--root", default="/mnt/f/WSL_ColdStorage/datasets/sr3m_v3")
    p.add_argument("--out", default=OUT); p.add_argument("--cache", default=str(Path.home() / "sr3mxhad_runs/v3/ground"))
    p.add_argument("--features", default=str(Path.home() / "sr3mxhad_runs/v3"), help="the folder holding features, features_grid4, features_mosaic")
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", choices=("test", "dev"), default="test", help="report only: dev exercises the report on dev flights")
    args = p.parse_args()
    {"check": check, "offline": offline, "fly": fly, "report": report}[args.stage](args)
