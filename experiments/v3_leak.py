"""Stage 0 of dataset v3 (protocol section 13): does a route design give the turn away through the pose? No images, no simulator.

Fits small models on routes from the A and D maps and measures them on the B maps. v2 runs through the same test as the control.
The perception proxies stand in for vision: they show whether the data supports stable flight, not that DINOv2 can see the object. No STOP is modelled.
"""
import argparse
import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.protocol import PreparedSplit, write_json
from sr3m.geometry import wrap
from sr3m.routes_v3 import Design, FRONT_HALF_DEG, generate

UE = Path("/mnt/e/UnrealProjects")
ROOTS = {"L_MTN": UE / "mountain_base/military_mountains/Saved/VLNDataset/full-200", "L_DES": UE / "desert/Saved/VLNDataset/des-200"}
FIT_MAPS = ("L_MTN_A", "L_MTN_D", "L_DES_A", "L_DES_D")
EVAL_MAPS = ("L_MTN_B", "L_DES_B")
DESIGNS = {"pursuit": Design(), "pursuit_kick": Design(kick_every=25.), "rotate_kick": Design(kick_every=25., rotate_first=True), "pursuit_kick_often": Design(kick_every=10.),
           "kick_start": Design(kick_every=25., start_kick_steps=3), "kick_start_relief": Design(kick_every=25., start_kick_steps=3, relief_share=.6)}
WORDS = ("left", "ahead", "right")
GRID_M = 4.
COARSE_STEP = math.radians(15)
COARSE_NOISE = math.radians(3)
DOWN_VIEW_M = 30.
GATES = {"response_deg": -.1, "pose_reach": .20, "clean_sign": .70, "clean_corr": .30, "perfect_reach": .90, "coarse_reach": .70}


class Ground:
    """Bilinear height from a cached grid of the map's terrain export; NaN outside it. Fast enough for many rollouts."""

    def __init__(self, map_id, cache):
        path = Path(cache) / f"{map_id}.npz"
        if path.exists():
            d = np.load(path)
            self.x0, self.y0, self.z = float(d["x0"]), float(d["y0"]), d["z"]
            return
        root = ROOTS[map_id[:5]]
        spec = importlib.util.spec_from_file_location("ground_" + map_id, root / "scripts/terrain_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        terrain = module.Terrain(root, map_id)
        design = json.load(open(root / f"design/{map_id}.final.json", encoding="utf-8"))
        way = np.concatenate([np.array(r["waypoints_xyz_m"])[:, :2] for r in design["routes"]])
        lo, hi = way.min(0) - 400, way.max(0) + 400
        xs, ys = np.arange(lo[0], hi[0], GRID_M), np.arange(lo[1], hi[1], GRID_M)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        z = terrain.sample(np.c_[gx.ravel(), gy.ravel()], strict=False).reshape(gx.shape)
        self.x0, self.y0, self.z = float(xs[0]), float(ys[0]), z.astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x0=self.x0, y0=self.y0, z=self.z)

    def __call__(self, xy):
        xy = np.atleast_2d(xy)
        u, v = (xy[:, 0] - self.x0) / GRID_M, (xy[:, 1] - self.y0) / GRID_M
        i, j = np.floor(u).astype(int), np.floor(v).astype(int)
        inside = (i >= 0) & (j >= 0) & (i < self.z.shape[0] - 1) & (j < self.z.shape[1] - 1)
        out = np.full(len(xy), np.nan)
        i, j, a, b = i[inside], j[inside], (u - np.floor(u))[inside], (v - np.floor(v))[inside]
        out[inside] = self.z[i, j] * (1 - a) * (1 - b) + self.z[i + 1, j] * a * (1 - b) + self.z[i, j + 1] * (1 - a) * b + self.z[i + 1, j + 1] * a * b
        return out


class Batch:
    """Route rows as arrays: poses in the start frame, plus what each proxy may read."""

    def __init__(self, routes):
        self.routes = routes
        steps = [len(r["actions"]) for r in routes]
        self.index = np.repeat(np.arange(len(routes)), steps)
        self.pos = np.concatenate([r["positions"][:-1, :2] for r in routes])
        self.yaw = np.concatenate([r["yaws"][:-1] for r in routes])
        self.actions = np.concatenate([r["actions"] for r in routes])
        self.start = np.array([r["positions"][0, :2] for r in routes])[self.index]
        self.start_yaw = np.array([r["yaws"][0] for r in routes])[self.index]
        self.target = np.array([r["target"] for r in routes])[self.index]
        self.word = np.array([WORDS.index(r["word"]) for r in routes])[self.index]
        self.flown = np.concatenate([np.arange(n) * 2. for n in steps])


def pose_features(pos, yaw, start, start_yaw):
    d = pos - start
    c, s = np.cos(start_yaw), np.sin(start_yaw)
    a = yaw - start_yaw
    return np.c_[(c * d[:, 0] + s * d[:, 1]) / 100, (-s * d[:, 0] + c * d[:, 1]) / 100, np.sin(a), np.cos(a)]


def perceive(pos, yaw, target, mode, rng):
    """What vision could report. Out of both views it reports nothing; the two 90 degree views meet at 45 degrees below the horizon."""
    d = target - pos
    bearing = np.arctan2(np.sin(np.arctan2(d[:, 1], d[:, 0]) - yaw), np.cos(np.arctan2(d[:, 1], d[:, 0]) - yaw))
    distance = np.hypot(d[:, 0], d[:, 1])
    visible = (np.abs(bearing) <= math.radians(FRONT_HALF_DEG)) | (distance <= DOWN_VIEW_M)
    if mode == "coarse":
        seen = np.round((bearing + rng.normal(0, COARSE_NOISE, len(bearing))) / COARSE_STEP) * COARSE_STEP
        return np.c_[seen * visible, visible.astype(float)]
    return np.c_[bearing * visible, distance / 100 * visible, visible.astype(float)]


FEATURES = {"pose": (True, False, None), "pose+word": (True, True, None), "pose+word+perfect": (True, True, "perfect"), "pose+word+coarse": (True, True, "coarse"), "word+perfect": (False, True, "perfect"), "word+coarse": (False, True, "coarse")}


def features(name, pos, yaw, start, start_yaw, target, word, rng):
    use_pose, use_word, view = FEATURES[name]
    parts = []
    if use_pose:
        parts.append(pose_features(pos, yaw, start, start_yaw))
    if use_word:
        parts.append(np.eye(3)[word])
    if view:
        parts.append(perceive(pos, yaw, target, view, rng))
    return np.concatenate(parts, axis=1).astype(np.float32)


def fit(x, forward, yaw, device, epochs=30, hidden=256):
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 2)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    X = torch.as_tensor(x, device=device)
    Y = torch.as_tensor(np.c_[forward / 2, yaw / .05], dtype=torch.float32, device=device)  # .05 rad is the scale of a turn; both outputs near one.
    for _ in range(epochs):
        order = torch.randperm(len(X), device=device)
        for i in range(0, len(X), 1024):
            b = order[i:i + 1024]
            loss = (model(X[b]) - Y[b]).abs().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model.eval()


def predict(model, x, device):
    with torch.no_grad():
        out = model(torch.as_tensor(x, device=device)).cpu().numpy()
    return np.clip(out[:, 0], 0, 1) * 2, out[:, 1] * .05


def agreement(label, pred):
    y, q = np.degrees(label), np.degrees(pred)
    big = np.abs(y) >= 1
    return {"yaw_mae": float(np.abs(y - q).mean()), "zero_mae": float(np.abs(y).mean()), "corr": float(np.corrcoef(y, q)[0, 1]) if q.std() > 1e-6 else 0., "sign": float((np.sign(y[big]) == np.sign(q[big])).mean()) if big.any() else float("nan"), "turning_rows": int(big.sum()), "rows": len(y)}


def heading_response(model, name, batch, device, rng, degrees):
    d = math.radians(degrees)
    plus = features(name, batch.pos, batch.yaw + d, batch.start, batch.start_yaw, batch.target, batch.word, np.random.default_rng(1))
    minus = features(name, batch.pos, batch.yaw - d, batch.start, batch.start_yaw, batch.target, batch.word, np.random.default_rng(1))
    back = -(predict(model, plus, device)[1] - predict(model, minus, device)[1]) / 2
    bins = {}
    for low, high in ((0, 40), (40, 80), (80, 1000)):
        keep = (batch.flown >= low) & (batch.flown < high)
        bins[f"{low}-{high}"] = float(np.degrees(np.median(back[keep]))) if keep.any() else None
    return {"median_deg": float(np.degrees(np.median(back))), "share_back": float((back > 0).mean()), "by_flown_m": bins}


def fly(model, name, routes, device, rng, kick_deg=0., factor=1.5, class_steps=None):
    """Closed loop in the plane from each route's start. Reach: ever within 20 m, or 5 m; a route ends at its own budget.

    class_stop: where a model that cannot see would stop, flying the training routes' median steps for this class of object.
    """
    n = len(routes)
    start = np.array([r["positions"][0, :2] for r in routes])
    start_yaw = np.array([r["yaws"][0] for r in routes])
    target = np.array([r["target"] for r in routes])
    word = np.array([WORDS.index(r["word"]) for r in routes])
    expert = np.array([len(r["actions"]) for r in routes])
    budget = np.maximum((factor * expert).astype(int), 10)
    pos, yaw = start.copy(), start_yaw + math.radians(kick_deg) * rng.choice((-1, 1), n)
    best = np.linalg.norm(pos - target, axis=1)
    at_expert = best.copy()
    at_class = best.copy()
    class_stop = np.array([(class_steps or {}).get(r.get("cls"), -1) for r in routes])
    alive = np.ones(n, dtype=bool)
    for t in range(int(budget.max())):
        alive = t < budget
        forward, dyaw = predict(model, features(name, pos, yaw, start, start_yaw, target, word, rng), device)
        step = np.c_[forward * np.cos(yaw), forward * np.sin(yaw)]
        pos = pos + step * alive[:, None]
        yaw = yaw + dyaw * alive
        distance = np.linalg.norm(pos - target, axis=1)
        best = np.minimum(best, distance)
        at_expert = np.where(t + 1 == expert, distance, at_expert)
        at_class = np.where(t + 1 == class_stop, distance, at_class)
    return {"reach20": float((best <= 20).mean()), "reach5": float((best <= 5).mean()), "at_expert_steps20": float((at_expert <= 20).mean()), "best_median_m": float(np.median(best)), "class_stop20": float((at_class <= 20).mean()), "class_stop10": float((at_class <= 10).mean())}


def v2_control(prepared, device):
    """The same test on v2: pose only, fitted on train, measured on val_seen."""
    routes = {}
    for split in ("train", "val_seen"):
        ds = PreparedSplit(prepared, split)
        built = []
        for r in ds.routes:
            o, n = r["offset"], r["count"]
            action = ds.arrays["action"][o:o + n - 1].astype(float)
            if not np.isfinite(action).all():
                continue
            p = ds.arrays["position"][o:o + n].astype(float)
            built.append(dict(positions=p, yaws=ds.arrays["yaw"][o:o + n].astype(float), actions=action, target=p[-1, :2].tolist(), word="ahead", episode=r["episode_id"]))
        routes[split] = built
    train, val = Batch(routes["train"]), Batch(routes["val_seen"])
    model = fit(features("pose", train.pos, train.yaw, train.start, train.start_yaw, train.target, train.word, None), train.actions[:, 0], train.actions[:, 3], device, epochs=40)
    rng = np.random.default_rng(0)
    pred = predict(model, features("pose", val.pos, val.yaw, val.start, val.start_yaw, val.target, val.word, None), device)[1]
    closed = json.loads((Path(prepared) / "closed_loop_routes.json").read_text())["routes"]
    flown = {c["episode_id"] for c in closed}
    return {"offline": agreement(val.actions[:, 3], pred), "response10": heading_response(model, "pose", val, device, rng, 10), "response20": heading_response(model, "pose", val, device, rng, 20), "closed_all": fly(model, "pose", routes["val_seen"], device, rng), "closed_16": fly(model, "pose", [r for r in routes["val_seen"] if r["episode"] in flown], device, rng)}


def route_stats(routes, rejected):
    bearing = np.abs(np.degrees([r["bearing"] for r in routes]))
    turn = [float(np.degrees(np.abs(r["actions"][:, 3]).sum())) for r in routes]
    return {"routes": len(routes), "rejected": rejected, "steps_median": float(np.median([len(r["actions"]) for r in routes])), "steps_total": int(sum(len(r["actions"]) for r in routes)), "distance_median": float(np.median([r["distance"] for r in routes])), "bearing_median": float(np.median(bearing)),
            "turn_median_deg": float(np.median(turn)), "relief_median_m": float(np.median([r["relief_m"] for r in routes])), "relief_p90_m": float(np.percentile([r["relief_m"] for r in routes], 90)), "follow_climb_median_m": float(np.median([r["climb_m"] for r in routes if r["mode"] == "follow"] or [0.])),
            "follow": sum(r["mode"] == "follow" for r in routes), "relief15": sum(r["relief_m"] >= 15 for r in routes), "words": {w: sum(r["word"] == w for r in routes) for w in WORDS}, "objects_used": len({r["object"] for r in routes}), "classes_used": len({r["cls"] for r in routes})}


def test_design(name, design, objects, grounds, args, device):
    rng = np.random.default_rng(args.seed)
    fit_routes, eval_routes, stats = [], [], {}
    for map_id in FIT_MAPS + EVAL_MAPS:
        routes, rejected = generate([o for o in objects if o["map"] == map_id], grounds[map_id], args.fit_routes if map_id in FIT_MAPS else args.eval_routes, design, rng)
        stats[map_id] = route_stats(routes, rejected)
        (fit_routes if map_id in FIT_MAPS else eval_routes).extend(routes)
    clean = dataclasses.replace(design, kick_every=0.)
    clean_routes = []
    for map_id in EVAL_MAPS:
        clean_routes.extend(generate([o for o in objects if o["map"] == map_id], grounds[map_id], args.eval_routes, clean, rng)[0])
    train, val, on_path = Batch(fit_routes), Batch(eval_routes), Batch(clean_routes)
    classes = {}
    for r in fit_routes:
        classes.setdefault(r["cls"], []).append(len(r["actions"]))
    class_steps = {c: int(np.median(v)) for c, v in classes.items()}
    class_steps.update({r["cls"]: int(np.median([len(x["actions"]) for x in fit_routes])) for r in eval_routes if r["cls"] not in classes})
    report = {"design": dataclasses.asdict(design), "maps": stats, "models": {}}
    for model_name in FEATURES:
        x = features(model_name, train.pos, train.yaw, train.start, train.start_yaw, train.target, train.word, rng)
        model = fit(x, train.actions[:, 0], train.actions[:, 3], device, epochs=args.epochs)
        row = {"closed": fly(model, model_name, eval_routes, device, rng, class_steps=class_steps), "closed_kick20": fly(model, model_name, eval_routes, device, rng, kick_deg=20.),
               "offline_clean": agreement(on_path.actions[:, 3], predict(model, features(model_name, on_path.pos, on_path.yaw, on_path.start, on_path.start_yaw, on_path.target, on_path.word, rng), device)[1]),
               "response10": heading_response(model, model_name, on_path, device, rng, 10), "response20": heading_response(model, model_name, on_path, device, rng, 20)}
        report["models"][model_name] = row
    m = report["models"]
    report["gates"] = {"G0-1 no positive feedback": min(m["pose"]["response10"]["median_deg"], m["pose"]["response20"]["median_deg"]) >= GATES["response_deg"],
                       "G0-2 cannot guess the target": m["pose"]["closed"]["reach20"] <= GATES["pose_reach"],
                       "G0-3 cannot guess the turn": m["pose"]["offline_clean"]["sign"] <= GATES["clean_sign"] and m["pose"]["offline_clean"]["corr"] <= GATES["clean_corr"],
                       "G0-4 perfect proxy flies": min(m["pose+word+perfect"]["closed"]["reach20"], m["pose+word+perfect"]["closed_kick20"]["reach20"]) >= GATES["perfect_reach"],
                       "G0-4 coarse proxy flies": min(m["pose+word+coarse"]["closed"]["reach20"], m["pose+word+coarse"]["closed_kick20"]["reach20"]) >= GATES["coarse_reach"]}
    return report


def show(name, report):
    print(f"\n== {name}: {report['design']['kick_every']:.0f} steps per kick, rotate first {report['design']['rotate_first']}")
    print(f"  {'map':8s} {'routes':>6s} {'steps':>6s} {'total':>6s} {'dist':>5s} {'turn':>5s} {'relief':>7s} {'p90':>5s} {'follow climb':>12s} {'objects':>7s} {'classes':>7s}  rejected")
    for map_id, s in report["maps"].items():
        print(f"  {map_id:8s} {s['routes']:6d} {s['steps_median']:6.0f} {s['steps_total']:6d} {s['distance_median']:5.0f} {s['turn_median_deg']:5.0f} {s['relief_median_m']:7.0f} {s['relief_p90_m']:5.0f} {s['follow_climb_median_m']:12.0f} {s['objects_used']:7d} {s['classes_used']:7d}  {dict(sorted(s['rejected'].items(), key=lambda x: -x[1])[:4])}")
    print(f"  {'model':20s} {'reach20':>7s} {'kick20':>7s} {'reach5':>7s} {'stop@expert':>11s} {'cls stop20':>10s} {'stop10':>6s} {'best m':>7s} | {'corr':>6s} {'sign':>6s} {'yaw MAE':>7s} {'zero':>6s} | {'resp10':>7s} {'back':>5s} {'resp20':>7s} {'back':>5s} {'early 0-40m':>11s}")
    for model_name, m in report["models"].items():
        o, a, b = m["offline_clean"], m["response10"], m["response20"]
        early = a["by_flown_m"]["0-40"]
        print(f"  {model_name:20s} {m['closed']['reach20']:7.2f} {m['closed_kick20']['reach20']:7.2f} {m['closed']['reach5']:7.2f} {m['closed']['at_expert_steps20']:11.2f} {m['closed']['class_stop20']:10.2f} {m['closed']['class_stop10']:6.2f} {m['closed']['best_median_m']:7.0f} | {o['corr']:6.2f} {o['sign']:6.2f} {o['yaw_mae']:7.2f} {o['zero_mae']:6.2f} | {a['median_deg']:7.2f} {a['share_back']:5.2f} {b['median_deg']:7.2f} {b['share_back']:5.2f} {early if early is None else round(early, 2):>11}")
    print("  gates:", {k: ("pass" if v else "FAIL") for k, v in report["gates"].items()})


def main(args):
    device = args.device
    torch.manual_seed(args.seed)
    objects = json.load(open(args.objects, encoding="utf-8"))
    grounds = {m: Ground(m, args.cache) for m in FIT_MAPS + EVAL_MAPS}
    out = Path(args.out)
    results = {}
    if not args.skip_control:
        control = v2_control(args.prepared, device)
        results["v2 control"] = control
        print("== v2 control (pose only, train -> val_seen):")
        print(f"  offline corr {control['offline']['corr']:.2f}, sign {control['offline']['sign']:.2f}, yaw MAE {control['offline']['yaw_mae']:.2f} vs zero {control['offline']['zero_mae']:.2f}; response 10 deg {control['response10']['median_deg']:.2f} (back {control['response10']['share_back']:.2f}), 20 deg {control['response20']['median_deg']:.2f} (back {control['response20']['share_back']:.2f}), early {control['response10']['by_flown_m']}")
        print(f"  closed loop, the 16 flown routes: reach20 {control['closed_16']['reach20']:.2f}, at expert steps {control['closed_16']['at_expert_steps20']:.2f}, best {control['closed_16']['best_median_m']:.0f} m; all val_seen: reach20 {control['closed_all']['reach20']:.2f}, at expert steps {control['closed_all']['at_expert_steps20']:.2f}")
    for name in args.designs:
        results[name] = test_design(name, DESIGNS[name], objects, grounds, args, device)
        show(name, results[name])
        write_json(out / "leak.json", results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="runs/terrain_v2_dino/prepared"); p.add_argument("--objects", default="configs/v3_objects.json")
    p.add_argument("--cache", default=str(Path.home() / "sr3mxhad_runs/v3/ground")); p.add_argument("--out", default="runs/v3/leak")
    p.add_argument("--designs", nargs="+", default=list(DESIGNS), choices=list(DESIGNS)); p.add_argument("--fit-routes", type=int, default=200); p.add_argument("--eval-routes", type=int, default=100)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--seed", type=int, default=20261102); p.add_argument("--device", default="cuda"); p.add_argument("--skip-control", action="store_true")
    main(p.parse_args())
