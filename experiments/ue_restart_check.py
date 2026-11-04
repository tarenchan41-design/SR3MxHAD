"""UE restart overhead of the K renderer (runs/infra/ue_restart/brief.md). Close the GUI editor; writes only under --out.

curve:       one UE session, shuffled poses from v3 pilot train routes on one map, memory every 8 renders; optional GC or
             console clean-up between renders. Stops at --stop-commit percent Windows commit charge.
consistency: renders 32 shuffled poses on each of 8 train routes per map, as the reference group (old policy, restart every
             32) or the test group (memory policy, after a warm-up of at least 200 renders in the same session).
report:      G1 of protocol section 7 between the two groups with the v3 pilot A/B seed 42 models (needs transformers).
throughput:  the same full routes checked (collision sweeps, ground) and rendered, old policy against the memory policy.
Training maps only (L_MTN_A, L_DES_A); never the dev or unseen maps.
"""
import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/f/WSL_ColdStorage/datasets/sr3m_v3_pilot")
PREPARED = Path("/mnt/e/DL/Transformers/Projects/SR3MxHAD/runs/v3_pilot/prepared")
MAPS = ("L_MTN_A", "L_DES_A")
CONSISTENCY_ROUTES, CONSISTENCY_POSES, WARMUP = 8, 32, 200
SAMPLE_EVERY = 8
ABORT_COMMIT = 90.
CLOSE_COMMIT = 72.


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    temporary.replace(path)


def train_routes(map_id):
    return [r for r in read_json(PREPARED / "train.json") if r["route_id"].startswith(map_id)]


def route_file(map_id, route_id):
    return read_json(ROOT / "routes" / map_id / route_id / "route.json")


def split_routes(map_id):
    """Fixed, seeded: 8 routes for the consistency check, the rest for curves and warm-ups."""
    routes = sorted(r["route_id"] for r in train_routes(map_id))
    long_enough = sorted(r["route_id"] for r in train_routes(map_id) if r["count"] >= 13 + CONSISTENCY_POSES + 1)
    order = np.random.default_rng(20261103).permutation(len(long_enough))
    chosen = sorted(long_enough[i] for i in order[:CONSISTENCY_ROUTES])
    return chosen, [r for r in routes if r not in chosen]


def curve_poses(map_id, count, seed):
    """Shuffled poses from the non-consistency routes (at least 10 routes)."""
    _, rest = split_routes(map_id)
    rng = np.random.default_rng(seed)
    pool = []
    for route_id in rest:
        r = route_file(map_id, route_id)
        pool += [(route_id, t, r["positions"][t], r["yaws"][t]) for t in range(len(r["positions"]))]
    picked = [pool[i] for i in rng.permutation(len(pool))[:count]]
    assert len({p[0] for p in picked}) >= min(10, count)
    return picked


def capture_of(map_id):
    r = route_file(map_id, split_routes(map_id)[1][0])
    return r["profile"], r["map_package"], r["capture"]


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return float(out.split()[0])
    except Exception:
        return None


def sample(backend, extra):
    m = backend.call("memory")
    row = {"renders": backend.renders, "since_start": backend.since_start, "process": backend.starts, "private_gb": m["private_bytes"] / 2**30,
           "working_set_gb": m["working_set_bytes"] / 2**30, "commit_percent": m["commit_percent"], "commit_total_gb": m["commit_total_bytes"] / 2**30,
           "commit_limit_gb": m["commit_limit_bytes"] / 2**30, "physical_available_gb": m["physical_available_bytes"] / 2**30, "gpu_used_mib": gpu_used_mib(),
           "time": time.time(), **extra}
    return row


def curve(args):
    from sim_k.backend import UEBackend
    out = Path(args.out) / f"curve_{args.map}_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    profile, map_package, capture = capture_of(args.map)
    capture = {**capture, **json.loads(args.capture_override)}  # e.g. {"persistent_state": false}; changes pixels, see --keep-images
    poses = curve_poses(args.map, args.renders, args.seed)
    if args.order == "same":  # diagnostic: one pose again and again, so nothing new comes into view
        poses = [poses[0]] * args.renders
    elif args.order == "sequential":  # whole routes in flight order, as captures and flights visit them
        _, rest = split_routes(args.map)
        poses = [(route_id, t, p, y) for route_id in rest for t, (p, y) in enumerate(zip(route_file(args.map, route_id)["positions"], route_file(args.map, route_id)["yaws"]))][:args.renders]
    rows, render_seconds, gc_seconds, stopped = [], [], [], None
    started = time.monotonic()
    with UEBackend(out / "session", profile, map_package, capture, restart_every=10 ** 9, keep_images=bool(args.keep_images)) as backend:
        backend.start()
        ready = time.monotonic() - started
        for command in args.startup_console:  # set before the first capture, e.g. r.RDG.TransientAllocator 0
            backend.call("console", command=command)
        kept = []
        rows.append(sample(backend, {"render_s_mean8": None, "cleanup_s": None}))
        print(json.dumps(rows[-1]), flush=True)
        for i, (route_id, t, position, yaw) in enumerate(poses):
            t0 = time.monotonic()
            images = backend.render(np.array(position), float(yaw))
            render_seconds.append(time.monotonic() - t0)
            if i < args.keep_images:
                kept.append({"route_id": route_id, "step": t, "render": backend.renders - 1, "files": {v: str(backend.images / f"render_{backend.renders - 1:06d}_{v}.png") for v in images}})
            if backend.since_start in args.dump_at:  # resource dumps go to the session's Unreal.log
                for command in ("r.DumpRenderTargetPoolMemory", "rhi.DumpMemory", "rhi.DumpResourceMemory"):
                    backend.call("console", command=command)
            cleanup = None
            if args.every and backend.since_start % args.every == 0:
                t1 = time.monotonic()
                if args.gc:
                    backend.call("collect_garbage")
                if args.barrier:
                    backend.call("barrier")
                for command in args.console:
                    backend.call("console", command=command)
                cleanup = time.monotonic() - t1
                gc_seconds.append(cleanup)
            close_to_stop = rows[-1]["commit_percent"] >= CLOSE_COMMIT  # 11/03: one step of 8 renders went from 82 % to 89 %
            if backend.since_start % SAMPLE_EVERY == 0 or close_to_stop:
                rows.append(sample(backend, {"render_s_mean8": float(np.mean(render_seconds[-SAMPLE_EVERY:])), "cleanup_s": cleanup}))
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
                if rows[-1]["commit_percent"] >= args.stop_commit:
                    stopped = f"commit {rows[-1]['commit_percent']:.1f}% >= {args.stop_commit}"
                    break
        exits = backend.unexpected_exits
        processes = backend.starts
    with open(out / "memory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    renders = len(render_seconds)
    private = [r["private_gb"] for r in rows]
    summary = {"map": args.map, "name": args.name, "gc": args.gc, "console": args.console, "every": args.every, "renders": renders, "stopped": stopped,
               "start_to_ready_s": ready, "render_s_median": float(np.median(render_seconds)), "render_s_first32_median": float(np.median(render_seconds[:32])),
               "render_s_last32_median": float(np.median(render_seconds[-32:])), "cleanup_s_median": float(np.median(gc_seconds)) if gc_seconds else None,
               "private_gb_ready": private[0], "private_gb_end": private[-1], "commit_percent_ready": rows[0]["commit_percent"], "commit_percent_end": rows[-1]["commit_percent"],
               "unexpected_exits": exits, "processes": processes, "routes": len({p[0] for p in poses[:renders]}), "order": args.order}
    if args.keep_images:
        write_json(out / "kept.json", kept)
    summary.update(capture_override=json.loads(args.capture_override), startup_console=args.startup_console, barrier=args.barrier,
                   private_mb_per_render=1024 * (private[-1] - private[0]) / max(renders, 1),
                   gpu_mib_per_render_first40=next(((r["gpu_used_mib"] - rows[0]["gpu_used_mib"]) / r["renders"] for r in rows if r["renders"] >= 40 and r["gpu_used_mib"] and rows[0]["gpu_used_mib"]), None))
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def pixels(args):
    """Kept renders of a curve run against the dataset images at the same poses: MAE per view (0-255), for whether a change moves pixels."""
    from PIL import Image
    kept = read_json(Path(args.out) / f"curve_{args.map}_{args.name}" / "kept.json")
    rows = []
    for k in kept:
        folder = ROOT / "routes" / args.map / k["route_id"] / f"step_{k['step']:06d}"
        row = {"route_id": k["route_id"], "step": k["step"], "render": k["render"]}
        for view, path in k["files"].items():
            a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            b = np.asarray(Image.open(folder / f"{view}_rgb.png").convert("RGB"), dtype=np.float32)
            row[f"{view}_mae"] = float(np.abs(a - b).mean())
        rows.append(row)
    out = {"map": args.map, "name": args.name, "n": len(rows), **{f"{v}_mae_median": float(np.median([r[f'{v}_mae'] for r in rows])) for v in ("front", "down")},
           **{f"{v}_mae_max": float(np.max([r[f'{v}_mae'] for r in rows])) for v in ("front", "down")}, "rows": rows}
    write_json(Path(args.out) / f"curve_{args.map}_{args.name}" / "pixels.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)


def consistency_poses(map_id):
    """Per route 32 poses after step 13, shuffled per route with the G1 generator; the test group reshuffles all of them."""
    chosen, _ = split_routes(map_id)
    rng = np.random.default_rng(0)
    out = []
    for route_id in chosen:
        r = route_file(map_id, route_id)
        n = len(r["positions"]) - 1  # the last pose has no action label
        start = 13
        order = [int(i) + start for i in rng.permutation(CONSISTENCY_POSES)]
        out += [(route_id, t, r["positions"][t], r["yaws"][t]) for t in order]
    return out


def consistency(args):
    from sim_k.backend import MemoryPolicy, UEBackend
    for map_id in args.maps:
        out = Path(args.out) / f"consistency_{args.group}" / map_id
        if (out / "done.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        profile, map_package, capture = capture_of(map_id)
        poses = consistency_poses(map_id)
        warmup = []
        if args.group == "test":
            poses = [poses[i] for i in np.random.default_rng(1).permutation(len(poses))]
            warmup = curve_poses(map_id, WARMUP, 7)
            policy = MemoryPolicy(**json.loads(args.policy))
            backend = UEBackend(out / "session", profile, map_package, capture, policy=policy)
        else:
            backend = UEBackend(out / "session", profile, map_package, capture, restart_every=32)
        log = []
        with backend:
            for route_id, t, position, yaw in warmup:
                backend.render(np.array(position), float(yaw))
            for route_id, t, position, yaw in poses:
                t0 = time.monotonic()
                images = backend.render(np.array(position), float(yaw))
                for view, image in images.items():
                    (out / route_id).mkdir(exist_ok=True)
                    image.save(out / route_id / f"step_{t:06d}_{view}.png")
                log.append({"route_id": route_id, "step": t, "render": backend.renders, "since_start": backend.since_start, "process": backend.starts, "seconds": time.monotonic() - t0})
        write_json(out / "done.json", {"group": args.group, "policy": args.policy if args.group == "test" else "restart_every=32", "warmup": len(warmup), "poses": log,
                                       "processes": backend.starts, "unexpected_exits": backend.unexpected_exits, "memory": getattr(backend, "memory_log", [])})
        print(map_id, args.group, "processes", backend.starts, "since_start min", min(p["since_start"] for p in log), "exits", backend.unexpected_exits, flush=True)


def report(args):
    """G1 between the groups (protocol section 7, thresholds unchanged), and each group against the dataset images."""
    import torch
    from PIL import Image
    from had.model import DinoFeatures, TerrainHAD, image_transform
    from had.tokenizer import Tokenizer
    device = args.device
    labels = np.load(PREPARED / "train.npz")
    routes = {r["route_id"]: r for r in read_json(PREPARED / "train.json")}
    backbone, transform = DinoFeatures().to(device), image_transform()

    @torch.inference_mode()
    def features(paths):
        return torch.cat([backbone(torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i + 16]]).to(device)).float() for i in range(0, len(paths), 16)])

    models = {}
    for variant in ("A", "B"):
        saved = torch.load(Path(args.models) / f"{variant}_seed42/best.pt", map_location="cpu", weights_only=True)
        tokenizer = Tokenizer.from_dict(saved["identity"]["tokenizer"])
        model = TerrainHAD(saved["identity"]["variant"], tokenizer, cached_only=True).to(device).eval()
        model.load_head_state(saved["head"])
        models[variant] = (model, tokenizer)
    sources = {"ref": lambda m, r, t, v: Path(args.out) / "consistency_ref" / m / r / f"step_{t:06d}_{v}.png",
               "test": lambda m, r, t, v: Path(args.out) / "consistency_test" / m / r / f"step_{t:06d}_{v}.png",
               "dataset": lambda m, r, t, v: ROOT / "routes" / m / r / f"step_{t:06d}" / f"{v}_rgb.png"}
    pairs = [("ref", "test"), ("dataset", "ref"), ("dataset", "test")]
    action, dz = {}, 2
    per_route, order = {}, []
    for map_id in MAPS:
        by_route = {}
        for route_id, t, _, _ in consistency_poses(map_id):
            by_route.setdefault(route_id, []).append(t)
        for route_id, steps in by_route.items():
            route = routes[route_id]
            index = [route["offset"] + s for s in steps]
            keep = [i for i, k in enumerate(index) if labels["action_valid"][k] and np.isfinite(labels["action"][k]).all()]
            steps, index = [steps[i] for i in keep], [index[i] for i in keep]
            paths = {s: {v: [str(sources[s](map_id, route_id, t, v)) for t in steps] for v in ("front", "down")} for s in sources}
            feats = {s: {v: features(paths[s][v]) for v in ("front", "down")} for s in sources}
            states = torch.tensor(labels["state"][index], dtype=torch.float32, device=device)
            row = {"map": map_id, "poses": len(steps)}
            for a, b in pairs:
                row[f"{a}-{b}"] = {"mae": {v: float(np.mean([np.abs(np.asarray(Image.open(x).convert("RGB"), dtype=float) - np.asarray(Image.open(y).convert("RGB"), dtype=float)).mean()
                                                              for x, y in zip(paths[a][v], paths[b][v])])) for v in ("front", "down")},
                                   "cosine_min": {v: float(torch.nn.functional.cosine_similarity(feats[a][v], feats[b][v]).min()) for v in ("front", "down")},
                                   "cosine_median": {v: float(torch.nn.functional.cosine_similarity(feats[a][v], feats[b][v]).median()) for v in ("front", "down")}}
            for variant, (model, tokenizer) in models.items():
                tokens = torch.tensor([tokenizer.encode(route["instruction"])["input_ids"]] * len(steps), device=device)
                for s in sources:
                    with torch.inference_mode():
                        out = model({v: feats[s][v] for v in model.views}, tokens, states)["action"].cpu().numpy()
                    action.setdefault((variant, s), []).append(out)
            action.setdefault("label", []).append(labels["action"][index])
            order += [(route_id, t) for t in steps]
            per_route[route_id] = row
            print(route_id, json.dumps({p: {k: {v: round(x, 3) for v, x in d.items()} for k, d in row[p].items()} for p in ("ref-test",)}), flush=True)
    action = {k: np.concatenate(v) for k, v in action.items()}
    label = action.pop("label")
    result = {"thresholds": {"median_abs_dz_change_m": .024, "gap_change_m": .024}, "poses": int(len(label)), "pairs": {}, "routes": per_route}
    for a, b in pairs:
        change = {v: np.abs(action[v, a][:, dz] - action[v, b][:, dz]) for v in models}
        gap = {s: float(np.mean(np.abs(action["B", s][:, dz] - label[:, dz])) - np.mean(np.abs(action["A", s][:, dz] - label[:, dz]))) for s in (a, b)}
        yaw = {v: np.abs(action[v, a][:, 3] - action[v, b][:, 3]) for v in models}
        entry = {"median_abs_dz_change_m": {v: float(np.median(change[v])) for v in change}, "p90_abs_dz_change_m": {v: float(np.percentile(change[v], 90)) for v in change},
                 "max_abs_dz_change_m": {v: float(change[v].max()) for v in change},
                 "gap_m": gap, "gap_change_m": gap[b] - gap[a],
                 "median_abs_yaw_change_deg": {v: float(np.degrees(np.median(yaw[v]))) for v in yaw}, "p90_abs_yaw_change_deg": {v: float(np.degrees(np.percentile(yaw[v], 90))) for v in yaw},
                 "mae": {v: float(np.mean([r[f"{a}-{b}"]["mae"][v] for r in per_route.values()])) for v in ("front", "down")},
                 "cosine_min": {v: float(min(r[f"{a}-{b}"]["cosine_min"][v] for r in per_route.values())) for v in ("front", "down")},
                 "cosine_median_of_routes": {v: float(np.median([r[f"{a}-{b}"]["cosine_median"][v] for r in per_route.values()])) for v in ("front", "down")}}
        entry["by_map_B_median_abs_dz_change_m"] = {}
        maps = np.array([r["map"] for r in per_route.values() for _ in range(r["poses"])])
        for m in MAPS:
            entry["by_map_B_median_abs_dz_change_m"][m] = float(np.median(change["B"][maps == m]))
        if "test" in (a, b):  # where in its UE process each test pose was rendered
            since = {}
            for m in MAPS:
                for pose in read_json(Path(args.out) / "consistency_test" / m / "done.json")["poses"]:
                    since[pose["route_id"], pose["step"]] = pose["since_start"]
            depth = np.array([since[k] for k in order])
            entry["test_since_start"] = {"min": int(depth.min()), "median": float(np.median(depth)), "max": int(depth.max())}
            entry["B_median_abs_dz_change_m_by_since_start"] = {f"{lo}-{hi}": {"n": int(((depth >= lo) & (depth <= hi)).sum()), "median": float(np.median(change["B"][(depth >= lo) & (depth <= hi)])) if ((depth >= lo) & (depth <= hi)).any() else None}
                                                                for lo, hi in ((1, 32), (33, 48), (49, 1000))}
        entry["passed"] = bool(entry["median_abs_dz_change_m"]["B"] < .024 and abs(entry["gap_change_m"]) < .024)
        result["pairs"][f"{a}-{b}"] = entry
        print(f"{a}-{b}", json.dumps({k: entry[k] for k in ("passed", "median_abs_dz_change_m", "p90_abs_dz_change_m", "gap_change_m", "median_abs_yaw_change_deg", "mae", "cosine_min")}), flush=True)
    write_json(Path(args.out) / "consistency.json", result)


def throughput(args):
    """Full routes as the v3 capture does them: UE checks (sight line, collision sweeps, ground), then every pose rendered."""
    from experiments.v3_capture import check
    from sim_k.backend import MemoryPolicy, UEBackend
    map_id = args.maps[0]
    _, rest = split_routes(map_id)
    chosen = rest[:3]
    out = Path(args.out) / f"throughput_{args.group}"
    out.mkdir(parents=True, exist_ok=True)
    profile, map_package, capture = capture_of(map_id)
    backend = UEBackend(out / "session", profile, map_package, capture, policy=MemoryPolicy(**json.loads(args.policy))) if args.group == "new" else \
        UEBackend(out / "session", profile, map_package, capture, restart_every=32)
    started, poses, rows = time.monotonic(), 0, []
    with backend:
        for route_id in chosen:
            route = route_file(map_id, route_id)
            t0 = time.monotonic()
            reason = check(backend, route)
            t1 = time.monotonic()
            for p, y in zip(route["positions"], route["yaws"]):
                backend.render(np.array(p), float(y))
            poses += len(route["positions"])
            rows.append({"route_id": route_id, "poses": len(route["positions"]), "check_s": t1 - t0, "render_s": time.monotonic() - t1, "check": reason})
            print(json.dumps(rows[-1]), flush=True)
        backend.close()
    total = time.monotonic() - started
    result = {"group": args.group, "policy": args.policy if args.group == "new" else "restart_every=32", "map": map_id, "routes": rows, "poses": poses, "total_s": total,
              "s_per_pose": total / poses, "processes": backend.starts, "unexpected_exits": backend.unexpected_exits, "memory": getattr(backend, "memory_log", [])}
    write_json(out / "throughput.json", result)
    print(json.dumps({k: result[k] for k in ("group", "poses", "total_s", "s_per_pose", "processes", "unexpected_exits")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("curve", "consistency", "report", "throughput", "pixels"))
    p.add_argument("--capture-override", default="{}"); p.add_argument("--startup-console", nargs="*", default=[]); p.add_argument("--barrier", action="store_true")
    p.add_argument("--keep-images", type=int, default=0, help="curve: keep the first N renders for the pixels stage"); p.add_argument("--dump-at", nargs="*", type=int, default=[])
    p.add_argument("--out", default="/mnt/e/DL/Transformers/Projects/SR3MxHAD/runs/infra/ue_restart")
    p.add_argument("--map", choices=MAPS); p.add_argument("--maps", nargs="+", default=list(MAPS), choices=MAPS)
    p.add_argument("--name", default="baseline"); p.add_argument("--renders", type=int, default=400); p.add_argument("--seed", type=int, default=3)
    p.add_argument("--stop-commit", type=float, default=85.); p.add_argument("--order", choices=("shuffled", "same", "sequential"), default="shuffled")
    p.add_argument("--gc", action="store_true"); p.add_argument("--console", nargs="*", default=[]); p.add_argument("--every", type=int, default=0)
    p.add_argument("--group", choices=("ref", "test", "old", "new"))
    p.add_argument("--policy", default="{}", help="MemoryPolicy fields as JSON")
    p.add_argument("--models", default="/mnt/e/DL/Transformers/Projects/SR3MxHAD/runs/v3_pilot/models"); p.add_argument("--device", default="cuda")
    args = p.parse_args()
    {"curve": curve, "consistency": consistency, "report": report, "throughput": throughput, "pixels": pixels}[args.stage](args)
