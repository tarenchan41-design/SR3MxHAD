"""K's G1 and the preview check for dataset v3 stage 2 (protocol sections 7 and 17.5), before the bulk render.

The stage 2 capture renders with the same UEBackend as K, so G1 here asks whether a pose rendered the way the capture renders it
(a fresh process, every route from step 0 in plan order, restarts by the memory policy of section 17.5) and the way K renders it
(another process, routes in another order, a shuffled window of 32 poses) give the same policy output. Criterion of section 7,
unchanged: the median |dz change| of B < 2.4 cm and the change of the B - A dz error gap < 2.4 cm (A, B x 3 seeds, the v2 DINOv2
models as in the desert G1 of section 14). Judged per map; the pooled numbers are reported too. On the test maps only changes
are written, never A's or B's own errors.

render:  per map, process 1 (capture way): UE checks of the plan candidates in order (as the bulk render will do), the 6 G1 routes
         in full, then the preview: 12 more routes, checked, start and end poses rendered (24 poses, front and down).
         Process 2 (K way): the G1 windows, shuffled. VSM off in both. Resumes per map and part. Close the GUI editor first.
report:  G1 per map and pooled (needs PYTHONPATH with transformers, no simulator), preview contact sheets, UE check tallies.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.protocol import code_version, read_json, write_json
from experiments.v3_ue_check import CAPTURE, PROFILE, VSM_OFF, maps

G1 = {"median_abs_dz_change_m": .024, "gap_change_m": .024, "start": 13, "poses": 32, "routes_per_mode": 3, "seeds": (42, 43, 44)}
PREVIEW_ROUTES = 12
DZ = 2
TEST_MAPS = ("L_MTR_C", "L_DES_C_v3")


def window(rng):
    """The G1 steps in the order K renders them: 32 consecutive steps from step 13, shuffled."""
    return [int(s) for s in rng.permutation(G1["poses"]) + G1["start"]]


def select(candidates, passes, need=G1["routes_per_mode"], previews=PREVIEW_ROUTES):
    """G1 routes (need per height mode, long enough for the window) and preview routes, all on distinct targets where possible.

    Candidates are taken in plan order; passes(route) runs the UE check and returns None or the reason it fails, as the bulk render
    would skip that route. A second pass allows repeated targets if the first falls short. Returns (g1, preview, checks) where
    checks maps route id -> "accepted" or the reason.
    """
    g1, preview, checks, used = [], [], {}, set()
    counts = {"follow": 0, "hold": 0}
    full = lambda: sum(counts.values()) == 2 * need and len(preview) >= previews
    for distinct in (True, False):
        for route in candidates:
            if full():
                return g1, preview, checks
            if route["id"] in checks or (distinct and route["object"] in used):
                continue
            wants_g1 = counts[route["mode"]] < need and len(route["positions"]) >= G1["start"] + G1["poses"] + 1
            if not wants_g1 and len(preview) >= previews:
                continue
            reason = passes(route)
            checks[route["id"]] = reason or "accepted"
            if reason:
                continue
            used.add(route["object"])
            if wants_g1:
                g1.append(route)
                counts[route["mode"]] += 1
            else:
                preview.append(route)
    return g1, preview, checks


def arrays(route):
    return {**route, "positions": np.asarray(route["positions"], dtype=float), "yaws": np.asarray(route["yaws"], dtype=float), "actions": np.asarray(route["actions"], dtype=float)}


def render_map(args, map_id, plan):
    from experiments.v3_capture import check
    from sim_k.backend import MemoryPolicy, UEBackend
    package = maps()[map_id][0]
    folder = Path(args.out) / map_id
    folder.mkdir(parents=True, exist_ok=True)
    entry = plan["maps"][map_id]
    candidates = [arrays(r) for r in entry["candidates"]]
    rng = np.random.default_rng(args.seed)
    windows = {r["id"]: window(rng) for r in candidates}  # fixed per route before any rendering
    selection_file = folder / "selection.json"
    if not (folder / "capture_done.json").exists():
        with UEBackend(folder / "session_capture", PROFILE, package, CAPTURE, policy=MemoryPolicy(), console=VSM_OFF) as backend:
            g1, preview, checks = select(candidates, lambda r: check(backend, r))
            write_json(selection_file, {"g1": [r["id"] for r in g1], "preview": [r["id"] for r in preview], "checks": checks, "windows": {r["id"]: windows[r["id"]] for r in g1}})
            for route in g1:
                keep = set(windows[route["id"]])
                for t in range(len(route["positions"])):  # the whole route, as the capture renders it
                    images = backend.render(route["positions"][t], float(route["yaws"][t]))
                    if t in keep:
                        for view, image in images.items():
                            image.save(folder / f"capture_{route['id']}_{t:03d}_{view}.png")
                print(map_id, "capture", route["id"], len(route["positions"]), "poses", flush=True)
            for route in preview:
                last = len(route["positions"]) - 1
                for t in (0, last):
                    for view, image in backend.render(route["positions"][t], float(route["yaws"][t])).items():
                        image.save(folder / f"preview_{route['id']}_{'start' if t == 0 else 'end'}_{view}.png")
            write_json(folder / "capture_done.json", {"renders": backend.renders, "starts": backend.starts, "unexpected_exits": backend.unexpected_exits, "code": code_version()})
        print(map_id, "capture process done", flush=True)
    if not (folder / "k_done.json").exists():
        chosen = read_json(selection_file)
        by_id = {r["id"]: r for r in candidates}
        with UEBackend(folder / "session_k", PROFILE, package, CAPTURE, policy=MemoryPolicy(), console=VSM_OFF) as backend:
            for route_id in reversed(chosen["g1"]):
                route = by_id[route_id]
                for t in chosen["windows"][route_id]:
                    for view, image in backend.render(route["positions"][t], float(route["yaws"][t])).items():
                        image.save(folder / f"k_{route_id}_{t:03d}_{view}.png")
            write_json(folder / "k_done.json", {"renders": backend.renders, "starts": backend.starts, "unexpected_exits": backend.unexpected_exits, "code": code_version()})
        print(map_id, "K process done", flush=True)


def render(args):
    plan = read_json(Path(args.plan))
    for map_id in args.maps:
        render_map(args, map_id, plan)


def g1_map(map_id, folder, routes, models, features, device):
    """Per map: dz change of each model between the capture and the K renders, and the B - A gap change."""
    import torch
    from PIL import Image
    from experiments.closed_loop_k import policy_state
    chosen = read_json(folder / "selection.json")
    change, err_capture, err_k, rows = {"A": [], "B": []}, {"A": [], "B": []}, {"A": [], "B": []}, {}
    image_mae = {"front": [], "down": []}
    for route_id in chosen["g1"]:
        route = routes[route_id]
        steps = sorted(chosen["windows"][route_id])
        paths = {side: {v: [str(folder / f"{side}_{route_id}_{t:03d}_{v}.png") for t in steps] for v in ("front", "down")} for side in ("capture", "k")}
        feats = {side: {v: features(paths[side][v]) for v in ("front", "down")} for side in paths}
        for v in ("front", "down"):
            image_mae[v] += [float(np.abs(np.asarray(Image.open(a).convert("RGB"), dtype=float) - np.asarray(Image.open(b).convert("RGB"), dtype=float)).mean()) for a, b in zip(paths["capture"][v], paths["k"][v])]
        P, Y = route["positions"], route["yaws"]
        states = torch.tensor([policy_state(P[t], Y[t], P[0], Y[0]) for t in steps], dtype=torch.float32, device=device)
        labels = route["actions"][steps, DZ]
        row = {"mode": route["mode"], "poses": len(steps)}
        for (variant, seed), (model, tokenizer) in models.items():
            tokens = torch.tensor([tokenizer.encode(route["instruction"])["input_ids"]] * len(steps), device=device)
            with torch.inference_mode():
                a = model({v: feats["capture"][v] for v in model.views}, tokens, states)["action"].cpu().numpy()[:, DZ]
                b = model({v: feats["k"][v] for v in model.views}, tokens, states)["action"].cpu().numpy()[:, DZ]
            change[variant].append(np.abs(b - a))
            err_capture[variant].append(np.abs(a - labels))
            err_k[variant].append(np.abs(b - labels))
            row.setdefault("median_abs_dz_change_cm", {}).setdefault(variant, []).append(round(float(np.median(np.abs(b - a))) * 100, 3))
        rows[route_id] = row
    stacked = {v: np.concatenate(change[v]) for v in change}
    gap_capture = float(np.mean(np.concatenate(err_capture["B"])) - np.mean(np.concatenate(err_capture["A"])))
    gap_k = float(np.mean(np.concatenate(err_k["B"])) - np.mean(np.concatenate(err_k["A"])))
    result = {"median_abs_dz_change_m": {v: float(np.median(stacked[v])) for v in stacked}, "p90_abs_dz_change_m": {v: float(np.percentile(stacked[v], 90)) for v in stacked},
              "gap_change_m": abs(gap_k - gap_capture), "image_mae": {v: {"median": float(np.median(x)), "max": float(np.max(x))} for v, x in image_mae.items()}, "routes": rows}
    if map_id not in TEST_MAPS:  # the test maps are opened once; no model errors on them before that
        result.update(gap_capture_m=gap_capture, gap_k_m=gap_k)
    result["passed"] = bool(result["median_abs_dz_change_m"]["B"] < G1["median_abs_dz_change_m"] and result["gap_change_m"] < G1["gap_change_m"])
    return result, stacked


def contact_sheet(folder, route_ids, out_path, tile=160):
    """One row per preview route: start front, start down, end front, end down."""
    from PIL import Image, ImageDraw
    cells = [("start", "front"), ("start", "down"), ("end", "front"), ("end", "down")]
    sheet = Image.new("RGB", (tile * len(cells), tile * len(route_ids)), "white")
    for i, route_id in enumerate(route_ids):
        for j, (when, view) in enumerate(cells):
            path = folder / f"preview_{route_id}_{when}_{view}.png"
            if path.exists():
                sheet.paste(Image.open(path).convert("RGB").resize((tile, tile)), (j * tile, i * tile))
        ImageDraw.Draw(sheet).text((4, i * tile + 4), route_id[-3:], fill=(255, 255, 0))
    sheet.save(out_path, quality=85)


def report(args):
    import torch
    from PIL import Image
    from had.model import DinoFeatures, TerrainHAD, image_transform
    from had.tokenizer import Tokenizer
    device = args.device
    out = Path(args.out)
    plan = read_json(Path(args.plan))
    backbone, transform = DinoFeatures().to(device), image_transform()

    @torch.inference_mode()
    def features(paths):
        return torch.cat([backbone(torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i + 16]]).to(device)).float() for i in range(0, len(paths), 16)])

    models = {}
    for variant in ("A", "B"):
        for seed in G1["seeds"]:
            saved = torch.load(Path(args.models) / f"{variant}_seed{seed}/best.pt", map_location="cpu", weights_only=True)
            tokenizer = Tokenizer.from_dict(saved["identity"]["tokenizer"])
            model = TerrainHAD(saved["identity"]["variant"], tokenizer, cached_only=True).to(device).eval()
            model.load_head_state(saved["head"])
            models[variant, seed] = (model, tokenizer)
    summary = {"code": code_version(), "thresholds": {k: v for k, v in G1.items() if k != "seeds"}, "models": args.models, "capture": CAPTURE, "console": VSM_OFF, "maps": {}}
    pooled = {"A": [], "B": []}
    for map_id in args.maps:
        folder = out / map_id
        if not (folder / "k_done.json").exists():
            print(map_id, "not rendered yet", flush=True)
            continue
        routes = {r["id"]: arrays(r) for r in plan["maps"][map_id]["candidates"]}
        result, stacked = g1_map(map_id, folder, routes, models, features, device)
        for v in pooled:
            pooled[v].append(stacked[v])
        chosen = read_json(folder / "selection.json")
        tally = {}
        for reason in chosen["checks"].values():
            key = reason.split(":")[0]
            tally[key] = tally.get(key, 0) + 1
        result["ue_checks"] = {"checked": len(chosen["checks"]), "tally": tally}
        result["process"] = {"capture": read_json(folder / "capture_done.json"), "k": read_json(folder / "k_done.json")}
        contact_sheet(folder, chosen["preview"], out / f"preview_{map_id}.jpg")
        brightness = [float(np.asarray(Image.open(p).convert("L"), dtype=float).mean() / 255) for p in sorted(folder.glob("preview_*_down.png"))]
        result["preview"] = {"routes": chosen["preview"], "down_brightness_mean": float(np.mean(brightness)) if brightness else None}
        summary["maps"][map_id] = result
        print(f"{map_id:11s} G1 {'pass' if result['passed'] else 'FAIL'}  B |dz| median {result['median_abs_dz_change_m']['B'] * 100:.2f} cm (A {result['median_abs_dz_change_m']['A'] * 100:.2f}),"
              f" gap change {result['gap_change_m'] * 100:.2f} cm, image MAE max front {result['image_mae']['front']['max']:.2f} down {result['image_mae']['down']['max']:.2f}, UE checks {tally}", flush=True)
    if pooled["B"]:
        stacked = {v: np.concatenate(pooled[v]) for v in pooled}
        summary["pooled"] = {"median_abs_dz_change_m": {v: float(np.median(stacked[v])) for v in stacked}, "maps": len(summary["maps"])}
        summary["passed"] = bool(summary["maps"]) and all(r["passed"] for r in summary["maps"].values()) and len(summary["maps"]) == len(args.maps)
    write_json(out / "g1.json", summary)
    print("G1 all maps passed:", summary.get("passed"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("render", "report"))
    p.add_argument("--plan", default="runs/v3/stage2/plan.json"); p.add_argument("--out", default="runs/v3/stage2/g1")
    p.add_argument("--maps", nargs="+", default=list(maps())); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models", default="runs/terrain_v2_dino/models"); p.add_argument("--device", default="cuda")
    args = p.parse_args()
    {"render": render, "report": report}[args.stage](args)
