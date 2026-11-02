"""K gates (protocol section 7). G2 needs no simulator; G1 and G3 start UnrealEditor-Cmd, so close the GUI editor first.

G1 judges by policy output, not by pixels or cosine alone. Its window is shuffled and does not line up with the dataset's
process restarts, because a model visits poses in an order the dataset never recorded.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.closed_loop_k import capture_settings, fly, policy_state, route_info
from experiments.protocol import PreparedSplit, read_json, write_json
from sr3m.geometry import integrate, wrap

G1 = {"median_abs_dz_change_m": .024, "gap_change_m": .024, "start": 13, "poses": 32, "seeds": (42, 43, 44)}
G2 = {"position_m": .01, "yaw_deg": .01}
DZ = 2


def closed_routes(prepared, routes=None):
    """The fixed closed-loop routes, or another list of val_seen routes (the desert G1 of protocol section 14)."""
    return read_json(Path(routes) if routes else Path(prepared) / "closed_loop_routes.json")["routes"]


def gate2(prepared):
    """The dataset's stored actions, integrated open loop from the start, must retrace the recorded poses."""
    ds = PreparedSplit(prepared, "val_seen")
    index = {r["episode_id"]: r for r in ds.routes}
    rows = {}
    for item in closed_routes(prepared):
        route = index[item["episode_id"]]
        window = slice(route["offset"], route["offset"] + route["count"])
        positions, yaws, states, actions = [ds.arrays[k][window].astype(float) for k in ("position", "yaw", "state", "action")]
        position, yaw = tuple(positions[0]), float(yaws[0])
        position_error, yaw_error, state_error = 0., 0., 0.
        for t in range(len(positions) - 1):
            state_error = max(state_error, float(np.abs(np.subtract(policy_state(position, yaw, positions[0], yaws[0]), states[t])).max()))
            position, yaw = integrate(position, yaw, actions[t])
            position_error = max(position_error, float(np.linalg.norm(np.subtract(position, positions[t + 1]))))
            yaw_error = max(yaw_error, abs(math.degrees(wrap(yaw - yaws[t + 1]))))
        rows[item["episode_id"]] = {"position_m": position_error, "yaw_deg": yaw_error, "state": state_error,
                                    "passed": position_error <= G2["position_m"] and yaw_error <= G2["yaw_deg"]}
    return {"thresholds": G2, "passed": all(r["passed"] for r in rows.values()), "routes": rows}


def gate1_render(args):
    from sim_k.backend import UEBackend
    out = Path(args.out)
    rng = np.random.default_rng(0)
    manifest = []
    routes = closed_routes(args.prepared, args.routes)
    for item in routes:
        episode = item["episode_id"]
        directory = out / episode.split("--")[-1]
        order = rng.permutation(G1["poses"]) + G1["start"]
        _, route, positions, yaws = route_info(args.prepared, "val_seen", episode)
        order = [int(i) for i in order if i < route["count"]]
        folder, capture, profile, map_package = capture_settings(args.data_root, episode)
        trajectory = read_json(folder / "trajectory.json")
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "done.json").exists():
            with UEBackend(directory / "session", profile, map_package, capture, restart_every=G1["poses"]) as backend:
                for step in order:
                    for view, image in backend.render(positions[step], float(yaws[step])).items():
                        image.save(directory / f"step_{step:06d}_{view}.png")
            (directory / "done.json").write_text("{}", encoding="utf-8")
        manifest.append({"episode": episode, "directory": str(directory), "folder": str(folder), "steps": sorted(order),
                         "images": {str(s): trajectory["steps"][s]["images"] for s in order}})
        print(f"rendered {episode.split('--')[-1]} {len(order)} poses", flush=True)
    write_json(out / "manifest.json", manifest)


def gate1_report(args):
    """Needs PYTHONPATH with transformers; no simulator."""
    import torch
    from PIL import Image
    from had.model import DinoFeatures, TerrainHAD, image_transform
    from had.tokenizer import Tokenizer
    device = args.device
    out = Path(args.out)
    ds = PreparedSplit(args.prepared, "val_seen")
    routes = {r["episode_id"]: r for r in ds.routes}
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
    per_route, change, error_new, error_old = {}, {"A": [], "B": []}, {"A": [], "B": []}, {"A": [], "B": []}
    for entry in read_json(out / "manifest.json"):
        route = routes[entry["episode"]]
        steps = [s for s in entry["steps"] if ds.arrays["action_valid"][route["offset"] + s] and np.isfinite(ds.arrays["action"][route["offset"] + s]).all()]
        new_paths = {v: [str(Path(entry["directory"]) / f"step_{s:06d}_{v}.png") for s in steps] for v in ("front", "down")}
        old_paths = {v: [str(Path(entry["folder"]) / entry["images"][str(s)][v]["rgb"]) for s in steps] for v in ("front", "down")}
        new, old = {v: features(new_paths[v]) for v in new_paths}, {v: features(old_paths[v]) for v in old_paths}
        mae = {v: float(np.mean([np.abs(np.asarray(Image.open(a).convert("RGB"), dtype=float) - np.asarray(Image.open(b).convert("RGB"), dtype=float)).mean() for a, b in zip(new_paths[v], old_paths[v])])) for v in new_paths}
        cosine = {v: torch.nn.functional.cosine_similarity(new[v], old[v]) for v in new}
        index = [route["offset"] + s for s in steps]
        labels, states = ds.arrays["action"][index], torch.tensor(ds.arrays["state"][index], dtype=torch.float32, device=device)
        row = {"poses": len(steps), "mae": mae, "cosine_min": {v: float(cosine[v].min()) for v in cosine}, "cosine_median": {v: float(cosine[v].median()) for v in cosine}}
        for (variant, seed), (model, tokenizer) in models.items():
            tokens = torch.tensor([tokenizer.encode(route["instruction"])["input_ids"]] * len(steps), device=device)
            with torch.inference_mode():
                a = model({v: new[v] for v in model.views}, tokens, states)["action"].cpu().numpy()[:, DZ]
                b = model({v: old[v] for v in model.views}, tokens, states)["action"].cpu().numpy()[:, DZ]
            change[variant].append(np.abs(a - b))
            error_new[variant].append(np.abs(a - labels[:, DZ]))
            error_old[variant].append(np.abs(b - labels[:, DZ]))
            row.setdefault("median_abs_dz_change_cm", {}).setdefault(variant, []).append(float(np.median(np.abs(a - b)) * 100))
        per_route[entry["episode"]] = row
        print(f"{entry['episode'].split('--')[-1][:34]:34s} n={len(steps):2d} mae f {mae['front']:.2f} d {mae['down']:.2f} cos min f {row['cosine_min']['front']:.3f} d {row['cosine_min']['down']:.3f} B |dz| {np.mean(row['median_abs_dz_change_cm']['B']):.2f} cm", flush=True)
    stacked = {v: np.concatenate(change[v]) for v in change}
    gap_new = float(np.mean(np.concatenate(error_new["B"])) - np.mean(np.concatenate(error_new["A"])))
    gap_old = float(np.mean(np.concatenate(error_old["B"])) - np.mean(np.concatenate(error_old["A"])))
    result = {"thresholds": {k: v for k, v in G1.items() if k != "seeds"}, "median_abs_dz_change_m": {v: float(np.median(stacked[v])) for v in stacked}, "p90_abs_dz_change_m": {v: float(np.percentile(stacked[v], 90)) for v in stacked},
              "gap_new_m": gap_new, "gap_old_m": gap_old, "routes": per_route}
    result["passed"] = bool(result["median_abs_dz_change_m"]["B"] < G1["median_abs_dz_change_m"] and abs(gap_new - gap_old) < G1["gap_change_m"])
    write_json(out / "g1.json", result)
    print(json.dumps({k: result[k] for k in ("passed", "median_abs_dz_change_m", "p90_abs_dz_change_m", "gap_new_m", "gap_old_m")}))


def gate3(args):
    """The expert through the whole K loop except rendering, which G1 covers: 16/16 within 0.5 m of the goal, no collision."""
    from sim_k.backend import UEBackend
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for item in closed_routes(args.prepared):
        episode = item["episode_id"]
        file = out / (episode.split("--")[-1] + ".json")
        if not file.exists():
            _, _, positions, yaws = route_info(args.prepared, "val_seen", episode)
            _, capture, profile, map_package = capture_settings(args.data_root, episode)
            with UEBackend(file.with_suffix(".session"), profile, map_package, capture) as backend:
                result = fly(backend, positions, yaws)
            result["unexpected_exits"] = backend.unexpected_exits
            write_json(file, result)
        result = read_json(file)
        agls = [h["agl_m"] for h in result["history"] if h["agl_m"] is not None]
        results[episode] = {k: result[k] for k in ("passed", "failure", "distance_goal_m", "steps")} | {"agl_min_m": min(agls) if agls else None, "agl_missing": sum(h["agl_m"] is None for h in result["history"])}
        print(f"{episode.split('--')[-1][:40]:40s} passed={result['passed']} failure={result['failure']} agl_min={results[episode]['agl_min_m']} missing={results[episode]['agl_missing']}", flush=True)
    summary = {"passed": all(r["passed"] for r in results.values()), "routes": results}
    write_json(out / "g3.json", summary)
    print(f"G3 passed={summary['passed']} {sum(r['passed'] for r in results.values())}/{len(results)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gate", choices=("g1-render", "g1-report", "g2", "g3"))
    p.add_argument("--prepared", default="runs/terrain_v2_dino/prepared")
    p.add_argument("--models", default="runs/terrain_v2_dino/models")
    p.add_argument("--data-root", default="/mnt/f/WSL_ColdStorage/datasets/DatasetCaptureBatches/workbench")
    p.add_argument("--out", default="runs/closed_loop/k")
    p.add_argument("--device", default="cuda")
    p.add_argument("--routes", help="g1-render: a JSON file with a routes list instead of closed_loop_routes.json")
    args = p.parse_args()
    if args.gate == "g2":
        result = gate2(args.prepared)
        write_json(Path(args.out) / "g2.json", result)
        worst = {k: max(r[k] for r in result["routes"].values()) for k in ("position_m", "yaw_deg", "state")}
        print(f"G2 passed={result['passed']} routes={len(result['routes'])} worst {worst}")
    else:
        {"g1-render": gate1_render, "g1-report": gate1_report, "g3": gate3}[args.gate](args)
