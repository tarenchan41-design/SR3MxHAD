"""Kinematic closed loop on the dataset's own renderer (protocol section 7): G2, G3 and model flights.

Run inside the repository with the GUI editor closed. Reported as a kinematic loop, never as flight-control closed loop:
actions move the pose by geometry, with no dynamics.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.protocol import PreparedSplit, read_json, write_json
from sr3m.geometry import adjacent_action, integrate, local_state, wrap

MIN_AGL_M = 2.
PROGRESS_RADIUS_M = 5.
SR_RADIUS_M = 20.
EXPERT_GOAL_M = .5


def route_info(prepared, split, episode):
    ds = PreparedSplit(prepared, split)
    route = next(r for r in ds.routes if r["episode_id"] == episode)
    window = slice(route["offset"], route["offset"] + route["count"])
    return ds, route, ds.arrays["position"][window].astype(float), ds.arrays["yaw"][window].astype(float)


def capture_settings(data_root, episode):
    folder = Path(data_root) / episode
    trajectory = read_json(folder / "trajectory.json")
    context = read_json(folder / "run-context.json")
    return folder, trajectory["capture"], context["task"]["profile_id"], trajectory["map_package"]


def policy_state(position, yaw, start, start_yaw):
    relative, angle = local_state(tuple(position), yaw, tuple(start), start_yaw)
    return [*(np.asarray(relative) / 100), math.sin(angle), math.cos(angle)]


def fly(backend, positions, yaws, policy=None):
    """One route. Without a policy the expert steps to each next recorded pose, checking collisions only."""
    position, yaw = positions[0].copy(), float(yaws[0])
    budget = 2 * (len(positions) - 1) + 20 if policy else len(positions) - 1
    history, failure, stopped, progress = [], None, False, 0
    for t in range(budget):
        if policy:
            decision = policy(backend.render(position, yaw), policy_state(position, yaw, positions[0], float(yaws[0])))
            if decision["stop"]:
                stopped = True
                history.append({"step": t, "position_m": position.tolist(), "yaw_rad": yaw, "stop_probability": decision["stop_probability"]})
                break
            action = decision["action"]
        else:
            action = adjacent_action(position, yaw, positions[t + 1], float(yaws[t + 1]))
        target, next_yaw = integrate(position, yaw, action)
        target = np.asarray(target, dtype=float)
        sweep = backend.sweep(position, target)
        ground = backend.ground_z_m(target)
        agl = None if ground is None else float(target[2] - ground)
        record = {"step": t, "position_m": position.tolist(), "yaw_rad": yaw, "action": [float(a) for a in action], "target_m": target.tolist(),
                  "next_yaw_rad": next_yaw, "agl_m": agl, "stop_probability": decision["stop_probability"] if policy else None}
        history.append(record)
        if sweep["hit"]:
            failure = f"collision: {sweep.get('actor')}"
            break
        if agl is not None and agl < MIN_AGL_M:
            failure = f"ground: AGL {agl:.2f} m"
            break
        position, yaw = target, next_yaw
        while progress < len(positions) - 1 and np.linalg.norm(position - positions[progress + 1]) <= PROGRESS_RADIUS_M:
            progress += 1
    distance = float(np.linalg.norm(position - positions[-1]))
    if policy is None:
        stopped = failure is None
    return {"expert": policy is None, "steps": len(history), "stopped": stopped, "failure": failure, "distance_goal_m": distance,
            "budget_exhausted": bool(policy and not stopped and failure is None), "ordered_progress_fraction": progress / (len(positions) - 1),
            "SR20": bool(failure is None and stopped and distance <= SR_RADIUS_M), "history": history,
            "passed": bool(failure is None and stopped and distance <= (EXPERT_GOAL_M if policy is None else SR_RADIUS_M))}


def load_policy(checkpoint, route, device):
    """Frozen model and its val_seen STOP threshold. Imported here: transformers is not in the shared venv."""
    import torch
    from had.model import TerrainHAD, image_transform
    from had.tokenizer import Tokenizer
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    selection = read_json(Path(checkpoint).parent / "selection.json")
    if not selection["stop"]["ready"]:
        raise RuntimeError("STOP calibration did not pass")
    tokenizer = Tokenizer.from_dict(saved["identity"]["tokenizer"])
    model = TerrainHAD(saved["identity"]["variant"], tokenizer).to(device).eval()
    model.load_head_state(saved["head"])
    tokens = torch.tensor([tokenizer.encode(route["instruction"])["input_ids"]], device=device)
    transform, threshold = image_transform(), selection["stop"]["threshold"]

    def policy(images, state):
        batch = {v: transform(images[v])[None].to(device) for v in model.views}
        current = torch.tensor([state], dtype=torch.float32, device=device)
        with torch.inference_mode():
            output = model.from_images(batch, tokens, current)
        probability = float(output["stop_logit"].sigmoid().item())
        return {"stop": probability >= threshold, "stop_probability": probability, "action": output["action"][0].cpu().numpy()}
    return policy, saved["identity"]


def run(args):
    from sim_k.backend import UEBackend
    ds, route, positions, yaws = route_info(args.prepared, args.split, args.episode)
    _, capture, profile, map_package = capture_settings(args.data_root, args.episode)
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f"{out} exists; skipping")
        return
    policy, identity = load_policy(args.checkpoint, route, args.device) if args.checkpoint else (None, None)
    with UEBackend(out.with_suffix(".session"), profile, map_package, capture, restart_every=args.restart_every) as backend:
        result = fly(backend, positions, yaws, policy)
    result.update(episode_id=args.episode, map_package=map_package, backend="kinematic", split=args.split, identity=identity and {k: identity[k] for k in identity if k != "tokenizer"})
    write_json(out, result)
    print(json.dumps({k: result[k] for k in ("passed", "SR20", "steps", "failure", "distance_goal_m", "ordered_progress_fraction", "stopped")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True); p.add_argument("--split", default="val_seen"); p.add_argument("--episode", required=True)
    p.add_argument("--data-root", required=True); p.add_argument("--out", required=True); p.add_argument("--checkpoint")
    p.add_argument("--device", default="cuda"); p.add_argument("--restart-every", type=int, default=32); p.add_argument("--overwrite", action="store_true")
    run(p.parse_args())
