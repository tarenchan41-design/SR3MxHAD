"""Closed-loop episode on Cosys-AirSim: expert replay gate, then frozen-model flight.

Live flight requires the manually configured UE 5.8.4 environment in docs/protocol.md.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from sr3m.geometry import adjacent_action, local_state, ned_to_ue_world_m, ue_world_m_to_ned, wrap
from sr3m.sim import EXECUTOR, Executor, camera_images, state
from experiments.protocol import PreparedSplit, read_json, write_json
from had.model import DEFAULT_WEIGHTS, TerrainHAD, image_transform
from had.tokenizer import Tokenizer


def run(args):
    import cosysairsim as api
    ds = PreparedSplit(args.prepared, "val_seen")
    registered = read_json(Path(args.prepared) / "closed_loop_routes.json")
    route_info = next(r for r in registered["routes"] if r["episode_id"] == args.episode)
    if args.map_package != route_info["map_package"]: raise ValueError("Operator-selected map does not match route")
    r = next(r for r in ds.routes if r["episode_id"] == args.episode)
    client = api.MultirotorClient(ip=args.host); client.confirmConnection()
    origin = np.asarray(args.origin_ue_m)
    # Map opening and camera placement are manual. Never silently change maps or teleport a running episode.
    sl = slice(r["offset"], r["offset"]+r["count"])
    positions, yaws = ds.arrays["position"][sl], ds.arrays["yaw"][sl]
    p, yaw = state(client)
    if np.linalg.norm(p-ue_world_m_to_ned(positions[0], origin)) > .5 or abs(wrap(yaw-yaws[0])) > math.radians(2):
        raise RuntimeError("Place drone at the registered route start before running")
    gate = Path(args.gate)
    previous = read_json(gate) if gate.exists() else {"prepared": ds.protocol,"executor":EXECUTOR, "routes": {}}
    if previous["prepared"] != ds.protocol or previous.get("executor")!=EXECUTOR: raise ValueError("Expert gate belongs to different data/executor")
    if args.checkpoint and not all(previous["routes"].get(v["episode_id"], {}).get("passed", False) for v in registered["routes"]):
        raise RuntimeError("All 16 expert routes must pass before model flight")
    model = tokenizer = threshold = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        identity = checkpoint["identity"]
        if identity["prepared"] != ds.protocol: raise ValueError("Model/data contract differs")
        selection = read_json(Path(args.checkpoint).parent / "selection.json")
        if not selection["stop"]["ready"]: raise RuntimeError("STOP calibration did not pass")
        threshold = selection["stop"]["threshold"]
        tokenizer = Tokenizer.from_dict(identity["tokenizer"])
        model = TerrainHAD(identity["variant"], tokenizer, args.weights).to(args.device).eval()
        model.load_head_state(checkpoint["head"])
    executor = Executor(client, api); initial_p, initial_yaw = p.copy(), yaw
    history = []; failure = None; stopped = False; progress=0
    client.enableApiControl(True); client.armDisarm(True)
    try:
        budget = 2*(r["count"]-1)+20 if model else r["count"]-1
        for t in range(budget):
            p, yaw = state(client)
            if model:
                relative, angle = local_state(ned_to_ue_world_m(p), yaw, ned_to_ue_world_m(initial_p), initial_yaw)
                current = torch.tensor([[*(np.asarray(relative)/100), math.sin(angle), math.cos(angle)]], dtype=torch.float32, device=args.device)
                tokens = torch.tensor([tokenizer.encode(r["instruction"])["input_ids"]], device=args.device)
                client.simPause(True)
                try:
                    images = {v: image_transform()(x)[None].to(args.device) for v, x in camera_images(client, api, model.views).items()}
                    with torch.inference_mode(): output = model.from_images(images, tokens, current)
                finally: client.simPause(False)
                probability = float(output["stop_logit"].sigmoid().item())
                if probability >= threshold:
                    stopped = True; break
                action = output["action"][0].cpu().numpy()
            else:
                # Expert uses the next recorded WORLD pose, corrected from actual current pose.
                action = adjacent_action(ned_to_ue_world_m(p, origin), yaw, positions[t+1], float(yaws[t+1]))
            reached, heading = executor.step(action)
            current_ue=ned_to_ue_world_m(reached,origin)
            while progress<len(positions)-1 and np.linalg.norm(current_ue-positions[progress+1])<=5.: progress+=1
            history.append({"step": t, "position_ned_m": reached.tolist(), "yaw_rad": heading,"tracking":dict(executor.tracking)})
        if not model: stopped = True
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        client.simPause(False); client.hoverAsync().join()
    p, yaw = state(client); distance = float(np.linalg.norm(p-ue_world_m_to_ned(positions[-1], origin)))
    result = {"episode_id": args.episode, "map_package": route_info["map_package"], "expert": model is None,
              "passed": failure is None and stopped and distance <= (.5 if model is None else 20.),
              "distance_goal_m": distance, "stopped": stopped, "failure": failure, "history": history,
              "ordered_expert_progress_fraction":progress/(len(positions)-1),
              "collided":bool(client.simGetCollisionInfo().has_collided),
              "budget_exhausted":bool(model is not None and not stopped and failure is None),
              "termination": "STOP_or_budget_or_executor_failure", "SR20": bool(failure is None and stopped and distance <= 20)}
    write_json(args.out, result)
    if model is None:
        previous["routes"][args.episode] = {"passed": result["passed"], "distance_goal_m": distance, "failure": failure}
        write_json(gate, previous)
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True); p.add_argument("--episode", required=True); p.add_argument("--map-package",required=True)
    p.add_argument("--gate", required=True); p.add_argument("--out", required=True)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--origin-ue-m", nargs=3, type=float, required=True)
    p.add_argument("--checkpoint"); p.add_argument("--weights", default=str(DEFAULT_WEIGHTS)); p.add_argument("--device", default="cuda")
    run(p.parse_args())

