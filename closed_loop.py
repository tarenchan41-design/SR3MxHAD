"""Cosys-AirSim executor: fixed world translation, then yaw, then observation.

The executor is testable without a simulator. Live flight requires the manually
configured UE 5.8.4 environment described in 闭环实验计划.md.
"""
import argparse
from io import BytesIO
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from coordinates import action_target_ned, adjacent_action, local_state, ned_to_ue_world_m, ue_world_m_to_ned, wrap
from experiment import PROTOCOL, EXECUTOR, PreparedSplit, read_json, write_json
from had_model import DEFAULT_WEIGHTS, TerrainHAD, Tokenizer, image_transform


def state(client):
    k = client.getMultirotorState().kinematics_estimated
    p, q = k.position, k.orientation
    yaw = math.atan2(2*(q.w_val*q.z_val+q.x_val*q.y_val), 1-2*(q.y_val*q.y_val+q.z_val*q.z_val))
    return np.array([p.x_val, p.y_val, p.z_val]), yaw


class Executor:
    def __init__(self, client, api, speed=EXECUTOR["speed_m_s"]):
        self.client, self.api, self.speed = client, api, speed
        self.tracking = {}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all(): raise ValueError("Invalid action")
        p, yaw = state(self.client)
        target, next_yaw = action_target_ned(p, yaw, action)
        self.client.moveToPositionAsync(*target, velocity=self.speed,
            timeout_sec=max(5., np.linalg.norm(target-p)/self.speed+5.),
            drivetrain=self.api.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=self.api.YawMode(is_rate=False, yaw_or_rate=math.degrees(yaw))).join()
        actual, _ = state(self.client)
        if np.linalg.norm(actual-target) > EXECUTOR["position_tolerance_m"]: raise RuntimeError("Translation did not reach target within 0.5m")
        # A 2-degree API margin would swallow the dataset's many sub-degree turns.
        self.client.rotateToYawAsync(math.degrees(next_yaw), timeout_sec=10., margin=EXECUTOR["yaw_api_margin_deg"]).join()
        self.client.hoverAsync().join()
        actual, actual_yaw = state(self.client)
        self.tracking={"target_ned_m":target.tolist(),"target_yaw_rad":next_yaw,
                       "position_error_m":float(np.linalg.norm(actual-target)),
                       "yaw_error_deg":abs(math.degrees(wrap(actual_yaw-next_yaw)))}
        if self.tracking["position_error_m"] > EXECUTOR["position_tolerance_m"] or self.tracking["yaw_error_deg"] > EXECUTOR["yaw_tolerance_deg"]:
            raise RuntimeError("Post-rotation pose is outside expert replay tolerances")
        if self.client.simGetCollisionInfo().has_collided: raise RuntimeError("Collision")
        return actual, actual_yaw


def observations(client, api, views):
    # PNG decoding gives explicit RGB; no assumption about raw API BGR/RGBA layout.
    replies = client.simGetImages([api.ImageRequest(v, api.ImageType.Scene, False, True) for v in views])
    if len(replies) != len(views): raise RuntimeError("Incomplete camera response")
    images = {}
    for view, reply in zip(views, replies):
        with Image.open(BytesIO(bytes(reply.image_data_uint8))) as image:
            images[view] = image_transform()(image.convert("RGB"))[None]
    return images


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
                    images = {v: x.to(args.device) for v, x in observations(client, api, model.views).items()}
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

