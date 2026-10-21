"""Cosys-AirSim executor: fixed world translation, then yaw, then observation.

Testable without a simulator; the caller passes a connected client and the
`cosysairsim` module. Live flight needs the manually configured UE 5.8.4 project.
"""
from io import BytesIO
import math

import numpy as np
from PIL import Image
from sr3m.geometry import action_target_ned, wrap

EXECUTOR = {"speed_m_s":2.,"position_tolerance_m":.5,"yaw_api_margin_deg":.01,
            "yaw_tolerance_deg":.05,"yaw_order":"translate_then_rotate"}


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


def camera_images(client, api, views):
    """RGB PIL images per view. PNG decoding avoids assumptions about raw BGR/RGBA layout."""
    replies = client.simGetImages([api.ImageRequest(v, api.ImageType.Scene, False, True) for v in views])
    if len(replies) != len(views): raise RuntimeError("Incomplete camera response")
    images = {}
    for view, reply in zip(views, replies):
        with Image.open(BytesIO(bytes(reply.image_data_uint8))) as image:
            images[view] = image.convert("RGB")
    return images
