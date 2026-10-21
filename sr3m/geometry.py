"""Action contract and coordinate conversions shared by labels, plots and the executor.

UE is left handed, +Z up (engine cm; capture position_m already meters).
Capture yaw is degrees, positive +X toward +Y. HAD actions are current-yaw
forward/right/up in meters, dyaw in radians; positive dyaw turns right.
Cosys-AirSim position is world NED meters; angle-based yaw APIs take degrees.
Points subtract the configured UE origin; displacement vectors never do.
"""
import math
import numpy as np

TASK = "capture_adjacent_yaw_fru_m_rad_v1"


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def local_vector(delta, yaw):
    x, y, z = delta
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * x + s * y, -s * x + c * y, z)


def local_state(position, yaw, origin, origin_yaw):
    return local_vector(tuple(x-y for x, y in zip(position, origin)), origin_yaw), wrap(yaw-origin_yaw)


def adjacent_action(position, yaw, next_position, next_yaw):
    return (*local_vector(tuple(x-y for x, y in zip(next_position, position)), yaw), wrap(next_yaw-yaw))


def integrate(position, yaw, action):
    dx, dy, dz = local_vector(action[:3], -yaw)
    return tuple(v+d for v, d in zip(position, (dx, dy, dz))), wrap(yaw+action[3])


def fru_to_world(delta, yaw_rad):
    delta = np.asarray(delta, dtype=float)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.stack((c * delta[..., 0] - s * delta[..., 1],
                     s * delta[..., 0] + c * delta[..., 1], delta[..., 2]), axis=-1)


def ue_world_m_to_ned(position_m, origin_ue_m=(0., 0., 0.)):
    return (np.asarray(position_m, dtype=float) - np.asarray(origin_ue_m)) * [1, 1, -1]


def ned_to_ue_world_m(position_ned_m, origin_ue_m=(0., 0., 0.)):
    return np.asarray(position_ned_m, dtype=float) * [1, 1, -1] + np.asarray(origin_ue_m)


def action_target_ned(current_ned_m, yaw_rad, action_m_rad):
    """Compute the world target ONCE, using yaw_t before any yaw command."""
    delta_world = fru_to_world(action_m_rad[:3], yaw_rad)
    return np.asarray(current_ned_m) + ue_world_m_to_ned(delta_world), wrap(yaw_rad + action_m_rad[3])


def check_climb(action_m_rad):
    if action_m_rad[2] <= 0 or ue_world_m_to_ned(action_m_rad[:3])[2] >= 0:
        raise ValueError("Climb must be positive in FRU and negative in NED")
