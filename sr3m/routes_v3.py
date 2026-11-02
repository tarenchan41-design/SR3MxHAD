"""Route rules for dataset v3 (protocol section 13): one object, stop directly above it, no shape templates.

The path comes out of a pursuit controller, so the next turn depends on where the target is rather than on the pose so far.
Geometry only; rendering and obstacle checks come later. Positive heading turns right, as everywhere in sr3m.geometry.
"""
from dataclasses import dataclass
import math

import numpy as np

from sr3m.geometry import integrate, wrap

STEP_M = 2.
ARRIVE_M = 1.
FRONT_HALF_DEG = 45.
AHEAD_DEG = 12.
SR_RADIUS_M = 20.
PATCH_FACTOR = 8.  # 256 px focal length / 32 px patch: a target of width w fills one patch out to 8 w.
PATCH_PX = 32.
FOCAL_PX = 256.
MIN_CLEARANCE_M = 10.
ABOVE_TOP_M = 8.
FRAMES = {"en": ("Fly to {what} {where} and stop directly above it. {height}", "{what} is {where}. Go there and stop right above it. {height}", "Your target is {what} {where}. Fly over to it and stop directly above it. {height}"),
          "zh": ("飞到{where}的{what}，并在其正上方停住。{height}", "{where}有{what}，飞过去，在它正上方停下。{height}", "目标是{where}的{what}，飞到它正上方后停住。{height}")}
WHERE = {"ahead": ("ahead of you", "前方"), "left": ("ahead on your left", "左前方"), "right": ("ahead on your right", "右前方")}
HEIGHT = {"follow": ("Follow the terrain along the way.", "全程随沿途地形调整高度。"), "hold": ("Keep a constant altitude.", "全程保持同一飞行高度、不随地形起伏升降。")}


@dataclass(frozen=True)
class Design:
    """gain and max_yaw set the pursuit; kick_every > 0 injects offsets; rotate_first turns on the spot before flying.

    start_kicks heading kicks always land within the first start_kick_steps, so that a turn so far says less about the turn to come.
    relief_share of a map's routes must cross at least relief_min_m of ground relief, or the height question has nothing to measure;
    after relief_patience flat proposals in a row the map is taken to have no more, and the shortfall is reported.
    """
    gain: float = .25
    max_yaw_deg: float = 5.
    kick_every: float = 0.
    kick_heading_deg: tuple = (5., 25.)
    kick_lateral_m: tuple = (2., 8.)
    kick_margin_m: float = 30.
    rotate_first: bool = False
    aligned_deg: float = 3.
    min_distance_m: float = 60.
    max_distance_m: float = 250.
    max_bearing_deg: float = 35.
    climb_limit_m: float = 2.
    budget_factor: float = 3.
    start_kick_steps: int = 0
    start_kicks: int = 1
    relief_share: float = 0.
    relief_min_m: float = 15.
    relief_patience: int = 300


def bearing_to(position, yaw, target):
    return wrap(math.atan2(target[1] - position[1], target[0] - position[0]) - yaw)


def direction_word(bearing):
    if abs(math.degrees(bearing)) < AHEAD_DEG:
        return "ahead"
    return "right" if bearing > 0 else "left"


def distance_range(width, design):
    """Empty when the object cannot fill a patch from the minimum distance."""
    return design.min_distance_m, min(PATCH_FACTOR * width, design.max_distance_m)


def apparent_px(width, height, distance, drop):
    """Side of the square with the same image area as the object, seen from distance metres away and drop metres above its base.

    A flat object seen at a shallow angle is foreshortened: its image height is width * sin(elevation) plus its own height.
    """
    elevation = math.atan2(max(drop, 0.), distance)
    across = FOCAL_PX * width / distance
    up = FOCAL_PX * (width * math.sin(elevation) + height * math.cos(elevation)) / distance
    return math.sqrt(across * up)


def misses_when_flying_straight(distance, bearing):
    """Rule 4: keeping the start heading must not reach the success radius."""
    return abs(bearing) >= math.pi / 2 or distance * math.sin(abs(bearing)) > SR_RADIUS_M


def line_of_sight(start, target_xy, target_z, ground, step_m=5.):
    """Terrain only; a small tolerance at the target end so a target sitting on the ground counts as visible."""
    distance = math.hypot(target_xy[0] - start[0], target_xy[1] - start[1])
    fractions = np.arange(step_m, distance - 10., step_m) / distance
    if not len(fractions):
        return True
    xy = np.c_[start[0] + fractions * (target_xy[0] - start[0]), start[1] + fractions * (target_xy[1] - start[1])]
    z = ground(xy)
    return bool(np.isfinite(z).all() and (start[2] + fractions * (target_z - start[2]) > z).all())


def ambiguous(target, others, position, yaw):
    """Another object of the same class inside the front view, under the same direction word."""
    word = direction_word(bearing_to(position, yaw, target["center"]))
    for other in others:
        if other["key"] == target["key"] or other["cls"] != target["cls"]:
            continue
        bearing = bearing_to(position, yaw, other["center"])
        if abs(math.degrees(bearing)) <= FRONT_HALF_DEG and direction_word(bearing) == word:
            return True
    return False


def pursuit_yaw(position, yaw, target, design):
    return float(np.clip(design.gain * bearing_to(position, yaw, target), -math.radians(design.max_yaw_deg), math.radians(design.max_yaw_deg)))


def rollout(start, start_yaw, target, height_at, design, rng):
    """Poses and the action the expert would take at each. Kicks move the pose between steps, so a label need not equal the pose difference."""
    position, yaw = np.array(start, dtype=float), float(start_yaw)
    positions, yaws, actions, kicks = [], [], [], []
    limit = int(design.budget_factor * math.hypot(target[0] - start[0], target[1] - start[1]) / STEP_M) + 20
    turning = design.rotate_first
    early_kicks = set(rng.choice(np.arange(1, design.start_kick_steps + 1), design.start_kicks, replace=False).tolist()) if design.start_kick_steps else set()
    for t in range(limit):
        distance = math.hypot(target[0] - position[0], target[1] - position[1])
        positions.append(position.copy()); yaws.append(yaw)
        if distance <= ARRIVE_M:
            return np.array(positions), np.array(yaws), np.array(actions).reshape(-1, 4), kicks
        bearing = bearing_to(position, yaw, target)
        turning = turning and abs(math.degrees(bearing)) > design.aligned_deg
        forward = 0. if turning else min(STEP_M, distance)
        dyaw = pursuit_yaw(position, yaw, target, design)
        moved = position + forward * np.array([math.cos(yaw), math.sin(yaw), 0.])
        moved[2] = height_at(moved[:2], position[2], math.atan2(target[1] - moved[1], target[0] - moved[0]))
        actions.append((forward, 0., moved[2] - position[2], dyaw))
        next_position, next_yaw = integrate(tuple(position), yaw, actions[-1])
        position, yaw = np.array(next_position), float(next_yaw)
        if t + 1 in early_kicks:
            offset = math.radians(rng.uniform(*design.kick_heading_deg)) * rng.choice((-1, 1))
            yaw = wrap(yaw + offset)
            kicks.append({"step": t + 1, "kind": "heading", "value": offset})
        elif design.kick_every > 0 and not turning and distance > design.kick_margin_m and rng.random() < 1 / design.kick_every:
            if rng.random() < .5:
                offset = math.radians(rng.uniform(*design.kick_heading_deg)) * rng.choice((-1, 1))
                yaw = wrap(yaw + offset)
                kicks.append({"step": t + 1, "kind": "heading", "value": offset})
            else:
                offset = rng.uniform(*design.kick_lateral_m) * rng.choice((-1, 1))
                position = position + offset * np.array([-math.sin(yaw), math.cos(yaw), 0.])
                position[2] = height_at(position[:2], position[2], yaw)
                kicks.append({"step": t + 1, "kind": "lateral", "value": offset})
    raise ValueError("Pursuit did not arrive within the step budget")


def follow_height(ground, agl, design, lookahead_m=10.):
    def height(xy, previous, heading):
        ahead = (xy[0] + lookahead_m * math.cos(heading), xy[1] + lookahead_m * math.sin(heading))
        wanted = float(np.nanmax(ground(np.array([xy, ahead])))) + agl
        return previous + float(np.clip(wanted - previous, -design.climb_limit_m, design.climb_limit_m))
    return height


def hold_height(z):
    return lambda xy, previous, heading: z


def instruction(obj, word, mode, rng):
    frame = int(rng.integers(len(FRAMES["en"])))
    en = FRAMES["en"][frame].format(what=obj["en"], where=WHERE[word][0], height=HEIGHT[mode][0])
    return en[0].upper() + en[1:], FRAMES["zh"][frame].format(what=obj["zh"], where=WHERE[word][1], height=HEIGHT[mode][1])


def propose(obj, others, ground, design, rng, require_relief=False):
    """One route for this object, or (None, reason). Rules follow protocol section 13."""
    low, high = distance_range(obj["width_m"], design)
    if high < low:
        return None, "too small"
    target = tuple(obj["center"])
    distance, around = rng.uniform(low, high), rng.uniform(-math.pi, math.pi)
    start_xy = (target[0] + distance * math.cos(around), target[1] + distance * math.sin(around))
    start_ground = float(ground(np.array([start_xy]))[0])
    if not math.isfinite(start_ground):
        return None, "outside terrain"
    bearing = math.radians(rng.uniform(-design.max_bearing_deg, design.max_bearing_deg))
    if not misses_when_flying_straight(distance, bearing):
        return None, "straight flight reaches it"
    yaw = wrap(math.atan2(target[1] - start_xy[1], target[0] - start_xy[0]) - bearing)
    mode = "follow" if rng.random() < .5 else "hold"
    agl = rng.uniform(20., 30.) if rng.random() < .6 else rng.uniform(35., 60.)
    start = (*start_xy, start_ground + agl)
    if mode == "hold" and start[2] < obj["top_z"] + ABOVE_TOP_M:
        return None, "below the target top"
    if abs(start[2] - obj["top_z"]) > distance:
        return None, "target outside the vertical view"
    if apparent_px(obj["width_m"], obj["height_m"], distance, start[2] - (obj["top_z"] - obj["height_m"])) < PATCH_PX:
        return None, "too small from the start"
    if not line_of_sight(start, target, obj["top_z"], ground):
        return None, "terrain blocks the view"
    if ambiguous(obj, others, start, yaw):
        return None, "ambiguous"
    height = follow_height(ground, agl, design) if mode == "follow" else hold_height(start[2])
    try:
        positions, yaws, actions, kicks = rollout(start, yaw, target, height, design, rng)
    except ValueError:
        return None, "did not arrive"
    below = ground(positions[:, :2])
    if not np.isfinite(below).all():
        return None, "left the terrain"
    if (positions[:, 2] - below).min() < MIN_CLEARANCE_M:
        return None, "too close to the ground"
    if positions[-1, 2] < obj["top_z"] + ABOVE_TOP_M:
        return None, "ends below the target top"
    relief = float(below.max() - below.min())
    if require_relief and relief < design.relief_min_m:
        return None, "too flat"
    en, zh = instruction(obj, direction_word(bearing), mode, rng)
    return dict(map=obj["map"], object=obj["key"], cls=obj["cls"], mode=mode, agl=agl, distance=distance, bearing=bearing, word=direction_word(bearing), target=list(target), top_z=obj["top_z"], width_m=obj["width_m"],
                positions=positions, yaws=yaws, actions=actions, kicks=kicks, relief_m=relief, climb_m=float(positions[:, 2].max() - positions[:, 2].min()), instruction=en, instruction_zh=zh), None


def generate(objects, ground, count, design, rng, tries_per_route=60):
    """Objects are drawn evenly so a map's rare classes are not crowded out. Returns routes and the count of each rejection."""
    routes, rejected = [], {}
    order = list(rng.permutation(len(objects)))
    i = tries = hilly = flat_streak = 0
    quota_open = design.relief_share > 0
    while len(routes) < count and tries < count * tries_per_route:
        obj = objects[order[i % len(order)]]
        i += 1
        tries += 1
        route, reason = propose(obj, objects, ground, design, rng, require_relief=quota_open and hilly < design.relief_share * (len(routes) + 1))
        if route is None:
            rejected[reason] = rejected.get(reason, 0) + 1
            flat_streak = flat_streak + 1 if reason == "too flat" else flat_streak
            if flat_streak >= design.relief_patience:
                quota_open = False
                rejected["relief quota gave up at route"] = len(routes)
            continue
        routes.append(route)
        flat_streak = 0
        hilly += route["relief_m"] >= design.relief_min_m
    return routes, rejected
