import math
import re

import numpy as np

from experiments.protocol import terrain_following
from sr3m.geometry import integrate
from sr3m.routes_v3 import Design, ambiguous, bearing_to, direction_word, distance_range, follow_height, generate, hold_height, instruction, line_of_sight, misses_when_flying_straight, propose, rollout


def flat(xy):
    return np.zeros(len(np.atleast_2d(xy)))


def building(key="a", cls="long_building", center=(0., 0.), width=30.):
    return dict(map="M", key=key, cls=cls, en="the long building", zh="长条建筑", center=list(center), top_z=8., height_m=8., width_m=width)


def test_positive_bearing_is_to_the_right_in_the_same_sense_as_a_positive_dyaw():
    # Yaw 0 faces +X; +Y is to the right, and a positive dyaw turns toward it.
    assert bearing_to((0., 0.), 0., (10., 10.)) > 0
    assert direction_word(math.radians(30)) == "right" and direction_word(math.radians(-30)) == "left" and direction_word(math.radians(5)) == "ahead"


def test_pursuit_arrives_above_the_target_and_labels_turn_toward_it():
    design = Design()
    target = (60., 40.)
    positions, yaws, actions, kicks = rollout((0., 0., 30.), 0., target, hold_height(30.), design, np.random.default_rng(0))
    assert np.hypot(*(positions[-1, :2] - target)) <= 1. and kicks == [] and len(actions) == len(positions) - 1
    assert actions[0, 3] > 0
    assert (np.abs(actions[:, 3]) <= math.radians(design.max_yaw_deg) + 1e-9).all() and (actions[:, 0] <= 2. + 1e-9).all()


def test_without_kicks_labels_equal_the_pose_differences():
    positions, yaws, actions, _ = rollout((0., 0., 30.), .3, (80., -30.), hold_height(30.), Design(), np.random.default_rng(0))
    for t in (0, 5, len(actions) - 1):
        p, y = integrate(tuple(positions[t]), yaws[t], actions[t])
        assert np.allclose(p, positions[t + 1]) and np.isclose(y, yaws[t + 1])


def test_kicks_keep_the_label_as_the_pursuit_action_at_the_kicked_pose_and_stay_away_from_the_end():
    design = Design(kick_every=5.)
    target = (200., 30.)
    positions, yaws, actions, kicks = rollout((0., 0., 30.), 0., target, hold_height(30.), design, np.random.default_rng(1))
    assert len(kicks) >= 3 and np.hypot(*(positions[-1, :2] - target)) <= 1.
    for kick in kicks:
        t = kick["step"]
        expected = np.clip(design.gain * bearing_to(positions[t], yaws[t], target), -math.radians(design.max_yaw_deg), math.radians(design.max_yaw_deg))
        assert np.isclose(actions[t, 3], expected)
        assert np.hypot(*(positions[t - 1, :2] - target)) > design.kick_margin_m


def test_rotate_first_turns_on_the_spot_before_moving():
    positions, _, actions, _ = rollout((0., 0., 30.), 0., (50., 50.), hold_height(30.), Design(rotate_first=True), np.random.default_rng(0))
    assert actions[0, 0] == 0. and np.allclose(positions[1, :2], positions[0, :2]) and actions[-1, 0] > 0.


def test_distance_follows_the_patch_rule():
    design = Design()
    assert distance_range(15., design) == (60., 120.)
    assert distance_range(100., design) == (60., 250.)
    low, high = distance_range(5., design)
    assert high < low


def test_straight_flight_must_miss_the_success_radius():
    assert not misses_when_flying_straight(60., math.radians(10))
    assert misses_when_flying_straight(200., math.radians(10))
    assert misses_when_flying_straight(60., math.radians(35))


def test_terrain_can_block_the_view():
    ridge = lambda xy: np.where(np.abs(np.atleast_2d(xy)[:, 0] - 50.) < 10., 100., 0.)
    assert line_of_sight((0., 0., 30.), (100., 0.), 0., flat)
    assert not line_of_sight((0., 0., 30.), (100., 0.), 0., ridge)


def test_same_class_under_the_same_direction_word_is_ambiguous():
    target = building("a", center=(100., 30.))
    assert ambiguous(target, [target, building("b", center=(100., 50.))], (0., 0., 30.), 0.)
    assert not ambiguous(target, [target, building("b", center=(100., -30.))], (0., 0., 30.), 0.)
    assert not ambiguous(target, [target, building("b", "hangar", center=(100., 50.))], (0., 0., 30.), 0.)
    assert not ambiguous(target, [target, building("b", center=(-100., 30.))], (0., 0., 30.), 0.)


def test_following_height_keeps_climbing_limited_and_holding_does_not_change():
    design = Design()
    wall = lambda xy: np.where(np.atleast_2d(xy)[:, 0] > 5., 50., 0.)
    assert follow_height(wall, 20., design)((0., 0.), 20., 0.) == 20. + design.climb_limit_m
    assert hold_height(40.)((0., 0.), 40., 0.) == 40.


def test_instructions_never_use_compass_or_numbers_and_keep_the_cohort_phrase():
    rng = np.random.default_rng(0)
    for mode in ("follow", "hold"):
        for word in ("ahead", "left", "right"):
            for _ in range(5):
                en, zh = instruction(building(), word, mode, rng)
                assert not re.search(r"\b(north|south|east|west)", en.lower()) and not re.search(r"[东南西北]", zh) and not re.search(r"\d", en + zh)
                assert en[0].isupper() and terrain_following({"instruction_zh": zh}) == (mode == "follow")


def test_generated_routes_obey_the_rules():
    objects = [building("a", center=(0., 0.)), building("b", "hangar", center=(400., 0.), width=20.)]
    routes, rejected = generate(objects, flat, 20, Design(kick_every=25.), np.random.default_rng(0))
    assert len(routes) == 20 and sum(rejected.values()) >= 0
    for r in routes:
        low, high = distance_range(r["width_m"], Design())
        assert low <= r["distance"] <= high and abs(math.degrees(r["bearing"])) <= 35.
        assert misses_when_flying_straight(r["distance"], r["bearing"])
        assert np.hypot(*(r["positions"][-1, :2] - r["target"])) <= 1. and r["positions"][-1, 2] >= r["top_z"] + 8.
        assert r["positions"][:, 2].min() >= 10.


def test_a_hold_route_cannot_end_below_the_target_top():
    tall = dict(building(center=(0., 0.), width=30.), top_z=500.)
    rng = np.random.default_rng(0)
    results = [propose(tall, [tall], flat, Design(), rng) for _ in range(40)]
    assert all(route is None for route, _ in results)


def test_start_kick_always_happens_early_and_arrival_still_works():
    design = Design(start_kick_steps=3)
    for seed in range(5):
        positions, yaws, actions, kicks = rollout((0., 0., 30.), 0., (150., 40.), hold_height(30.), design, np.random.default_rng(seed))
        assert len(kicks) == 1 and 1 <= kicks[0]["step"] <= 3 and kicks[0]["kind"] == "heading" and np.hypot(*(positions[-1, :2] - (150., 40.))) <= 1.


def test_relief_quota_is_met_where_the_ground_allows_and_gives_up_where_it_does_not():
    obj = building(center=(0., 0.))
    routes, rejected = generate([obj], flat, 10, Design(relief_share=.5, relief_patience=20), np.random.default_rng(0))
    assert len(routes) == 10 and rejected["too flat"] >= 20 and "relief quota gave up at route" in rejected
    wavy = lambda xy: 10 + 10 * np.sin(np.atleast_2d(xy)[:, 0] / 15.)
    routes, rejected = generate([obj], wavy, 20, Design(relief_share=.5), np.random.default_rng(0))
    assert sum(r["relief_m"] >= 15 for r in routes) >= 10 and "relief quota gave up at route" not in rejected


def test_two_early_kicks_land_on_different_steps():
    design = Design(start_kick_steps=6, start_kicks=2)
    _, _, _, kicks = rollout((0., 0., 30.), 0., (200., 40.), hold_height(30.), design, np.random.default_rng(3))
    assert len({k["step"] for k in kicks}) == 2 and all(k["step"] <= 6 for k in kicks)


def test_a_flat_object_far_away_looks_too_small_and_a_tall_one_does_not():
    from sr3m.routes_v3 import apparent_px
    # The reviewed helipad: 27 m wide, flat, 133 m away, 42 m below.
    assert apparent_px(27., .5, 133., 42.) < 32 < apparent_px(27., .5, 60., 30.)
    assert apparent_px(27., .5, 200., 25.) < apparent_px(27., 10., 200., 25.)
    assert np.isclose(apparent_px(10., 10., 80., 0.), 256 * 10 / 80)


def test_routes_to_flat_objects_start_closer():
    pad = dict(building("pad", "helipad", width=27.), top_z=.5, height_m=.5)
    routes, rejected = generate([pad], flat, 20, Design(), np.random.default_rng(0))
    assert rejected.get("too small from the start", 0) > 0 and max(r["distance"] for r in routes) < 130

