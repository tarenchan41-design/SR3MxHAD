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



def test_sight_line_stops_short_of_the_target_and_splits_follow_rank():
    from experiments.v3_capture import sight_end, split_of
    end = sight_end((0., 0., 30.), (100., 0.), 10., 20.)
    assert np.isclose(np.linalg.norm(end - np.array([100., 0., 10.])), 13.) and end[0] < 100.
    assert [split_of("L_MTN_A", k) for k in range(10)].count("val") == 1 and split_of("L_MTN_B", 9) == "dev"


def test_b0_never_sees_the_pose():
    import torch
    from had.model import NO_POSE, TerrainHAD
    from had.tokenizer import Tokenizer
    assert "B0" in NO_POSE
    tok = Tokenizer.fit(["fly to the tank"], max_length=8)
    model = TerrainHAD("B0", tok, cached_only=True).eval()
    features = {v: torch.randn(2, 3840) for v in ("front", "down")}
    tokens = torch.tensor([tok.encode("fly to the tank")["input_ids"]] * 2)
    a = model(features, tokens, torch.zeros(2, 5))["action"]
    b = model(features, tokens, torch.randn(2, 5))["action"]
    assert torch.allclose(a, b)


def test_pose_read_back_from_the_policy_state_and_the_reference_oracle_arrives():
    from experiments.closed_loop_k import policy_state
    from experiments.v3_fly import pose_from_state, reference_policy
    start, start_yaw = np.array([10., -5., 40.]), .7
    pos, yaw = np.array([60., 30., 44.]), 1.1
    back, back_yaw = pose_from_state(policy_state(pos, yaw, start, start_yaw), start, start_yaw)
    assert np.allclose(back, pos) and np.isclose(math.remainder(back_yaw - yaw, 2 * math.pi), 0)
    positions, yaws, _, _ = rollout(tuple(start), start_yaw, (150., 80.), hold_height(40.), Design(), np.random.default_rng(0))
    route = {"mode": "hold", "target": [150., 80.]}
    policy = reference_policy("oracle", route, positions, yaws, flat)
    p, y = positions[0].copy(), float(yaws[0])
    for _ in range(400):
        d = policy({}, policy_state(p, y, positions[0], float(yaws[0])))
        if d["stop"]:
            break
        p, y = integrate(tuple(p), y, d["action"]); p = np.array(p)
    assert d["stop"] and np.hypot(p[0] - 150., p[1] - 80.) <= 1.


def test_plural_objects_take_are_and_them():
    rng = np.random.default_rng(0)
    sheds = dict(building(), en="the sheds with orange roofs", zh="橙色屋顶的棚屋")
    texts = {instruction(sheds, "left", "hold", rng)[0] for _ in range(30)}
    assert any(" are " in t for t in texts) and all(" is ahead" not in t and "above it" not in t for t in texts)
    camp = dict(building(), en="the camp of three black tents")
    assert all(" are " not in instruction(camp, "left", "hold", rng)[0] for _ in range(20))


def test_targets_far_below_are_rejected_before_they_reach_the_bottom_edge():
    deep = dict(building(center=(0., 0.)), top_z=-150., height_m=5.)
    rng = np.random.default_rng(0)
    reasons = [propose(deep, [deep], lambda xy: np.where(np.hypot(*np.atleast_2d(xy).T) < 40, -155., 0.), Design(), rng)[1] for _ in range(40)]
    assert "target near the bottom of the view" in reasons



def test_rule_8_rejects_a_same_class_object_near_the_path_but_not_other_classes_or_far_ones():
    from sr3m.routes_v3 import same_class_near
    target = building("a", center=(200., 0.))
    path = np.c_[np.linspace(0., 200., 101), np.zeros(101)]
    near = dict(building("b", center=(100., 45.)), footprint_m=[20., 20.])  # edge 35 m from the path
    far = dict(building("c", center=(100., 55.)), footprint_m=[20., 20.])  # edge 45 m
    assert same_class_near(target, [target, near], path, 40.) == "b"
    assert same_class_near(target, [target, far], path, 40.) is None
    assert same_class_near(target, [target, dict(near, cls="hangar")], path, 40.) is None
    beside_start = dict(building("d", center=(-30., 0.)), width_m=10.)  # behind the start, still within 40 m of it
    assert same_class_near(target, [target, beside_start], path, 40.) == "d"


def test_rule_8_in_generation_and_distractors_are_never_targets():
    target = building("a", center=(0., 0.))
    beside = dict(building("d", center=(0., 30.)), role="distractor", width_m=10.)  # every path ends at the target, 25 m from its edge
    routes, rejected = generate([target, beside], flat, 5, Design(same_class_clear_m=40.), np.random.default_rng(0), tries_per_route=100)
    assert routes == [] and rejected.get("same class near the path", 0) > 0
    routes, _ = generate([target, dict(beside, cls="hangar")], flat, 10, Design(same_class_clear_m=40.), np.random.default_rng(0))
    assert len(routes) == 10 and {r["object"] for r in routes} == {"a"}


def test_hold_deviation_is_the_ground_change_from_the_start():
    from sr3m.routes_v3 import hold_deviation
    median, mean = hold_deviation([10., 10., 16., 20., 30.])
    assert median == 6. and np.isclose(mean, (0 + 0 + 6 + 10 + 20) / 5)


def test_rule_9_quota_holds_for_terrain_following_routes_and_gives_up_on_flat_ground():
    obj = building(center=(0., 0.))
    slope = lambda xy: .15 * np.atleast_2d(xy)[:, 0]
    design = Design(follow_dev_share=.5, follow_dev_min_m=5.)
    routes, rejected = generate([obj], slope, 30, design, np.random.default_rng(0))
    follow = [r for r in routes if r["mode"] == "follow"]
    assert len(routes) == 30 and follow and sum(r["hold_dev_median_m"] >= 5. for r in follow) >= len(follow) / 2
    assert "rule 9 quota gave up at route" not in rejected
    routes, rejected = generate([obj], flat, 30, Design(follow_dev_share=.5, relief_patience=5), np.random.default_rng(0))
    assert len(routes) == 30 and rejected["follow route too flat"] >= 5 and "rule 9 quota gave up at route" in rejected


def test_an_explicit_plural_flag_wins_over_the_head_noun_guess():
    rng = np.random.default_rng(0)
    rows = dict(building(), en="the rows of four long white arched greenhouses on a valley terrace", plural=True)
    assert all("them" in instruction(rows, "left", "hold", rng)[0] for _ in range(20))
    dome = dict(building(), en="the white radar dome on a small grey building on a mountain shoulder", plural=False)
    assert all(" are " not in t and "them" not in t for t in (instruction(dome, "left", "hold", rng)[0] for _ in range(20)))


def test_balanced_rounds_spread_routes_evenly_and_a_hopeless_target_sits_out():
    easy = [building(k, cls=k, center=(400. * i, 0.)) for i, k in enumerate("abcd")]
    routes, _ = generate(easy, flat, 20, Design(balance_rounds=True), np.random.default_rng(0))
    counts = [sum(r["object"] == k for r in routes) for k in "abcd"]
    assert counts == [5, 5, 5, 5]
    for start in range(0, 20, 4):  # every round gives each target one route before any gets another
        assert {r["object"] for r in routes[start:start + 4]} == set("abcd")
    tiny = dict(building("t", cls="t", center=(2000., 0.)), width_m=3.)  # never fits the patch rule
    routes, rejected = generate([*easy, tiny], flat, 12, Design(balance_rounds=True, round_patience=30), np.random.default_rng(0))
    assert len(routes) == 12 and rejected["sat out a round"] >= 2 and "t" not in {r["object"] for r in routes}
    assert [sum(r["object"] == k for r in routes) for k in "abcd"] == [3, 3, 3, 3]


def test_short_names_use_the_appearance_alone_for_about_half_the_routes():
    obj = dict(building(), en="the long building on a terrace", zh="台地上的长条建筑", en_short="the long building", zh_short="长条建筑", plural=False)
    routes, _ = generate([obj], flat, 60, Design(short_name_share=.5), np.random.default_rng(0))
    short = [r for r in routes if r["name_form"] == "short"]
    assert 15 <= len(short) <= 45
    assert all("terrace" not in r["instruction"] and "台地" not in r["instruction_zh"] for r in short)
    assert all("terrace" in r["instruction"] and "台地" in r["instruction_zh"] for r in routes if r["name_form"] == "full")


def test_stage_2_object_table_names_have_no_compass_words_and_short_names_are_prefixes_of_meaning():
    import json
    from pathlib import Path
    rows = json.loads(Path("configs/v3s2_objects.json").read_text(encoding="utf-8"))
    targets = [r for r in rows if r["role"] == "target"]
    assert len(targets) >= 8 * 15
    for r in targets:
        assert not re.search(r"\b(north|south|east|west)", (r["en"] + r["en_short"]).lower()) and not re.search(r"[东南西北]", r["zh"] + r["zh_short"])
        assert r["en"] == f'{r["en_short"]} {r["site_en"]}' and r["zh"] == r["site_zh"] + r["zh_short"]
