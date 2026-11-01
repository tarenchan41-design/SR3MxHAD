import json
import math
import numpy as np
from experiments.closed_loop_k import policy_state
from experiments.recovery_probe import OFFSETS, close_back, group_passes, offset_pose, pairs, plain, probe_point, probe_steps, required, spectral_radius, turn_back
from sr3m.geometry import integrate, local_vector, wrap
from tests.test_closed_loop_k import Fake, route


def pursuit(position, yaw, kind_free_gain=.1):
    """Steers toward a straight route along +X at y=0: right is +Y for yaw 0, so turning back is a negative dyaw."""
    heading_error, lateral_error = wrap(yaw), local_vector((0., position[1], 0.), yaw)[1]
    return np.array([2., -kind_free_gain * lateral_error, 0., -kind_free_gain * heading_error - .01 * lateral_error])


def test_positive_offsets_are_to_the_right_in_the_same_sense_as_a_positive_dyaw():
    position = np.array([1., 2., 3.])
    moved, yaw = offset_pose(position, 0., "lateral", 2.)
    assert np.allclose(moved, [1., 4., 3.]) and yaw == 0.
    same_direction, _ = integrate(tuple(position), math.radians(1), (2., 0., 0., 0.))
    assert same_direction[1] > position[1]
    turned_position, turned = offset_pose(position, 0., "heading", .2)
    assert np.allclose(turned_position, position) and np.isclose(turned, .2)


def test_a_policy_that_steers_back_reads_as_positive_and_one_that_steers_away_as_negative():
    origin = np.zeros(3)
    for kind, magnitude in OFFSETS:
        plus = pursuit(*offset_pose(origin, 0., kind, magnitude))
        minus = pursuit(*offset_pose(origin, 0., kind, -magnitude))
        assert turn_back(plus, minus) > 0
        assert turn_back(-plus, -minus) < 0
        if kind == "lateral":
            assert close_back(plus, minus) > 0


def test_the_route_turning_by_itself_cancels():
    plus, minus = np.array([2., 0., 0., .03]), np.array([2., 0., 0., .03])
    assert turn_back(plus, minus) == 0


def test_steering_back_really_reduces_the_error():
    position, yaw = offset_pose(np.zeros(3), 0., "lateral", 5.)
    for _ in range(80):
        position, yaw = integrate(tuple(position), yaw, pursuit(position, yaw))
    assert abs(position[1]) < 1.


def test_thresholds_match_the_written_table():
    needed = {(k, m): math.degrees(required(k, m)) for k, m in OFFSETS}
    assert np.isclose(needed[("heading", math.radians(10))], .5) and np.isclose(needed[("heading", math.radians(20))], 1.)
    assert round(needed[("lateral", 2.)], 2) == .29 and round(needed[("lateral", 5.)], 2) == .70


def test_half_pursuit_is_stable_and_steering_away_is_not():
    assert spectral_radius(.05, math.atan(.1) / 20 / 2) < 1
    assert spectral_radius(0., 0.) >= 1
    assert spectral_radius(-.05, -.0025) > 1


def test_group_needs_both_share_and_medians():
    values = {1: np.full(20, .2), 2: np.full(20, .2)}
    assert group_passes(values, {1: .1, 2: .1})["passed"]
    assert not group_passes(values, {1: .1, 2: .3})["passed"]
    assert not group_passes({1: np.r_[np.full(15, .2), np.full(5, -.2)], 2: np.r_[np.full(10, .2), np.full(10, -.2)]}, {1: .0, 2: .0})["passed"]


def test_probe_point_renders_once_per_pose_and_keeps_conditions_apart():
    positions, yaws = route(31)
    seen = {}
    def recording(name):
        def policy(images, state):
            seen.setdefault(name, []).append(state)
            return {"stop": False, "stop_probability": 0., "action": np.array([2., 0., 0., 0.], dtype=np.float32)}
        return policy
    backend = Fake()
    renders = []
    original = backend.render
    backend.render = lambda position, yaw: renders.append(1) or original(position, yaw)
    answer = probe_point(backend, {"B": recording("B"), "N": recording("N")}, positions, yaws, 20)
    assert len(renders) == 9 and len(answer["offsets"]) == 8
    on_path = policy_state(positions[20], float(yaws[20]), positions[0], float(yaws[0]))
    # B: full, image only, state only, per offset, after the on-path call.
    assert np.allclose(seen["B"][0], on_path) and np.allclose(seen["B"][2], on_path)
    assert not np.allclose(seen["B"][1], on_path) and np.allclose(seen["B"][1], seen["B"][3])
    json.dumps(plain(answer))


def test_a_pair_is_dropped_when_either_side_hits_or_is_too_low():
    positions, yaws = route(31)
    policy = lambda images, state: {"stop": False, "stop_probability": 0., "action": np.array([2., 0., 0., 0.])}
    answer = plain(probe_point(Fake(), {"B": policy, "N": policy}, positions, yaws, 20))
    rows, dropped = pairs({"points": [answer]})
    assert len(rows) == len(OFFSETS) and dropped == 0
    answer["offsets"][4]["hit"] = True
    answer["offsets"][7]["agl_m"] = 1.
    rows, dropped = pairs({"points": [answer]})
    assert len(rows) == 2 and dropped == 2


def test_probe_steps_stay_on_recorded_poses_with_an_action():
    assert probe_steps(101) == [25, 50, 75]
    assert max(probe_steps(4)) < 3
