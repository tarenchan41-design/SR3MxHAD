import json
import numpy as np
from experiments.closed_loop_k import policy_state
from experiments.short_rollout import rollout, window_errors, window_starts
from tests.test_closed_loop_k import Fake, route


def test_windows_end_on_recorded_poses():
    assert window_starts(45) == [0, 20]
    assert window_starts(11) == [0]
    assert window_starts(10) == []
    assert window_starts(46)[-1] + 10 <= 45


def test_expert_actions_have_zero_error_and_state_uses_the_route_start():
    positions, yaws = route(31)
    seen = []
    def policy(images, state):
        seen.append(state)
        return {"stop": False, "stop_probability": 0., "action": np.array([2., 0., 0., 0.], dtype=np.float32)}
    window = rollout(Fake(), positions, yaws, policy, 20)
    errors = window_errors({"windows": [window]}, positions, yaws)[0]
    assert errors["z"] == 0 and errors["h"] == 0 and errors["yaw"] == 0
    assert np.allclose(seen[0], policy_state(positions[20], 0., positions[0], 0.)) and np.isclose(seen[0][0], .4)
    json.dumps(window)


def test_stop_does_not_end_a_window_and_height_error_is_measured_at_the_end():
    positions, yaws = route(31)
    policy = lambda images, state: {"stop": True, "stop_probability": 1., "action": np.array([2., 0., .1, 0.])}
    window = rollout(Fake(), positions, yaws, policy, 0)
    errors = window_errors({"windows": [window]}, positions, yaws)[0]
    assert len(window["steps"]) == 10 and np.isclose(errors["z"], 1.) and np.isclose(errors["climb"], 1.)
