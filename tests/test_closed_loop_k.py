import ast
import math
from pathlib import Path
import numpy as np
from experiments.closed_loop_k import fly, policy_state
from sr3m.geometry import integrate

ROOT = Path(__file__).resolve().parents[1]


class Fake:
    def __init__(self, ground=0., hit_after=None):
        self.ground, self.hit_after, self.sweeps, self.renders = ground, hit_after, 0, 0
    def render(self, position, yaw):
        self.renders += 1
        return {"front": position, "down": position}
    def sweep(self, start, end):
        self.sweeps += 1
        return {"hit": self.hit_after is not None and self.sweeps > self.hit_after, "actor": "rock"}
    def ground_z_m(self, position):
        return self.ground


def route(n=6):
    return np.array([[2. * i, 0., 100.] for i in range(n)]), np.zeros(n)


def test_expert_retraces_route_without_rendering():
    positions, yaws = route()
    backend = Fake()
    result = fly(backend, positions, yaws)
    assert result["passed"] and result["distance_goal_m"] < 1e-9 and backend.renders == 0 and result["steps"] == 5
    assert result["ordered_progress_fraction"] == 1.


def test_policy_moves_by_integration_and_stops():
    positions, yaws = route()
    calls = []
    def policy(images, state):
        calls.append(state)
        return {"stop": len(calls) == 6, "stop_probability": float(len(calls) == 6), "action": np.array([2., 0., 0., 0.])}
    result = fly(Fake(), positions, yaws, policy)
    assert result["SR20"] and result["stopped"] and result["distance_goal_m"] < 1e-9
    assert np.allclose(calls[1], [.02, 0., 0., 0., 1.])


def test_collision_and_ground_fail_the_flight():
    positions, yaws = route()
    assert fly(Fake(hit_after=1), positions, yaws)["failure"] == "collision: rock"
    low = fly(Fake(ground=99.), positions, yaws)
    assert low["failure"].startswith("ground") and not low["passed"]


def test_budget_runs_out_without_stop():
    positions, yaws = route(3)
    policy = lambda images, state: {"stop": False, "stop_probability": 0., "action": np.zeros(4)}
    result = fly(Fake(), positions, yaws, policy)
    assert result["budget_exhausted"] and result["steps"] == 2 * 2 + 20 and not result["SR20"]


def test_policy_state_matches_cached_convention():
    assert np.allclose(policy_state((100., 0., 50.), math.pi / 2, (0., 0., 0.), 0.), [1., 0., .5, 1., 0.])


def test_integrate_turns_right_after_translating():
    position, yaw = integrate((0., 0., 0.), 0., (2., 0., 1., math.pi / 2))
    assert np.allclose(position, (2., 0., 1.)) and math.isclose(yaw, math.pi / 2)


def test_ue_server_imports_stay_inside_the_engine_environment():
    tree = ast.parse((ROOT / "sim_k/ue_server.py").read_text(encoding="utf8"))
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not names & {"numpy", "torch", "PIL", "had", "experiments", "sr3m"}
