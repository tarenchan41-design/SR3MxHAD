"""Stage 2 dev flights: map name from route_group_id, E of section 17.7, the grid rule of 17.9."""
import json

import numpy as np

import experiments.v3s2_fly as fly


def test_map_of_long_names():
    assert fly.map_of({"route_group_id": "L_DES_B_v3/DES_B_tent_1"}) == "L_DES_B_v3"
    assert fly.map_of({"route_group_id": "L_MTR_B/MTR_B_solar_array_1"}) == "L_MTR_B"


def test_route_e_is_mean_abs_agl_change():
    result = {"start_agl_m": 30., "history": [{"agl_m": 32.}, {"agl_m": 26.}, {"stop_probability": .9}]}
    assert fly.route_e(result) == 3.
    assert fly.route_e({"start_agl_m": None, "history": [{"agl_m": 1.}]}) is None


def flight(rid, sr20, e):
    return {"route_id": rid, "SR20": sr20, "SR10": sr20, "SR5": False, "stopped": True, "failure": None, "horizontal_m": 3., "e_m": e, "history": []}


def run_report(tmp_path, monkeypatch, b, b4):
    routes = [({"route_id": f"L_MTR_B_{i:03d}", "route_group_id": "L_MTR_B/x", "mode": "follow", "unseen_class": False, "name_form": "full"}, None, None) for i in range(4)]
    monkeypatch.setattr(fly, "dev_routes", lambda prepared: routes)
    for name, rows in (("B", b), ("B4", b4)):
        (tmp_path / name).mkdir()
        for i, (sr, e) in enumerate(rows):
            (tmp_path / name / f"L_MTR_B_{i:03d}.json").write_text(json.dumps(flight(f"L_MTR_B_{i:03d}", sr, e)))
    args = type("A", (), {"out": str(tmp_path), "prepared": None, "summary": "summary.json"})
    fly.report(args)
    return json.loads((tmp_path / "summary.json").read_text())["decisions"]


def test_grid_rule_picks_4x4_only_when_higher_and_e_within_tolerance(tmp_path, monkeypatch):
    d = run_report(tmp_path, monkeypatch, [(True, 2.), (False, 2.), (False, 2.), (False, 2.)], [(True, 2.1), (True, 2.1), (False, 2.1), (False, 2.1)])
    assert d["grid"]["choice"] == "4x4" and d["dev_gate"]["model"] == "B4" and d["dev_gate"]["SR20 >= 0.5"]


def test_grid_rule_tie_and_worse_e_keep_2x2(tmp_path, monkeypatch):
    d = run_report(tmp_path / "tie", monkeypatch, [(True, 2.), (False, 2.)] * 2, [(True, 1.), (False, 1.)] * 2) if (tmp_path / "tie").mkdir() is None else None
    assert d["grid"]["choice"] == "2x2"
    d = run_report(tmp_path / "e", monkeypatch, [(False, 2.)] * 4, [(True, 2.3)] * 4) if (tmp_path / "e").mkdir() is None else None
    assert d["grid"]["choice"] == "2x2" and np.isclose(d["grid"]["E_ratio_B4_over_B"], 1.15)
