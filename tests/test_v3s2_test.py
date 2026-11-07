"""Test opening (protocol 17.10): per-route quantities of 17.7 and the 17.8 judging of T1/T2/T3."""
import numpy as np
import pytest

import experiments.v3s2_test as t


def step(x, z, agl):
    return {"position_m": [x, 0., z], "agl_m": agl}


def flight(history, agl0=10., **kw):
    base = {"SR20": True, "SR10": True, "SR5": False, "stopped": True, "horizontal_m": 3., "failure": None,
            "start_agl_m": agl0, "e_m": 1., "history": history}
    return dict(base, **kw)


def test_follow_quantities():
    route = {"mode": "follow", "target": [100., 0.]}
    # ground rises 0 -> 4 -> 8 under the path; the drone climbs half of it (z0 = 10, AGL0 = 10, ground0 = 0)
    q = t.route_quantities(flight([step(10, 12, 8), step(20, 14, 6)]), route, (0., 0., 10.))
    assert q["E_hold"] == pytest.approx(6.)                     # mean |ground0 - ground_t| = (4 + 8) / 2
    assert q["beta"] == pytest.approx(.5)                       # z - z0 = 2, 4 against ground - ground0 = 4, 8
    assert not q["reached"] and "hold_dz" not in q


def test_follow_without_ground_change_has_no_beta():
    q = t.route_quantities(flight([step(10, 11, 11)]), {"mode": "follow", "target": [15., 0.]}, (0., 0., 10.))
    assert q["beta"] is None and q["E_hold"] == 0. and q["reached"]


def test_hold_quantities():
    q = t.route_quantities(flight([step(10, 10.5, 10), step(20, 9., 10)]), {"mode": "hold", "target": [500., 0.]}, (0., 0., 10.))
    assert q["hold_dz"] == pytest.approx(.75) and "E" not in q


def test_relative_improvement_and_bootstrap():
    r = t.bootstrap(t.relative_improvement, [1., 2., 3.], [2., 4., 6.])
    assert r["estimate"] == pytest.approx(.5) and r["ci"] == pytest.approx([.5, .5]) and r["routes"] == 3


def rows(values):
    return {f"r{i}": {"E": v} for i, v in enumerate(values)}


def test_compare_averages_seeds_inside_routes_and_reports_each_seed():
    flights = {"B4_seed42": rows([1., 1.]), "B4_seed43": rows([3., 3.]), "B4_seed44": rows([2., 2.]),
               "A_seed42": rows([4., 4.]), "A_seed43": rows([4., 4.]), "A_seed44": rows([2., 2.])}
    r = t.compare(flights, "B4", "A", "E", ["r0", "r1"], t.relative_improvement)
    assert r["estimate"] == pytest.approx(.4)                   # 1 - 2 / (10 / 3)
    assert r["per_seed"] == pytest.approx({"42": .75, "43": .25, "44": 0.})
    assert not t.positive(r)                                    # seed 44 is not > 0
    flights["B4_seed44"] = rows([1., 1.])
    assert t.positive(t.compare(flights, "B4", "A", "E", ["r0", "r1"], t.relative_improvement))  # every seed > 0 now


def test_policies_cover_registered_runs():
    names = [n for n, _ in t.policies()]
    assert names[0] == "oracle" and len(names) == 22 and len(set(names)) == 22
    assert {"B40_seed42", "B4_frac25", "B4_frac50"} <= set(names)
    assert t.variant_of("B4_frac25") == "B4" and t.variant_of("DG_seed43") == "DG"
