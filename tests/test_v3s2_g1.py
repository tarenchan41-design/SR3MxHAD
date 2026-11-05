import numpy as np

from experiments.v3s2_g1 import G1, select, window


def route(i, mode, obj, steps=60):
    return {"id": f"M_{i:03d}", "mode": mode, "object": obj, "positions": np.zeros((steps, 3))}


def test_window_is_32_shuffled_steps_from_13():
    w = window(np.random.default_rng(0))
    assert sorted(w) == list(range(13, 45)) and w != sorted(w)


def test_selection_takes_three_per_mode_on_distinct_targets_skips_failed_checks_and_short_routes():
    candidates = [route(i, "follow" if i % 2 else "hold", f"o{i % 20}") for i in range(80)]
    candidates[1]["positions"] = np.zeros((40, 3))  # too short for the window
    failing = {"M_003"}
    g1, preview, checks = select(candidates, lambda r: "collision: x" if r["id"] in failing else None)
    assert [r["mode"] for r in g1].count("follow") == G1["routes_per_mode"] and [r["mode"] for r in g1].count("hold") == G1["routes_per_mode"]
    assert len(preview) == 12 and checks["M_003"] == "collision: x" and all(r["id"] not in ("M_001", "M_003") for r in g1)
    objects = [r["object"] for r in g1 + preview]
    assert len(objects) == len(set(objects))


def test_selection_repeats_targets_rather_than_fall_short():
    candidates = [route(i, "follow" if i % 2 else "hold", f"o{i % 5}") for i in range(60)]
    g1, preview, _ = select(candidates, lambda r: None)
    assert len(g1) == 6 and len(preview) == 12


def test_bulk_render_splits_every_tenth_training_route_to_val_and_keeps_dev_and_test_apart():
    from experiments.v3s2_capture import split_of
    assert [split_of("L_MTR_A", k) for k in range(20)].count("val") == 2 and split_of("L_DES_D_v3", 9) == "val"
    assert split_of("L_MTR_B", 9) == "dev" and split_of("L_DES_C_v3", 9) == "test" and split_of("L_MTR_A", 0) == "train"
