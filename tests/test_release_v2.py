import numpy as np
import pytest

from sr3m.release_v2 import PREFIX, build

CONFIG = {"seed": 1, "seen_validation_fraction": 0.15, "near_duplicate_xy_tolerance_m": 2.0,
          "near_duplicate_bidirectional_coverage": 0.9, "near_duplicate_length_ratio": 0.85}


def base_row(i, category, scene, split):
    return {"episode_id": f"old/{category}{i:03d}", "category": category, "scene_group_id": scene,
            "route_group_id": f"{category}{i}", "split": split}


def desert_row(name, i, x0, y0=0.0):
    xy = np.c_[np.linspace(x0, x0 + 100, 20), np.full(20, y0)]
    return {"episode_id": f"new/{name}-{i:03d}", "category": "desert", "scene_group_id": PREFIX + name,
            "route_id": f"r{i}", "source_schema_version": 3, "_positions": np.c_[xy, np.zeros(20)],
            "_hashes": [f"{name}{i}-{k}" for k in range(40)]}


def fixture():
    base = ([base_row(i, "mountain", "M_A", "train") for i in range(8)] + [base_row(8, "mountain", "M_A", "val_seen"),
            base_row(9, "mountain", "M_C", "val_unseen"), base_row(10, "forest", "F_B", "test")]
            + [base_row(i, "desert", "OLD_DESERT", "train") for i in range(20, 25)])
    desert = [desert_row(m, i, 1000.0 * i) for m in ("L_DES_A", "L_DES_B", "L_DES_C", "L_DES_D") for i in range(10)]
    # Same horizontal path at another altitude must stay with its twin.
    twin = desert_row("L_DES_A", 99, 0.0)
    twin["_positions"][:, 2] = 50.0
    return base, desert + [twin]


def test_v1_rows_kept_old_desert_removed_heldout_whole():
    base, desert = fixture()
    rows, edges = build(base, desert, "L_DES_C", CONFIG)
    by_id = {r["episode_id"]: r for r in rows}
    assert all(by_id[r["episode_id"]] == r for r in base if r["category"] != "desert")
    assert not any(r["scene_group_id"] == "OLD_DESERT" for r in rows)
    assert {r["split"] for r in rows if r["scene_group_id"] == PREFIX + "L_DES_C"} == {"val_unseen"}
    for name in ("L_DES_A", "L_DES_B", "L_DES_D"):
        assert {r["split"] for r in rows if r["scene_group_id"] == PREFIX + name} == {"train", "val_seen"}
    a = [r for r in rows if r["episode_id"] in ("new/L_DES_A-000", "new/L_DES_A-099")]
    assert a[0]["route_group_id"] == a[1]["route_group_id"] and a[0]["split"] == a[1]["split"]
    assert any(e["reason"] == "near_route_xy" for e in edges)


def test_seen_map_with_one_route_group_is_rejected():
    base, desert = fixture()
    lonely = [r for r in desert if not r["episode_id"].startswith("new/L_DES_B")] + [desert_row("L_DES_B", 0, 0.0)]
    with pytest.raises(ValueError, match="only one indivisible route group"):
        build(base, lonely, "L_DES_C", CONFIG)
