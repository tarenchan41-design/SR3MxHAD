import numpy as np
from experiments.data_audit import angle_gap, bearing_from, in_views, parse_region


def test_region_text_gives_the_last_bearing_and_distance():
    assert parse_region("生活区北排长条营房东侧约44米的上空") == (0., 44.)
    assert parse_region("西北侧约55米") == (225., 55.)
    assert parse_region("跑道南侧约127米处") == (90., 127.)
    assert parse_region("院内出发点 (427,-400,27) m") is None


def test_compass_matches_the_design_files_with_y_pointing_south():
    # mtn_a_low_u_001: landmark centre (-120, 195); 东侧约44米 ends at (-76, 197.2), 西北侧约55米 starts at (-164, 162).
    assert angle_gap(parse_region("东侧约44米")[0], bearing_from((-120, 195), (-76, 197.2))) <= 22.5
    assert angle_gap(parse_region("西北侧约55米")[0], bearing_from((-120, 195), (-164, 162))) <= 22.5


def test_views_split_ahead_below_and_behind():
    low, high = np.array([99., -1., 0.]), np.array([101., 1., 2.])
    ahead = in_views((0., 0., 1.), 0., low, high)
    assert ahead["front"] and not ahead["down"] and 3 < ahead["front_px"] < 8
    assert not in_views((0., 0., 1.), np.pi, low, high)["front"]
    below = in_views((100., 0., 50.), 0., low, high)
    assert below["down"] and not below["front"]


def test_a_big_box_counts_when_only_its_edge_is_in_frame():
    low, high = np.array([50., -5., 0.]), np.array([90., 300., 10.])
    assert in_views((0., 0., 5.), 0., low, high)["front"]


def test_review_sample_covers_every_category_and_type_and_is_repeatable():
    from experiments.data_review import TYPES, stratified
    rows = [{"category": c, "type": t, "map": f"{c}_{m}", "route": f"{c}_{t}_{m}_{i}"} for c in "abcdefgh" for t in TYPES for m in "xyz" for i in range(3)]
    first, second = stratified(rows), stratified(rows)
    assert [r["route"] for r in first] == [r["route"] for r in second]
    assert len(first) == 30 and len({r["route"] for r in first}) == 30
    assert {(r["category"], r["type"]) for r in first} == {(c, t) for c in "abcdefgh" for t in TYPES}
