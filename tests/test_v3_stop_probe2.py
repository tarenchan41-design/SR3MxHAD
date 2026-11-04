import numpy as np

from experiments import v3_stop_vlm
from experiments.v3_stop_probe2 import VARIANTS, VLM_QUESTION, p_yes, positive_weight, seen_classes, soft_target


def test_soft_target_is_one_at_the_target_and_falls_with_horizontal_distance():
    y = soft_target(np.array([0., 5., 10., 40.]), 5.)
    assert y[0] == 1 and abs(y[1] - np.exp(-.5)) < 1e-6 and y[2] < y[1] and y[3] < 1e-6


def test_positive_weight_is_the_negative_to_positive_ratio():
    assert positive_weight([1, 0, 0, 0, 1, 0]) == 2.


def test_seen_classes_come_only_from_the_other_maps():
    objects = {"a": {"map": "L_MTN_A", "cls": "helipad"}, "b": {"map": "L_DES_A", "cls": "hut"}, "c": {"map": "L_MTN_D", "cls": "bridge"}}
    assert seen_classes(objects, ["L_MTN_A", "L_MTN_D"]) == {"helipad", "bridge"}


def test_p_yes_is_the_yes_share_of_the_yes_and_no_mass_and_matches_the_standalone_copy():
    yes, no = np.array([[2., 0.], [-50., -50.]]), np.array([[0., 0.], [10., 10.]])
    p = p_yes(yes, no)
    assert abs(p[0] - (np.e ** 2 + 1) / (np.e ** 2 + 3)) < 1e-9 and p[1] < 1e-20
    assert np.allclose(p, v3_stop_vlm.p_yes(yes, no))


def test_round_two_variants_all_build_on_v1_and_the_question_names_the_phrase():
    assert all(v["parts"][:4] == VARIANTS["V1"]["parts"] for v in VARIANTS.values())
    assert '"round helipad"' in VLM_QUESTION.format("round helipad")
