import numpy as np

from experiments.v3_stop_probe import auroc, bag_of_words, calibrate, first_trigger, phrase_of


def test_first_trigger_reports_the_distance_at_the_first_crossing():
    p, d = np.array([.1, .6, .2, .9]), np.array([80., 50., 30., 2.])
    assert first_trigger(p, d, .5) == 50. and first_trigger(p, d, .95) is None


def test_calibration_prefers_the_threshold_that_stops_inside_the_radius_and_keeps_the_higher_on_ties():
    # Route 1 has an early false spike at .6 and a late true peak at .8; route 2 peaks only at the end at .5.
    routes = [(np.array([.6, .1, .8]), np.array([100., 50., 3.])), (np.array([.1, .2, .5]), np.array([90., 40., 1.]))]
    assert calibrate(routes) == .8  # .8 and .5 both stop one route inside (1 or 2); the tie keeps the higher
    routes.append((np.array([.1, .5]), np.array([60., 4.])))
    assert calibrate(routes) == .5  # now .5 stops routes 2 and 3 inside, .8 only route 1
    single = [(np.array([.1, .7, .9]), np.array([90., 10., 2.]))]
    assert calibrate(single) == .9


def test_auroc_handles_ties_and_perfect_order():
    assert auroc([.1, .2, .8, .9], [0, 0, 1, 1]) == 1.
    assert auroc([.5, .5, .5, .5], [0, 1, 0, 1]) == .5


def test_word_bag_drops_words_outside_the_fitted_instructions():
    x, vocab = bag_of_words(["fly to the bridge", "fly to the hut"], ["fly to the hut"])
    assert "bridge" not in vocab and x[0].sum() == 3 and x[1].sum() == 4


def test_target_phrase_must_appear_in_the_instruction():
    objects = {"k": {"en": "the steel bridge over the river"}}
    route = {"route_id": "r", "object": "k", "instruction": "The steel bridge over the river is ahead on your right."}
    assert phrase_of(route, objects) == "steel bridge over the river"


def test_chunked_standardization_matches_numpy():
    from experiments.v3_stop_probe import standardize
    x = np.random.default_rng(0).normal(3., 2., (5000, 7)).astype(np.float32)
    expected = (x - x.mean(0)) / x.std(0)
    standardize(x, chunk=333)
    assert np.allclose(x, expected, atol=1e-4)


def test_b4_pools_the_down_view_on_a_4x4_grid_and_leaves_other_variants_unchanged():
    import torch
    from had.model import FEATURE_DIM, TerrainHAD, feature_dim, pooled_tokens
    from had.tokenizer import Tokenizer
    hidden = torch.arange(1 + 64, dtype=torch.float32).reshape(1, 65, 1)  # CLS 0, then an 8x8 grid valued 1..64 row-major
    out = pooled_tokens(hidden, 4)[0]
    assert out.shape == (17,) and out[0] == 0 and out[1] == (1 + 2 + 9 + 10) / 4 and out[16] == (55 + 56 + 63 + 64) / 4
    assert torch.equal(pooled_tokens(hidden), pooled_tokens(hidden, 2))
    tok = Tokenizer.fit(["fly to the tank"], max_length=8)
    b4, b = TerrainHAD("B4", tok, cached_only=True), TerrainHAD("B", tok, cached_only=True)
    assert b4.down_encoder.proj[0].in_features == feature_dim(4) == 13056 and b4.front_encoder.proj[0].in_features == FEATURE_DIM
    assert b.down_encoder.proj[0].in_features == FEATURE_DIM
    calls = []

    class Fake(torch.nn.Module):
        def forward(self, x, grid=2):
            calls.append(grid)
            return torch.zeros(len(x), feature_dim(grid))
    b4.backbone = Fake()
    tokens = torch.tensor([tok.encode("fly to the tank")["input_ids"]])
    b4.eval().from_images({"front": torch.zeros(1, 3, 8, 8), "down": torch.zeros(1, 3, 8, 8)}, tokens, torch.zeros(1, 5))
    assert sorted(calls) == [2, 4]
