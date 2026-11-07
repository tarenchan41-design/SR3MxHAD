"""D4 baselines AV (mosaic) and DG (token fusion), docs/d4_baseline_spec.md: shapes, view checks, trainable parts,
checkpoint round trip, one backward step, the padding mask, and that the other variants keep their initialization."""
import numpy as np
import pytest
import torch

from had.model import FEATURE_DIM, GRIDS, VARIANTS, TerrainHAD, feature_dim
from had.tokenizer import Tokenizer


def inputs(model, n=3):
    torch.manual_seed(0)
    widths = {v: feature_dim(model.grids[v]) for v in model.views}
    return {v: torch.randn(n, w) for v, w in widths.items()}, torch.tensor([[2, 3, 4, 0, 0, 0]] * n), torch.randn(n, 5)


@pytest.mark.parametrize("variant", ["AV", "DG"])
def test_forward_backward_and_checkpoint(variant):
    torch.set_num_threads(2)
    tok = Tokenizer.fit(["fly to the red roof"], max_length=6)
    model = TerrainHAD(variant, tok, cached_only=True, small=True)
    features, tokens, state = inputs(model)
    out = model(features, tokens, state)
    assert out["action"].shape == (3, 4) and out["stop_logit"].shape == (3,)
    (out["action"].sum() + out["stop_logit"].sum()).backward()
    new = model.mosaic_fusion if variant == "AV" else model.token_fusion
    assert all(p.grad is not None for p in new.parameters())
    assert not any(p.requires_grad for p in model.fusion.parameters())   # ConcatFusion of B is not used
    assert not any(p.requires_grad for p in model.front_encoder.proj.parameters())
    with pytest.raises(ValueError):
        model({"front": torch.zeros(3, FEATURE_DIM)}, tokens, state)
    model.eval()
    restored = TerrainHAD(variant, tok, cached_only=True, small=True).eval()
    restored.load_head_state(model.head_state())
    with torch.no_grad():
        assert torch.equal(model(features, tokens, state)["action"], restored(features, tokens, state)["action"])


def test_dg_reads_4x4_down_and_ignores_padding():
    tok = Tokenizer.fit(["fly to the red roof"], max_length=6)
    model = TerrainHAD("DG", tok, cached_only=True, small=True).eval()
    assert VARIANTS["DG"] == ("front", "down") and GRIDS["DG"] == {"down": 4} and model.token_fusion.tokens == {"front": 5, "down": 17}
    features, tokens, state = inputs(model, 1)
    with torch.no_grad():
        words = model.text_encoder(tokens)[1]
        mask = tokens.eq(0)
        a = model.token_fusion(features, words, mask)
        noisy = words.clone(); noisy[mask] += 100.   # padded word features must not reach [ACT]
        b = model.token_fusion(features, noisy, mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_av_is_one_mosaic_view():
    tok = Tokenizer.fit(["fly"], max_length=4)
    model = TerrainHAD("AV", tok, cached_only=True, small=True)
    assert model.views == ("mosaic",) and model.grids["mosaic"] == 2
    assert model.mosaic_encoder.proj[0].in_features == FEATURE_DIM
