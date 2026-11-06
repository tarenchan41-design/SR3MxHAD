"""--mosaic-out (D4 baseline AV, docs/d4_baseline_spec.md): end to end on CPU with a deterministic stand-in backbone."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import experiments.cache as cache
from had.model import DINO, image_transform, mosaic_image, pooled_tokens

N = 5


class FakeNet(torch.nn.Module):
    """DINOv2-shaped output: CLS + 16x16 patch tokens of width 768 from a fixed patch embedding."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.patch = torch.nn.Conv2d(3, DINO["width"], 14, 14)

    def forward(self, pixel_values):
        patches = self.patch(pixel_values).flatten(2).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=torch.cat([patches.mean(1, keepdim=True), patches], 1))


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = FakeNet()

    def forward(self, images, grid=2):
        return pooled_tokens(self.net(pixel_values=images).last_hidden_state, grid)


class FakeSplit:
    def __init__(self, prepared, split):
        self.root, self.n, self.protocol = prepared, N, {"name": "fixture", "split": split}

    def payload(self, index, views, root):
        return {"images": {v: f"{root}/{v}_{index}.png" for v in views}}


@pytest.fixture
def frames(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for i in range(N):
        for view in ("front", "down"):
            Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)).save(tmp_path / f"{view}_{i}.png")
    monkeypatch.setattr(cache, "DinoFeatures", FakeBackbone)
    monkeypatch.setattr(cache, "PreparedSplit", FakeSplit)
    return tmp_path


def run(root, out, grid4=None, mosaic=None):
    cache.extract(SimpleNamespace(prepared=str(root), root=str(root), out=str(out), device="cpu", splits=["train"],
                                  grid4_out=grid4 and str(grid4), mosaic_out=mosaic and str(mosaic), batch_size=2, workers=0))


def test_mosaic_is_front_over_down_and_squeezed_whole():
    front, down = Image.new("RGB", (512, 512), (255, 0, 0)), Image.new("RGB", (512, 512), (0, 0, 255))
    m = mosaic_image(front, down)
    assert m.size == (224, 448) and m.getpixel((100, 100)) == (255, 0, 0) and m.getpixel((100, 300)) == (0, 0, 255)
    x = image_transform()(m)
    assert x.shape == (3, 224, 224)
    assert x[0, :100].mean() > x[2, :100].mean() and x[2, 124:].mean() > x[0, 124:].mean()  # top rows front, bottom down


def test_mosaic_out_leaves_the_2x2_and_4x4_caches_byte_identical(frames):
    run(frames, frames / "plain", grid4=frames / "plain4")
    run(frames, frames / "with", grid4=frames / "with4", mosaic=frames / "mosaic")
    for a, b in (("plain", "with"), ("plain4", "with4")):
        for view in ("front", "down"):
            assert (frames / a / "train" / f"{view}.npy").read_bytes() == (frames / b / "train" / f"{view}.npy").read_bytes()


def test_mosaic_cache_holds_the_backbone_on_the_mosaic(frames):
    run(frames, frames / "c", mosaic=frames / "mosaic")
    values = np.load(frames / "mosaic" / "train" / "mosaic.npy")
    meta = cache.read_json(frames / "mosaic" / "train" / "cache.json")
    assert values.shape == (N, 3840) and values.dtype == np.float16 and meta["completed_rows"] == N
    assert meta["prepared"] == FakeSplit(str(frames), "train").protocol and "mosaic_image" in meta["input"]
    backbone, transform = FakeBackbone(), image_transform()
    with torch.inference_mode():
        for i in range(N):
            with Image.open(frames / f"front_{i}.png") as f, Image.open(frames / f"down_{i}.png") as d:
                expected = backbone(transform(mosaic_image(f.convert("RGB"), d.convert("RGB")))[None])[0].numpy()
            np.testing.assert_allclose(values[i].astype(np.float32), expected, rtol=2e-3, atol=2e-3)
    front = np.load(frames / "c" / "train" / "front.npy")
    assert not np.allclose(values.astype(np.float32), front.astype(np.float32))  # not a copy of a single view


def test_mosaic_cache_must_start_with_the_2x2_cache(frames):
    run(frames, frames / "c")
    with pytest.raises(ValueError, match="start the mosaic cache together"):
        run(frames, frames / "c", mosaic=frames / "mosaic")
