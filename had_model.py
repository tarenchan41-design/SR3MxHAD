"""A/B/C share the same HAD modules and initialization; only view masks differ."""
from pathlib import Path
from functools import lru_cache
import hashlib
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import source  # noqa: F401
from vln_benchmark.unified.had import HadConfig, HadPolicy
from vln_benchmark.unified.tokenizer import ChineseTokenizer as Tokenizer
from experiment import PROTOCOL

DEFAULT_WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/resnet50-0676ba61.pth"


def configure_precision():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


@lru_cache(maxsize=4)
def validate_weights(path):
    # Bind external pretrained weights to torchvision's published download hash.
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""): h.update(block)
    if not h.hexdigest().startswith("0676ba61"):
        raise ValueError("Expected torchvision ResNet50 IMAGENET1K_V1 weights")


def image_transform():
    return transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
                               transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


class TerrainHAD(HadPolicy):
    def __init__(self, variant, tokenizer, weights=DEFAULT_WEIGHTS, *, cached_only=False, small=False):
        configure_precision()
        if variant not in PROTOCOL["variants"]:
            raise ValueError("Variant must be A, B or C")
        config = HadConfig(profile="dual_relative_v1", fusion="concat", init_mode="imagenet", legacy_fusion=False,
                           train_backbone=False, dropout=.2)
        if small:
            config.visual_dim = 8; config.text_embedding_dim = 8; config.text_hidden_dim = 8
            config.text_layers = 1; config.height_dim = 4; config.fusion_dim = 8
            config.state_dim = 4; config.policy_hidden_dims = [8]
        super().__init__(config, tokenizer)
        self.variant = variant
        self.views = tuple(PROTOCOL["variants"][variant])
        if not cached_only:
            validate_weights(str(Path(weights).resolve()))
            weights = torch.load(weights, map_location="cpu", weights_only=True)
            state = {k: v for k, v in weights.items() if not k.startswith("fc.")}
            for encoder in (self.front_encoder, self.down_encoder):
                encoder.cnn.load_state_dict(state, strict=True)
        for view in ("front", "down"):
            if view not in self.views:
                getattr(self, f"{view}_encoder").proj.requires_grad_(False)

    def forward(self, features, tokens, state):
        if set(features) != set(self.views):
            raise ValueError(f"{self.variant} requires exactly {self.views}")
        if state.ndim != 2 or state.shape[1] != 5:
            raise ValueError("Policy state must contain only initial-yaw-relative position/sin/cos")
        zero = state.new_zeros((len(state), self.config.visual_dim))
        encoded = {v: getattr(self, f"{v}_encoder").proj(x) for v, x in features.items()}
        text, _ = self.text_encoder(tokens)
        height = state.new_zeros((len(state), self.config.height_dim))
        fused, _ = self.fusion(encoded.get("front", zero), encoded.get("down", zero), text, height)
        hidden = self.state_fusion(torch.cat((fused, self.state_encoder(state)), dim=-1))
        out = self.policy(hidden)
        return {"action": out["pred_action"], "stop_logit": out["stop_logit"].reshape(-1)}

    def from_images(self, images, tokens, state):
        if set(images) != set(self.views):
            raise ValueError("Unexpected image views")
        with torch.no_grad():
            features = {v: getattr(self, f"{v}_encoder").cnn(x) for v, x in images.items()}
        return self(features, tokens, state)

    def head_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if ".cnn." not in k}

    def load_head_state(self, state):
        expected = {k for k in self.state_dict() if ".cnn." not in k}
        if set(state) != expected:
            raise ValueError("Checkpoint head schema differs")
        self.load_state_dict(state, strict=False)

