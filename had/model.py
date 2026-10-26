"""HAD dual-view policy. A/B/C share the same modules and initialization; only view masks differ.

AD/BD add frozen monocular-depth features per view: a linear map of the 16x16 depth grid
is added before the view projection's LayerNorm, i.e. exactly a linear projection of the
concatenated [CNN feature; depth] vector. Depth modules are built last, so every module
shared with A/B keeps its seeded initialization.

Visual/text/fusion/policy modules are the pinned HAD implementation in had/vendor.
Submodules are constructed in the same order as SR3M's former HadPolicy adapter
(profile dual_relative_v1, concat fusion), so seeded initialization is unchanged.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from had.vendor.encoders import FrontEncoder, DownEncoder, TextEncoder
from had.vendor.fusion import ConcatFusion
from had.vendor.policy_head import MultiHeadPolicy

DEFAULT_WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
VARIANTS = {"A": ("front",), "B": ("front", "down"), "C": ("down",), "AD": ("front",), "BD": ("front", "down")}
DEPTH_VARIANTS = ("AD", "BD")
DEPTH_DIM = 256


@dataclass
class HadDims:
    visual_dim: int = 512
    text_embedding_dim: int = 300
    text_hidden_dim: int = 512
    text_layers: int = 2
    height_dim: int = 64
    fusion_dim: int = 512
    state_dim: int = 64
    policy_hidden_dims: list = field(default_factory=lambda: [512, 256])
    dropout: float = .2


SMALL = HadDims(visual_dim=8, text_embedding_dim=8, text_hidden_dim=8, text_layers=1, height_dim=4,
                fusion_dim=8, state_dim=4, policy_hidden_dims=[8])


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


class TerrainHAD(nn.Module):
    def __init__(self, variant, tokenizer, weights=DEFAULT_WEIGHTS, *, cached_only=False, small=False):
        super().__init__()
        configure_precision()
        if variant not in VARIANTS:
            raise ValueError(f"Variant must be one of {sorted(VARIANTS)}")
        self.config = c = SMALL if small else HadDims()
        self.tokenizer = tokenizer
        vision = dict(backbone="resnet50", pretrained=False, output_dim=c.visual_dim, freeze_bn=True)
        self.front_encoder = FrontEncoder(**vision)
        self.front_encoder.set_train_backbone(False)
        self.down_encoder = DownEncoder(**vision)
        self.down_encoder.set_train_backbone(False)
        self.text_encoder = TextEncoder(vocab_size=tokenizer.vocab_size, embedding_dim=c.text_embedding_dim,
                                        hidden_dim=c.text_hidden_dim, num_layers=c.text_layers, bidirectional=True,
                                        dropout=c.dropout, encoder_type="lstm", max_len=tokenizer.max_length)
        self.fusion = ConcatFusion(vis_dim=c.visual_dim, text_dim=self.text_encoder.output_dim, height_dim=c.height_dim,
                                   hidden_dim=c.fusion_dim, dropout=c.dropout)
        self.state_encoder = nn.Sequential(nn.Linear(5, c.state_dim), nn.LayerNorm(c.state_dim), nn.ReLU())
        self.state_fusion = nn.Sequential(nn.Linear(c.fusion_dim + c.state_dim, c.fusion_dim),
                                          nn.LayerNorm(c.fusion_dim), nn.ReLU(), nn.Dropout(c.dropout))
        self.policy = MultiHeadPolicy(input_dim=c.fusion_dim, policy_hidden_dims=tuple(c.policy_hidden_dims),
                                      dropout=c.dropout, yaw_strategy="baseline", dz_strategy="baseline",
                                      use_progress_monitor=False, use_dz_sign_aux=False)
        self.variant = variant
        self.views = VARIANTS[variant]
        if not cached_only:
            validate_weights(str(Path(weights).resolve()))
            weights = torch.load(weights, map_location="cpu", weights_only=True)
            state = {k: v for k, v in weights.items() if not k.startswith("fc.")}
            for encoder in (self.front_encoder, self.down_encoder):
                encoder.cnn.load_state_dict(state, strict=True)
        for view in ("front", "down"):
            if view not in self.views:
                getattr(self, f"{view}_encoder").proj.requires_grad_(False)
        self.depth = variant in DEPTH_VARIANTS
        if self.depth:
            self.depth_proj = nn.ModuleDict({v: nn.Linear(DEPTH_DIM, c.visual_dim) for v in self.views})
        self.inputs = self.views + (tuple(f"{v}_depth" for v in self.views) if self.depth else ())

    def encode(self, view, features):
        proj = getattr(self, f"{view}_encoder").proj
        if not self.depth:
            return proj(features[view])
        linear, *rest = proj
        x = linear(features[view]) + self.depth_proj[view](features[f"{view}_depth"])
        for layer in rest:
            x = layer(x)
        return x

    def forward(self, features, tokens, state):
        if set(features) != set(self.inputs):
            raise ValueError(f"{self.variant} requires exactly {self.inputs}")
        if state.ndim != 2 or state.shape[1] != 5:
            raise ValueError("Policy state must contain only initial-yaw-relative position/sin/cos")
        zero = state.new_zeros((len(state), self.config.visual_dim))
        encoded = {v: self.encode(v, features) for v in self.views}
        text, _ = self.text_encoder(tokens)
        height = state.new_zeros((len(state), self.config.height_dim))
        fused, _ = self.fusion(encoded.get("front", zero), encoded.get("down", zero), text, height)
        hidden = self.state_fusion(torch.cat((fused, self.state_encoder(state)), dim=-1))
        out = self.policy(hidden)
        return {"action": out["pred_action"], "stop_logit": out["stop_logit"].reshape(-1)}

    def from_images(self, images, tokens, state):
        if self.depth:
            raise NotImplementedError("Depth variants need precomputed depth features")
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
