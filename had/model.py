"""HAD dual-view policy. A/B/C share the same modules and initialization; only view masks differ.

Text/fusion/policy modules are the pinned HAD implementation in had/vendor. The visual
backbone is a frozen DINOv2-B (docs/protocol.md §11) shared by both views; each view keeps
HAD's trainable projection (Linear + LayerNorm + ReLU), now from the 3840-d DINOv2 feature.
The backbone is built after every trainable module, so seeded head initialization does not
depend on whether it is loaded (cached features) or not (raw images).
"""
from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from had.vendor.encoders import TextEncoder
from had.vendor.fusion import ConcatFusion
from had.vendor.policy_head import MultiHeadPolicy

DINO = {"model": "facebook/dinov2-base", "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415", "width": 768}
FEATURE_DIM = 5 * DINO["width"]   # CLS + 2x2 average of the last-layer patch tokens
# N (no image input) is a diagnostic: missing views are zero-filled exactly as for A and C.
VARIANTS = {"A": ("front",), "B": ("front", "down"), "C": ("down",), "N": ()}


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


def image_transform():
    # Whole frame resized (no crop) so the image layout the slope depends on is kept.
    return transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
                               transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def pooled_tokens(hidden):
    """(b, 1 + h*w, d) last-layer tokens -> (b, 5d): CLS, then the 2x2 average of the patch grid (row-major)."""
    patches = hidden[:, 1:]; side = int(round(patches.shape[1] ** .5))
    if side * side != patches.shape[1]:
        raise ValueError("Patch tokens do not form a square grid")
    grid = F.adaptive_avg_pool2d(patches.transpose(1, 2).reshape(len(hidden), -1, side, side), 2)
    return torch.cat([hidden[:, 0], grid.flatten(2).transpose(1, 2).flatten(1)], 1)


class DinoFeatures(nn.Module):
    """Frozen DINOv2-B; `config` builds a randomly initialized model instead (tests only)."""
    def __init__(self, config=None):
        super().__init__()
        from transformers import Dinov2Model   # kept outside the shared venv: PYTHONPATH=~/sr3mxhad_runs/pylib
        self.net = Dinov2Model(config) if config is not None else \
            Dinov2Model.from_pretrained(DINO["model"], revision=DINO["revision"], local_files_only=True)
        self.net.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode); self.net.eval()
        return self

    def forward(self, images):
        return pooled_tokens(self.net(pixel_values=images).last_hidden_state)


class ViewProjection(nn.Module):
    """HAD VisualEncoder.proj with the DINOv2 feature as input."""
    def __init__(self, output_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(FEATURE_DIM, output_dim), nn.LayerNorm(output_dim), nn.ReLU(inplace=True))


class TerrainHAD(nn.Module):
    def __init__(self, variant, tokenizer, *, cached_only=False, small=False, backbone=None):
        super().__init__()
        configure_precision()
        if variant not in VARIANTS:
            raise ValueError(f"Variant must be one of {sorted(VARIANTS)}")
        self.config = c = SMALL if small else HadDims()
        self.tokenizer = tokenizer
        self.front_encoder = ViewProjection(c.visual_dim)
        self.down_encoder = ViewProjection(c.visual_dim)
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
        for view in ("front", "down"):
            if view not in self.views:
                getattr(self, f"{view}_encoder").proj.requires_grad_(False)
        self.backbone = None if cached_only else (backbone if backbone is not None else DinoFeatures())

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
        if self.backbone is None:
            raise ValueError("Built with cached_only=True; raw images need the backbone")
        if set(images) != set(self.views):
            raise ValueError("Unexpected image views")
        with torch.no_grad():
            features = {v: self.backbone(x) for v, x in images.items()}
        return self(features, tokens, state)

    def head_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("backbone.")}

    def load_head_state(self, state):
        expected = {k for k in self.state_dict() if not k.startswith("backbone.")}
        if set(state) != expected:
            raise ValueError("Checkpoint head schema differs")
        self.load_state_dict(state, strict=False)
