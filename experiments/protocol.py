"""The preregistered offline protocol and compact prepared-data interface."""
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import tomllib

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

PROTOCOL = {
    "name": "sr3m_had_terrain_v2_dinov2", "index": "benchmark_split_v2", "language": "en",
    "terrain_categories": ["mountain", "desert"],
    "action_contract": "capture_adjacent_yaw_fru_m_rad_v1",
    "stop_radius_m": 10., "success_radius_m": 20., "stop_positive": "terminal_contiguous_suffix",
    "stop_pos_weight": 1., "stop_loss_weight": .1,
    "turn_threshold_rad": math.pi / 180, "std_floor": [.05, .05, .05, math.pi / 180],
    "seeds": [42, 43, 44], "variants": {"A": ["front"], "B": ["front", "down"], "C": ["down"]},
    "image_size": 224, "image_resize": "bicubic_antialias_whole_frame", "backbone": "dinov2-base",
    "weights": "facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    "feature": "last_layer_cls_plus_2x2_patch_mean", "feature_dim": 3840, "feature_dtype": "float16", "dropout": .2,
    "precision": "float32_ieee_no_tf32",
    "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "batch_size": 32,
                  "max_epochs": 40, "min_epochs": 10, "patience": 5, "gradient_clip": 5.},
    "selection": "equal_normalized_route_macro_translation_turn_yaw_cohort_macro_terrain_dz",
}
EXPECTED = {"train": 1028, "val_seen": 184, "val_unseen": 247, "test": 147}
# Categories with terrain exports: AGL/slope labels, follow/control cohorts, H1/H2.
TERRAIN = tuple(PROTOCOL["terrain_categories"])
# Variants trained, reported and decided on. C keeps its PROTOCOL definition but is not run
# (protocol.md §6); kept outside PROTOCOL so existing preparations and caches stay valid.
MAIN_VARIANTS = ("A", "B")
# Diagnostic variants defined outside PROTOCOL for the same reason. N: instruction + state only.
DIAGNOSTIC_VARIANTS = {"N": []}


def machine_paths():
    """Per-machine locations from configs/paths.toml (copied from configs/paths.example.toml)."""
    path = Path(os.environ.get("SR3MXHAD_PATHS", PROJECT / "configs/paths.toml"))
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; copy configs/paths.example.toml and edit it")
    with path.open("rb") as f:
        return {k: PROJECT / Path(v).expanduser() for k, v in tomllib.load(f).items()}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def code_version():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(PROJECT), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "had_source": "f01440a7648e8c9ce89ea4f925a74c32ecc97120"}


def stop_suffix(positions, radius_m=10.):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all() or radius_m <= 0:
        raise ValueError("Invalid positions or STOP radius")
    distance = np.linalg.norm(positions - positions[-1], axis=1)
    stop = np.zeros(len(positions), dtype=bool)
    outside = np.flatnonzero(distance > radius_m)
    start = int(outside[-1] + 1) if len(outside) else 0
    if start == 0 and len(positions) > 1:
        raise ValueError("Whole route lies inside STOP radius; semantic completion needs review")
    stop[start:] = True
    return stop, distance


def terrain_following(record):
    text = record.get("instruction_zh")
    if not isinstance(text, str):
        raise ValueError("Original instruction_zh is required for cohort assignment")
    return "随沿途地形调整高度" in text


def action_statistics(actions, valid):
    a = np.asarray(actions)[np.asarray(valid, dtype=bool)]
    if not len(a):
        raise ValueError("No action-valid training labels")
    mean = a.mean(0)
    mean[3] = math.atan2(np.sin(a[:, 3]).mean(), np.cos(a[:, 3]).mean())
    std = a.std(0)
    turn_fraction = float((np.abs(a[:, 3]) >= PROTOCOL["turn_threshold_rad"]).mean())
    if not 0 < turn_fraction < 1:
        raise ValueError("Both turn and non-turn training actions are required")
    return {"mean": mean.tolist(), "std": std.tolist(),
            "scale": np.maximum(std, PROTOCOL["std_floor"]).tolist(),
            "turn_fraction": turn_fraction, "yaw_abs_lt_1deg_fraction": 1 - turn_fraction,
            "actions": len(a), "fit_split": "train", "stop_actions_excluded": True}


class PreparedSplit:
    """Arrays contain labels; payload() alone constructs policy observations."""
    def __init__(self, prepared, split):
        self.directory = Path(prepared)
        self.protocol = read_json(self.directory / "protocol.json")
        if self.protocol["protocol"] != PROTOCOL:
            raise ValueError("Prepared protocol differs from current implementation")
        self.split = split
        self.routes = read_json(self.directory / f"{split}.json")
        with np.load(self.directory / f"{split}.npz", allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files}
        self.n = len(self.arrays["state"])
        if len(self.routes) != EXPECTED[split] or self.n != sum(r["count"] for r in self.routes):
            raise ValueError("Incomplete prepared split")
        self.route_index = np.repeat(np.arange(len(self.routes)), [r["count"] for r in self.routes])

    def payload(self, index, views, root):
        r = self.routes[int(self.route_index[index])]
        t = index - r["offset"]
        return {"instruction": r["instruction"],
                "images": {v: str(Path(root) / r["directory"] / r["images"][v][t]) for v in views},
                "state": self.arrays["state"][index].copy()}

    def cohort_counts(self):
        return dict(Counter("follow" if r["terrain_following"] else "control"
                            for r in self.routes if r["category"] in TERRAIN))


def read_features(cache, split, prepared):
    path = Path(cache) / split
    metadata = read_json(path / "cache.json")
    if metadata["completed_rows"] != prepared.n or metadata["prepared"] != prepared.protocol:
        raise ValueError("Feature cache is incomplete or belongs to different preparation")
    if metadata["backbone"] != PROTOCOL["weights"] or metadata["image_size"] != PROTOCOL["image_size"]:
        raise ValueError("Feature cache backbone/preprocessing mismatch")
    features = {v: np.load(path / f"{v}.npy", mmap_mode="r", allow_pickle=False) for v in ("front", "down")}
    if any(x.shape != (prepared.n, PROTOCOL["feature_dim"]) or x.dtype != np.dtype(PROTOCOL["feature_dtype"]) for x in features.values()):
        raise ValueError("Invalid cached feature array")
    return features, metadata

