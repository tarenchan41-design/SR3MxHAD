"""Read-only access to the published SR3M capture split. No simulator or model dependencies."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re

import numpy as np

SPLITS = ("train", "val_seen", "val_unseen", "test")
INDEX_DIR = "benchmark_split_v1"


def local_path(value):
    """Accept native Windows paths or WSL /mnt/<drive>/ paths on either OS."""
    value = str(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", value):
        return Path("/mnt") / value[0].lower() / value[3:].replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        return Path(value[5].upper() + ":/" + value[7:])
    return Path(value)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_hashes(indices):
    return {f"{s}.jsonl": sha256_file(Path(indices)/f"{s}.jsonl") for s in SPLITS}


def relative_path(value):
    """Validate stored portable paths without silently accepting drive/.. paths."""
    if not isinstance(value, str) or not value:
        raise ValueError("Missing relative path")
    p = PurePosixPath(value.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or ":" in value:
        raise ValueError(f"Unsafe relative path: {value}")
    return p.as_posix()


def resolve_inside(root, relative):
    root = local_path(root).resolve()
    p = (root / relative_path(relative)).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"Path escapes root: {relative}")
    return p


def captured_instruction(episode):
    text = episode.get("instruction")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing captured instruction; never synthesize it from a route name")
    plan_text = episode.get("source_plan", {}).get("instruction")
    if plan_text is not None and plan_text != text:
        raise ValueError("Captured instruction conflicts with source_plan.instruction")
    return text


def read_capture(path):
    """Keep raw optional evidence as-is. Missing v2 cameras/source indices stay absent."""
    path = local_path(path).resolve()
    raw = path.read_bytes()
    episode = json.loads(raw)
    if episode.get("schema_version") not in (2, 3):
        raise ValueError("Only audited capture schema 2/3 are supported")
    for key in ("map_package", "trajectory_name", "run_id"):
        if not isinstance(episode.get(key), str) or not episode[key]:
            raise ValueError(f"Missing {key}")
    captured_instruction(episode)
    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps or episode.get("num_steps") != len(steps):
        raise ValueError("Empty steps or num_steps mismatch")
    prev_source = None
    for t, step in enumerate(steps):
        if step.get("step_id") != t:
            raise ValueError("Noncontiguous step_id; refusing to reorder observations")
        pos = np.array([step["position_m"][k] for k in "xyz"], dtype=float)
        cm = np.array([step["position_ue_cm"][k] for k in "xyz"], dtype=float)
        rot = [step["rotation_deg"][k] for k in ("roll", "pitch", "yaw")]
        quat = [step["quaternion"][k] for k in "xyzw"]
        if not np.isfinite(np.r_[pos, cm, rot, quat]).all() or not np.allclose(pos, cm / 100, atol=1e-7, rtol=0):
            raise ValueError("Nonfinite pose or cm/m disagreement")
        if not math.isclose(float(np.linalg.norm(quat)), 1, abs_tol=1e-5):
            raise ValueError("Non-unit quaternion")
        src = step.get("source_pose_index")
        if src is not None:
            if type(src) is not int or src < 0 or (prev_source is not None and src <= prev_source):
                raise ValueError("Invalid/non-increasing source_pose_index")
            prev_source = src
        if set(step.get("images", {})) != {"front", "down"}:
            raise ValueError("Expected actual front/down observations")
        for view in ("front", "down"):
            relative_path(step["images"][view]["rgb"])
    source_indices = episode.get("source_indices")
    if source_indices is not None and source_indices != [s.get("source_pose_index") for s in steps]:
        raise ValueError("source_indices disagree with steps")
    return episode, hashlib.sha256(raw).hexdigest()


def iter_split(root, index_dir, split):
    """Yield (index_record, episode) for exactly one frozen split, verify JSON SHA."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    index_dir = local_path(index_dir)
    audit = json.loads((index_dir / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "published":
        raise ValueError("Split directory is not published")
    selected = index_dir / f"{split}.jsonl"
    if sha256_file(selected) != audit["output_sha256"][selected.name]:
        raise ValueError("Split file changed after publication")
    with selected.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["split"] != split:
                raise ValueError("Mixed split records")
            path = resolve_inside(root, record["trajectory_relpath"])
            episode, digest = read_capture(path)
            if digest != record["trajectory_sha256"]:
                raise ValueError(f"Source trajectory changed: {record['episode_id']}")
            yield record, episode


class CaptureRoutes:
    """Validated poses and image names for the selected routes of one published split."""
    def __init__(self, root, split, selected=None):
        self.root = local_path(root).resolve()
        self.indices = self.root / INDEX_DIR
        self.split, self.routes = split, []
        requested = None if selected is None else set(selected)
        for record, ep in iter_split(self.root, self.indices, split):
            if requested is not None and record["episode_id"] not in requested:
                continue
            if record["instruction"] != ep["instruction"]:
                raise ValueError("Frozen instruction differs from source")
            steps = ep["steps"]
            if len(steps) < 2:
                raise ValueError("Route has no adjacent transition")
            if any(abs(s["rotation_deg"][a]) > 1e-6 for s in steps for a in ("roll", "pitch")):
                raise ValueError("Capture contract expects zero roll/pitch; do not reinterpret 3D poses")
            source = [s.get("source_pose_index") for s in steps]
            if any(a is not None and b is not None and b != a+1 for a, b in zip(source, source[1:])):
                raise ValueError("Sparse captures must not reconnect missing source transitions")
            p = np.asarray([[s["position_m"][k] for k in "xyz"] for s in steps], dtype=np.float64)
            yaw = np.radians([s["rotation_deg"]["yaw"] for s in steps])
            # One canonical pose convention, verified independently of the old adapters.
            quat = np.asarray([[s["quaternion"][k] for k in "xyzw"] for s in steps])
            expected = np.column_stack((np.zeros(len(yaw)), np.zeros(len(yaw)), np.sin(yaw/2), np.cos(yaw/2)))
            if np.any(np.minimum(np.linalg.norm(quat-expected, axis=1), np.linalg.norm(quat+expected, axis=1)) > 1e-5):
                raise ValueError("Quaternion/yaw mismatch")
            self.routes.append({"record": record, "directory": resolve_inside(self.root, record["trajectory_relpath"]).parent,
                                "positions": p, "yaw": yaw,
                                "images": [{v: s["images"][v]["rgb"] for v in ("front", "down")} for s in steps],
                                "scene_fingerprint": ep.get("scene_fingerprint"), "map_package": ep["map_package"]})
        if not self.routes or (requested is not None and requested != {r["record"]["episode_id"] for r in self.routes}):
            raise ValueError("Empty split or missing selected episode")
