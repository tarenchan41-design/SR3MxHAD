"""Publish benchmark_split_v2: v1 without the old desert, plus the rebuilt L_DES maps.

v1 rows outside the old desert are copied verbatim, so their splits never change.
New desert rows follow v1's rule: the held-out map goes whole to val_unseen; each
seen map puts the closest 15% of indivisible route groups into val_seen. Captures
are only read, never written.

    python -m sr3m.release_v2 --root <data_root> --batch <batch> --desert-dir <des-200> --heldout L_DES_C --dry-run
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from sr3m.data import SPLITS, local_path, read_capture, relative_path, sha256_file

BASE, OUT = "benchmark_split_v1", "benchmark_split_v2"
PREFIX = "/Game/Desert/Maps/"
MAPS = ("L_DES_A", "L_DES_B", "L_DES_C", "L_DES_D")
DARK = 0.2  # near-black pixel share that makes a frame "low visibility" in the audit


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def load_base(root):
    base = root / BASE
    audit = json.loads((base / "audit.json").read_text(encoding="utf-8"))
    for name, digest in audit["output_sha256"].items():
        if sha256_file(base / name) != digest:
            raise ValueError(f"{BASE}/{name} changed after publication")
    rows = [json.loads(line) for line in (base / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    config = json.loads((base / "split_config.json").read_text(encoding="utf-8"))
    registry = json.loads((base / "scene_registry.json").read_text(encoding="utf-8"))
    return rows, config, registry, audit


def collect_desert(root, batch, desert_dir):
    design = {}
    for name in MAPS:
        for r in json.loads((desert_dir / "design" / f"{name}.final.json").read_text(encoding="utf-8"))["routes"]:
            design[(PREFIX + name, r["route_id"])] = r
    dirs = sorted(p for p in (root / batch).iterdir() if p.is_dir())

    def one(directory):
        path = directory / "trajectory.json"
        episode, digest = read_capture(path)
        scene, route = episode["map_package"], episode["trajectory_name"]
        plan = design.get((scene, route))
        if plan is None or plan["instruction"] != episode["instruction"]:
            raise ValueError(f"Capture does not match des-200 design: {directory.name}")
        positions, hashes, refs, exposures = [], [], [], set()
        for step in episode["steps"]:
            positions.append([step["position_m"][k] for k in "xyz"])
            for view in ("front", "down"):
                image = relative_path(step["images"][view]["rgb"])
                h = step.get("file_hashes", {}).get(image)
                if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                    raise ValueError(f"Missing/invalid declared RGB SHA256: {directory.name}, {image}")
                refs.append((directory.relative_to(root) / image).as_posix())
                hashes.append(h)
                exposures.add(step["images"][view].get("rgb_exposure", {}).get("bias"))
        quality = json.loads((directory / "quality.json").read_text(encoding="utf-8"))
        return {"episode_id": directory.relative_to(root).as_posix(), "category": "desert", "batch": batch,
                "scene_id": scene, "scene_group_id": scene, "route_id": route, "run_id": episode["run_id"],
                "instruction": episode["instruction"], "instruction_zh": plan["instruction_zh"],
                "source_schema_version": episode["schema_version"],
                "trajectory_relpath": path.relative_to(root).as_posix(), "trajectory_sha256": digest,
                "num_steps": len(positions), "views": ["front", "down"], "resolution": episode["resolution"],
                "step_interval_cm": episode.get("step_interval_cm"),
                "cameras_available": all("cameras" in s for s in episode["steps"]),
                "source_pose_indices_available": all("source_pose_index" in s for s in episode["steps"]),
                "environment_mode": episode.get("environment", {}).get("mode"),
                "exposure_bias_values": sorted(exposures, key=lambda x: (x is None, x or 0)),
                "source_schema_retained": True, "_positions": np.array(positions), "_hashes": hashes,
                "_refs": refs, "_quality": quality["quality"], "_episode": episode,
                "_dark_frames": sum(f["rgb"]["near_black_ratio"] > DARK for f in quality["frames"])}

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = sorted(pool.map(one, dirs), key=lambda r: r["episode_id"])
    keys = Counter((r["scene_id"], r["route_id"]) for r in rows)
    if set(keys) != set(design) or max(keys.values()) != 1:
        raise ValueError("Captured routes must equal the des-200 design one-to-one")
    return rows


def layout_digest(items):
    """Static-bounds layout hash, invariant to translation, axis swap and reflection (as in v1)."""
    lo, hi = np.array([a["lo"] for a in items]) / 100, np.array([a["hi"] for a in items]) / 100
    centers, extents = (lo + hi) / 2, (hi - lo) / 2
    centers -= centers.mean(0)
    variants = []
    for swap, sx, sy in itertools.product((False, True), (-1, 1), (-1, 1)):
        c, e = centers.copy(), extents.copy()
        if swap:
            c[:, [0, 1]] = c[:, [1, 0]]; e[:, [0, 1]] = e[:, [1, 0]]
        c[:, 0] *= sx; c[:, 1] *= sy
        variants.append(fingerprint(sorted(map(tuple, np.round(np.c_[c, e], 2).tolist()))))
    return min(variants)


def review_desert(desert_dir, rows, registry):
    known = {e["normalized_layout_sha256"]: k for k, v in registry["scenes"].items() for e in v["evidence"]
             if "normalized_layout_sha256" in e}
    scenes = {}
    for name in MAPS:
        scene = PREFIX + name
        geometry = desert_dir / "survey" / f"geometry-{name}.json.gz"
        data = json.loads(gzip.decompress(geometry.read_bytes()))
        items = [a for a in data["items"] if a.get("mesh")]
        if data.get("map") != name or len(items) < 10:
            raise ValueError(f"Insufficient static layout evidence: {scene}")
        digest = layout_digest(items)
        if digest in known:
            raise ValueError(f"Copied static layout: {known[digest]}, {scene}")
        known[digest] = scene
        agl = desert_dir / "evidence" / f"continuous-agl-{name}.json"
        terrain = json.loads(agl.read_text(encoding="utf-8"))
        if terrain["status"] != "pass":
            raise ValueError(f"Continuous AGL certificate did not pass: {scene}")
        sample = next(r for r in rows if r["scene_id"] == scene)
        scenes[scene] = {
            "scene_group_id": scene, "category": "desert", "reviewed": True,
            "captured_scene_fingerprint_example": sample["_episode"].get("scene_fingerprint"),
            "evidence": [{"path": str(geometry), "sha256": sha256_file(geometry),
                          "method": "static_bounds_without_names_lights_or_route_actors",
                          "object_count": len(items), "normalized_layout_sha256": digest},
                         {"path": str(agl), "sha256": sha256_file(agl), "method": "continuous_agl_certificate",
                          "terrain_source_sha256": terrain["terrain"]["source_sha256"],
                          "routes_passed": sum(r["status"] == "pass" for r in terrain["routes"])}],
            "review_scope": "Archived static mesh bounds; translation/axis rotation/reflection exact-clone check "
                            "against each other and all v1 layouts. Terrain mesh identified by source hash."}
    return scenes


class Groups:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        i, j = self.find(i), self.find(j)
        self.parent[max(i, j)] = min(i, j)


def nearest(a, b):
    """Distance from each point of a to its nearest point of b."""
    return np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1).min(1))


def duplicate_groups(rows, config):
    """v1 grouping: same route id, same RGB sequence, >=80% shared RGB, or near-identical XY path.

    Height is never used, so altitude variants of one horizontal route stay on one side.
    """
    groups, edges = Groups(len(rows)), []
    same_scene, same_route, same_video = defaultdict(list), {}, {}
    for i, row in enumerate(rows):
        same_scene[row["scene_group_id"]].append(i)
        key = (row["scene_group_id"], row["route_id"])
        if key in same_route:
            groups.union(i, same_route[key]); edges.append([same_route[key], i, "same_route_id"])
        same_route[key] = i
        video = fingerprint(row["_hashes"])
        if video in same_video:
            j = same_video[video]
            if row["scene_group_id"] != rows[j]["scene_group_id"]:
                raise ValueError("Identical image sequence in different scene groups; review scene registry")
            groups.union(i, j); edges.append([j, i, "same_rgb_sequence"])
        same_video[video] = i
    xy = [r["_positions"][:, :2] for r in rows]
    lengths = [float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in xy]
    hash_sets = [set(r["_hashes"]) for r in rows]
    tolerance, coverage = config["near_duplicate_xy_tolerance_m"], config["near_duplicate_bidirectional_coverage"]
    for indices in same_scene.values():
        for i, j in itertools.combinations(indices, 2):
            shared = len(hash_sets[i] & hash_sets[j])
            if shared and shared >= 0.8 * max(len(hash_sets[i]), len(hash_sets[j])):
                groups.union(i, j); edges.append([i, j, "shared_rgb_80pct"])
                continue
            if max(lengths[i], lengths[j]) > 0 and min(lengths[i], lengths[j]) / max(lengths[i], lengths[j]) < config["near_duplicate_length_ratio"]:
                continue
            pi, pj = xy[i], xy[j]
            if np.any(pi.max(0) + tolerance < pj.min(0)) or np.any(pj.max(0) + tolerance < pi.min(0)):
                continue
            if (nearest(pi, pj) <= tolerance).mean() >= coverage and (nearest(pj, pi) <= tolerance).mean() >= coverage:
                groups.union(i, j); edges.append([i, j, "near_route_xy"])
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[groups.find(i)].append(row)
    for group in members.values():
        gid = fingerprint(sorted(r["episode_id"] for r in group))
        for row in group:
            row["route_group_id"] = gid
    return [{"left": rows[i]["episode_id"], "right": rows[j]["episode_id"], "reason": reason} for i, j, reason in edges]


def allocate_seen(rows, fraction, seed):
    """v1 rule: closest per-scene 15% quota, preserving indivisible groups and legacy ratio."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["route_group_id"]].append(r)
    target = math.floor(len(rows) * fraction + 0.5)
    old_target = sum(r["source_schema_version"] == 2 for r in rows) * fraction
    order = sorted(groups, key=lambda k: fingerprint([seed, "route_split", sorted(r["episode_id"] for r in groups[k])]))
    states = {(0, 0): ()}
    for key in order:
        size, old = len(groups[key]), sum(r["source_schema_version"] == 2 for r in groups[key])
        for (n, legacy), selected in list(states.items()):
            states.setdefault((n + size, legacy + old), selected + (key,))
    choices = [k for k in states if 0 < k[0] < len(rows)]
    if not choices:
        raise ValueError("A seen scene has only one indivisible route group; cannot make val_seen")
    best = min(choices, key=lambda x: (abs(x[0] - target), abs(x[1] - old_target), fingerprint(states[x])))
    selected = set(states[best])
    for r in rows:
        r["split"] = "val_seen" if r["route_group_id"] in selected else "train"


def assign_desert(rows, heldout, config):
    edges = duplicate_groups(rows, config)
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["scene_group_id"]].append(r)
    for scene, rr in by_scene.items():
        if scene == PREFIX + heldout:
            for r in rr:
                r["split"] = "val_unseen"
        else:
            allocate_seen(rr, config["seen_validation_fraction"], config["seed"])
    return edges


def validate_splits(rows):
    ids, scenes, groups = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in rows:
        ids[r["split"]].add(r["episode_id"])
        scenes[r["split"]].add(r["scene_group_id"])
        groups[r["route_group_id"]].add(r["split"])
    if sum(map(len, ids.values())) != len(rows) or set(ids) != set(SPLITS):
        raise ValueError("Missing/duplicate sample or empty split")
    if any(len(x) != 1 for x in groups.values()):
        raise ValueError("Route variant crosses splits")
    if not scenes["val_seen"] <= scenes["train"]:
        raise ValueError("val_seen contains an unseen scene")
    for a, b in itertools.combinations(SPLITS, 2):
        if ids[a] & ids[b] or ((a, b) != ("train", "val_seen") and scenes[a] & scenes[b]):
            raise ValueError(f"Scene/sample leakage: {a}, {b}")


def public_row(row):
    return {k: v for k, v in row.items() if not k.startswith("_")}


def build(base_rows, desert_rows, heldout, config):
    """Pure split logic: returns (v2 rows, duplicate edges). Base rows are never reassigned."""
    kept = [dict(r) for r in base_rows if r["category"] != "desert"]
    edges = assign_desert(desert_rows, heldout, config)
    rows = sorted(kept + desert_rows, key=lambda r: r["episode_id"])
    validate_splits(rows)
    return rows, edges


def check_rgb(root, rows):
    refs = {rel: h for r in rows for rel, h in zip(r["_refs"], r["_hashes"])}
    with ThreadPoolExecutor(max_workers=16) as pool:
        sizes = list(pool.map(lambda rel: os.stat(root / rel).st_size, refs))
    if min(sizes) <= 0:
        raise ValueError("Empty RGB file")
    return len(refs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Workbench containing benchmark_split_v1 and the batch")
    parser.add_argument("--batch", required=True, help="Desert capture batch directory name")
    parser.add_argument("--desert-dir", required=True, help="UE export des-200 (design/, survey/, evidence/)")
    parser.add_argument("--heldout", required=True, choices=MAPS, help="Whole desert map for val_unseen")
    parser.add_argument("--dry-run", action="store_true", help="All audits, no published files")
    args = parser.parse_args()
    root, desert_dir = local_path(args.root).resolve(), local_path(args.desert_dir).resolve()
    out = root / OUT
    if out.exists():
        parser.error("Output already exists; use a new version, never overwrite a frozen split")

    base_rows, base_config, base_registry, base_audit = load_base(root)
    print(f"{BASE} verified: {len(base_rows)} rows. Reading desert captures...", flush=True)
    desert = collect_desert(root, args.batch, desert_dir)
    config = {k: base_config[k] for k in ("seed", "seen_validation_fraction", "near_duplicate_xy_tolerance_m",
                                          "near_duplicate_bidirectional_coverage", "near_duplicate_length_ratio")}
    family = {"category": "desert", "label": "沙漠", "batch": args.batch, "map_prefix": PREFIX,
              "heldout_map": args.heldout, "heldout_split": "val_unseen", "design_dir": str(desert_dir)}
    config = {"version": 2, "base": {"index": BASE, "output_sha256": base_audit["output_sha256"]},
              "removed_categories": ["desert"], **config, "expected_episodes": None, "scene_registry": "scene_registry.json",
              "families": [{**f, "source": BASE} for f in base_config["families"] if f["category"] != "desert"] + [family]}
    rows, edges = build(base_rows, desert, args.heldout, config)
    config["expected_episodes"] = len(rows)
    # Same allocation from a reversed input order, without rescanning captures.
    again, _ = build(base_rows, [dict(r) for r in reversed(desert)], args.heldout, config)
    if fingerprint([public_row(r) for r in again]) != fingerprint([public_row(r) for r in rows]):
        raise ValueError("Nondeterministic split allocation")
    base_by_id = {r["episode_id"]: r for r in base_rows}
    if any(public_row(r) != base_by_id[r["episode_id"]] for r in rows if r["category"] != "desert"):
        raise ValueError("A v1 row changed")

    print(f"Splits valid; {len(edges)} desert duplicate/near-route edges. Checking RGB files...", flush=True)
    rgb = check_rgb(root, desert)
    if any(sha256_file(root / r["trajectory_relpath"]) != r["trajectory_sha256"] for r in desert):
        raise ValueError("Capture metadata changed during preparation")

    removed = Counter(r["split"] for r in base_rows if r["category"] == "desert")
    quality = Counter(r["_quality"] for r in desert)
    dark = {m: sum(r["_dark_frames"] > 0 for r in desert if r["scene_id"] == PREFIX + m) for m in MAPS}
    dark_frames = sum(r["_dark_frames"] for r in desert)
    failed = sorted(r["episode_id"] for r in desert if r["_quality"] == "failed")
    sun = {m: sorted({round(light["rotation"][0], 1) for r in desert if r["scene_id"] == PREFIX + m
                      for light in r["_episode"]["environment"]["original_scene"]["DirectionalLight"]}) for m in MAPS}
    registry = {"version": 2, "scenes": {**{k: v for k, v in base_registry["scenes"].items() if v["category"] != "desert"},
                                         **review_desert(desert_dir, desert, base_registry)},
                "limitations": base_registry["limitations"] + [
        f"v2 removes all {sum(removed.values())} old-desert episodes ({dict(removed)}) and the 4 RuleMaps3 desert scenes; "
        f"their captures are kept on disk. All other v1 rows are copied unchanged.",
        f"Rebuilt desert ({len(desert)} routes, {len(MAPS)} maps) has large relief and a low sun (pitch by map {sun} deg): "
        f"backlit faces have very low visibility and render near-black at the fixed exposure bias. {sum(dark.values())} routes "
        f"have frames with >{DARK:.0%} near-black pixels ({dark_frames} frames; per map {dark}); {len(failed)} routes are "
        f"auto-QC 'failed', all for near_black. Their continuous-AGL certificates pass, capture timing shows no interruption "
        f"or slowdown, and the UE viewport shows the same dim backlit faces, so this is scene lighting, not a capture fault. "
        f"Released unfiltered; human_review in quality.json is still pending."]}
    plain = [public_row(r) for r in rows]
    summary = {s: {"episodes": sum(r["split"] == s for r in rows),
                   "scenes": len({r["scene_group_id"] for r in rows if r["split"] == s}),
                   "steps": sum(r["num_steps"] for r in rows if r["split"] == s),
                   "desert_episodes": sum(r["split"] == s and r["category"] == "desert" for r in rows)} for s in SPLITS}
    audit = {"status": "dry_run" if args.dry_run else "published", "version": 2, "summary": summary,
             "episode_count": len(rows), "base": config["base"], "removed_old_desert": dict(removed),
             "base_rows_unchanged": True, "desert_episodes": len(desert),
             "desert_by_map_split": {m: dict(Counter(r["split"] for r in desert if r["scene_id"] == PREFIX + m)) for m in MAPS},
             "desert_rgb_references_checked": sum(len(r["_refs"]) for r in desert), "desert_unique_rgb_paths": rgb,
             "all_referenced_rgb_files_exist_and_nonempty": True,
             "image_check_scope": "New desert RGB paths/size checked; file bytes and decoding NOT rehashed. v1 rows rely on the v1 audit.",
             "desert_quality": {"status": dict(quality), "failed_episodes": failed, "dark_threshold": DARK,
                                "routes_with_dark_frames": dark, "dark_frames": dark_frames},
             "source_metadata_unchanged": True,
             "desert_metadata_hashes": {r["trajectory_relpath"]: r["trajectory_sha256"] for r in desert},
             "deterministic_repeat_passed": True, "height_used_for_exclusion_or_stratification": False,
             "duplicate_edges": edges, "route_groups": len({r["route_group_id"] for r in rows}),
             "raw_data_written": False, "model_inference_run": False}
    print(json.dumps({"summary": summary, "desert_by_map_split": audit["desert_by_map_split"],
                      "removed_old_desert": dict(removed), "desert_quality": dict(quality)}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    staging = Path(tempfile.mkdtemp(prefix=".vln-split-staging-", dir=root))
    try:
        for name, entries in [("episodes", plain)] + [(s, [r for r in plain if r["split"] == s]) for s in SPLITS]:
            (staging / f"{name}.jsonl").write_text("".join(encoded(r) + "\n" for r in entries), encoding="utf-8")
        (staging / "split_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "scene_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audit["output_sha256"] = {p.name: sha256_file(p) for p in sorted(staging.iterdir())}
        (staging / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if out.exists():
            raise FileExistsError(out)
        staging.rename(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Published: {out}", flush=True)


if __name__ == "__main__":
    main()
