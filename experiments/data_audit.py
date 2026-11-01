"""Read-only audit of v2 routes: is what the instruction says true, and can the policy know it from what it gets?

Uses train and val_seen only. Joins each route to its design file; nothing here writes outside --out.
Compass convention, checked against the design's own region text: +X east, +Y south (UE is left handed).
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re

import numpy as np

from experiments.protocol import PreparedSplit, read_json, terrain_following, write_json
from sr3m.geometry import wrap

UE = Path("/mnt/e/UnrealProjects")
DESIGNS = {"mountain": "mountain_base/military_mountains/Saved/VLNDataset/full-200/design", "airfields": "airfields/military_airfields/Saved/VLNDataset/full-200/design",
           "harbors": "harbors/military_harbors/Saved/VLNDataset/full-197/design", "industry": "industry/industry/Saved/VLNDataset/full-200/design",
           "integrated_base": "integrated_base/integrated_base/Saved/VLNDataset/full-200/design", "prairie": "prairie_vln_training/prairie_vln_training/Saved/VLNDataset/full-200/design",
           "desert": "desert/Saved/VLNDataset/des-200/design"}
FORESTS = "forests_new/military_forests/forest_json/forest"
COMPASS = {"东": 0., "东南": 45., "南": 90., "西南": 135., "西": 180., "西北": 225., "北": 270., "东北": 315.}  # Degrees of atan2(y, x) with +Y south.
REGION = re.compile(r"(东南|东北|西南|西北|东|西|南|北)[侧方向]*约?\s*(\d+(?:\.\d+)?)\s*米")
CLAIMS_VIEW = re.compile(r"看到|可见|能看|观察到")
USES_COMPASS = re.compile(r"[东南西北]")
PICTURE = re.compile(r"图中|图片|画面|屏幕|照片|截图|图左|图右|图上|图下|图中央")
IMAGE_CORNER = re.compile(r"左上|右上|左下|右下")
FOCAL_PX = 256.  # 512 px at 90 degrees.
PATCH_PX = 512 / 224 * 14  # One DINOv2 patch at the policy's 224 px input, in capture pixels.
YAW_JUMP_DEG = 20.
SECTOR_DEG = 22.5
TOTAL_BLACK = .9
PLAIN_STD = 15.  # Descriptive: a frame this uniform has almost nothing to read.


def parse_region(text):
    """(bearing from the landmark, metres) from text like 西北侧约55米; None when the text has neither."""
    match = None
    for match in REGION.finditer(text or ""):
        pass
    return None if match is None else (COMPASS[match.group(1)], float(match.group(2)))


def bearing_from(anchor_xy, point_xy):
    return math.degrees(math.atan2(point_xy[1] - anchor_xy[1], point_xy[0] - anchor_xy[0])) % 360.


def angle_gap(a, b):
    return abs((a - b + 180.) % 360. - 180.)


def in_views(position, yaw, low, high):
    """Whether any of a grid of points on a box falls inside each 90 degree view, and its approximate width in capture pixels. No occlusion test."""
    grid = np.array([(x, y, z) for x in np.linspace(low[0], high[0], 5) for y in np.linspace(low[1], high[1], 5) for z in (low[2], high[2])])
    dx, dy, dz = (grid - np.asarray(position, dtype=float)).T
    forward, right = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    front, down = (forward > 0) & (np.abs(right) <= forward) & (np.abs(dz) <= forward), (dz < 0) & (np.abs(forward) <= -dz) & (np.abs(right) <= -dz)
    size = max(high[0] - low[0], high[1] - low[1])
    return {"front": bool(front.any()), "down": bool(down.any()), "front_px": float(min(512., FOCAL_PX * size / max(forward[front].min(), 1.))) if front.any() else 0., "down_px": float(min(512., FOCAL_PX * size / max((-dz[down]).min(), 1.))) if down.any() else 0.}


def load_designs():
    designs = {}
    for category, folder in DESIGNS.items():
        for path in sorted((UE / folder).glob("*.final.json")):
            final = read_json(path)
            meta_path = path.with_name(path.name.replace(".final.", ".metadata."))
            metadata = {r["id"]: r for r in read_json(meta_path)["routes"]} if meta_path.exists() else {}
            for route in final["routes"]:
                designs[(final["map"], route["route_id"])] = (route, metadata.get(route["route_id"]))
    for path in sorted((UE / FORESTS).glob("Forest_*.json")):
        final = read_json(path)
        for route in final["routes"]:
            designs[(final["map"], route["route_id"])] = (route, None)
    return designs


def anchor_of(metadata, end_region):
    """The landmark named at the end of the route, in metres; the first anchor when none is named."""
    if not metadata or not metadata.get("anchor_refs"):
        return None
    anchors = metadata["anchor_refs"]
    chosen = next((a for a in anchors if a.get("description") and a["description"] in (end_region or "")), anchors[0])
    low, high = np.array(chosen["lo"], dtype=float) / 100, np.array(chosen["hi"], dtype=float) / 100
    return {"center": (low + high) / 2, "low": low, "high": high}


def route_row(split, route, ds, design, data_root):
    o, n = route["offset"], route["count"]
    P, Y, A = ds.arrays["position"][o:o + n].astype(float), ds.arrays["yaw"][o:o + n].astype(float), ds.arrays["action"][o:o + n - 1].astype(float)
    agl = ds.arrays["agl"][o:o + n].astype(float)
    text = route["instruction_zh"]
    map_name = route["map_package"].split("/")[-1]
    yaw_deg = np.degrees(np.abs(A[:, 3]))
    deltas = np.diff(P, axis=0)
    motion = np.arctan2(deltas[:, 1], deltas[:, 0])
    row = dict(split=split, category=route["category"], map=map_name, route=route["route_id"], episode_id=route["episode_id"], steps=n - 1, length_m=route["length_m"], start_goal_m=route["start_goal_m"],
               detour=route["length_m"] / max(route["start_goal_m"], 1.), total_turn_deg=math.degrees(route["total_turn_rad"]), max_yaw_deg=float(yaw_deg.max()), yaw_jumps=int((yaw_deg > YAW_JUMP_DEG).sum()),
               yaw_rms_deg=float(np.sqrt((yaw_deg ** 2).mean())), max_heading_motion_deg=float(np.degrees(np.abs([wrap(x) for x in motion - Y[:-1]])).max()), compass=bool(USES_COMPASS.search(text)), claims_view=bool(CLAIMS_VIEW.search(text)),
               picture=bool(PICTURE.search(text)), image_corner=bool(IMAGE_CORNER.search(text)), following=terrain_following(route), climb_range_m=float(P[:, 2].max() - P[:, 2].min()), agl_min=float(np.nanmin(agl)) if np.isfinite(agl).any() else None, agl_max=float(np.nanmax(agl)) if np.isfinite(agl).any() else None)
    # Bearing from the goal to the start in the start's body frame: where the goal is when the flight begins.
    goal = P[-1] - P[0]
    row["goal_bearing_at_start_deg"] = abs(math.degrees(wrap(math.atan2(goal[1], goal[0]) - Y[0])))
    if design is None:
        row.update(design=False)
        return row
    plan, metadata = design
    via = plan.get("via_regions") or []
    row.update(design=True, type=plan.get("instruction_type"), shape=(metadata or {}).get("shape") or route["route_id"].split("_")[3] if metadata else None, instruction_same=plan["instruction_zh"] == text, via_count=len(via),
               via_named=sum(bool(v) and v in text for v in via), goal_named=bool(plan.get("end_region")) and plan["end_region"] in text, success=((metadata or {}).get("success_condition") or {}).get("kind"))
    waypoints = np.array(plan["waypoints_xyz_m"], dtype=float)
    row["design_goal_gap_m"] = float(np.linalg.norm(waypoints[-1] - P[-1]))
    row["design_start_gap_m"] = float(np.linalg.norm(waypoints[0] - P[0]))
    anchor = anchor_of(metadata, plan.get("end_region"))
    if anchor is not None:
        claim = parse_region(plan.get("end_region"))
        if claim is not None:
            row["claim_bearing_gap_deg"] = angle_gap(claim[0], bearing_from(anchor["center"], P[-1]))
            row["claim_distance_m"], row["true_distance_m"] = claim[1], float(np.linalg.norm(P[-1, :2] - anchor["center"][:2]))
        views = [in_views(p, y, anchor["low"], anchor["high"]) for p, y in zip(P, Y)]
        seen = [v["front"] or v["down"] for v in views]
        big = [v["front_px"] >= PATCH_PX or v["down_px"] >= PATCH_PX for v in views]
        row.update(anchor=True, anchor_seen_start=seen[0], anchor_seen_goal=seen[-1], anchor_seen_share=float(np.mean(seen)), anchor_big_start=big[0], anchor_big_goal=big[-1], anchor_big_share=float(np.mean(big)),
                   anchor_goal_px=max(views[-1]["front_px"], views[-1]["down_px"]))
    folder = Path(data_root) / route["directory"]
    quality = read_json(folder / "quality.json")
    frames = quality["frames"]
    stats = {v: [f["rgb"] for f in frames if f["view"] == v] for v in ("front", "down")}
    black = {v: [f["near_black_ratio"] > .2 for f in stats[v]] for v in stats}
    flat = {v: [f["rgb"]["laplacian_variance"] for f in frames if f["view"] == v] for v in ("front", "down")}
    row.update(quality=quality["quality"], human_review=quality["human_review"], issues=len(quality["issues"]), frames_per_view=len(black["down"]), black_front=int(sum(black["front"])), black_down=int(sum(black["down"])), total_black_front=sum(f["near_black_ratio"] > TOTAL_BLACK for f in stats["front"]), total_black_down=sum(f["near_black_ratio"] > TOTAL_BLACK for f in stats["down"]),
               plain_front=sum(f["std"] < PLAIN_STD for f in stats["front"]), plain_down=sum(f["std"] < PLAIN_STD for f in stats["down"]), down_sharp_median=float(np.median(flat["down"])), front_sharp_median=float(np.median(flat["front"])), interrupted=sum(p.name.startswith("_interrupted") for p in folder.iterdir()))
    return row


def share(rows, key, condition=bool):
    values = [r.get(key) for r in rows if r.get(key) is not None]
    return (sum(condition(v) for v in values), len(values))


def show(name, pair):
    print(f"  {name:58s} {pair[0]:5d} / {pair[1]:5d}  {100 * pair[0] / max(pair[1], 1):5.1f}%")


def run(args):
    designs = load_designs()
    rows = []
    for split in ("train", "val_seen"):
        ds = PreparedSplit(args.prepared, split)
        for route in ds.routes:
            rows.append(route_row(split, route, ds, designs.get((route["map_package"].split("/")[-1], route["route_id"])), args.data_root))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with open(out / "routes.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "routes.json", rows)
    report(rows)


def report(rows):
    print(f"== coverage: {len(rows)} routes, with a design file {share(rows, 'design')[0]}, with landmark coordinates {share(rows, 'anchor')[0]}")
    with_design = [r for r in rows if r.get("design")]
    print("\n== capture against design")
    show("instruction text identical to the design", share(with_design, "instruction_same"))
    show("start within 1 m of the first waypoint", share(with_design, "design_start_gap_m", lambda v: v <= 1))
    show("end within 2 m of the last waypoint", share(with_design, "design_goal_gap_m", lambda v: v <= 2))
    print("  quality flags:", {k: sum(r.get("quality") == k for r in rows) for k in ("pass", "warning", "failed")}, "failed:", [r["route"] for r in rows if r.get("quality") == "failed"])
    show("human review pending", share(with_design, "human_review", lambda v: v == "pending"))
    show("route folder with an interrupted-step leftover", share(with_design, "interrupted", lambda v: v > 0))
    print("\n== is the claim about the goal true (landmark centre to the recorded end, where parsable)")
    show("bearing within one compass sector", share(rows, "claim_bearing_gap_deg", lambda v: v <= SECTOR_DEG))
    gaps = [abs(r["claim_distance_m"] - r["true_distance_m"]) for r in rows if "claim_distance_m" in r]
    print(f"  distance error, median {np.median(gaps):.1f} m, p90 {np.percentile(gaps, 90):.1f} m, n={len(gaps)}")
    print("\n== can the instruction determine the path, by instruction type")
    print(f"  {'type':12s} {'n':>5s} {'via total':>9s} {'via named':>9s} {'goal named':>10s} {'detour':>7s} {'turn deg':>8s} {'goal off start view':>19s}")
    for kind in sorted({r.get("type") or "?" for r in rows}):
        group = [r for r in rows if (r.get("type") or "?") == kind]
        via, named = sum(r.get("via_count", 0) for r in group), sum(r.get("via_named", 0) for r in group)
        off = sum(r["goal_bearing_at_start_deg"] > 45 for r in group)
        print(f"  {kind:12s} {len(group):5d} {via:9d} {named:9d} {share(group, 'goal_named')[0]:10d} {np.median([r['detour'] for r in group]):7.2f} {np.median([r['total_turn_deg'] for r in group]):8.0f} {off:19d}")
    print("\n== what the policy cannot read")
    show("instruction uses compass words", share(rows, "compass"))
    show("instruction refers to a picture or screen", share(rows, "picture"))
    show("instruction uses image corners (upper left and so on)", share(rows, "image_corner"))
    show("  ...of which forests", (sum(r["image_corner"] and r["category"] == "forests" for r in rows), sum(r["image_corner"] for r in rows)))
    print("\n== landmark in frame, any point of the landmark box, without occlusion (upper bound)")
    for key, name in (("anchor_seen_start", "in a view at the start"), ("anchor_seen_goal", "in a view at the goal"), ("anchor_big_start", f"in a view at the start and >= one patch ({PATCH_PX:.0f} px)"), ("anchor_big_goal", "in a view at the goal and >= one patch")):
        show(name, share(rows, key))
    claims = [r for r in rows if r.get("claims_view")]
    show("instruction says it can be seen, and it is in a view at the goal", share(claims, "anchor_seen_goal"))
    for category in sorted({r["category"] for r in rows if r.get("anchor")}):
        show(f"  {category}: in a view at the goal", share([r for r in rows if r["category"] == category], "anchor_seen_goal"))
    for key in ("anchor_seen_share", "anchor_big_share", "anchor_goal_px"):
        values = [r[key] for r in rows if key in r]
        print(f"  {key}: median {np.median(values):.2f}, p10 {np.percentile(values, 10):.2f}, p90 {np.percentile(values, 90):.2f}")
    print("\n== labels")
    show(f"route with a single-step yaw above {YAW_JUMP_DEG:.0f} deg", share(rows, "yaw_jumps", lambda v: v > 0))
    print("  by category:", {c: sum(r["yaw_jumps"] > 0 for r in rows if r["category"] == c) for c in sorted({r["category"] for r in rows})})
    show("detour above 1.5 x start-goal", share(rows, "detour", lambda v: v > 1.5))
    show("start-goal under 20 m (SR20 is free at the start)", share(rows, "start_goal_m", lambda v: v <= 20))
    print(f"  share of frames: over 20% near black front/down; over {TOTAL_BLACK:.0%} near black front/down; plain (std < {PLAIN_STD:.0f}) front/down")
    for category in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == category]
        frames = sum(r["frames_per_view"] for r in group)
        values = [100 * sum(r[k] for r in group) / frames for k in ("black_front", "black_down", "total_black_front", "total_black_down", "plain_front", "plain_down")]
        print(f"    {category:16s} {values[0]:5.1f} {values[1]:5.1f} | {values[2]:5.2f} {values[3]:5.2f} | {values[4]:5.1f} {values[5]:5.1f}")
    show(f"route with a frame over {TOTAL_BLACK:.0%} near black, either view", (sum(r["total_black_front"] + r["total_black_down"] > 0 for r in rows), len(rows)))
    worst = sorted(rows, key=lambda r: r["total_black_front"] + r["total_black_down"], reverse=True)[:8]
    print("  worst:", [(r["route"], r["total_black_front"], r["total_black_down"], r["frames_per_view"]) for r in worst])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="runs/terrain_v2_dino/prepared")
    p.add_argument("--data-root", default="/mnt/f/WSL_ColdStorage/datasets/DatasetCaptureBatches/workbench")
    p.add_argument("--out", default="runs/data_audit")
    p.add_argument("--report", action="store_true", help="Only print the tables from an earlier run")
    args = p.parse_args()
    report(read_json(Path(args.out) / "routes.json")) if args.report else run(args)
