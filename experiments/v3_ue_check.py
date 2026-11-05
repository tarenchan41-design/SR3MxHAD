"""Stage 2 (protocol section 17): check the levels imported into UE against the Blender deliverables, before stage 0.

Per map, in one UE process (VSM off, as section 17.5):
  terrain: line trace down on a grid over the inner map, against the exported 16-bit heightmap (the first hit can be a tree or a
           building, so the report gives the median and the share within 1 m, not the maximum);
  targets: the hit straight down at each target centre (should be near the target top) and a sphere sweep down through it
           (should hit something);
  renders: the down view 40 m above each target, and the front view from the Blender takeoff camera towards the target.
--biases renders the same poses again at other exposure biases (one UE process each), to set the exposure of the new mountain.
Close the GUI editor first. Writes only under --out; never saves a package.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.protocol import code_version, write_json

BLENDER = Path("/mnt/e/blender3dProjects")
PROFILE = "desert_mountain"
VSM_OFF = ["r.Shadow.Virtual.Enable 0"]
# Stage 2 capture and K closed loop (protocol 17.5, 11/05): VSM off plus a milder conventional shadow bias, which removes the
# triangle facets of the 6 m terrain mesh in slope shadows and keeps the shadows of fences and animals (runs/v3/stage2/shadow).
# For K flights set SR3MXHAD_K_CONSOLE to the same list.
STAGE2_CONSOLE = VSM_OFF + ["r.Shadow.CSMDepthBias 20", "r.Shadow.CSMSlopeScaleDepthBias 4.5"]
# Capture settings of the v2 desert dataset (trajectory.json "capture"), front and down views only.
CAPTURE = {"step_interval_cm": 200.0, "resolution": [512, 512], "fov_deg": 90.0, "exposure_bias": -10.0, "persistent_state": True, "warmup_frames": 3,
           "settle_ms": 50.0, "capture_passes": 2, "views": ["front", "down"], "modalities": ["rgb"], "environment_mode": "preserve_scene"}
DOWN_AGL_M = 40.
GRID = 25


def maps():
    """map id -> (UE package, scene config, heightmap .r16)."""
    out = {}
    for m in "ABCD":
        out[f"L_MTR_{m}"] = (f"/Game/Mountain/Maps/L_MTR_{m}", BLENDER / f"mountain/MTR_{m}/scene_config.json", BLENDER / f"mountain/MTR_{m}/export_ue5/MTR_{m}_heightmap_2017.r16")
        out[f"L_DES_{m}_v3"] = (f"/Game/Desert/Maps/L_DES_{m}_v3", BLENDER / f"desert/DES_{m}/scene_config_v3.json",
                                BLENDER / f"desert/DES_{m}/export_ue5_v3/DES_{m}_v3_heightmap_2017.r16")
    return out


class Heightmap:
    """The exported landscape heightmap in UE metres (x east, y south): z = location + (v - 32768) * scale_z / 128 cm."""

    def __init__(self, path, landscape):
        n = landscape["resolution"]
        self.v = np.fromfile(path, dtype="<u2").reshape(n, n).astype(np.float64)
        self.x0, self.y0, self.z0 = (c / 100 for c in landscape["ue_landscape_location_cm"])
        self.sx, self.sy, self.sz = (s / 100 for s in landscape["ue_landscape_scale"])

    def __call__(self, x, y):
        u, w = (x - self.x0) / self.sx, (y - self.y0) / self.sy  # column, row
        c, r = int(math.floor(u)), int(math.floor(w))
        a, b = u - c, w - r
        v = self.v[r, c] * (1 - a) * (1 - b) + self.v[r, c + 1] * a * (1 - b) + self.v[r + 1, c] * (1 - a) * b + self.v[r + 1, c + 1] * a * b
        return self.z0 + (v - 32768) * self.sz / 128

    def many(self, x, y):
        """Vectorised __call__; NaN outside the heightmap."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        u, w = (x - self.x0) / self.sx, (y - self.y0) / self.sy
        c, r = np.floor(u).astype(int), np.floor(w).astype(int)
        n = self.v.shape[0]
        inside = (c >= 0) & (r >= 0) & (c < n - 1) & (r < n - 1)
        out = np.full(x.shape, np.nan)
        c, r, a, b = c[inside], r[inside], (u - np.floor(u))[inside], (w - np.floor(w))[inside]
        v = self.v[r, c] * (1 - a) * (1 - b) + self.v[r, c + 1] * a * (1 - b) + self.v[r + 1, c] * (1 - a) * b + self.v[r + 1, c + 1] * a * b
        out[inside] = self.z0 + (v - 32768) * self.sz / 128
        return out


def ue_xy(blender_xy):
    return float(blender_xy[0]), -float(blender_xy[1])


def poses(config):
    """Per target: the down pose 40 m above the target base and the front pose at the takeoff camera, yawed to the target."""
    out = []
    for t in config["targets"]:
        x, y = ue_xy(t["position"])
        cam = t["takeoff_camera"]["position"]
        cx, cy = ue_xy(cam)
        out.append({"id": t["id"], "down": ([x, y, t["position"][2] + DOWN_AGL_M], 0.), "front": ([cx, cy, float(cam[2])], math.atan2(y - cy, x - cx))})
    return out


def brightness(image):
    a = np.asarray(image, dtype=np.float64) / 255
    luma = a @ np.array([.2126, .7152, .0722])
    return {"mean": round(float(luma.mean()), 4), "dark_share": round(float((luma < .08).mean()), 4), "bright_share": round(float((luma > .95).mean()), 4)}


def check_map(map_id, package, config, heightmap, out, bias, geometry):
    from sim_k.backend import UEBackend
    capture = dict(CAPTURE, exposure_bias=bias)
    folder = out / map_id / f"bias_{bias:+.0f}"
    report = {"map": map_id, "package": package, "profile": PROFILE, "console": VSM_OFF, "capture": capture}
    with UEBackend(folder / "session", PROFILE, package, capture, keep_images=True, restart_every=10 ** 6, console=VSM_OFF) as backend:
        if geometry:
            half = min(config["extent_m"]) / 2 - 300
            diffs = []
            for x in np.linspace(-half, half, GRID):
                for y in np.linspace(-half, half, GRID):
                    expected = heightmap(x, y)
                    z = backend.ground_z_m([x, y, expected])
                    diffs.append(np.nan if z is None else z - expected)
            d = np.array(diffs)
            report["terrain"] = {"points": len(d), "no_hit": int(np.isnan(d).sum()), "median_m": round(float(np.nanmedian(d)), 3),
                                 "within_1m": round(float(np.nanmean(np.abs(d) < 1)), 3), "above_1m": round(float(np.nanmean(d > 1)), 3),
                                 "below_1m": round(float(np.nanmean(d < -1)), 3), "p05_p95_m": [round(float(q), 2) for q in np.nanpercentile(d, [5, 95])]}
            targets = []
            for t in config["targets"]:
                x, y = ue_xy(t["position"])
                base, top = float(t["position"][2]), float(t["top_z"])
                z = backend.ground_z_m([x, y, top])
                sweep = backend.sweep([x, y, top + 15], [x, y, base - 3])
                targets.append({"id": t["id"], "class": t["class"], "base_z": base, "top_z": top, "hit_z": z, "hit_minus_top": None if z is None else round(z - top, 2),
                                "hit_minus_base": None if z is None else round(z - base, 2), "heightmap_minus_base": round(heightmap(x, y) - base, 2),
                                "sweep_hit": sweep["hit"], "sweep_actor": sweep.get("actor")})
            report["targets"] = targets
        shots = []
        for p in poses(config):
            row = {"id": p["id"]}
            for view in ("down", "front"):
                position, yaw = p[view]
                images = backend.render(np.array(position), yaw)
                row[view] = {"position_m": position, "yaw_deg": round(math.degrees(yaw), 2), "file": str(sorted((folder / "session/images").glob("render_*"))[-1].name),
                             **brightness(images[view])}
                images[view].save(folder / f"{p['id']}_{view}.png")
            shots.append(row)
        report["renders"] = shots
    for f in (folder / "session/images").glob("render_*"):
        f.unlink()
    write_json(folder / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/v3/ue_check")
    p.add_argument("--maps", nargs="*", default=None)
    p.add_argument("--biases", type=float, nargs="*", default=[CAPTURE["exposure_bias"]], help="the first one also runs the geometry checks")
    args = p.parse_args()
    out = Path(args.out)
    table = maps()
    summary = {"code": code_version(), "maps": {}}
    for map_id in args.maps or list(table):
        package, config_path, heightmap_path = table[map_id]
        config = json.load(open(config_path, encoding="utf-8"))
        heightmap = Heightmap(heightmap_path, config["ue5_landscape"])
        for i, bias in enumerate(args.biases):
            r = check_map(map_id, package, config, heightmap, out, bias, geometry=i == 0)
            s = summary["maps"].setdefault(map_id, {})
            if "terrain" in r:
                s["terrain"] = r["terrain"]
                s["targets_without_sweep_hit"] = [t["id"] for t in r["targets"] if not t["sweep_hit"]]
                s["targets_hit_far_from_top"] = [(t["id"], t["hit_minus_top"]) for t in r["targets"] if t["hit_minus_top"] is None or abs(t["hit_minus_top"]) > 3]
            s[f"bias_{bias:+.0f}"] = {view: round(float(np.mean([x[view]["mean"] for x in r["renders"]])), 4) for view in ("down", "front")}
            print(map_id, bias, json.dumps(s), flush=True)
            write_json(out / "summary.json", summary)


if __name__ == "__main__":
    main()
