"""Frozen monocular-depth features: Depth Anything V2 Small relative inverse depth, 16x16 per view.

Reads a prepared directory's route/image lists directly (any protocol version) and writes
raw (unstandardized) float32 arrays aligned with that preparation's rows. Rows already
computed elsewhere (`--reuse`, a directory with rows.npy and r2_<view>.npy per split) are
copied; a sample of them is recomputed and must agree.

Needs `transformers` (kept outside the shared venv: PYTHONPATH=~/sr3mxhad_runs/pylib).
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
SIZE, GRID, VIEWS = 266, 16, ("front", "down")
MEAN = torch.tensor([.485, .456, .406]).view(3, 1, 1); STD = torch.tensor([.229, .224, .225]).view(3, 1, 1)


def image_paths(prepared, split, root):
    routes = json.loads((Path(prepared) / f"{split}.json").read_text(encoding="utf8"))
    return [{v: str(Path(root) / r["directory"] / r["images"][v][t]) for v in VIEWS} for r in routes for t in range(r["count"])]


class Images(Dataset):
    def __init__(self, paths, rows):
        self.paths, self.rows = paths, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        out = []
        for v in VIEWS:
            with Image.open(self.paths[int(self.rows[k])][v]) as im:
                x = torch.from_numpy(np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)).copy())
            out.append((x.permute(2, 0, 1).float() / 255 - MEAN) / STD)
        return torch.stack(out)


@torch.inference_mode()
def compute(model, paths, rows, device, batch_size=32, workers=8, label=""):
    result = np.zeros((len(rows), len(VIEWS), GRID * GRID), np.float32)
    loader = DataLoader(Images(paths, rows), batch_size=batch_size, num_workers=workers, pin_memory=device.startswith("cuda"))
    cursor, started = 0, time.monotonic()
    for images in loader:
        n = len(images)
        depth = model(pixel_values=images.flatten(0, 1).to(device, non_blocking=True)).predicted_depth
        result[cursor:cursor + n] = F.adaptive_avg_pool2d(depth[:, None], GRID).flatten(1).reshape(n, len(VIEWS), -1).cpu().numpy()
        cursor += n
        if cursor % 3200 < n or cursor == len(rows):
            print(f"depth {label} {cursor}/{len(rows)}, {cursor/max(time.monotonic()-started, 1e-3):.1f} observations/s", flush=True)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite depth features")
    return result


def run(args):
    from transformers import AutoModelForDepthEstimation
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(args.device).eval().requires_grad_(False)
    for split in args.splits:
        out = Path(args.out) / split
        if (out / "depth.json").exists():
            print(f"depth {split}: already complete", flush=True); continue
        paths = image_paths(args.prepared, split, args.root)
        values = np.full((len(paths), len(VIEWS), GRID * GRID), np.nan, np.float32)
        if args.reuse:
            rows = np.load(Path(args.reuse) / split / "rows.npy")
            for j, v in enumerate(VIEWS):
                values[rows, j] = np.load(Path(args.reuse) / split / f"r2_{v}.npy")
            sample = rows[np.linspace(0, len(rows) - 1, 64).astype(int)]
            check = compute(model, paths, sample, args.device, label=f"{split} reuse-check")
            worst = float(np.abs(check - values[sample]).max() / np.abs(values[sample]).max())
            if worst > 1e-3:
                raise ValueError(f"Reused depth rows disagree with recomputation: relative {worst:.2e}")
            print(f"depth {split}: reused {len(rows)} rows, recomputed sample relative error {worst:.1e}", flush=True)
        todo = np.flatnonzero(np.isnan(values[:, 0, 0]))
        values[todo] = compute(model, paths, todo, args.device, label=split)
        out.mkdir(parents=True, exist_ok=True)
        for j, v in enumerate(VIEWS):
            np.save(out / f"{v}_depth.npy", values[:, j])
        meta = {"model": MODEL, "input_size": SIZE, "grid": GRID, "output": "relative_inverse_depth_mean_pooled",
                "rows": len(paths), "reused_rows": len(paths) - len(todo), "prepared": str(Path(args.prepared).resolve())}
        (out / "depth.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared", required=True); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val_seen", "val_unseen"]); p.add_argument("--reuse")
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
