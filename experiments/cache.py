"""Resumeable frozen DINOv2 feature extraction (fp32 compute, float16 storage); projections are never cached.

--grid4-out also writes, from the same forward pass, the down view as CLS + 4x4 patch mean into a second cache laid out like the
B4 cache of protocol section 16 (front.npy links to the 2x2 front view); stage 2 compares 2x2 and 4x4 (section 17.6).
--mosaic-out also writes CLS + 2x2 of the AerialVLA-style mosaic (had.model.mosaic_image; D4 baseline AV,
docs/d4_baseline_spec.md) into mosaic.npy, from the images read for the 2x2 cache. The mosaic gets its own forward pass so
the 2x2 and 4x4 caches see exactly the batches they see without it.

Needs `transformers` (kept outside the shared venv: PYTHONPATH=~/sr3mxhad_runs/pylib)."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from experiments.protocol import PROTOCOL, PreparedSplit, read_json, write_json
from had.model import FEATURE_DIM, DinoFeatures, feature_dim, image_transform, configure_precision, mosaic_image, pooled_tokens


class Images(Dataset):
    def __init__(self, prepared, root, start=0, mosaic=False):
        self.ds, self.root, self.start, self.mosaic = prepared, root, start, mosaic
        self.transform = image_transform()

    def __len__(self):
        return self.ds.n - self.start

    def __getitem__(self, index):
        p = self.ds.payload(index + self.start, ("front", "down"), self.root)
        out, raw = [], {}
        for view in ("front", "down"):
            with Image.open(p["images"][view]) as image:
                raw[view] = image.convert("RGB")
                out.append(self.transform(raw[view]))
        if not self.mosaic:
            return torch.stack(out)
        return torch.stack(out), self.transform(mosaic_image(raw["front"], raw["down"]))


def extract_locked(args):
    torch.set_num_threads(4)
    configure_precision()
    if FEATURE_DIM != PROTOCOL["feature_dim"]:
        raise ValueError("Model feature width differs from PROTOCOL")
    backbone = DinoFeatures().to(args.device).eval()
    dtype = np.dtype(PROTOCOL["feature_dtype"])
    for split in args.splits:
        ds = PreparedSplit(args.prepared, split)
        out = Path(args.out) / split; out.mkdir(parents=True, exist_ok=True)
        meta_path = out / "cache.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta["prepared"] != ds.protocol or meta["backbone"] != PROTOCOL["weights"] or meta["image_size"] != PROTOCOL["image_size"]:
                raise ValueError("Refusing to reuse a different feature cache")
            start = meta["completed_rows"]
            features = {v: np.load(out / f"{v}.npy", mmap_mode="r+") for v in ("front", "down")}
        else:
            if any(out.iterdir()):
                raise ValueError("Unrecognized partial cache; use a fresh directory")
            start = 0
            features = {v: np.lib.format.open_memmap(out / f"{v}.npy", mode="w+", dtype=dtype, shape=(ds.n, FEATURE_DIM)) for v in ("front", "down")}
            meta = {"prepared": ds.protocol, "backbone": PROTOCOL["weights"], "image_size": PROTOCOL["image_size"],
                    "feature_stage": "frozen_backbone_before_trainable_projection", "feature": PROTOCOL["feature"],
                    "dtype": PROTOCOL["feature_dtype"], "rows": ds.n, "completed_rows": 0}
            write_json(meta_path, meta)
        if any(x.shape != (ds.n, FEATURE_DIM) or x.dtype != dtype for x in features.values()):
            raise ValueError("Bad cache shape/dtype")
        grid4 = open_grid4(args, split, ds, meta, start, out, dtype) if args.grid4_out else None
        mosaic = open_mosaic(args, split, ds, meta, start, dtype) if args.mosaic_out else None
        loader = DataLoader(Images(ds, args.root, start, mosaic is not None), batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=args.device.startswith("cuda"),
                            **({"prefetch_factor": 1} if args.workers else {}))
        started = time.monotonic(); cursor = start; logged = started
        with torch.inference_mode():
            for images in loader:
                if mosaic is not None:
                    images, mosaics = images
                count = len(images)
                hidden = backbone.net(pixel_values=images.flatten(0, 1).to(args.device, non_blocking=True)).last_hidden_state
                values = pooled_tokens(hidden, 2).reshape(count, 2, FEATURE_DIM).cpu().numpy()
                if grid4 is not None:
                    down4 = pooled_tokens(hidden.reshape(count, 2, *hidden.shape[1:])[:, 1], 4).cpu().numpy()
                    if not np.isfinite(down4).all() or np.abs(down4).max() >= np.finfo(dtype).max:
                        raise ValueError("Nonfinite 4x4 features or out of storage range")
                    grid4["down"][cursor:cursor+count] = down4.astype(dtype)
                if mosaic is not None:
                    tiles = backbone(mosaics.to(args.device, non_blocking=True)).cpu().numpy()
                    if not np.isfinite(tiles).all() or np.abs(tiles).max() >= np.finfo(dtype).max:
                        raise ValueError("Nonfinite mosaic features or out of storage range")
                    mosaic["mosaic"][cursor:cursor+count] = tiles.astype(dtype)
                if not np.isfinite(values).all() or np.abs(values).max() >= np.finfo(dtype).max:
                    raise ValueError("Nonfinite backbone features or out of storage range")
                values = values.astype(dtype)
                for j, view in enumerate(("front", "down")):
                    features[view][cursor:cursor+count] = values[:, j]
                cursor += count
                if time.monotonic() - logged > 20 or cursor == ds.n:
                    for extra in (grid4, mosaic):  # the extra caches never claim more rows than the 2x2 one
                        if extra is not None:
                            flush_extra(extra, cursor)
                    for value in features.values(): value.flush()
                    meta["completed_rows"] = cursor; write_json(meta_path, meta)
                    print(f"cache {split}: {cursor}/{ds.n}, {(cursor-start)/max(time.monotonic()-started,.001):.1f} observations/s", flush=True)
                    logged = time.monotonic()
        for extra in (grid4, mosaic):
            if extra is not None:
                flush_extra(extra, cursor)
        for value in features.values(): value.flush()
        meta["completed_rows"] = cursor; write_json(meta_path, meta)


def open_grid4(args, split, ds, meta, start, source, dtype):
    """The 4x4 down cache for this split, resumed in step with the 2x2 cache."""
    out = Path(args.grid4_out) / split; out.mkdir(parents=True, exist_ok=True)
    path, width = out / "cache.json", feature_dim(4)
    if path.exists():
        grid_meta = read_json(path)
        if grid_meta["prepared"] != ds.protocol or grid_meta["completed_rows"] < start:
            raise ValueError("The 4x4 cache belongs to another preparation or is behind the 2x2 cache")
        down = np.load(out / "down.npy", mmap_mode="r+")
    else:
        if start:
            raise ValueError("The 2x2 cache is already partly filled; start the 4x4 cache together with it")
        down = np.lib.format.open_memmap(out / "down.npy", mode="w+", dtype=dtype, shape=(ds.n, width))
        (out / "front.npy").symlink_to((source / "front.npy").resolve())
        grid_meta = {**meta, "feature": {"front": meta["feature"], "down": "last_layer_cls_plus_4x4_patch_mean"}, "feature_dims": {"front": FEATURE_DIM, "down": width},
                     "sources": {"front": str(source.resolve()), "down": "same forward pass as the 2x2 cache"}, "completed_rows": 0}
        write_json(path, grid_meta)
    if down.shape != (ds.n, width) or down.dtype != dtype:
        raise ValueError("Bad 4x4 cache shape/dtype")
    return {"down": down, "meta": grid_meta, "path": path}


def open_mosaic(args, split, ds, meta, start, dtype):
    """The mosaic cache for this split, resumed in step with the 2x2 cache."""
    out = Path(args.mosaic_out) / split; out.mkdir(parents=True, exist_ok=True)
    path = out / "cache.json"
    if path.exists():
        mosaic_meta = read_json(path)
        if mosaic_meta["prepared"] != ds.protocol or mosaic_meta["completed_rows"] < start:
            raise ValueError("The mosaic cache belongs to another preparation or is behind the 2x2 cache")
        values = np.load(out / "mosaic.npy", mmap_mode="r+")
    else:
        if start:
            raise ValueError("The 2x2 cache is already partly filled; start the mosaic cache together with it")
        values = np.lib.format.open_memmap(out / "mosaic.npy", mode="w+", dtype=dtype, shape=(ds.n, FEATURE_DIM))
        mosaic_meta = {**meta, "feature": {"mosaic": meta["feature"]}, "feature_dims": {"mosaic": FEATURE_DIM},
                       "input": "had.model.mosaic_image: front over down, each PIL bicubic 224x224, then the whole 224x448 frame resized to 224x224",
                       "sources": {"mosaic": "same images as the 2x2 cache, own forward pass"}, "completed_rows": 0}
        write_json(path, mosaic_meta)
    if values.shape != (ds.n, FEATURE_DIM) or values.dtype != dtype:
        raise ValueError("Bad mosaic cache shape/dtype")
    return {"mosaic": values, "meta": mosaic_meta, "path": path}


def flush_extra(extra, cursor):
    for key in ("down", "mosaic"):
        if key in extra:
            extra[key].flush()
    extra["meta"]["completed_rows"] = cursor; write_json(extra["path"], extra["meta"])


def extract(args):
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    with (root / ".cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        extract_locked(args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--splits", nargs="+", choices=("train", "val_seen", "val_unseen", "dev", "test"), default=["train", "val_seen", "val_unseen"])
    p.add_argument("--grid4-out", default=None, help="also write the down view on a 4x4 grid here (same forward pass)")
    p.add_argument("--mosaic-out", default=None, help="also write the AerialVLA-style mosaic (CLS + 2x2) here (same images, own forward pass)")
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--workers", type=int, default=8)
    extract(p.parse_args())
