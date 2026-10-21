"""Resumeable ImageNet CNN feature extraction; projections are never cached."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from experiment import PROTOCOL, PreparedSplit, read_json, write_json
from had_model import DEFAULT_WEIGHTS, image_transform, validate_weights, configure_precision


class Images(Dataset):
    def __init__(self, prepared, root, start=0):
        self.ds, self.root, self.start = prepared, root, start
        self.transform = image_transform()

    def __len__(self):
        return self.ds.n - self.start

    def __getitem__(self, index):
        p = self.ds.payload(index + self.start, ("front", "down"), self.root)
        out = []
        for view in ("front", "down"):
            with Image.open(p["images"][view]) as image:
                out.append(self.transform(image.convert("RGB")))
        return torch.stack(out)


def extract_locked(args):
    torch.set_num_threads(4)
    configure_precision()
    validate_weights(str(Path(args.weights).resolve()))
    backbone = resnet50(weights=None)
    backbone.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)
    backbone.fc = torch.nn.Identity()
    backbone.to(args.device).eval().requires_grad_(False)
    for split in args.splits:
        ds = PreparedSplit(args.prepared, split)
        out = Path(args.out) / split; out.mkdir(parents=True, exist_ok=True)
        meta_path = out / "cache.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta["prepared"] != ds.protocol or meta["backbone"] != PROTOCOL["weights"] or meta["image_size"] != 224:
                raise ValueError("Refusing to reuse a different feature cache")
            start = meta["completed_rows"]
            features = {v: np.load(out / f"{v}.npy", mmap_mode="r+") for v in ("front", "down")}
        else:
            if any(out.iterdir()):
                raise ValueError("Unrecognized partial cache; use a fresh directory")
            start = 0
            features = {v: np.lib.format.open_memmap(out / f"{v}.npy", mode="w+", dtype="float32", shape=(ds.n, 2048)) for v in ("front", "down")}
            meta = {"prepared": ds.protocol, "backbone": PROTOCOL["weights"], "image_size": 224,
                    "feature_stage": "cnn_before_trainable_projection", "dtype": "float32", "rows": ds.n, "completed_rows": 0}
            write_json(meta_path, meta)
        if any(x.shape != (ds.n, 2048) or x.dtype != np.float32 for x in features.values()):
            raise ValueError("Bad cache shape/dtype")
        loader = DataLoader(Images(ds, args.root, start), batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=args.device.startswith("cuda"),
                            **({"prefetch_factor": 1} if args.workers else {}))
        started = time.monotonic(); cursor = start; logged = started
        with torch.inference_mode():
            for images in loader:
                count = len(images)
                values = backbone(images.flatten(0, 1).to(args.device, non_blocking=True)).reshape(count, 2, 2048).cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite CNN features")
                for j, view in enumerate(("front", "down")):
                    features[view][cursor:cursor+count] = values[:, j]
                cursor += count
                if time.monotonic() - logged > 20 or cursor == ds.n:
                    for value in features.values(): value.flush()
                    meta["completed_rows"] = cursor; write_json(meta_path, meta)
                    print(f"cache {split}: {cursor}/{ds.n}, {(cursor-start)/max(time.monotonic()-started,.001):.1f} observations/s", flush=True)
                    logged = time.monotonic()
        for value in features.values(): value.flush()
        meta["completed_rows"] = cursor; write_json(meta_path, meta)


def extract(args):
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    with (root / ".cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        extract_locked(args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS)); p.add_argument("--device", default="cuda")
    p.add_argument("--splits", nargs="+", choices=("train", "val_seen", "val_unseen"), default=["train", "val_seen", "val_unseen"])
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--workers", type=int, default=8)
    extract(p.parse_args())
