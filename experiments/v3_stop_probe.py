"""Model check after the v3 pilot (protocol section 15): leave-one-training-map-out STOP probes on frozen features.

Only the train and val_seen routes of the four training maps are used; no dev map, no UE, no policy training.
extract: per observation, the down view through DINOv2-B (CLS + 4x4 patch grid) and SigLIP-so400m (image embedding of
         the whole view and of its centre half); per route, SigLIP text embeddings of the instruction and of the target phrase.
probe:   for each held-out map, MLP STOP heads (fixed recipe, 3 seeds, probabilities averaged) trained on the other three
         maps' train routes, threshold calibrated on their val routes as in training, scored along the expert trajectory of
         every route of the held-out map. Variants differ only in their inputs (VARIANTS).
Run with SR3MXHAD_PROTOCOL=v3_pilot and PYTHONPATH=~/sr3mxhad_runs/pylib.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.protocol import PROTOCOL, PreparedSplit, code_version, read_features, read_json, write_json
from had.tokenizer import tokenize

SPLITS = ("train", "val_seen")
MAPS = ("L_MTN_A", "L_MTN_D", "L_DES_A", "L_DES_D")
SIGLIP = "google/siglip-so400m-patch14-384"
PROMPT = "an aerial view of {}"
VARIANTS = {"V0": ("front", "down2", "bow", "state"),            # the B inputs as they are
            "V1": ("front", "down_grid", "bow", "state"),        # finer down view: CLS + 4x4 grid instead of CLS + 2x2
            "V2": ("front", "down2", "text", "state"),           # pretrained text: SigLIP instruction embedding instead of the word bag
            "V3": ("front", "down2", "bow", "sim", "state"),     # image-text agreement: SigLIP down view vs target phrase
            "V4": ("front", "down_grid", "text", "sim", "state")}
SEEDS, EPOCHS, HIDDEN, BATCH, LR, WEIGHT_DECAY, DROPOUT = (0, 1, 2), 10, 256, 256, 1e-3, 1e-2, .2
RADIUS_M = 20.
GATE_GAIN = .10
RESAMPLES = 10000


def phrase_of(route, objects):
    """The target phrase exactly as the instruction says it, without the leading article."""
    en = objects[route["object"]]["en"]
    if en.lower() not in route["instruction"].lower():
        raise ValueError(f"{route['route_id']}: target phrase not in the instruction")
    return en[4:] if en.lower().startswith("the ") else en


def horizontal_to_target(ds):
    out = np.empty(ds.n)
    for r in ds.routes:
        sl = slice(r["offset"], r["offset"] + r["count"])
        p = ds.arrays["position"][sl]
        out[sl] = np.hypot(p[:, 0] - r["target"][0], p[:, 1] - r["target"][1])
    return out


def first_trigger(p, distance, threshold):
    """Distance at the first step with p >= threshold along one route, or None if it never triggers."""
    hit = np.flatnonzero(np.asarray(p) >= threshold)
    return None if len(hit) == 0 else float(distance[hit[0]])


def calibrate(routes, radius=RADIUS_M):
    """Threshold maximising routes whose first trigger lies within the radius; ties keep the higher threshold.
    routes: list of (probabilities, horizontal distances). Same rule as experiments.metrics.calibrate_stop."""
    records = []
    for p, d in routes:
        prefix = np.maximum.accumulate(p)
        idx = np.flatnonzero(np.r_[True, prefix[1:] > prefix[:-1]])
        records.append((p[idx], d[idx] <= radius))
    best, best_success = None, -1
    for threshold in sorted({float(v) for values, _ in records for v in values}, reverse=True):
        success = 0
        for values, inside in records:
            k = np.searchsorted(values, threshold)
            success += bool(k < len(values) and inside[k])
        if success > best_success:
            best, best_success = threshold, success
    return best


def auroc(score, label):
    score, label = np.asarray(score, dtype=float), np.asarray(label, dtype=bool)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    for value in np.unique(score):  # average ranks of ties
        tied = score == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    n_pos, n_neg = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bag_of_words(instructions, fit_on):
    vocab = sorted({w for text in fit_on for w in tokenize(text)})
    index = {w: i for i, w in enumerate(vocab)}
    out = np.zeros((len(instructions), len(vocab)), dtype=np.float32)
    for i, text in enumerate(instructions):
        for w in tokenize(text):
            if w in index:
                out[i, index[w]] = 1.
    return out, set(vocab)


def extract(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoProcessor, Dinov2Model, SiglipModel
    from had.model import DINO, image_transform

    objects = {o["key"]: o for o in json.load(open(args.objects, encoding="utf-8"))}
    dino = Dinov2Model.from_pretrained(DINO["model"], revision=DINO["revision"], local_files_only=True).to(args.device).eval().requires_grad_(False)
    siglip = SiglipModel.from_pretrained(SIGLIP, local_files_only=True, torch_dtype=torch.float16).to(args.device).eval().requires_grad_(False)  # WSL has 5 GB RAM
    processor = AutoProcessor.from_pretrained(SIGLIP, local_files_only=True)
    dino_transform = image_transform()

    def siglip_pixels(image):
        x = torch.from_numpy(np.asarray(image.resize((384, 384), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255
        return (x - .5) / .5

    class Down(Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return self.ds.n

        def __getitem__(self, i):
            with Image.open(self.ds.payload(i, ("down",), args.root)["images"]["down"]) as image:
                image = image.convert("RGB")
                w, h = image.size
                centre = image.crop((w // 4, h // 4, w - w // 4, h - h // 4))
                return dino_transform(image), siglip_pixels(image), siglip_pixels(centre)

    with torch.inference_mode():
        def text_features(texts):
            batch = processor.tokenizer(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(args.device)
            return siglip.get_text_features(**batch).float().cpu().numpy()

        meta = {"code": code_version(), "siglip": SIGLIP, "prompt": PROMPT, "dino": DINO, "logit_scale": float(siglip.logit_scale.exp()), "logit_bias": float(siglip.logit_bias), "splits": {}}
        for split in SPLITS:
            ds = PreparedSplit(args.prepared, split)
            out = Path(args.out) / split
            if (out / "done.json").exists():
                print(split, "already extracted", flush=True)
                meta["splits"][split] = read_json(out / "done.json")
                continue
            out.mkdir(parents=True, exist_ok=True)
            arrays = {"dino_grid": np.lib.format.open_memmap(out / "dino_grid.npy", "w+", np.float16, (ds.n, 17 * DINO["width"])),
                      "sig_down": np.lib.format.open_memmap(out / "sig_down.npy", "w+", np.float16, (ds.n, 1152)),
                      "sig_centre": np.lib.format.open_memmap(out / "sig_centre.npy", "w+", np.float16, (ds.n, 1152))}
            np.save(out / "text_instruction.npy", np.concatenate([text_features([r["instruction"] for r in ds.routes[i:i + 32]]) for i in range(0, len(ds.routes), 32)]))
            phrases = [PROMPT.format(phrase_of(r, objects)) for r in ds.routes]
            np.save(out / "text_phrase.npy", np.concatenate([text_features(phrases[i:i + 32]) for i in range(0, len(phrases), 32)]))
            cursor, started = 0, time.monotonic()
            for dino_x, full, centre in DataLoader(Down(ds), batch_size=args.batch_size, num_workers=args.workers):
                n = len(dino_x)
                hidden = dino(pixel_values=dino_x.to(args.device)).last_hidden_state  # fp32, as the feature cache
                side = int(round((hidden.shape[1] - 1) ** .5))
                grid = F.adaptive_avg_pool2d(hidden[:, 1:].transpose(1, 2).reshape(n, -1, side, side), 4).flatten(2).transpose(1, 2).flatten(1)
                arrays["dino_grid"][cursor:cursor + n] = torch.cat([hidden[:, 0], grid], 1).cpu().numpy().astype(np.float16)
                with torch.autocast("cuda", dtype=torch.float16):
                    arrays["sig_down"][cursor:cursor + n] = siglip.get_image_features(pixel_values=full.to(args.device)).float().cpu().numpy().astype(np.float16)
                    arrays["sig_centre"][cursor:cursor + n] = siglip.get_image_features(pixel_values=centre.to(args.device)).float().cpu().numpy().astype(np.float16)
                cursor += n
                if cursor % (args.batch_size * 50) < args.batch_size or cursor == ds.n:
                    print(f"{split} {cursor}/{ds.n}, {cursor / (time.monotonic() - started):.1f} observations/s", flush=True)
            for a in arrays.values():
                if not np.isfinite(np.asarray(a[:cursor], dtype=np.float32)).all():
                    raise ValueError("Non-finite features")
                a.flush()
            done = {"rows": ds.n, "routes": len(ds.routes), "prepared": ds.protocol["code"]}
            write_json(out / "done.json", done)
            meta["splits"][split] = done
        write_json(Path(args.out) / "meta.json", meta)


def load(args):
    data = {}
    meta = read_json(Path(args.features) / "meta.json")
    for split in SPLITS:
        ds = PreparedSplit(args.prepared, split)
        cached, _ = read_features(args.cache, split, ds)
        f = Path(args.features) / split
        route_of_row = np.repeat(np.arange(len(ds.routes)), [r["count"] for r in ds.routes])
        sig_down, sig_centre = (np.load(f / n, mmap_mode="r").astype(np.float32) for n in ("sig_down.npy", "sig_centre.npy"))
        phrase = np.load(f / "text_phrase.npy")[route_of_row]

        def logit(image):
            a = image / np.linalg.norm(image, axis=1, keepdims=True)
            b = phrase / np.linalg.norm(phrase, axis=1, keepdims=True)
            return (a * b).sum(1) * meta["logit_scale"] + meta["logit_bias"]
        data[split] = {"ds": ds, "route_of_row": route_of_row, "map": np.array([ds.routes[i]["route_id"][:7] for i in route_of_row]),
                       "front": cached["front"], "down2": cached["down"], "down_grid": np.load(f / "dino_grid.npy", mmap_mode="r"),
                       "text": np.load(f / "text_instruction.npy")[route_of_row], "sim": np.column_stack((logit(sig_down), logit(sig_centre))).astype(np.float32),
                       "state": ds.arrays["state"], "stop": ds.arrays["stop"], "distance": horizontal_to_target(ds)}
    return data, meta


def matrix(data, split, rows, parts, bow):
    d = data[split]
    blocks = []
    for part in parts:
        if part == "bow":
            blocks.append(bow[split][d["route_of_row"][rows]])
        else:
            blocks.append(np.asarray(d[part][rows], dtype=np.float32))
    return np.concatenate(blocks, 1)


def train_head(x, y, seed, device):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 1)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y.astype(np.float32)).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(EPOCHS):
        net.train()
        for batch in torch.randperm(len(xt), generator=generator).split(BATCH):
            batch = batch.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(net(xt[batch])[:, 0], yt[batch])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    net.eval()
    return net


def predict(nets, x, device):
    import torch
    xt = torch.from_numpy(x).to(device)
    with torch.inference_mode():
        return np.mean([torch.sigmoid(n(xt)[:, 0]).cpu().numpy() for n in nets], 0)


def probe(args):
    data, meta = load(args)
    objects = {o["key"]: o for o in json.load(open(args.objects, encoding="utf-8"))}
    results = {"code": code_version(), "features": meta, "recipe": {"seeds": SEEDS, "epochs": EPOCHS, "hidden": HIDDEN, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT},
               "variants": {k: list(v) for k, v in VARIANTS.items()}, "folds": {}, "routes": {}}
    for held in MAPS:
        others = [m for m in MAPS if m != held]
        fit_on = [r["instruction"] for r in data["train"]["ds"].routes if r["route_id"][:7] in others]
        bow, vocab = {}, None
        for split in SPLITS:
            bow[split], vocab = bag_of_words([r["instruction"] for r in data[split]["ds"].routes], fit_on)
        rows = {"fit": np.flatnonzero(np.isin(data["train"]["map"], others)), "calib": np.flatnonzero(np.isin(data["val_seen"]["map"], others)),
                "test": {s: np.flatnonzero(data[s]["map"] == held) for s in SPLITS}}
        fold = {"held_out": held, "fit_rows": int(len(rows["fit"])), "vocab": len(vocab), "variants": {}}
        for name, parts in VARIANTS.items():
            x_fit = matrix(data, "train", rows["fit"], parts, bow)
            mean, std = x_fit.mean(0), np.maximum(x_fit.std(0), 1e-6)
            x_fit = (x_fit - mean) / std
            nets = [train_head(x_fit, data["train"]["stop"][rows["fit"]], seed, args.device) for seed in SEEDS]
            del x_fit

            def scored(split, idx):
                return predict(nets, (matrix(data, split, idx, parts, bow) - mean) / std, args.device)

            p_cal = scored("val_seen", rows["calib"])
            cal_routes = []
            for k in np.unique(data["val_seen"]["route_of_row"][rows["calib"]]):
                m = data["val_seen"]["route_of_row"][rows["calib"]] == k
                cal_routes.append((p_cal[m], data["val_seen"]["distance"][rows["calib"]][m]))
            threshold = calibrate(cal_routes)
            outcome, probabilities, labels = {}, [], []
            for split in SPLITS:
                idx = rows["test"][split]
                p = scored(split, idx)
                probabilities.append(p); labels.append(data[split]["stop"][idx])
                for k in np.unique(data[split]["route_of_row"][idx]):
                    m = data[split]["route_of_row"][idx] == k
                    route = data[split]["ds"].routes[k]
                    first = first_trigger(p[m], data[split]["distance"][idx][m], threshold)
                    known = all(w in vocab for w in tokenize(phrase_of(route, objects)))
                    outcome[route["route_id"]] = {"first_m": first, "within": first is not None and first <= RADIUS_M, "known_phrase": known}
            share = float(np.mean([o["within"] for o in outcome.values()]))
            fold["variants"][name] = {"threshold": threshold, "first_within_20m": share, "never": float(np.mean([o["first_m"] is None for o in outcome.values()])),
                                      "auroc": auroc(np.concatenate(probabilities), np.concatenate(labels))}
            for route_id, o in outcome.items():
                results["routes"].setdefault(route_id, {})[name] = o
            print(held, name, {k: round(v, 3) for k, v in fold["variants"][name].items()}, flush=True)
        results["folds"][held] = fold
    route_ids = sorted(results["routes"])
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(route_ids), (RESAMPLES, len(route_ids)))
    base = np.array([results["routes"][r]["V0"]["within"] for r in route_ids], dtype=float)
    known = np.array([results["routes"][r]["V0"]["known_phrase"] for r in route_ids])
    results["pooled"], results["gate"] = {}, {}
    for name in VARIANTS:
        v = np.array([results["routes"][r][name]["within"] for r in route_ids], dtype=float)
        d = v - base
        ci = np.percentile(d[draws].mean(1), [2.5, 97.5])
        results["pooled"][name] = {"first_within_20m": float(v.mean()), "known_phrase": float(v[known].mean()), "unknown_phrase": float(v[~known].mean()),
                                   "minus_V0": float(d.mean()), "ci": [float(ci[0]), float(ci[1])]}
        if name != "V0":
            results["gate"][name] = bool(d.mean() >= GATE_GAIN and ci[0] > 0)
        print(name, {k: (round(x, 3) if isinstance(x, float) else [round(c, 3) for c in x]) for k, x in results["pooled"][name].items()}, flush=True)
    results["routes_known_phrase"] = int(known.sum()); results["routes_total"] = len(route_ids)
    print("gate (V0 + 10 points, CI above 0):", results["gate"], flush=True)
    write_json(Path(args.out), results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("extract", "probe"))
    p.add_argument("--prepared", default="runs/v3_pilot/prepared"); p.add_argument("--root", default="/mnt/f/WSL_ColdStorage/datasets/sr3m_v3_pilot")
    p.add_argument("--cache", default=str(Path.home() / "sr3mxhad_runs/v3_pilot/features"))
    p.add_argument("--features", default=str(Path.home() / "sr3mxhad_runs/v3_pilot/stop_probe"))
    p.add_argument("--objects", default="configs/v3_objects.json"); p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if PROTOCOL["name"] != "sr3m_had_v3_pilot":
        raise SystemExit("Run with SR3MXHAD_PROTOCOL=v3_pilot")
    if args.stage == "extract":
        args.out = args.out or args.features
        extract(args)
    else:
        args.out = args.out or "runs/v3_pilot/model_check/stop_probe.json"
        probe(args)
