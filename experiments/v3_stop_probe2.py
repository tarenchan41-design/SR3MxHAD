"""STOP probe, round 2 (protocol section 18): leave-one-training-map-out, as section 15, with V1 (down CLS + 4x4) as the base.

Only the train and val_seen routes of the four pilot training maps; no dev map, no UE, no policy training.
extract-rc:   RemoteCLIP ViT-L/14 image embeddings of the down view (whole and centre half) and text embeddings of the target phrase.
vlm-manifest: the down-view frames and questions for the zero-shot VLM (W4), which runs on the 5090 (experiments/v3_stop_vlm.py).
probe:        V1 and W1-W3 trained as section 15 (fixed recipe, 3 seeds, probabilities averaged, threshold calibrated on the
              other maps' val routes); W4, if its scores exist, uses P(yes) directly with the same calibration.
Run with SR3MXHAD_PROTOCOL=v3_pilot and PYTHONPATH=~/sr3mxhad_runs/pylib.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments import v3_stop_probe as base
from experiments.protocol import PROTOCOL, PreparedSplit, code_version, read_json, write_json

REMOTECLIP = {"arch": "ViT-L-14", "url": "https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-L-14.pt"}
PROMPT = base.PROMPT  # "an aerial view of {}", as V3 in section 15
VLM_QUESTION = 'This is a top-down view from a drone camera. Is the target "{}" directly below the camera? Answer yes or no.'
V1 = ("front", "down_grid", "bow", "state")
VARIANTS = {"V1": {"parts": V1},
            "W1_s5": {"parts": V1, "soft_sigma": 5.},     # soft target exp(-d^2 / 2 sigma^2), d = horizontal distance to the target
            "W1_s10": {"parts": V1, "soft_sigma": 10.},
            "W2": {"parts": V1, "pos_weight": True},      # positives weighted by negatives / positives of the fit rows
            "W3": {"parts": V1 + ("rc_sim",)}}            # + two RemoteCLIP image-text logits (whole and centre down view)
BASELINE = "V1"


def soft_target(distance, sigma):
    return np.exp(-np.square(distance) / (2 * sigma ** 2)).astype(np.float32)


def positive_weight(stop):
    stop = np.asarray(stop, dtype=bool)
    return float((~stop).sum() / max(stop.sum(), 1))


def seen_classes(objects, maps):
    return {o["cls"] for o in objects.values() if o["map"] in maps}


def p_yes(yes_logits, no_logits):
    """P(yes) from next-token logits: softmax mass on the yes tokens over yes + no tokens (rows = frames)."""
    yes, no = np.atleast_2d(yes_logits).astype(np.float64), np.atleast_2d(no_logits).astype(np.float64)
    m = np.maximum(yes.max(1, keepdims=True), no.max(1, keepdims=True))
    a, b = np.exp(yes - m).sum(1), np.exp(no - m).sum(1)
    return a / (a + b)


def remoteclip(device, weights):
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms(REMOTECLIP["arch"])
    state = torch.load(weights, map_location="cpu", mmap=True)
    missing = model.load_state_dict(state)
    del state
    return model.to(device).eval().requires_grad_(False), preprocess, open_clip.get_tokenizer(REMOTECLIP["arch"]), str(missing)


def extract_rc(args):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    objects = {o["key"]: o for o in json.load(open(args.objects, encoding="utf-8"))}
    model, preprocess, tokenizer, loaded = remoteclip(args.device, args.remoteclip)

    class Down(Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return self.ds.n

        def __getitem__(self, i):
            with Image.open(self.ds.payload(i, ("down",), args.root)["images"]["down"]) as image:
                image = image.convert("RGB")
                w, h = image.size
                return preprocess(image), preprocess(image.crop((w // 4, h // 4, w - w // 4, h - h // 4)))

    meta = {"code": code_version(), "remoteclip": REMOTECLIP, "weights": args.remoteclip, "load": loaded, "prompt": PROMPT,
            "logit_scale": float(model.logit_scale.exp()), "splits": {}}
    with torch.inference_mode():
        for split in base.SPLITS:
            ds = PreparedSplit(args.prepared, split)
            out = Path(args.features) / split
            if (out / "rc_done.json").exists():
                print(split, "already extracted", flush=True)
                meta["splits"][split] = read_json(out / "rc_done.json")
                continue
            phrases = [PROMPT.format(base.phrase_of(r, objects)) for r in ds.routes]
            text = torch.cat([model.encode_text(tokenizer(phrases[i:i + 32]).to(args.device)).float().cpu() for i in range(0, len(phrases), 32)])
            np.save(out / "rc_phrase.npy", text.numpy())
            arrays = {k: np.lib.format.open_memmap(out / f"{k}.npy", "w+", np.float16, (ds.n, text.shape[1])) for k in ("rc_down", "rc_centre")}
            cursor, started = 0, time.monotonic()
            for full, centre in DataLoader(Down(ds), batch_size=args.batch_size, num_workers=args.workers):
                n = len(full)
                with torch.autocast("cuda", dtype=torch.float16):
                    arrays["rc_down"][cursor:cursor + n] = model.encode_image(full.to(args.device)).float().cpu().numpy().astype(np.float16)
                    arrays["rc_centre"][cursor:cursor + n] = model.encode_image(centre.to(args.device)).float().cpu().numpy().astype(np.float16)
                cursor += n
                if cursor % (args.batch_size * 50) < args.batch_size or cursor == ds.n:
                    print(f"{split} {cursor}/{ds.n}, {cursor / (time.monotonic() - started):.1f} observations/s", flush=True)
            for a in arrays.values():
                if cursor != ds.n or not np.isfinite(np.asarray(a, dtype=np.float32)).all():
                    raise ValueError("Incomplete or non-finite features")
                a.flush()
            done = {"rows": ds.n, "routes": len(ds.routes), "prepared": ds.protocol["code"]}
            write_json(out / "rc_done.json", done)
            meta["splits"][split] = done
    write_json(Path(args.features) / "rc_meta.json", meta)


def vlm_manifest(args):
    """One line per observation: split, row, image path relative to the dataset root, target phrase, question."""
    objects = {o["key"]: o for o in json.load(open(args.objects, encoding="utf-8"))}
    root, lines = Path(args.root), []
    for split in base.SPLITS:
        ds = PreparedSplit(args.prepared, split)
        for i in range(ds.n):
            route = ds.routes[int(ds.route_index[i])]
            phrase = base.phrase_of(route, objects)
            image = Path(ds.payload(i, ("down",), root)["images"]["down"]).relative_to(root)
            lines.append({"split": split, "row": i, "image": image.as_posix(), "phrase": phrase, "question": VLM_QUESTION.format(phrase)})
    out = Path(args.vlm_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    (out / "files.txt").write_text("".join(f"{line['image']}\n" for line in lines), encoding="utf-8")
    print(len(lines), "frames ->", out, flush=True)


def load(args):
    data, meta = base.load(args)
    rc_scale = read_json(Path(args.features) / "rc_meta.json")["logit_scale"] if (Path(args.features) / "rc_meta.json").exists() else None
    vlm = Path(args.vlm_dir)
    for split in base.SPLITS:
        d, f = data[split], Path(args.features) / split
        if rc_scale is not None:
            phrase = np.load(f / "rc_phrase.npy")[d["route_of_row"]]
            phrase = phrase / np.linalg.norm(phrase, axis=1, keepdims=True)

            def logit(name):
                image = np.load(f / name).astype(np.float32)
                return (image / np.linalg.norm(image, axis=1, keepdims=True) * phrase).sum(1) * rc_scale
            d["rc_sim"] = np.column_stack((logit("rc_down.npy"), logit("rc_centre.npy"))).astype(np.float32)
        if (vlm / f"{split}.npy").exists():
            d["vlm"] = np.load(vlm / f"{split}.npy").astype(np.float64)
            if d["vlm"].shape != (d["ds"].n,) or not np.isfinite(d["vlm"]).all():
                raise ValueError(f"{split}: VLM scores do not match the prepared rows")
    return data, meta


def train_head(x, y, seed, device, pos_weight=None):
    """Section 15 recipe; y may be soft, positives may be weighted."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], base.HIDDEN), nn.ReLU(), nn.Dropout(base.DROPOUT), nn.Linear(base.HIDDEN, 1)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    weight = None if pos_weight is None else torch.tensor(pos_weight, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(base.EPOCHS):
        net.train()
        for batch in torch.randperm(len(xt), generator=generator).split(base.BATCH):
            batch = batch.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(net(xt[batch])[:, 0], yt[batch], pos_weight=weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    net.eval()
    return net


def evaluate(data, rows, score, threshold, objects, vocab, seen):
    """Per held-out route: first-trigger distance, whether it is within 20 m, phrase in vocabulary, target class seen on the other maps."""
    outcome, probabilities, labels = {}, [], []
    for split in base.SPLITS:
        idx = rows["test"][split]
        p = score(split, idx)
        probabilities.append(p); labels.append(data[split]["stop"][idx])
        for k in np.unique(data[split]["route_of_row"][idx]):
            m = data[split]["route_of_row"][idx] == k
            route = data[split]["ds"].routes[k]
            first = base.first_trigger(p[m], data[split]["distance"][idx][m], threshold)
            outcome[route["route_id"]] = {"first_m": first, "within": first is not None and first <= base.RADIUS_M,
                                          "known_phrase": all(w in vocab for w in base.tokenize(base.phrase_of(route, objects))),
                                          "seen_class": objects[route["object"]]["cls"] in seen}
    summary = {"threshold": threshold, "first_within_20m": float(np.mean([o["within"] for o in outcome.values()])),
               "never": float(np.mean([o["first_m"] is None for o in outcome.values()])),
               "auroc": base.auroc(np.concatenate(probabilities), np.concatenate(labels))}
    return outcome, summary


def calibrate_on(data, rows, p_cal):
    routes = []
    r = data["val_seen"]["route_of_row"][rows["calib"]]
    for k in np.unique(r):
        routes.append((p_cal[r == k], data["val_seen"]["distance"][rows["calib"]][r == k]))
    return base.calibrate(routes)


def probe(args):
    data, meta = load(args)
    objects = {o["key"]: o for o in json.load(open(args.objects, encoding="utf-8"))}
    variants = {k: v for k, v in VARIANTS.items() if k != "W3" or "rc_sim" in data["train"]}
    if "vlm" in data["train"] and "vlm" in data["val_seen"]:
        variants["W4"] = {"score": "vlm"}
    results = {"code": code_version(), "features": meta, "baseline": BASELINE, "vlm_question": VLM_QUESTION,
               "recipe": {"seeds": base.SEEDS, "epochs": base.EPOCHS, "hidden": base.HIDDEN, "batch": base.BATCH, "lr": base.LR,
                          "weight_decay": base.WEIGHT_DECAY, "dropout": base.DROPOUT},
               "variants": variants, "folds": {}, "routes": {}}
    for held in base.MAPS:
        others = [m for m in base.MAPS if m != held]
        fit_on = [r["instruction"] for r in data["train"]["ds"].routes if r["route_id"][:7] in others]
        bow, vocab = {}, None
        for split in base.SPLITS:
            bow[split], vocab = base.bag_of_words([r["instruction"] for r in data[split]["ds"].routes], fit_on)
        rows = {"fit": np.flatnonzero(np.isin(data["train"]["map"], others)), "calib": np.flatnonzero(np.isin(data["val_seen"]["map"], others)),
                "test": {s: np.flatnonzero(data[s]["map"] == held) for s in base.SPLITS}}
        seen = seen_classes(objects, others)
        fold = {"held_out": held, "fit_rows": int(len(rows["fit"])), "vocab": len(vocab), "seen_classes": sorted(seen), "variants": {}}
        for name, spec in variants.items():
            if "score" in spec:  # zero-shot score, no training
                def scored(split, idx, key=spec["score"]):
                    return data[split][key][idx]
            else:
                x_fit = base.matrix(data, "train", rows["fit"], spec["parts"], bow)
                mean, std = base.standardize(x_fit)
                stop = data["train"]["stop"][rows["fit"]]
                y = soft_target(data["train"]["distance"][rows["fit"]], spec["soft_sigma"]) if "soft_sigma" in spec else stop
                weight = positive_weight(stop) if spec.get("pos_weight") else None
                nets = [train_head(x_fit, y, seed, args.device, weight) for seed in base.SEEDS]
                del x_fit

                def scored(split, idx, nets=nets, mean=mean, std=std, parts=spec["parts"]):
                    x = base.matrix(data, split, idx, parts, bow)
                    x -= mean; x /= std
                    return base.predict(nets, x, args.device)
            threshold = calibrate_on(data, rows, scored("val_seen", rows["calib"]))
            outcome, summary = evaluate(data, rows, scored, threshold, objects, vocab, seen)
            fold["variants"][name] = summary
            for route_id, o in outcome.items():
                results["routes"].setdefault(route_id, {})[name] = o
            print(held, name, {k: round(v, 3) for k, v in summary.items()}, flush=True)
        results["folds"][held] = fold
    route_ids = sorted(results["routes"])
    draws = np.random.default_rng(0).integers(0, len(route_ids), (base.RESAMPLES, len(route_ids)))
    ref = np.array([results["routes"][r][BASELINE]["within"] for r in route_ids], dtype=float)
    seen = np.array([results["routes"][r][BASELINE]["seen_class"] for r in route_ids])
    known = np.array([results["routes"][r][BASELINE]["known_phrase"] for r in route_ids])
    results["pooled"], results["gate"] = {}, {}
    for name in variants:
        v = np.array([results["routes"][r][name]["within"] for r in route_ids], dtype=float)
        d = v - ref
        ci = np.percentile(d[draws].mean(1), [2.5, 97.5])
        results["pooled"][name] = {"first_within_20m": float(v.mean()), "seen_class": float(v[seen].mean()), "unseen_class": float(v[~seen].mean()),
                                   "known_phrase": float(v[known].mean()), "unknown_phrase": float(v[~known].mean()),
                                   "minus_V1": float(d.mean()), "ci": [float(ci[0]), float(ci[1])],
                                   "minus_V1_unseen_class": float(d[~seen].mean())}
        if name != BASELINE:
            results["gate"][name] = bool(d.mean() >= base.GATE_GAIN and ci[0] > 0)
        print(name, {k: (round(x, 3) if isinstance(x, float) else [round(c, 3) for c in x]) for k, x in results["pooled"][name].items()}, flush=True)
    results.update(routes_total=len(route_ids), routes_seen_class=int(seen.sum()), routes_known_phrase=int(known.sum()))
    print(f"routes {len(route_ids)}, seen class {int(seen.sum())}; gate (V1 + 10 points, CI above 0):", results["gate"], flush=True)
    write_json(Path(args.out), results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("extract-rc", "vlm-manifest", "probe"))
    p.add_argument("--prepared", default="runs/v3_pilot/prepared"); p.add_argument("--root", default="/mnt/f/WSL_ColdStorage/datasets/sr3m_v3_pilot")
    p.add_argument("--cache", default=str(Path.home() / "sr3mxhad_runs/v3_pilot/features"))
    p.add_argument("--features", default=str(Path.home() / "sr3mxhad_runs/v3_pilot/stop_probe"))
    p.add_argument("--remoteclip", default=str(Path.home() / "sr3mxhad_runs/weights/RemoteCLIP-ViT-L-14.pt"))
    p.add_argument("--vlm-dir", default=str(Path.home() / "sr3mxhad_runs/v3_pilot/stop_probe/vlm"))
    p.add_argument("--objects", default="configs/v3_objects.json"); p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="runs/v3_pilot/model_check/stop_probe2.json")
    args = p.parse_args()
    if PROTOCOL["name"] != "sr3m_had_v3_pilot":
        raise SystemExit("Run with SR3MXHAD_PROTOCOL=v3_pilot")
    {"extract-rc": extract_rc, "vlm-manifest": vlm_manifest, "probe": probe}[args.stage](args)
