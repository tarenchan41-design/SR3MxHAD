"""Train A/B/C x three seeds using cached frozen-CNN features and val_seen only."""
import argparse
import copy
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from experiments.protocol import DIAGNOSTIC_VARIANTS, MAIN_VARIANTS, PROTOCOL, TERRAIN, PreparedSplit, code_version, read_features, read_json, write_json
from had.model import GRIDS, VARIANTS, TerrainHAD, feature_dim
from had.tokenizer import Tokenizer
from experiments.metrics import calibrate_stop, evaluate


def token_rows(ds, tokenizer):
    return torch.tensor([tokenizer.encode(r["instruction"])["input_ids"] for r in ds.routes], dtype=torch.long)


# R1 (docs/protocol.md §12): the train-route altitude condition, FOLLOW phrase -> HOLD phrase.
HOLD_PHRASES = {"adjusting the flight height to follow the terrain along the way":
                    "holding one constant flight altitude without climbing or descending with the terrain",
                "adjusting the height according to the terrain along the way":
                    "holding one constant flight altitude without climbing or descending with the terrain",
                "adjust the height according to the terrain along the way":
                    "hold one constant flight altitude without climbing or descending with the terrain"}


def hold_instruction(text):
    found = [p for p in HOLD_PHRASES if p in text]
    if len(found) != 1: raise ValueError(f"Expected one terrain-following phrase, found {found}")
    return text.replace(found[0], HOLD_PHRASES[found[0]])


def with_hold_copies(ds, tokens, tokenizer):
    """Append a copy of every terrain-following route with the HOLD condition; its dz target is 0, the
    exact hold action from any state. Images (via feature_index), horizontal actions and STOP are unchanged."""
    picked = [r for r in ds.routes if r["category"] in TERRAIN and r["terrain_following"]]
    rows = np.concatenate([np.arange(r["offset"], r["offset"]+r["count"]) for r in picked])
    out = copy.copy(ds)
    out.routes = ds.routes + [dict(r, instruction=hold_instruction(r["instruction"]), augmented="hold") for r in picked]
    out.arrays = {k: np.concatenate([v, v[rows]]) for k, v in ds.arrays.items()}
    action = out.arrays["action"]; valid = out.arrays["action_valid"]
    action[ds.n:, 2] = np.where(valid[ds.n:], 0., action[ds.n:, 2])
    out.feature_index = np.r_[np.arange(ds.n), rows]
    out.route_index = np.r_[ds.route_index, len(ds.routes)+np.repeat(np.arange(len(picked)), [r["count"] for r in picked])]
    out.n = len(out.feature_index)
    extra = torch.tensor([tokenizer.encode(r["instruction"])["input_ids"] for r in out.routes[len(ds.routes):]], dtype=torch.long)
    return out, torch.cat([tokens, extra])


SUBSET_SEED = 20261107


def route_subset(ds, fraction, seed=SUBSET_SEED):
    """Data curve (protocol 17.6): keep round(fraction * n), at least one, of each target's train routes, drawn with a fixed
    seed, so every target stays in training. Features keep their rows through feature_index; statistics, tokenizer and
    the val_seen selection stay those of the full preparation."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, r in enumerate(ds.routes):
        groups.setdefault(r["route_group_id"], []).append(i)
    keep = sorted(i for g in sorted(groups) for i in rng.permutation(groups[g])[:max(1, round(fraction * len(groups[g])))].tolist())
    rows = np.concatenate([np.arange(ds.routes[i]["offset"], ds.routes[i]["offset"] + ds.routes[i]["count"]) for i in keep])
    out = copy.copy(ds)
    out.routes = [ds.routes[i] for i in keep]
    out.arrays = {k: v[rows] for k, v in ds.arrays.items()}
    out.feature_index = rows
    out.route_index = np.repeat(np.arange(len(keep)), [ds.routes[i]["count"] for i in keep])
    out.n = len(rows)
    return out, [ds.routes[i]["route_id"] for i in keep]


def batch_inputs(ds, features, tokens, indices, views, device):
    rows = getattr(ds, "feature_index", None)
    rows = indices if rows is None else rows[indices]
    values = {v: torch.from_numpy(np.asarray(features[v][rows], dtype=np.float32)).to(device) for v in views}
    return values, tokens[ds.route_index[indices]].to(device), torch.from_numpy(ds.arrays["state"][indices]).to(device)


def loss_for_batch(output, action, valid, stop, statistics):
    scale = action.new_tensor(statistics["scale"])
    if valid.any():
        truth = action[valid]
        error = output["action"][valid] - truth
        yaw = torch.atan2(torch.sin(error[:, 3]), torch.cos(error[:, 3])) / scale[3]
        turns = truth[:, 3].abs() >= PROTOCOL["turn_threshold_rad"]
        fraction = statistics["turn_fraction"]
        weights = torch.where(turns, .5/fraction, .5/(1-fraction))
        regression = (((error[:, :3]/scale[:3])**2).mean(0).sum() + (weights*yaw.square()).mean())/4
    else:
        # No subtraction against NaN placeholders; action-head gradient is exactly zero.
        regression = output["action"].sum()*0
    stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(output["stop_logit"], stop.float())
    return regression + PROTOCOL["stop_loss_weight"]*stop_loss


@torch.inference_mode()
def predict(model, ds, features, tokens, device, batch_size=128):
    model.eval()
    actions, probabilities = [], []
    for start in range(0, ds.n, batch_size):
        ids = np.arange(start, min(start+batch_size, ds.n))
        out = model(*batch_inputs(ds, features, tokens, ids, model.views, device))
        actions.append(out["action"].cpu().numpy())
        probabilities.append(out["stop_logit"].sigmoid().cpu().numpy())
    actions, probabilities = np.concatenate(actions), np.concatenate(probabilities)
    if not np.isfinite(actions).all() or not np.isfinite(probabilities).all():
        raise FloatingPointError("Nonfinite model prediction")
    actions[:,3]=np.arctan2(np.sin(actions[:,3]),np.cos(actions[:,3]))
    return actions, probabilities


def save_torch(path, value):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp); tmp.replace(path)


def train_run(args, variant, seed, data, features, tokenizer, tokens, statistics):
    out = Path(args.out) / f"{variant}_seed{seed}"; out.mkdir(parents=True, exist_ok=True)
    identity = {"variant": variant, "seed": seed, "augment": getattr(args, "augment", None), "prepared": data["train"].protocol,
                "protocol": PROTOCOL, "statistics": statistics, "tokenizer": tokenizer.to_dict(), "code": args.code_version,
                "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None}}
    if getattr(args, "train_subset", None):
        identity["train_subset"] = args.train_subset
    if (out / "completion.json").exists():
        old = read_json(out / "run.json")
        if any(old.get(k) != identity.get(k) for k in ("variant", "seed", "augment", "prepared", "protocol", "statistics", "tokenizer", "code", "train_subset")):
            raise ValueError("Existing run uses a different experiment")
        print(f"already completed {out.name}", flush=True); return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = TerrainHAD(variant, tokenizer, cached_only=True).to(args.device)
    opt_config = PROTOCOL["optimizer"]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=opt_config["lr"], weight_decay=opt_config["weight_decay"])
    epoch = 0; best = float("inf"); stale = 0; best_epoch = 0
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        old = state["identity"]
        if any(old.get(k) != identity.get(k) for k in ("variant", "seed", "augment", "prepared", "protocol", "statistics", "tokenizer", "code", "train_subset")):
            raise ValueError("Resume identity mismatch")
        model.load_head_state(state["head"]); optimizer.load_state_dict(state["optimizer"])
        for item in optimizer.state.values():
            for k, value in item.items():
                if isinstance(value, torch.Tensor): item[k] = value.to(args.device)
        epoch, best, stale, best_epoch = state["epoch"], state["best_score"], state["stale"], state["best_epoch"]
        torch.set_rng_state(state["torch_rng"])
        if args.device.startswith("cuda"): torch.cuda.set_rng_state_all(state["cuda_rng"])
    elif (out / "run.json").exists():
        old = read_json(out / "run.json")
        if any(old.get(k) != identity.get(k) for k in ("variant", "seed", "augment", "prepared", "protocol", "statistics", "tokenizer", "code", "train_subset")):
            raise ValueError("Interrupted first epoch belongs to another run")
        if (out / "training.jsonl").exists() and (out / "training.jsonl").stat().st_size:
            raise ValueError("Completed epoch log exists but resume checkpoint is missing")
        print(f"restarting interrupted first epoch: {out.name}", flush=True)
    else:
        write_json(out / "run.json", identity)
    ds = data["train"]; started = time.monotonic()
    already_stopped = epoch >= opt_config["min_epochs"] and stale >= opt_config["patience"]
    final_epoch = epoch if already_stopped else opt_config["max_epochs"]
    for epoch in range(epoch+1, final_epoch+1):
        model.train(); tick = time.monotonic(); total = 0.
        order = np.random.default_rng(seed + epoch).permutation(ds.n)
        for start in range(0, ds.n, opt_config["batch_size"]):
            ids = order[start:start+opt_config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            output = model(*batch_inputs(ds, features["train"], tokens["train"], ids, model.views, args.device))
            loss = loss_for_batch(output, torch.from_numpy(ds.arrays["action"][ids]).to(args.device),
                                  torch.from_numpy(ds.arrays["action_valid"][ids]).to(args.device),
                                  torch.from_numpy(ds.arrays["stop"][ids]).to(args.device), statistics)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), opt_config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach())*len(ids)
        pred, probability = predict(model, data["val_seen"], features["val_seen"], tokens["val_seen"], args.device)
        result = evaluate(data["val_seen"], pred, statistics)
        score = result["selection_score"]
        if score is None or not np.isfinite(score): raise ValueError("Incomplete validation selection score")
        improved = score < best
        if improved: best, best_epoch, stale = score, epoch, 0
        else: stale += 1
        checkpoint = {"head": model.head_state(), "identity": identity, "epoch": epoch, "best_score": best,
                      "best_epoch": best_epoch, "stale": stale}
        if improved:
            save_torch(out / "best.pt", checkpoint)
        checkpoint.update(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [])
        save_torch(out / "last.pt", checkpoint)
        row = {"epoch": epoch, "loss": total/ds.n, "selection_score": score, "best_epoch": best_epoch,
               "seconds": time.monotonic()-tick, "improved": improved,
               "turn_yaw_mae_deg":result["overall"]["turn_yaw_mae_deg"],
               "terrain_follow_dz_m":result["terrain"]["follow/all"]["dz_mae_m"],
               "terrain_control_dz_m":result["terrain"]["control/all"]["dz_mae_m"],
               "baseline_scores":getattr(args,"baseline_scores",{})}
        with (out / "training.jsonl").open("a", encoding="utf8") as f: f.write(json.dumps(row)+"\n")
        print(f"{variant} seed{seed}: {json.dumps(row)}", flush=True)
        if epoch >= opt_config["min_epochs"] and stale >= opt_config["patience"]: break
    selected = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.load_head_state(selected["head"])
    pred, probability = predict(model, data["val_seen"], features["val_seen"], tokens["val_seen"], args.device)
    stop = calibrate_stop(data["val_seen"], probability)
    result = evaluate(data["val_seen"], pred, statistics, probability, stop["threshold"])
    result.update(variant=variant, seed=seed, epoch=selected["epoch"], stop_calibration=stop)
    write_json(out / "val_seen.json", result)
    np.savez(out / "val_seen_predictions.npz", action=pred, stop_probability=probability)
    write_json(out / "selection.json", {"split": "val_seen", "epoch": selected["epoch"], "score": best,
        "criterion": PROTOCOL["selection"], "stop": stop, "val_unseen_read": False, "test_read": False})
    write_json(out / "completion.json", {"epochs": epoch, "best_epoch": selected["epoch"],
        "seconds_this_invocation": time.monotonic()-started, "status": "selected_on_val_seen"})
    del model, optimizer, checkpoint, selected; gc.collect()
    if args.device.startswith("cuda"): torch.cuda.empty_cache()


def train(args):
    torch.set_num_threads(4)
    args.code_version = code_version()
    data = {s: PreparedSplit(args.prepared, s) for s in ("train", "val_seen")}
    widths = {}  # the views the variants read and their widths; variants trained together must agree on each view
    for v in args.variants:
        for view in VARIANTS[v]:
            if widths.setdefault(view, feature_dim(GRIDS.get(v, {}).get(view, 2))) != feature_dim(GRIDS.get(v, {}).get(view, 2)):
                raise ValueError("Variants trained together must read features of the same widths")
    widths = widths or None  # N reads no view; the default cache check still runs
    features = {s: read_features(args.cache, s, ds, widths)[0] for s, ds in data.items()}
    tokenizer = Tokenizer.load(Path(args.prepared) / "tokenizer.json")
    tokens = {s: token_rows(ds, tokenizer) for s, ds in data.items()}
    statistics = read_json(Path(args.prepared) / "statistics.json")
    if getattr(args, "train_fraction", 1.) < 1:
        data["train"], kept = route_subset(data["train"], args.train_fraction)
        tokens["train"] = token_rows(data["train"], tokenizer)
        args.train_subset = {"fraction": args.train_fraction, "seed": SUBSET_SEED, "routes": kept}
        print(f"train subset: {len(kept)} routes, {data['train'].n} observations", flush=True)
    if args.augment == "hold":
        data["train"], tokens["train"] = with_hold_copies(data["train"], tokens["train"], tokenizer)
    baselines={name:evaluate(data["val_seen"],np.broadcast_to(action,(data["val_seen"].n,4)),statistics)
               for name,action in (("fixed_forward",[2.,0.,0.,0.]),("train_mean",statistics["mean"]))}
    write_json(Path(args.out)/"val_seen_baselines.json",baselines)
    args.baseline_scores={name:r["selection_score"] for name,r in baselines.items()}
    for seed in args.seeds:
        for variant in args.variants:
            train_run(args, variant, seed, data, features, tokenizer, tokens, statistics)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True); p.add_argument("--cache", required=True); p.add_argument("--out", required=True)
    p.add_argument("--variants", nargs="+", choices=("A", "B", "C", *DIAGNOSTIC_VARIANTS), default=list(MAIN_VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44]); p.add_argument("--device", default="cuda")
    p.add_argument("--augment", choices=("hold",), default=None, help="R1 counterfactual HOLD copies (docs/protocol.md §12)")
    p.add_argument("--train-fraction", type=float, default=1., help="data curve (protocol 17.6): share of each target's train routes")
    train(p.parse_args())

