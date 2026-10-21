"""Route-macro action metrics; every action mask is evaluator-owned ground truth."""
import math
import numpy as np
from experiment import PROTOCOL

NUMERIC = ("translation_m", "dx_mae_m", "dy_mae_m", "dz_mae_m", "yaw_mae_deg", "turn_yaw_mae_deg",
           "turn_sign_accuracy", "predicted_yaw_lt_1deg_fraction", "turn_predicted_yaw_lt_1deg_fraction",
           "target_abs_dz_m", "normalized_translation", "normalized_turn_yaw", "normalized_dz")


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def route_errors(target, prediction, valid, scale):
    if not np.any(valid):
        return {"actions": 0}
    truth, pred = target[valid], prediction[valid].copy()
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Nonfinite action in evaluated ground-truth mask")
    pred[:,3]=np.arctan2(np.sin(pred[:,3]),np.cos(pred[:,3]))
    error = pred - truth
    yaw = np.abs(np.arctan2(np.sin(error[:, 3]), np.cos(error[:, 3])))
    turns = np.abs(truth[:, 3]) >= PROTOCOL["turn_threshold_rad"]
    return {"actions": len(truth), "turns": int(turns.sum()),
            "translation_m": float(np.linalg.norm(error[:, :3], axis=1).mean()),
            "dx_mae_m": float(np.abs(error[:, 0]).mean()), "dy_mae_m": float(np.abs(error[:, 1]).mean()),
            "dz_mae_m": float(np.abs(error[:, 2]).mean()),
            "yaw_mae_deg": float(np.degrees(yaw).mean()),
            "turn_yaw_mae_deg": float(np.degrees(yaw[turns]).mean()) if turns.any() else None,
            "turn_sign_accuracy": float((np.where(np.abs(pred[turns,3])<PROTOCOL["turn_threshold_rad"],0,np.sign(pred[turns,3])) == np.sign(truth[turns, 3])).mean()) if turns.any() else None,
            "predicted_yaw_lt_1deg_fraction": float((np.abs(pred[:, 3]) < PROTOCOL["turn_threshold_rad"]).mean()),
            "turn_predicted_yaw_lt_1deg_fraction": float((np.abs(pred[turns, 3]) < PROTOCOL["turn_threshold_rad"]).mean()) if turns.any() else None,
            "target_abs_dz_m": float(np.abs(truth[:, 2]).mean()),
            "normalized_translation": float(np.sqrt(np.mean((error[:, :3]/scale[:3])**2, axis=1)).mean()),
            "normalized_turn_yaw": float((yaw[turns]/scale[3]).mean()) if turns.any() else None,
            "normalized_dz": float(np.abs(error[:, 2]/scale[2]).mean())}


def macro(routes):
    eligible = [r for r in routes if r.get("actions", 0)]
    out = {"routes": len(routes), "eligible_routes": len(eligible),
           "actions": sum(r.get("actions", 0) for r in eligible), "turns": sum(r.get("turns", 0) for r in eligible)}
    for key in NUMERIC:
        out[key] = mean([r.get(key) for r in eligible])
    return out


def selection_score(by_route):
    values = list(by_route.values())
    xyz = mean([r.get("normalized_translation") for r in values])
    yaw = mean([r.get("normalized_turn_yaw") for r in values])
    dz = [mean([r.get("normalized_dz") for r in values if r["category"] == "mountain" and r["terrain_following"] == follow])
          for follow in (False, True)]
    if xyz is None or yaw is None or None in dz:
        return None
    return (xyz + yaw + sum(dz)/2)/3


def stop_metrics(ds, probabilities, threshold):
    p = np.asarray(probabilities)
    if p.shape != (ds.n,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid STOP probabilities")
    target = ds.arrays["stop"]; predicted = p >= threshold
    tp, fp, fn = (int((predicted & target).sum()), int((predicted & ~target).sum()), int((~predicted & target).sum()))
    first_outside = first_before = triggered = 0
    for r in ds.routes:
        sl = slice(r["offset"], r["offset"] + r["count"])
        active = np.flatnonzero(predicted[sl])
        if len(active):
            triggered += 1; i = r["offset"] + int(active[0])
            first_outside += ds.arrays["distance_goal"][i] > PROTOCOL["success_radius_m"]
            first_before += not target[i]
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "triggered_routes": triggered, "routes": len(ds.routes),
            "first_stop_outside_20m": int(first_outside), "first_stop_before_terminal_suffix": int(first_before)}


def calibrate_stop(ds, probabilities):
    if ds.split != "val_seen":
        raise ValueError("STOP calibration is permitted only on val_seen")
    p = np.asarray(probabilities)
    if p.shape != (ds.n,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid STOP probabilities")
    order = np.argsort(-p, kind="stable")
    target = ds.arrays["stop"]; positives = int(target.sum())
    first = np.full(len(ds.routes), ds.n, dtype=int)
    outside = np.zeros(len(ds.routes), dtype=bool)
    tp = fp = bad = 0; best = None; j = 0
    while j < ds.n:
        threshold = p[order[j]]
        end = j + 1
        while end < ds.n and p[order[end]] == threshold:
            end += 1
        for i in order[j:end]:
            tp += int(target[i]); fp += int(not target[i])
            rid = ds.route_index[i]
            if i < first[rid]:
                bad -= int(outside[rid]); first[rid] = i
                outside[rid] = ds.arrays["distance_goal"][i] > PROTOCOL["success_radius_m"]
                bad += int(outside[rid])
        precision, recall = tp/(tp+fp), tp/max(positives, 1)
        if bad == 0 and precision >= PROTOCOL["stop_min_precision"] and tp > 0:
            if best is None or recall > best["recall"]:
                best = {"threshold": float(threshold), "recall": recall}
        j = end
    if best is None:
        return {"ready": False, "threshold": None, "calibration_split": "val_seen", "reason": "no_nonempty_precision_95_zero_outside20_threshold"}
    return dict(stop_metrics(ds, p, best["threshold"]), ready=True, calibration_split="val_seen")


def evaluate(ds, prediction, statistics, probabilities=None, threshold=None):
    prediction = np.asarray(prediction)
    if prediction.shape != (ds.n, 4):
        raise ValueError("Incomplete or malformed predictions")
    scale = np.asarray(statistics["scale"])
    by_route = {}
    for r in ds.routes:
        sl = slice(r["offset"], r["offset"]+r["count"])
        errors = route_errors(ds.arrays["action"][sl], prediction[sl], ds.arrays["action_valid"][sl], scale)
        by_route[r["episode_id"]] = dict(errors, **{k: r[k] for k in ("category", "height_layer", "terrain_following", "map_package")})
    numeric = {k: {kk: vv for kk, vv in v.items() if kk not in ("category", "height_layer", "terrain_following", "map_package")}
               for k, v in by_route.items()}
    groups = {}
    for follow in (False, True):
        cohort = "follow" if follow else "control"
        for layer in (None, "low", "medium", "high"):
            ids = [r["episode_id"] for r in ds.routes if r["category"] == "mountain" and r["terrain_following"] == follow
                   and (layer is None or r["height_layer"] == layer)]
            groups[f"{cohort}/{layer or 'all'}"] = macro([numeric[i] for i in ids])
    slope_groups = {}
    edges = [-math.inf, *statistics["slope_bins_deg"], math.inf]
    for name, lo, hi in zip(("low", "medium", "high"), edges[:-1], edges[1:]):
        results = []
        for r in ds.routes:
            if r["category"] != "mountain": continue
            sl = slice(r["offset"], r["offset"]+r["count"])
            slope = ds.arrays["slope"][sl]
            mask = ds.arrays["action_valid"][sl] & (slope > lo) & (slope <= hi)
            results.append(route_errors(ds.arrays["action"][sl], prediction[sl], mask, scale))
        slope_groups[name] = macro(results)
    categories={category:macro([numeric[r["episode_id"]] for r in ds.routes if r["category"]==category])
                for category in sorted({r["category"] for r in ds.routes})}
    return {"split": ds.split, "prepared":getattr(ds,"protocol",None), "selection_score": selection_score(by_route), "overall": macro(list(numeric.values())),
            "by_category":categories,
            "mountain": groups, "mountain_slope": slope_groups, "slope_edges_deg": statistics["slope_bins_deg"],
            "by_route": by_route, "action_mask": "ground_truth_transition_valid_AND_NOT_stop_target",
            "stop": stop_metrics(ds, probabilities, threshold) if probabilities is not None and threshold is not None else None,
            "closed_loop": False}

