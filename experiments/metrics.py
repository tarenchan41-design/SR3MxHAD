"""Route-macro action metrics; every action mask is evaluator-owned ground truth."""
import math
import numpy as np
from experiments.protocol import PROTOCOL, TERRAIN

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
    dz = [mean([r.get("normalized_dz") for r in values if r["category"] in TERRAIN and r["terrain_following"] == follow])
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
            "first_stop_within_20m": triggered-int(first_outside), "never_triggered": len(ds.routes)-triggered,
            "first_stop_outside_20m": int(first_outside), "first_stop_before_terminal_suffix": int(first_before)}


def calibrate_stop(ds, probabilities):
    """Choose the val_seen threshold maximising routes whose FIRST trigger along the expert
    trajectory lies within the success radius (offline SR proxy); ties keep the higher threshold."""
    if ds.split != "val_seen":
        raise ValueError("STOP calibration is permitted only on val_seen")
    p = np.asarray(probabilities)
    if p.shape != (ds.n,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid STOP probabilities")
    # The first index with p >= t is always a running-maximum record, so each route reduces
    # to its strictly increasing record values and whether each record lies inside the radius.
    records = []
    for r in ds.routes:
        sl = slice(r["offset"], r["offset"] + r["count"]); q = p[sl]
        prefix = np.maximum.accumulate(q)
        idx = np.flatnonzero(np.r_[True, prefix[1:] > prefix[:-1]])
        records.append((q[idx], ds.arrays["distance_goal"][sl][idx] <= PROTOCOL["success_radius_m"]))
    best_threshold, best_success = None, 0
    for threshold in sorted({float(v) for values, _ in records for v in values}, reverse=True):
        success = 0
        for values, inside in records:
            k = np.searchsorted(values, threshold)
            success += bool(k < len(values) and inside[k])
        if success > best_success:
            best_threshold, best_success = threshold, success
    if best_threshold is None:
        return {"ready": False, "threshold": None, "calibration_split": "val_seen", "reason": "no_route_stops_within_success_radius"}
    return dict(stop_metrics(ds, p, best_threshold), ready=True, calibration_split="val_seen",
                criterion="max_routes_first_stop_within_success_radius")


def follow_slope_edges(train):
    """H2 slope tertiles, fitted on action-valid steps of train terrain-following routes only."""
    if train.split != "train":
        raise ValueError("H2 slope bins are fitted on train only")
    values = []
    for r in train.routes:
        if r["category"] in TERRAIN and r["terrain_following"]:
            sl = slice(r["offset"], r["offset"]+r["count"])
            values.append(train.arrays["slope"][sl][train.arrays["action_valid"][sl]])
    return np.quantile(np.concatenate(values), [1/3, 2/3]).tolist()


def evaluate(ds, prediction, statistics, probabilities=None, threshold=None, follow_slope_edges=None):
    prediction = np.asarray(prediction)
    if prediction.shape != (ds.n, 4):
        raise ValueError("Incomplete or malformed predictions")
    scale = np.asarray(statistics["scale"])
    by_route = {}
    for r in ds.routes:
        sl = slice(r["offset"], r["offset"]+r["count"])
        errors = route_errors(ds.arrays["action"][sl], prediction[sl], ds.arrays["action_valid"][sl], scale)
        by_route[r["episode_id"]] = dict(errors, **{k: r[k] for k in ("category", "height_layer", "terrain_following", "map_package")})
        if follow_slope_edges is not None and r["category"] in TERRAIN:
            # Per-route dz MAE inside each H2 slope bin; None when the route has no steps there.
            edges = [-math.inf, *follow_slope_edges, math.inf]; slope = ds.arrays["slope"][sl]
            by_route[r["episode_id"]]["slope_dz"] = {
                name: route_errors(ds.arrays["action"][sl], prediction[sl],
                                   ds.arrays["action_valid"][sl] & (slope > lo) & (slope <= hi), scale).get("dz_mae_m")
                for name, lo, hi in zip(("low", "medium", "high"), edges[:-1], edges[1:])}
    numeric = {k: {kk: vv for kk, vv in v.items() if kk not in ("category", "height_layer", "terrain_following", "map_package")}
               for k, v in by_route.items()}
    def cohorts(categories):
        groups = {}
        for follow in (False, True):
            cohort = "follow" if follow else "control"
            for layer in (None, "low", "medium", "high"):
                ids = [r["episode_id"] for r in ds.routes if r["category"] in categories and r["terrain_following"] == follow
                       and (layer is None or r["height_layer"] == layer)]
                groups[f"{cohort}/{layer or 'all'}"] = macro([numeric[i] for i in ids])
        return groups
    slope_groups = {}
    edges = [-math.inf, *statistics["slope_bins_deg"], math.inf]
    for name, lo, hi in zip(("low", "medium", "high"), edges[:-1], edges[1:]):
        results = []
        for r in ds.routes:
            if r["category"] not in TERRAIN: continue
            sl = slice(r["offset"], r["offset"]+r["count"])
            slope = ds.arrays["slope"][sl]
            mask = ds.arrays["action_valid"][sl] & (slope > lo) & (slope <= hi)
            results.append(route_errors(ds.arrays["action"][sl], prediction[sl], mask, scale))
        slope_groups[name] = macro(results)
    categories={category:macro([numeric[r["episode_id"]] for r in ds.routes if r["category"]==category])
                for category in sorted({r["category"] for r in ds.routes})}
    return {"split": ds.split, "prepared":getattr(ds,"protocol",None), "selection_score": selection_score(by_route), "overall": macro(list(numeric.values())),
            "by_category":categories,
            "terrain": cohorts(TERRAIN), "terrain_by_category": {c: cohorts((c,)) for c in TERRAIN},
            "terrain_slope": slope_groups, "slope_edges_deg": statistics["slope_bins_deg"],
            "by_route": by_route, "action_mask": "ground_truth_transition_valid_AND_NOT_stop_target",
            "follow_slope_edges_deg": follow_slope_edges,
            "stop": stop_metrics(ds, probabilities, threshold) if probabilities is not None and threshold is not None else None,
            "closed_loop": False}

