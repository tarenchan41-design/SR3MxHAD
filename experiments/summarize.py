"""Paired route analysis and the November 3 one-page decision report."""
import argparse
from pathlib import Path
import numpy as np
from experiments.protocol import PROTOCOL, read_json, write_json
from experiments.metrics import mean, selection_score


def averaged_routes(reports):
    ids = list(reports[0]["by_route"])
    if any(set(r["by_route"]) != set(ids) for r in reports):
        raise ValueError("Reports do not cover the same held-out routes")
    out = {}
    for episode in ids:
        values = [r["by_route"][episode] for r in reports]
        result = dict(values[0])
        for key in result:
            if key not in ("category", "height_layer", "terrain_following", "map_package"):
                result[key] = mean([r[key] for r in values])
        out[episode] = result
    return out


def draw_routes(routes, rng):
    groups = {}
    for episode, r in routes.items():
        groups.setdefault((r["category"], r["terrain_following"], r["height_layer"]), []).append(episode)
    return [e for group in groups.values() for e in rng.choice(group, len(group), replace=True)]


def gain(a, b, ids=None):
    ids = list(a) if ids is None else ids
    def cohort(follow, layer=None):
        selected = [e for e in ids if a[e]["category"] == "mountain" and a[e]["terrain_following"] == follow
                    and (layer is None or a[e]["height_layer"] == layer)]
        return mean([a[e]["dz_mae_m"]-b[e]["dz_mae_m"] for e in selected])
    follow, control = cohort(True), cohort(False)
    matched = [cohort(True, layer)-cohort(False, layer) for layer in ("low", "medium")
               if cohort(True, layer) is not None and cohort(False, layer) is not None]
    return {"follow_gain_m": follow, "control_gain_m": control,
            "raw_D_m": follow-control if follow is not None and control is not None else None,
            "agl_matched_D_m": mean(matched) if len(matched) == 2 else None}


def relative_gain(a,b):
    result={}
    for follow in (False,True):
        for layer in (None,"low","medium","high"):
            ids=[e for e,r in a.items() if r["category"]=="mountain" and r["terrain_following"]==follow
                 and (layer is None or r["height_layer"]==layer)]
            first=mean([a[e]["dz_mae_m"] for e in ids]);second=mean([b[e]["dz_mae_m"] for e in ids])
            result[f"{'follow' if follow else 'control'}/{layer or 'all'}"]=(first-second)/first if first is not None and first>0 else None
    return result


def metric(routes, ids, key):
    selected = {str(i): routes[e] for i, e in enumerate(ids)}
    return selection_score(selected) if key == "selection_score" else mean([r.get(key) for r in selected.values()])


def paired_interval(a, b, key, draws=2000):
    if set(a) != set(b): raise ValueError("Unpaired reports")
    rng = np.random.default_rng(20261021)
    points = []
    for _ in range(draws):
        ids = draw_routes(a, rng)
        points.append(metric(a, ids, key)-metric(b, ids, key))
    return np.quantile(points, [.025, .975]).tolist()


def stable_support(interaction):
    d=interaction["agl_matched_D_m"]
    return (all(v>0 for v in d["by_seed"]) and d["mean"]>d["seed_std"]
            and all(v>0 for v in interaction["follow_gain_m"]["by_seed"])
            and d["route_95ci"][0]>0 and interaction["follow_gain_m"]["route_95ci"][0]>0
            and interaction["raw_D_m"]["mean"]>0)


def run(args):
    root = Path(args.results)
    reports = {v: [read_json(root / f"{v}_seed{s}.json") for s in PROTOCOL["seeds"]] for v in PROTOCOL["variants"]}
    if any(r["split"] != "val_unseen" for rr in reports.values() for r in rr):
        raise ValueError("The decision report requires frozen val_unseen results")
    reference=reports["A"][0]
    if (reference.get("prepared") or {}).get("protocol")!=PROTOCOL or len(reference["by_route"])!=199:
        raise ValueError("Incomplete or incompatible held-out evaluation")
    for follow,expected in ((True,37),(False,11)):
        if sum(r["category"]=="mountain" and r["terrain_following"]==follow for r in reference["by_route"].values())!=expected:
            raise ValueError("Held-out cohort counts differ")
    if any(r.get("prepared")!=reference["prepared"] or r.get("selection",{}).get("split")!="val_seen"
           for rr in reports.values() for r in rr):
        raise ValueError("Missing frozen selection or mixed protocols")
    for variant,rr in reports.items():
        if any(r["variant"]!=variant or r["seed"]!=seed for r,seed in zip(rr,PROTOCOL["seeds"])):
            raise ValueError("Run filename and variant/seed differ")
    averaged = {v: averaged_routes(rr) for v, rr in reports.items()}
    baselines = {n: read_json(root / f"{n}.json") for n in ("fixed_forward", "train_mean")}
    if any(r.get("prepared")!=reference["prepared"] for r in baselines.values()): raise ValueError("Baseline protocol differs")
    gates = {}
    for variant, routes in averaged.items():
        checks = {}
        for name, baseline in baselines.items():
            checks[name] = {k: paired_interval(baseline["by_route"], routes, k) for k in ("selection_score", "turn_yaw_mae_deg")}
        per_seed={str(seed):all(r["selection_score"]<b["selection_score"] and r["overall"]["turn_yaw_mae_deg"]<b["overall"]["turn_yaw_mae_deg"]
                               for b in baselines.values()) for seed,r in zip(PROTOCOL["seeds"],reports[variant])}
        gates[variant] = {"passed": all(per_seed.values()) and all(ci[0] > 0 for check in checks.values() for ci in check.values()),
                          "each_seed_beats_both_baselines":per_seed,"improvement_95ci": checks}
    gains = [gain(a["by_route"], b["by_route"]) for a, b in zip(reports["A"], reports["B"])]
    rng = np.random.default_rng(20261021)
    bootstrap = [gain(averaged["A"], averaged["B"], draw_routes(averaged["A"], rng)) for _ in range(2000)]
    interaction = {k: {"mean": float(np.mean([g[k] for g in gains])), "seed_std": float(np.std([g[k] for g in gains], ddof=1)),
                       "route_95ci": np.quantile([g[k] for g in bootstrap], [.025, .975]).tolist(),
                       "by_seed": [g[k] for g in gains]} for k in gains[0]}
    probe = read_json(Path(args.probe) / "probe.json")
    readable = all(value["mae"] < value["train_mean_mae"] and value["r2"] is not None and value["r2"] > 0
                   for value in probe["splits"]["val_unseen"].values())
    supported = stable_support(interaction)
    if not gates["A"]["passed"] or not gates["B"]["passed"]:
        decision = "训练尚未通过基线检查，先处理动作学习，暂不解释视角差异。"
    elif supported:
        decision = "核心假设获得首轮支持；继续双视角主线，用重建 desert 检验未见地形泛化。"
    elif not readable:
        decision = "当前结果不足以判断视角价值；俯视冻结特征线性可读性较差，优先检查深度表示。"
    else:
        decision = "当前配置未支持稳定的地形分组增益；转向 benchmark 主线，保留 HAD 基线。"
    relative=[relative_gain(a["by_route"],b["by_route"]) for a,b in zip(reports["A"],reports["B"])]
    result = {"decision": decision, "baseline_gates": gates, "interaction": interaction,"relative_dz_gain_by_seed":relative,
              "criteria":"Each A/B seed beats both baselines; follow gain and matched D positive in all seeds; matched D mean > seed std; route CIs positive; raw D mean positive",
              "probe_linearly_readable": readable, "uncertainty": "route CI averages the three seeds first; seed std reported separately",
              "limitation": "one held-out mountain map; no high-AGL control; not a causal terrain experiment"}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); write_json(out / "summary.json", result)
    lines = ["# SR3M × HAD：11/3 离线决策", "", decision, "",
             "val_unseen：199 条路线，其中 mountain 随地形调整 37 条、对照 11 条；低/中/高 AGL 为 17/7、13/4、7/0。", "",
             "| 方法 | 平移 cm | 转弯 yaw ° | 随地形 dz cm | 对照 dz cm | 选模分数 |", "|---|---:|---:|---:|---:|---:|"]
    def fields(report):
        return [report["overall"]["translation_m"]*100, report["overall"]["turn_yaw_mae_deg"],
                report["mountain"]["follow/all"]["dz_mae_m"]*100, report["mountain"]["control/all"]["dz_mae_m"]*100,
                report["selection_score"]]
    for name, report in baselines.items(): lines.append("| "+name+" | "+" | ".join(f"{v:.3f}" for v in fields(report))+" |")
    for variant, rr in reports.items():
        values = np.asarray([fields(r) for r in rr]); avg, sd = values.mean(0), values.std(0, ddof=1)
        lines.append("| "+variant+" | "+" | ".join(f"{m:.3f}±{s:.3f}" for m, s in zip(avg, sd))+" |")
    lines.extend(["", "双视角增益定义为 A 的 dz MAE 减去 B；D 为随地形组增益减去对照组增益。", "",
                  "| 比较 | 均值±种子标准差 cm | 路线配对 95% CI cm |", "|---|---:|---:|"])
    for key, value in interaction.items():
        lines.append(f"| {key} | {100*value['mean']:.3f}±{100*value['seed_std']:.3f} | [{100*value['route_95ci'][0]:.3f}, {100*value['route_95ci'][1]:.3f}] |")
    lines.extend(["", "| AGL | 随地形 A→B dz cm | 对照 A→B dz cm | 路线 随地形/对照 |", "|---|---:|---:|---:|"])
    for layer in ("low","medium","high"):
        values=[];counts=[]
        for cohort in ("follow","control"):
            key=f"{cohort}/{layer}";counts.append(reference["mountain"][key]["routes"])
            av=mean([r["mountain"][key]["dz_mae_m"] for r in reports["A"]]);bv=mean([r["mountain"][key]["dz_mae_m"] for r in reports["B"]])
            values.append(f"{100*av:.3f}→{100*bv:.3f}" if av is not None else "NA")
        lines.append(f"| {layer} | {values[0]} | {values[1]} | {counts[0]}/{counts[1]} |")
    lines.append("")
    for cohort in ("follow","control"):
        value=mean([r[f"{cohort}/all"] for r in relative])
        lines.append(f"{cohort} 相对 dz 改善：{100*value:.2f}%。" if value is not None else f"{cohort} 相对改善：NA。")
    lines.extend(["", "AGL 匹配结果只对低/中层等权；高 AGL 无对照，不能外推。完整各层、坡度和逐路线数据见结果 JSON。", "",
                  "俯视线性探针（val_unseen，训练均值基线）："])
    for name, value in probe["splits"]["val_unseen"].items():
        r2=f"{value['r2']:.3f}" if value['r2'] is not None else "NA"
        lines.append(f"- {name}: MAE {value['mae']:.3f}，基线 {value['train_mean_mae']:.3f}，R² {r2}。")
    lines.extend(["", "| STOP | 通过校准种子数 | val_unseen precision / recall / F1 | 首次误停>20m（种子平均次数） |", "|---|---:|---:|---:|"])
    for variant,rr in reports.items():
        ready=[r for r in rr if r.get("stop") is not None]
        scores=[mean([r["stop"][k] for r in ready]) for k in ("precision","recall","f1")]
        text=" / ".join(f"{v:.3f}" if v is not None else "NA" for v in scores)
        early=mean([r["stop"]["first_stop_outside_20m"] for r in ready])
        lines.append(f"| {variant} | {len(ready)}/3 | {text} | {early if early is not None else 'NA'} |")
    lines.extend(["", "STOP 阈值全部来自 val_seen。未通过精确率/首次误停校准的模型不宣称具备闭环停止能力。",
                  "下一阶段：Cosys-AirSim 专家回放通过后，运行预先固定的 16 条路线；新 desert 不参与调参。", ""])
    (out / "decision.md").write_text("\n".join(lines), encoding="utf8")
    print(decision, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True); p.add_argument("--probe", required=True); p.add_argument("--out", required=True)
    run(p.parse_args())
