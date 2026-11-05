"""Stage 0 of dataset v3 stage 2 (protocol sections 13 and 17.4): geometry only, no rendering, no UE.

plan:   the route plan, every candidate in a fixed order (design kick_start2_relief_s2, rules 1-9). Written once.
leak:   G0-1 to G0-4 as in section 13 (v3_leak.test_design): fit on the four training maps, measure on the two development maps.
        The test maps are not used here.
gates:  G0-6 (terrain discrimination on the development and test maps) and G0-7 (class coverage, section 17.2) from the plan and
        the object table, plus the leak results; writes gates.json.
review: the G0-5 page: the object table and five routes per map from the plan.
"""
import argparse
import dataclasses
import html
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from experiments.protocol import code_version, read_json, write_json
from experiments.v3_leak import DESIGNS, GATES, Ground, route_stats, test_design
from sr3m.routes_v3 import generate

TRAIN = ("L_MTR_A", "L_MTR_D", "L_DES_A_v3", "L_DES_D_v3")
DEV = ("L_MTR_B", "L_DES_B_v3")
TEST = ("L_MTR_C", "L_DES_C_v3")
QUOTA = {**{m: 200 for m in TRAIN}, **{m: 40 for m in DEV}, **{m: 60 for m in TEST}}
DESIGN = "kick_start2_relief_s2"
SEED = 20261108
LEAK_SEEDS = (20261105, 20261106, 20261107)
CANDIDATES = 1.6  # as stage 1: candidates per quota slot, so UE rejections cannot run a map dry
DEV_MIN_M = 5.
SHARED_CLASSES = 12
SHARED_SHARE = .75
PREVIEWS_PER_MAP = 5


def plain(route):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in route.items()}


def arrays(route):
    return {**route, **{k: np.asarray(route[k]) for k in ("positions", "yaws", "actions")}}


def plan(args):
    path = Path(args.out) / "plan.json"
    if path.exists():
        raise ValueError(f"{path} exists; the plan is fixed once written")
    objects = json.load(open(args.objects, encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    design = DESIGNS[DESIGN]
    out = {"seed": args.seed, "design": DESIGN, "design_fields": dataclasses.asdict(design), "objects_file": args.objects, "code": code_version(), "maps": {}}
    for map_id, quota in QUOTA.items():
        routes, rejected = generate([o for o in objects if o["map"] == map_id], Ground(map_id, args.cache), int(quota * CANDIDATES), design, rng)
        for i, r in enumerate(routes):
            r["id"] = f"{map_id}_{i:03d}"
        out["maps"][map_id] = {"quota": quota, "rejected_by_design": rejected, "candidates": [plain(r) for r in routes]}
        print(map_id, len(routes), "candidates", dict(sorted(rejected.items(), key=lambda x: -x[1])[:6]), flush=True)
    write_json(path, out)


def leak(args):
    objects = json.load(open(args.objects, encoding="utf-8"))
    grounds = {m: Ground(m, args.cache) for m in TRAIN + DEV}
    for seed in args.seeds:
        report = test_design(DESIGN, DESIGNS[DESIGN], objects, grounds, SimpleNamespace(seed=seed, fit_routes=200, eval_routes=100, epochs=30), args.device, TRAIN, DEV)
        report["code"] = code_version()
        write_json(Path(args.out) / f"leak/leak_{seed}.json", report)
        m = report["models"]
        print(seed, {k: ("pass" if v else "FAIL") for k, v in report["gates"].items()},
              f"resp10 {m['pose']['response10']['median_deg']:.3f} resp20 {m['pose']['response20']['median_deg']:.3f} reach20 {m['pose']['closed']['reach20']:.2f}"
              f" class-stop SR20 {m['pose']['closed']['class_stop20']:.2f} sign {m['pose']['offline_clean']['sign']:.2f} corr {m['pose']['offline_clean']['corr']:.2f}"
              f" perfect {m['pose+word+perfect']['closed']['reach20']:.2f}/{m['pose+word+perfect']['closed_kick20']['reach20']:.2f}"
              f" coarse {m['pose+word+coarse']['closed']['reach20']:.2f}/{m['pose+word+coarse']['closed_kick20']['reach20']:.2f}", flush=True)


def discrimination(routes):
    follow = [r for r in routes if r["mode"] == "follow"]
    med = [r["hold_dev_median_m"] for r in follow]
    return {"routes": len(routes), "follow": len(follow), "median_of_step_median_m": float(np.median(med)) if med else None,
            "median_of_E_hold_m": float(np.median([r["hold_dev_mean_m"] for r in follow])) if follow else None,
            "share_ge_5m": float(np.mean(np.array(med) >= DEV_MIN_M)) if med else None,
            "follow_climb_median_m": float(np.median([r["climb_m"] for r in follow])) if follow else None}


def coverage(objects, planned):
    targets = [o for o in objects if o["role"] == "target"]
    by = lambda split: [o for o in targets if o["split"] == split]
    train_classes = sorted({o["cls"] for o in by("train")})
    scene = lambda o: "mountain" if o["map"].startswith("L_MTR") else "desert"
    shared = {}
    for c in train_classes:
        inst = [o for o in by("train") if o["cls"] == c]
        shared[c] = {"train_maps": sorted({o["map"] for o in inst}), "scenes": sorted({scene(o) for o in inst}), "variants": sorted({f"{scene(o)}:{o['variant']}" for o in inst}),
                     "instances_train": len(inst), "instances_all": sum(o["cls"] == c for o in targets)}
    seen_dev = set(train_classes) | {o["cls"] for o in by("dev")}
    maps = {}
    for map_id in TRAIN + DEV + TEST:
        mine = [o for o in targets if o["map"] == map_id]
        split = mine[0]["split"]
        unseen = sorted({o["cls"] for o in mine if o["cls"] not in (train_classes if split == "dev" else seen_dev)}) if split != "train" else []
        routes = planned["maps"][map_id]["candidates"][:QUOTA[map_id]] if planned else []
        maps[map_id] = {"split": split, "targets": len(mine), "classes": len({o["cls"] for o in mine}), "shared_share": float(np.mean([o["cls"] in shared for o in mine])),
                        "unseen_classes": unseen, "routes_first_quota": len(routes), "targets_without_route": sorted({o["key"] for o in mine} - {r["object"] for r in routes}),
                        "unseen_routes": sum(r["unseen_class"] for r in routes)}
    checks = {
        "12 shared classes": len(shared) >= SHARED_CLASSES,
        "each shared class on >= 2 training maps": all(len(v["train_maps"]) >= 2 for v in shared.values()),
        "each shared class in both scenes": all(len(v["scenes"]) == 2 for v in shared.values()),
        "each shared class has >= 2 looks on the training maps": all(len(v["variants"]) >= 2 for v in shared.values()),
        ">= 75% shared targets on dev and test maps": all(maps[m]["shared_share"] >= SHARED_SHARE for m in DEV + TEST),
        ">= 2 unseen classes on each test map": all(len(maps[m]["unseen_classes"]) >= 2 for m in TEST),
        "1-2 unseen classes on each dev map": all(1 <= len(maps[m]["unseen_classes"]) <= 2 for m in DEV),
        ">= 15 targets on each map": all(v["targets"] >= 15 for v in maps.values()),
    }
    return {"shared_classes": shared, "maps": maps, "checks": checks, "pass": all(checks.values())}


def gates(args):
    out = Path(args.out)
    planned = read_json(out / "plan.json")
    objects = json.load(open(args.objects, encoding="utf-8"))
    report = {"code": code_version(), "plan_seed": planned["seed"], "design": planned["design"], "maps": {}, "G0-6": {}, "leak": {}}
    for map_id, entry in planned["maps"].items():
        first = entry["candidates"][:entry["quota"]]
        stats = route_stats([arrays(r) for r in entry["candidates"]], entry["rejected_by_design"])
        stats["first_quota"] = discrimination(first)
        stats["all_candidates"] = discrimination(entry["candidates"])
        stats["unseen_routes_first_quota"] = sum(r["unseen_class"] for r in first)
        report["maps"][map_id] = stats
    per_map = {m: report["maps"][m]["first_quota"]["median_of_step_median_m"] for m in DEV + TEST}
    pooled = discrimination([r for m in DEV + TEST for r in planned["maps"][m]["candidates"][:QUOTA[m]]])
    report["G0-6"] = {"per_map_median_m": per_map, "pooled": pooled, "pass": all(v is not None and v >= DEV_MIN_M for v in per_map.values())}
    report["G0-7"] = coverage(objects, planned)
    for path in sorted((out / "leak").glob("leak_*.json")):
        r = read_json(path)
        m = r["models"]
        report["leak"][path.stem[5:]] = {"gates": r["gates"], "pose_response10": m["pose"]["response10"]["median_deg"], "pose_response20": m["pose"]["response20"]["median_deg"],
                                         "pose_reach20": m["pose"]["closed"]["reach20"], "pose_class_stop20": m["pose"]["closed"]["class_stop20"], "pose_class_stop10": m["pose"]["closed"]["class_stop10"],
                                         "pose_sign": m["pose"]["offline_clean"]["sign"], "pose_corr": m["pose"]["offline_clean"]["corr"],
                                         "perfect": [m["pose+word+perfect"]["closed"]["reach20"], m["pose+word+perfect"]["closed_kick20"]["reach20"]],
                                         "coarse": [m["pose+word+coarse"]["closed"]["reach20"], m["pose+word+coarse"]["closed_kick20"]["reach20"]],
                                         "n_class_stop20": m["pose+word"]["closed"]["class_stop20"], "n_class_stop10": m["pose+word"]["closed"]["class_stop10"], "maps": r["maps"]}
    report["thresholds"] = {**GATES, "G0-6_m": DEV_MIN_M}
    write_json(out / "gates.json", report)
    print("G0-6", "pass" if report["G0-6"]["pass"] else "FAIL", {k: round(v, 2) for k, v in per_map.items()}, "pooled", round(pooled["median_of_step_median_m"], 2))
    print("G0-7", "pass" if report["G0-7"]["pass"] else "FAIL", report["G0-7"]["checks"])
    for seed, row in report["leak"].items():
        print(seed, {k: ("pass" if v else "FAIL") for k, v in row["gates"].items()})


def thumbnail(source, out, name, size=256):
    Image.open(source).convert("RGB").resize((size, size)).save(out / "img" / name, quality=82)
    return name


def object_card(obj, out):
    from experiments.v3_review import OBJECT_CHOICES
    identity = f'{obj["map"]}/{obj["key"]}'
    pics = "".join(f'<a href="img/{n}" target="_blank"><img loading="lazy" src="img/{n}" alt="{html.escape(obj["key"])} {v}"></a>'
                   for v, n in (("top", thumbnail(obj["preview_top"], out, f'{obj["key"]}_top.jpg')), ("takeoff", thumbnail(obj["preview_takeoff"], out, f'{obj["key"]}_takeoff.jpg'))))
    options = "".join(f'<label><input type="radio" name="obj__{html.escape(identity)}" value="{v}" data-id="{html.escape(identity)}" data-q="verdict">{t}</label>' for v, t in OBJECT_CHOICES)
    badges = "".join(f'<span class="badge">{t}</span> ' for t in (["草稿，请重点看"] if obj["draft"] else []) + (["未见类别"] if obj["unseen_class"] else []) + [f"已修：{f}" for f in obj["fixes"]])
    return f"""<div class="obj" data-id="{html.escape(identity)}"><div class="pair">{pics}</div><div class="objtext">{badges}<br><b>{html.escape(obj["zh"])}</b><br><span class="en">{html.escape(obj["en"])}</span><br>
<span class="meta">外观：{html.escape(obj["description_zh"])} ｜ 位置：{html.escape(obj["site_zh"])}<br>{obj["map"]} · {html.escape(obj["cls"])} · 外观 {html.escape(str(obj["variant"]))} · {obj["footprint_m"][0]:.0f}×{obj["footprint_m"][1]:.0f} m · 高 {obj["height_m"]:.1f} m · {"复数" if obj["plural"] else "单数"}</span></div>
<div class="opts">{options}</div><input class="note" data-id="{html.escape(identity)}" data-q="note" placeholder="改成什么（可不填）"></div>"""


def route_card(number, route, ground, out):
    from experiments.v3_review import CHOICES, ROUTE_QUESTIONS, height_chart, route_map
    identity = f"route_{number:02d}"
    options = []
    for key, text in ROUTE_QUESTIONS:
        radios = "".join(f'<label><input type="radio" name="{identity}__{key}" value="{v}" data-id="{identity}" data-q="{key}">{t}</label>' for v, t in CHOICES)
        options.append(f'<div class="q"><span>{text}</span><span class="opts">{radios}</span></div>')
    facts = (f'{route["id"]} · 目标 {html.escape(route["cls"])}{"（未见类别）" if route["unseen_class"] else ""}，等效宽 {route["width_m"]:.0f} m · 起点距离 {route["distance"]:.0f} m · 方位 {math.degrees(route["bearing"]):+.0f}° · '
             f'{"随地形" if route["mode"] == "follow" else "定高"}，起飞离地 {route["agl"]:.0f} m · 共 {len(route["actions"])} 步 · 注入 {len(route["kicks"])} 次 · 路线下地形起伏 {route["relief_m"]:.0f} m · '
             f'保持起飞高度时离地偏差：步中位 {route["hold_dev_median_m"]:.1f} m，均值 {route["hold_dev_mean_m"]:.1f} m')
    return f"""<section class="card" data-id="{identity}" data-needs="{",".join(k for k, _ in ROUTE_QUESTIONS)}"><h2>{number}. {html.escape(route["instruction_zh"])}</h2><p class="en">{html.escape(route["instruction"])}</p><div class="meta">{facts}</div>
<div class="row"><div class="charts">{route_map(route, ground, f"{identity}.jpg", out)}</div><div class="charts">{height_chart(route, ground)}<div class="caption">目标在 Blender 预览里的样子（正上方 60 m）</div><img loading="lazy" src="img/{route["object"]}_top.jpg" alt="目标"></div></div>
<div class="questions">{"".join(options)}<textarea data-id="{identity}" data-q="note" placeholder="备注（可不填）"></textarea></div></section>"""


def pick(routes, objects, count, seed):
    """Spread over objects, modes and words; one unseen-class route and one draft-target route when the map has them."""
    from experiments.v3_review import choose
    drafts = {o["key"] for o in objects if o["role"] == "target" and o["draft"]}
    picked = []
    for wanted in (lambda r: r["unseen_class"], lambda r: r["object"] in drafts):
        pool = [r for r in routes if wanted(r) and r not in picked]
        if pool:
            picked.append(pool[0])
    rest = [r for r in routes if r["object"] not in {p["object"] for p in picked}]
    return picked + choose(rest, count - len(picked), seed)


def review(args):
    from experiments.data_review import STYLE
    from experiments.v3_review import EXTRA_STYLE, SCRIPT
    out = Path(args.out) / "review"
    (out / "img").mkdir(parents=True, exist_ok=True)
    planned = read_json(Path(args.out) / "plan.json")
    objects = json.load(open(args.objects, encoding="utf-8"))
    targets = [o for o in objects if o["role"] == "target"]
    cards = "".join(object_card(o, out) for o in sorted(targets, key=lambda o: (not o["draft"], o["map"], o["key"])))
    previews, number = [], 0
    for map_id in TRAIN + DEV + TEST:
        routes = [arrays(r) for r in planned["maps"][map_id]["candidates"][:QUOTA[map_id]]]
        ground = Ground(map_id, args.cache)
        for r in pick(routes, objects, PREVIEWS_PER_MAP, args.seed):
            number += 1
            previews.append(route_card(number, r, ground, out))
    distractors = [o for o in objects if o["role"] == "distractor"]
    counts = {}
    for o in distractors:
        counts.setdefault(o["map"], {}).setdefault(o["cls"], 0)
        counts[o["map"]][o["cls"]] += 1
    rows = "".join(f"<li>{m}：{html.escape(', '.join(f'{c} ×{n}' for c, n in sorted(v.items())))}</li>" for m, v in sorted(counts.items()))
    page = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>v3 阶段 2 · 阶段 0 核对</title><style>{STYLE}{EXTRA_STYLE}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:4px}}</style></head><body>
<header><h1>v3 阶段 2 · 阶段 0 核对（G0-5）</h1><span class="grow" id="progress"></span><button id="allyes">未看的物体全标"可以"</button><button id="export">导出 review.json</button></header><main>
<p class="intro">先看物体：指令里的称呼是"位置 + 外观"的完整描述，看它是否准确、能否只指向这一个物体。7 个"草稿"是沿用 v2 的 desert 目标，原配置里只有占位文字，由我按阶段 1 已核对的描述和 v2 地标起草，排在最前面。图是 Blender 预览：左边是正上方 60 m 的俯视，右边是 150 m 外的起飞视角，不是新路线的 UE 画面，UE 画面在下一步预览检查时出。
再看 {number} 条路线（每张图 {PREVIEWS_PER_MAP} 条，取自路线计划，开发图和测试图各含 1 条未见类别路线）。地图上方为北，底图为地形阴影，粗短线是起飞朝向，橙色圈是注入偏移的位置。打分自动保存在本浏览器，看完导出，放到 <code>runs/v3/stage2/review/review.json</code>。</p>
<h2 class="group">一、目标物体（{len(targets)} 个，8 张图）</h2><div class="objects">{cards}</div>
<h2 class="group">干扰物（{len(distractors)} 个，不作目标，参与歧义检查和规则 8）</h2><ul>{rows}</ul><p class="intro">desert 是 class_like_v2_objects 中 counted=false 的物体；small_houses 是村落、农舍、山屋和 DES_A 聚落的房屋（11/05 决定）。</p>
<h2 class="group">二、路线预览（{number} 条，设计 {DESIGN}）</h2>{"".join(previews)}</main><script>{SCRIPT.replace("sr3m-v3-stage0", "sr3m-v3s2-stage0")}</script></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(targets)} objects, {number} routes: {out / 'index.html'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "leak", "gates", "review"))
    p.add_argument("--out", default="runs/v3/stage2"); p.add_argument("--objects", default="configs/v3s2_objects.json")
    p.add_argument("--cache", default=str(Path.home() / "sr3mxhad_runs/v3/ground")); p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--seeds", type=int, nargs="+", default=list(LEAK_SEEDS)); p.add_argument("--device", default="cuda")
    args = p.parse_args()
    {"plan": plan, "leak": leak, "gates": gates, "review": review}[args.stage](args)
