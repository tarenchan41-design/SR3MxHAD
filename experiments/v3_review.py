"""Human check for stage 0 of dataset v3 (gate G0-5): the object table and 20 route previews. No rendering, nothing written outside --out.

The thumbnails are existing v2 frames with the object boxed, so they show the object itself but not the view from a new route.
Scores stay in the browser until exported to review.json.
"""
import argparse
import html
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from experiments.data_review import STYLE, polyline
from experiments.v3_leak import DESIGNS, EVAL_MAPS, FIT_MAPS, Ground
from sr3m.routes_v3 import generate

ROUTES_PER_MAP = 40
PREVIEWS = 20
SEED = 20261102
OBJECT_CHOICES = (("yes", "可以"), ("fix", "要改"), ("no", "不用"))
ROUTE_QUESTIONS = (("text", "指令清楚、只指向这一个物体"), ("route", "起点和路线合理"))
CHOICES = (("yes", "是"), ("partly", "勉强"), ("no", "否"))


def choose(routes, count=PREVIEWS, seed=SEED):
    """Spread over maps, then prefer objects, modes and direction words not shown yet."""
    rng = np.random.default_rng(seed)
    maps = sorted({r["map"] for r in routes})
    picked, seen = [], set()
    order = 0
    while len(picked) < count:
        map_id = maps[order % len(maps)]
        order += 1
        pool = [r for r in routes if r["map"] == map_id and r not in picked]
        if not pool:
            continue
        scores = [(r["object"] in seen) * 4 + ((r["mode"], r["word"]) in seen) * 2 + rng.random() for r in pool]
        best = pool[int(np.argmin(scores))]
        picked.append(best)
        seen.add(best["object"]); seen.add((best["mode"], best["word"]))
    return picked


def hillshade(ground, x0, y0, span, path, size=256):
    xs, ys = np.linspace(x0, x0 + span, size), np.linspace(y0, y0 + span, size)
    gx, gy = np.meshgrid(xs, ys)
    z = ground(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    step = span / (size - 1)
    dy, dx = np.gradient(z, step)
    shade = 1 / np.sqrt(1 + dx ** 2 + dy ** 2) * (1 - .55 * (dx - dy) / np.sqrt(2) / np.sqrt(1 + dx ** 2 + dy ** 2))
    tone = np.clip((shade - .35) / .9, 0, 1)
    Image.fromarray((150 + 95 * tone).astype(np.uint8)).save(path, quality=80)


def route_map(route, ground, name, out, size=320, margin=30):
    """North up: +X is east and +Y is south, so the image's own axes already match."""
    P, Y = route["positions"], route["yaws"]
    target, half = np.array(route["target"]), route["width_m"] / 2
    xs, ys = [*P[:, 0], target[0] - half, target[0] + half], [*P[:, 1], target[1] - half, target[1] + half]
    span_inner = max(max(xs) - min(xs), max(ys) - min(ys), 40.)
    scale = (size - 2 * margin) / span_inner
    x0, y0 = (min(xs) + max(xs)) / 2 - size / 2 / scale, (min(ys) + max(ys)) / 2 - size / 2 / scale
    hillshade(ground, x0, y0, size / scale, out / "img" / name)
    to = lambda x, y: ((x - x0) * scale, (y - y0) * scale)
    parts = [f'<svg viewBox="0 0 {size} {size}" class="map" role="img" aria-label="俯视路线图，上方为北，底图为地形阴影"><image href="img/{name}" width="{size}" height="{size}"/>']
    ax, ay = to(target[0] - half, target[1] - half)
    parts.append(f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{route["width_m"] * scale:.1f}" height="{route["width_m"] * scale:.1f}" class="anchor"><title>目标，宽 {route["width_m"]:.0f} m</title></rect>')
    for t in range(0, len(P), 20):
        x, y = to(P[t, 0], P[t, 1])
        parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 11 * math.cos(Y[t]):.1f}" y2="{y + 11 * math.sin(Y[t]):.1f}" class="tick"/>')
    parts.append(f'<polyline points="{polyline(to(p[0], p[1]) for p in P)}" class="path"/>')
    for kick in route["kicks"]:
        x, y = to(P[kick["step"], 0], P[kick["step"], 1])
        value = f'{math.degrees(kick["value"]):+.0f}°' if kick["kind"] == "heading" else f'{kick["value"]:+.1f} m'
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="jump"><title>第 {kick["step"]} 步注入偏移 {value}</title></circle>')
    for t in range(0, len(P), 5):
        x, y = to(P[t, 0], P[t, 1])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" class="hit"><title>第 {t} 步，离地面 {P[t, 2] - float(ground(P[t:t + 1, :2])[0]):.0f} m</title></circle>')
    sx, sy = to(P[0, 0], P[0, 1])
    ex, ey = to(P[0, 0] + 22 / scale * math.cos(Y[0]), P[0, 1] + 22 / scale * math.sin(Y[0]))
    gx, gy = to(P[-1, 0], P[-1, 1])
    parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" class="heading"/><circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" class="start"/><text x="{sx + 8:.1f}" y="{max(sy - 7, 12):.1f}" class="label">起点</text><circle cx="{gx:.1f}" cy="{gy:.1f}" r="5" class="goal"/><text x="{gx + 8 if gx < size - 40 else gx - 32:.1f}" y="{max(gy - 7, 12):.1f}" class="label">终点</text>')
    metres = next((m for m in (10, 20, 50, 100, 200, 500) if m * scale >= 40), 500)
    parts.append(f'<line x1="14" y1="{size - 9}" x2="{14 + metres * scale:.1f}" y2="{size - 9}" class="scale"/><text x="14" y="{size - 13}" class="label">{metres} m</text><text x="{size - 26}" y="16" class="label">北 ↑</text></svg>')
    return "".join(parts)


def height_chart(route, ground, width=320, height=110, left=38, bottom=8, top=18):
    """Flight and ground share one height axis; both are labelled directly and every sampled step has its value."""
    z, g = route["positions"][:, 2], ground(route["positions"][:, :2])
    low, high = float(np.nanmin(g)), float(z.max())
    span = max(high - low, 2.)
    x = lambda i: left + i * (width - left - 4) / max(len(z) - 1, 1)
    y = lambda v: top + (height - top - bottom) * (1 - (v - low) / span)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="spark" role="img" aria-label="高度剖面"><text x="{left}" y="11" class="label">高度（m）：蓝线飞行，灰线地面</text><text x="{left - 4}" y="{y(high) + 8:.1f}" class="axis" text-anchor="end">{high:.0f}</text><text x="{left - 4}" y="{y(low):.1f}" class="axis" text-anchor="end">{low:.0f}</text>',
             f'<polyline points="{polyline((x(i), y(v)) for i, v in enumerate(g))}" class="groundline"/><polyline points="{polyline((x(i), y(v)) for i, v in enumerate(z))}" class="path"/>']
    for i in range(0, len(z), max(1, len(z) // 50)):
        parts.append(f'<rect x="{x(i) - 3:.1f}" y="{top}" width="6" height="{height - top - bottom}" class="hit"><title>第 {i} 步：飞行 {z[i]:.1f} m，地面 {g[i]:.1f} m，离地 {z[i] - g[i]:.1f} m</title></rect>')
    return "".join(parts) + "</svg>"


def object_card(obj, out):
    frame = ""
    if obj.get("frame"):
        source = Path(out).parent / "objects" / obj["frame"]
        if source.exists():
            shutil.copy(source, out / "img" / obj["frame"])
            frame = f'<a href="img/{obj["frame"]}" target="_blank"><img loading="lazy" src="img/{obj["frame"]}" alt="{html.escape(obj["key"])}"></a>'
    identity = f'{obj["map"]}/{obj["key"]}'
    options = "".join(f'<label><input type="radio" name="obj__{html.escape(identity)}" value="{v}" data-id="{html.escape(identity)}" data-q="verdict">{t}</label>' for v, t in OBJECT_CHOICES)
    limit = min(8 * obj["width_m"], 250)
    warning = f'<br><span class="badge">注意：{html.escape(obj["warn"])}</span>' if obj.get("warn") else ""
    return f"""<div class="obj" data-id="{html.escape(identity)}">{frame}<div class="objtext"><b>{html.escape(obj["zh"])}</b><br><span class="en">{html.escape(obj["en"])}</span><br><span class="meta">{obj["map"]} · {html.escape(obj["cls"])} · 宽 {obj["width_m"]:.0f} m · 起点最远 {limit:.0f} m<br>旧描述：{html.escape(str(obj.get("old") or "无"))}</span>{warning}</div><div class="opts">{options}</div><input class="note" data-id="{html.escape(identity)}" data-q="note" placeholder="改成什么（可不填）"></div>"""


def route_card(number, route, ground, thumbs, out):
    identity = f"route_{number:02d}"
    options = []
    for key, text in ROUTE_QUESTIONS:
        radios = "".join(f'<label><input type="radio" name="{identity}__{key}" value="{v}" data-id="{identity}" data-q="{key}">{t}</label>' for v, t in CHOICES)
        options.append(f'<div class="q"><span>{text}</span><span class="opts">{radios}</span></div>')
    px_start = 256 * route["width_m"] / route["distance"]
    above = route["positions"][-1, 2] - route["top_z"]
    px_end = min(512, 256 * route["width_m"] / max(above, 1))
    facts = f'{route["map"]} · 目标 {html.escape(route["cls"])}，宽 {route["width_m"]:.0f} m · 起点距离 {route["distance"]:.0f} m · 方位 {math.degrees(route["bearing"]):+.0f}° · {"随地形" if route["mode"] == "follow" else "定高"} · 共 {len(route["actions"])} 步 · 路线下地形起伏 {route["relief_m"]:.0f} m · 注入 {len(route["kicks"])} 次 · 目标宽度约：起飞 {px_start:.0f} px，到达 {px_end:.0f} px（512 px 画面）'
    thumb = f'<img loading="lazy" src="img/{thumbs[route["object"]]}" alt="目标">' if route["object"] in thumbs else ""
    return f"""<section class="card" data-id="{identity}" data-needs="{",".join(k for k, _ in ROUTE_QUESTIONS)}"><h2>{number}. {html.escape(route["instruction_zh"])}</h2><p class="en">{html.escape(route["instruction"])}</p><div class="meta">{facts}</div>
<div class="row"><div class="charts">{route_map(route, ground, f"{identity}.jpg", out)}</div><div class="charts">{height_chart(route, ground)}<div class="caption">目标在 v2 帧里的样子（红框）</div>{thumb}</div></div>
<div class="questions">{"".join(options)}<textarea data-id="{identity}" data-q="note" placeholder="备注（可不填）"></textarea></div></section>"""


EXTRA_STYLE = """
.objects{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}.obj{border:1px solid var(--line);border-radius:8px;padding:8px}.obj img{width:100%;aspect-ratio:1;border-radius:4px;display:block}.obj.done{border-color:var(--series)}.objtext{font-size:13px;margin:6px 0}.objtext .en{color:var(--muted)}.objtext .meta{font-size:12px}.opts label{margin-right:10px;white-space:nowrap}.note{width:100%;font:inherit;font-size:13px;background:var(--surface);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:3px 6px;margin-top:4px}
.charts img{width:100%;max-width:320px;border-radius:4px}.groundline{fill:none;stroke:var(--muted);stroke-width:2;stroke-linejoin:round}.heading{stroke:var(--text);stroke-width:2}.q .opts label{margin-left:12px;margin-right:0}.map{background:none}.path{stroke-width:2.5}
"""

SCRIPT = """
const KEY='sr3m-v3-stage0', saved=JSON.parse(localStorage.getItem(KEY)||'{}');
function done(el){const s=saved[el.dataset.id]||{};const needs=(el.dataset.needs||'verdict').split(',');return needs.every(q=>s[q]);}
function refresh(){const items=[...document.querySelectorAll('[data-id].obj,[data-id].card')];items.forEach(e=>e.classList.toggle('done',done(e)));const objects=[...document.querySelectorAll('.obj')],routes=[...document.querySelectorAll('.card')];document.getElementById('progress').textContent=`物体 ${objects.filter(done).length} / ${objects.length}，路线 ${routes.filter(done).length} / ${routes.length}`;}
document.querySelectorAll('input[data-q],textarea[data-q]').forEach(f=>{const s=saved[f.dataset.id]||{};if(f.type==='radio'){if(s[f.dataset.q]===f.value)f.checked=true;}else if(s[f.dataset.q])f.value=s[f.dataset.q];
f.addEventListener(f.type==='radio'?'change':'input',()=>{(saved[f.dataset.id]=saved[f.dataset.id]||{})[f.dataset.q]=f.value;localStorage.setItem(KEY,JSON.stringify(saved));refresh();});});
document.getElementById('allyes').onclick=()=>{document.querySelectorAll('.obj').forEach(o=>{const id=o.dataset.id;if(!(saved[id]||{}).verdict){(saved[id]=saved[id]||{}).verdict='yes';o.querySelector('input[value=yes]').checked=true;}});localStorage.setItem(KEY,JSON.stringify(saved));refresh();};
document.getElementById('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(saved,null,1)],{type:'application/json'}));a.download='review.json';a.click();};
refresh();
"""


def main(args):
    objects = json.load(open(args.objects, encoding="utf-8"))
    design = DESIGNS[args.design]
    rng = np.random.default_rng(args.seed)
    grounds = {m: Ground(m, args.cache) for m in FIT_MAPS + EVAL_MAPS}
    routes = []
    for map_id in FIT_MAPS + EVAL_MAPS:
        routes.extend(generate([o for o in objects if o["map"] == map_id], grounds[map_id], ROUTES_PER_MAP, design, rng)[0])
    picked = choose(routes)
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    cards = "".join(object_card(o, out) for o in objects)
    thumbs = {o["key"]: o["frame"] for o in objects if o.get("frame")}
    previews = "".join(route_card(i + 1, r, grounds[r["map"]], thumbs, out) for i, r in enumerate(picked))
    page = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>v3 阶段 0 核对</title><style>{STYLE}{EXTRA_STYLE}</style></head><body>
<header><h1>v3 阶段 0 核对</h1><span class="grow" id="progress"></span><button id="allyes">未看的物体全标"可以"</button><button id="export">导出 review.json</button></header><main>
<p class="intro">分两部分：先看物体描述对不对、适不适合当目标，再看 {len(picked)} 条路线。打分自动保存在本浏览器。看完点右上角导出，放到 <code>runs/v3/review/review.json</code>。物体图是 v2 已有的帧，红框是物体位置，不代表新路线的视角；新路线的真实画面要到阶段 1 才有。地图上方为北，底图是地形阴影，粗短线是起飞朝向，橙色圈是注入偏移的位置。</p>
<h2 class="group">一、物体表（{len(objects)} 个，mountain 和 desert 的 A、B、D 图）</h2><div class="objects">{cards}</div>
<h2 class="group">二、路线预览（{len(picked)} 条，设计 {html.escape(args.design)}）</h2>{previews}</main><script>{SCRIPT}</script></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(objects)} objects, {len(picked)} routes: {out / 'index.html'}")
    print({m: sum(r["map"] == m for r in picked) for m in sorted({r["map"] for r in picked})}, {k: sum(r["mode"] == k for r in picked) for k in ("follow", "hold")}, {k: sum(r["word"] == k for r in picked) for k in ("left", "ahead", "right")}, "objects", len({r["object"] for r in picked}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objects", default="configs/v3_objects.json"); p.add_argument("--design", default="pursuit_kick", choices=list(DESIGNS))
    p.add_argument("--cache", default=str(Path.home() / "sr3mxhad_runs/v3/ground")); p.add_argument("--out", default="runs/v3/review"); p.add_argument("--seed", type=int, default=SEED)
    main(p.parse_args())
