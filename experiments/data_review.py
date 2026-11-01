"""Human review page for the v2 audit: 30 stratified train routes plus a few the audit flagged. Read-only on the data.

Scores stay in the browser (localStorage) until exported to scores.json, so the page needs no server.
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.data_audit import anchor_of, load_designs
from experiments.protocol import PreparedSplit, read_json

SEED = 20261101
SAMPLE = 30
TYPES = ("goal", "instruction", "requirement")
FRACTIONS = (0., 1 / 3, 2 / 3, 1.)
THUMB_PX = 256
QUESTIONS = (("read", "指令读得懂（不需要额外的图或地图）"), ("path", "指令能确定这条路线，或路线形状说得通"), ("arrive", "到终点时能看出到了（目标在画面里）"), ("scene", "路线放在这个场景里合理（不乱绕、不穿奇怪的地方）"), ("image", "图像可用（不过暗、不模糊、无异常）"))
CHOICES = (("yes", "是"), ("partly", "勉强"), ("no", "否"))


def stratified(rows, seed=SEED, count=SAMPLE):
    """One route per category and instruction type, rotating over maps, then the remainder from random categories."""
    rng = np.random.default_rng(seed)
    chosen = []
    categories = sorted({r["category"] for r in rows})
    for c, category in enumerate(categories):
        maps = sorted({r["map"] for r in rows if r["category"] == category})
        for t, kind in enumerate(TYPES):
            wanted = maps[(c + t) % len(maps)]
            pool = [r for r in rows if r["category"] == category and r["type"] == kind and r["map"] == wanted] or [r for r in rows if r["category"] == category and r["type"] == kind]
            chosen.append(pool[rng.integers(len(pool))])
    for category in rng.permutation(categories)[:max(0, count - len(chosen))]:
        taken = {r["route"] for r in chosen}
        pool = [r for r in rows if r["category"] == category and r["route"] not in taken]
        chosen.append(pool[rng.integers(len(pool))])
    return chosen


def flagged(rows, taken):
    """The audit's worst examples, which a random sample would rarely reach."""
    free = [r for r in rows if r["route"] not in taken]
    picks = []
    for reason, key, limit in (("质量标记 failed", lambda r: r.get("quality") == "failed", 2), ("单步转向最大", lambda r: r["max_yaw_deg"], 2), ("俯视暗帧最多", lambda r: r.get("black_down", 0) / max(r.get("frames_per_view", 1), 1), 2)):
        ranked = sorted(free, key=key, reverse=True)
        category_seen = set()
        for r in ranked:
            if r["category"] not in category_seen and r["route"] not in {p[1]["route"] for p in picks}:
                category_seen.add(r["category"])
                picks.append((reason, r))
            if len(category_seen) == limit:
                break
    return picks


def polyline(points):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def route_map(P, Y, waypoints, anchor, jumps, size=320, margin=26):
    """Map with north up: +X is east and +Y is south, so the SVG's own axes already match."""
    xs = [*P[:, 0], *waypoints[:, 0]]
    ys = [*P[:, 1], *waypoints[:, 1]]
    if anchor is not None:
        xs += [anchor["low"][0], anchor["high"][0]]
        ys += [anchor["low"][1], anchor["high"][1]]
    x0, y0, span = min(xs), min(ys), max(max(xs) - min(xs), max(ys) - min(ys), 1.)
    scale = (size - 2 * margin) / span
    to = lambda x, y: (margin + (x - x0) * scale, margin + (y - y0) * scale)
    parts = [f'<svg viewBox="0 0 {size} {size}" class="map" role="img" aria-label="俯视路线图，上方为北">']
    if anchor is not None:
        ax, ay = to(anchor["low"][0], anchor["low"][1])
        bx, by = to(anchor["high"][0], anchor["high"][1])
        parts.append(f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{max(bx - ax, 3):.1f}" height="{max(by - ay, 3):.1f}" class="anchor"><title>指令的地标</title></rect><text x="{ax:.1f}" y="{max(ay - 4, 10):.1f}" class="label">地标</text>')
    for t in range(0, len(P), 20):
        x, y = to(P[t, 0], P[t, 1])
        parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 11 * np.cos(Y[t]):.1f}" y2="{y + 11 * np.sin(Y[t]):.1f}" class="tick"/>')
    parts.append(f'<polyline points="{polyline(to(p[0], p[1]) for p in P)}" class="path"/>')
    for w in waypoints[1:-1]:
        x, y = to(w[0], w[1])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="way"><title>设计航点</title></circle>')
    for t in jumps:
        x, y = to(P[t, 0], P[t, 1])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="jump"><title>第 {t} 步：单步转向超过 20°</title></circle>')
    for t in range(0, len(P), 5):
        x, y = to(P[t, 0], P[t, 1])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" class="hit"><title>第 {t} 步，航向 {np.degrees(Y[t]):.0f}°</title></circle>')
    sx, sy, gx, gy = *to(P[0, 0], P[0, 1]), *to(P[-1, 0], P[-1, 1])
    parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" class="start"/><text x="{sx + 8 if sx < size - 40 else sx - 32:.1f}" y="{max(sy - 7, 10):.1f}" class="label">起点</text><circle cx="{gx:.1f}" cy="{gy:.1f}" r="5" class="goal"/><text x="{gx + 8 if gx < size - 40 else gx - 32:.1f}" y="{max(gy - 7, 10):.1f}" class="label">终点</text>')
    metres = next((m for m in (10, 20, 50, 100, 200, 500) if m * scale >= 40), 500)
    parts.append(f'<line x1="{margin}" y1="{size - 9}" x2="{margin + metres * scale:.1f}" y2="{size - 9}" class="scale"/><text x="{margin}" y="{size - 13}" class="label">{metres} m</text><text x="{size - 22}" y="16" class="label">北 ↑</text></svg>')
    return "".join(parts)


def small_chart(values, unit, title, zero=False, width=320, height=84, left=38, bottom=6, top=16):
    """One series on its own axis; each bar of a hit target carries the exact value."""
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if zero:
        low, high = min(low, 0.), max(high, 0.)
    if high - low < 2:
        low, high = (low + high) / 2 - 1, (low + high) / 2 + 1
    span = high - low
    x = lambda i: left + i * (width - left - 4) / max(len(values) - 1, 1)
    y = lambda v: top + (height - top - bottom) * (1 - (v - low) / span)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="spark" role="img" aria-label="{html.escape(title)}"><text x="{left}" y="11" class="label">{html.escape(title)}</text>',
             f'<text x="{left - 4}" y="{y(high) + 8:.1f}" class="axis" text-anchor="end">{high:.1f}</text><text x="{left - 4}" y="{y(low):.1f}" class="axis" text-anchor="end">{low:.1f}</text>']
    if zero:
        parts.append(f'<line x1="{left}" y1="{y(0):.1f}" x2="{width - 4}" y2="{y(0):.1f}" class="grid"/>')
    parts.append(f'<polyline points="{polyline((x(i), y(v)) for i, v in enumerate(values))}" class="path"/>')
    for i in range(0, len(values), max(1, len(values) // 60)):
        parts.append(f'<rect x="{x(i) - 3:.1f}" y="{top}" width="6" height="{height - top - bottom}" class="hit"><title>第 {i} 步：{values[i]:.2f} {unit}</title></rect>')
    return "".join(parts) + "</svg>"


def thumbnail(source, target):
    if not target.exists():
        with Image.open(source) as image:
            image.convert("RGB").resize((THUMB_PX, THUMB_PX), Image.BICUBIC).save(target, quality=82)


def card(number, row, route, ds, plan, metadata, data_root, out, reason=None):
    o, n = route["offset"], route["count"]
    P, Y = ds.arrays["position"][o:o + n].astype(float), ds.arrays["yaw"][o:o + n].astype(float)
    A = ds.arrays["action"][o:o + n - 1].astype(float)
    yaw_deg = np.degrees(A[:, 3])
    waypoints = np.array(plan["waypoints_xyz_m"], dtype=float)
    anchor = anchor_of(metadata, plan.get("end_region"))
    frames = []
    for fraction in FRACTIONS:
        step = round(fraction * (n - 1))
        cells = []
        for view in ("front", "down"):
            name = f"{row['route']}_{step:04d}_{view}.jpg"
            thumbnail(Path(data_root) / route["directory"] / route["images"][view][step], out / "img" / name)
            cells.append(f'<a href="img/{name}" target="_blank"><img loading="lazy" src="img/{name}" alt="第 {step} 步 {view}"></a>')
        frames.append(f'<div class="frame"><div class="caption">第 {step} 步</div>{"".join(cells)}</div>')
    questions = []
    for key, text in QUESTIONS:
        options = "".join(f'<label><input type="radio" name="{row["route"]}__{key}" value="{value}" data-route="{row["route"]}" data-q="{key}">{label}</label>' for value, label in CHOICES)
        questions.append(f'<div class="q"><span>{text}</span><span class="opts">{options}</span></div>')
    via = "；".join(plan.get("via_regions") or []) or "无"
    facts = {"类型": row["type"], "形状": row.get("shape") or "无记录", "路线长度": f"{row['length_m']:.0f} m", "起终点直线": f"{row['start_goal_m']:.0f} m", "绕行比": f"{row['detour']:.2f}", "总转角": f"{row['total_turn_deg']:.0f}°", "单步最大转向": f"{row['max_yaw_deg']:.1f}°", "用东南西北": "是" if row["compass"] else "否", "俯视暗帧": f"{row.get('black_down', 0)}/{row.get('frames_per_view', 0)}", "质量标记": row.get("quality"), "成功条件": row.get("success") or "无记录", "地标在终点画面内": {True: "是", False: "否", None: "无坐标"}[row.get("anchor_seen_goal")]}
    table = "".join(f"<tr><th>{k}</th><td>{html.escape(str(v))}</td></tr>" for k, v in facts.items())
    badge = f'<span class="badge">自动标记：{reason}</span>' if reason else ""
    return f"""<section class="card" id="{row['route']}" data-route="{row['route']}"><h2>{number}. {html.escape(row['route'])} {badge}</h2><div class="meta">{row['category']} · {row['map']} · {row['type']} · 共 {n - 1} 步</div>
<p class="zh">{html.escape(route['instruction_zh'])}</p><p class="en">{html.escape(route['instruction'])}</p>
<div class="row"><div class="charts">{route_map(P, Y, waypoints, anchor, [int(t) for t in np.nonzero(np.abs(yaw_deg) > 20)[0]])}{small_chart(yaw_deg, "°/步", "专家每步转向（°，正为右转）", zero=True)}{small_chart(P[:, 2], "m", "世界高度（m）")}</div><div class="frames">{"".join(frames)}</div></div>
<div class="row lower"><div class="questions">{"".join(questions)}<textarea data-route="{row['route']}" data-q="note" placeholder="备注（可不填）"></textarea></div>
<details><summary>设计信息与自动指标（打分后再看，避免先入为主）</summary><p class="plan"><b>途经：</b>{html.escape(via)}<br><b>终点：</b>{html.escape(plan.get('end_region') or '无')}<br><b>任务：</b>{html.escape(plan.get('task') or '无')}</p><table>{table}</table></details></div></section>"""


STYLE = """
:root{color-scheme:light;--surface:#fcfcfb;--panel:#f3f2ef;--text:#0b0b0b;--muted:#52514e;--line:#d9d8d3;--series:#2a78d6;--accent:#eb6834}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--surface:#1a1a19;--panel:#252523;--text:#f0efec;--muted:#c3c2b7;--line:#42413d;--series:#3987e5;--accent:#d95926}}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:2;display:flex;gap:16px;align-items:center;padding:10px 20px;background:var(--panel);border-bottom:1px solid var(--line)}header h1{font-size:16px;margin:0}header .grow{flex:1;color:var(--muted);font-size:13px}
button{font:inherit;padding:5px 12px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--text);cursor:pointer}
main{max-width:1320px;margin:0 auto;padding:16px 20px}.intro{color:var(--muted);max-width:900px}.group{margin:28px 0 8px;font-size:17px}
.card{border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:14px 0}.card.done{border-color:var(--series)}.card h2{font-size:15px;margin:0;word-break:break-all}.meta{color:var(--muted);font-size:13px}
.badge{font-weight:400;font-size:12px;border:1px solid var(--accent);border-radius:4px;padding:0 6px;margin-left:6px}.zh{font-size:16px;margin:10px 0 2px}.en{color:var(--muted);font-size:13px;margin:0 0 10px}
.row{display:flex;gap:16px;flex-wrap:wrap}.charts{width:320px;flex:none}.frames{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:8px;flex:1;min-width:640px}.frame img{display:block;width:100%;aspect-ratio:1;margin-bottom:4px;border-radius:4px;background:var(--panel)}.caption{font-size:12px;color:var(--muted)}
.lower{margin-top:12px}.questions{flex:1;min-width:520px}.q{display:flex;justify-content:space-between;gap:12px;padding:3px 0;border-bottom:1px solid var(--line)}.opts label{margin-left:12px;white-space:nowrap}textarea{width:100%;margin-top:8px;min-height:44px;font:inherit;background:var(--surface);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px}
details{flex:0 0 340px;font-size:13px}summary{cursor:pointer;color:var(--muted)}table{border-collapse:collapse;width:100%;margin-top:6px}th{text-align:left;font-weight:400;color:var(--muted);padding:1px 8px 1px 0}td{padding:1px 0}
svg{display:block;width:100%}.map{background:var(--panel);border-radius:4px}.path{fill:none;stroke:var(--series);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.tick{stroke:var(--muted);stroke-width:1}.way{fill:var(--surface);stroke:var(--muted);stroke-width:1.5}.anchor{fill:none;stroke:var(--text);stroke-width:1.5;stroke-dasharray:3 2}.jump{fill:none;stroke:var(--accent);stroke-width:2}.start{fill:var(--surface);stroke:var(--text);stroke-width:2}.goal{fill:var(--text);stroke:var(--surface);stroke-width:2}.hit{fill:transparent}.scale{stroke:var(--text);stroke-width:2}.grid{stroke:var(--line);stroke-width:1}.label{font-size:11px;fill:var(--text)}.axis{font-size:10px;fill:var(--muted)}.spark{margin-top:6px}
"""

SCRIPT = """
const KEY='sr3m-data-audit-v2', saved=JSON.parse(localStorage.getItem(KEY)||'{}');
const fields=document.querySelectorAll('[data-route][data-q]');
function refresh(){const needed=%d;let done=0;document.querySelectorAll('.card').forEach(c=>{const s=saved[c.dataset.route]||{};const full=%s.every(q=>s[q]);c.classList.toggle('done',full);if(full)done++;});document.getElementById('progress').textContent=`已打分 ${done} / ${needed}`;}
fields.forEach(f=>{const s=saved[f.dataset.route]||{};if(f.type==='radio'){if(s[f.dataset.q]===f.value)f.checked=true;}else if(s[f.dataset.q])f.value=s[f.dataset.q];
f.addEventListener(f.type==='radio'?'change':'input',()=>{(saved[f.dataset.route]=saved[f.dataset.route]||{})[f.dataset.q]=f.value;localStorage.setItem(KEY,JSON.stringify(saved));refresh();});});
document.getElementById('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(saved,null,1)],{type:'application/json'}));a.download='scores.json';a.click();};
refresh();
"""


def run(args):
    rows = read_json(Path(args.audit) / "routes.json")
    train = [r for r in rows if r["split"] == "train" and r.get("type")]
    sample = stratified(train)
    extra = flagged(train, {r["route"] for r in sample})
    ds = PreparedSplit(args.prepared, "train")
    routes = {r["episode_id"]: r for r in ds.routes}
    designs = load_designs()
    out = Path(args.audit) / "review"
    (out / "img").mkdir(parents=True, exist_ok=True)
    cards, number = [], 0
    for row in sample:
        number += 1
        plan, metadata = designs[(row["map"], row["route"])]
        cards.append(card(number, row, routes[row["episode_id"]], ds, plan, metadata, args.data_root, out))
    flagged_cards = []
    for reason, row in extra:
        number += 1
        plan, metadata = designs[(row["map"], row["route"])]
        flagged_cards.append(card(number, row, routes[row["episode_id"]], ds, plan, metadata, args.data_root, out, reason))
    page = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>v2 数据抽检</title><style>{STYLE}</style></head><body>
<header><h1>v2 数据抽检</h1><span class="grow" id="progress"></span><button id="export">导出 scores.json</button></header><main>
<p class="intro">只读抽检，不改数据。打分自动保存在本浏览器。看完点右上角导出，把文件放到 <code>runs/data_audit/review/scores.json</code>。地图上方为北，蓝线是专家路线，短灰线是相机朝向，虚线框是指令的地标（部分类别没有坐标）。每列上下两张是同一步的前视和俯视，点图看大图；缩略图为 256 px，模型实际输入是 224 px。</p>
<h2 class="group">分层抽样（{len(sample)} 条，train，种子 {SEED}）</h2>{"".join(cards)}<h2 class="group">自动标记（{len(extra)} 条，选看）</h2>{"".join(flagged_cards)}</main>
<script>{SCRIPT % (number, json.dumps([k for k, _ in QUESTIONS]))}</script></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(sample)} sampled, {len(extra)} flagged; {out / 'index.html'}")
    print({c: sum(r["category"] == c for r in sample) for c in sorted({r["category"] for r in sample})}, {t: sum(r["type"] == t for r in sample) for t in TYPES})
    print([(reason, r["route"]) for reason, r in extra])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="runs/terrain_v2_dino/prepared")
    p.add_argument("--data-root", default="/mnt/f/WSL_ColdStorage/datasets/DatasetCaptureBatches/workbench")
    p.add_argument("--audit", default="runs/data_audit")
    run(p.parse_args())
