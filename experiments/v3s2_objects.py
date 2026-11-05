"""Stage 2 object table (protocol section 17.2): configs/v3s2_objects.json from the Blender scene configs. Reads only; writes --out.

Coordinates are UE metres (x east, y south = -Blender y), the frame of the ground cache and of K.
Targets come from each map's scene config. The instruction name is appearance + site (decision of 11/05): description_en/zh hold the
appearance alone, site_en/zh the site, en/zh the full name the generator uses. Distractors are never targets but take part in the
ambiguity check (rule 7) and rule 8: desert class_like_v2_objects with counted=false, and village, farm and hut houses as small_houses
(decision of 11/05). Text fixes and drafts for the seven v2 targets are listed here so the table can be rebuilt; the drafts passed G0-5
on 11/05. Sites carry no compass words: the policy only knows its pose relative to the takeoff, never north (rule 7, decision of 11/05).
en_short / zh_short name the target by its appearance alone, for the routes that use the short name (option C, decision of 11/05).
"""
import argparse
import json
import math
import re
from pathlib import Path

from experiments.v3_ue_check import maps

SPLIT = {"A": "train", "D": "train", "B": "dev", "C": "test"}
HOUSE_PREFIXES = ("Village_House", "AlpineHut", "FarmHouse", "Settlement_House")
HOUSE_WIDTH_M = 10.
# The v2 targets kept in the desert carry placeholders ("(original)"). Appearance from the stage 1 table reviewed at G0-5 and the
# Blender previews; site from the v2 landmarks. All are drafts until the G0-5 review.
DRAFTS = {
    "DES_A_helipad_1": ("round light grey concrete helipad with a white H", "in the western dune valley", "带白色H的浅灰色圆形混凝土停机坪", "西侧沙丘谷地中的"),
    "DES_A_ruins_1": ("square stone ruins without a roof, with a row of columns and an arched gate", "in a dune valley", "带一排石柱和拱门的无顶方形石砌遗迹", "沙丘谷地中的"),
    "DES_B_helipad_1": ("round light grey concrete helipad with a white H", "on the north plateau", "带白色H的浅灰色圆形混凝土停机坪", "北侧台地上的"),
    "DES_B_tent_camp_1": ("camp of small orange, yellow, blue and green dome tents in rows next to white containers and parked cars", "on the canyon floor",
                          "白色集装箱和车辆旁成排的橙、黄、蓝、绿色小圆顶帐篷营地", "峡谷底部的"),
    "DES_C_helipad_1": ("round light grey concrete helipad with a white H", "on the rock bench above a gap in the cliffs", "带白色H的浅灰色圆形混凝土停机坪", "峭壁缺口上方岩石台地上的"),
    "DES_D_helipad_1": ("round light grey concrete helipad with a white H", "below the fenced weather station", "带白色H的浅灰色圆形混凝土停机坪", "围栏气象站下方的"),
    "DES_D_helipad_2": ("round light grey concrete helipad with a white H", "on the northern alluvial fan", "带白色H的浅灰色圆形混凝土停机坪", "北侧冲积扇上的"),
}
# Compass words removed from sites (11/05): (site_en, site_zh) after the change.
SITE_FIXES = {
    "DES_A_helipad_1": ("in a dune valley", "沙丘谷地中的"),
    "MTR_B_radar_dome_1": ("on a ridge shoulder", "山脊肩部的"),
    "MTR_B_solar_array_1": ("on a meadow shoulder", "草甸肩部的"),
    "MTR_B_container_yard_1": ("on a terrace above a valley", "山谷上方台地上的"),
    "MTR_B_helipad_2": ("on a ridge shoulder", "山脊肩部的"),
    "MTR_B_sports_field_1": ("on a terrace above a valley", "山谷上方台地上的"),
    "MTR_B_tent_camp_2": ("at the forest edge above a valley", "山谷上方林缘的"),
    "DES_B_helipad_1": ("on the plateau", "台地上的"),
    "DES_D_helipad_2": ("on an alluvial fan", "冲积扇上的"),
}
COMPASS_EN = re.compile(r"\b(north|south|east|west)", re.I)
COMPASS_ZH = re.compile("[东南西北]")
HEAD_BREAK = re.compile(r",| (with|without|of|on|in|at|by|next|beside|below|above|near|split|parked|stacked)\b")


def ue_xy(blender_xy):
    return [float(blender_xy[0]), -float(blender_xy[1])]


def plural_of(appearance_en):
    """The head noun is the last word before the first modifier: 'two small white houses with ...' -> houses."""
    head = HEAD_BREAK.split(appearance_en, maxsplit=1)[0].split()[-1]
    return head.endswith("s")


def split_text(target):
    """Appearance and site, with the fixes applied; returns (description_en, site_en, description_zh, site_zh, fixes, draft)."""
    if target["id"] in DRAFTS:
        en, site_en, zh, site_zh = DRAFTS[target["id"]]
        fixes = ["v2 target: placeholder text replaced by a draft (G0-5 passed 11/05)"]
        if target["id"] in SITE_FIXES:
            site_en, site_zh = SITE_FIXES[target["id"]]
            fixes.append("compass word removed from the site")
        return en, site_en, zh, site_zh, fixes, True
    full_en, site_en, full_zh, site_zh = target["description_en"], target["site_en"], target["description_zh"], target["site_zh"]
    if not full_en.endswith(" " + site_en) or not full_zh.startswith(site_zh):
        raise ValueError(f"{target['id']}: the description does not end with the site")
    en, zh, fixes = full_en[:-len(site_en)].rstrip(), full_zh[len(site_zh):], []
    if en.endswith(" by the road") and site_en.startswith("by the"):
        en = en[:-len(" by the road")]
        fixes.append("removed the repeated 'by the road'")
    if zh.startswith("路边") and site_zh.endswith("旁的"):
        zh = zh[len("路边"):]
        fixes.append("removed the repeated '路边'")
    if target["id"] in SITE_FIXES:
        site_en, site_zh = SITE_FIXES[target["id"]]
        fixes.append("compass word removed from the site")
    return en, site_en, zh, site_zh, fixes, False


def target_row(map_id, config_path, target):
    en, site_en, zh, site_zh, fixes, draft = split_text(target)
    w, l = target["footprint_m"]
    folder = config_path.parent
    return {"map": map_id, "key": target["id"], "cls": target["class"], "role": "target", "split": SPLIT[map_id[6]], "variant": target["variant"],
            "unseen_class": bool(target["unseen_class"]), "center": ue_xy(target["position"]), "base_z": float(target["position"][2]), "top_z": float(target["top_z"]),
            "height_m": float(target["height_m"]), "footprint_m": [float(w), float(l)], "width_m": round(math.sqrt(w * l), 2),
            "description_en": en, "site_en": site_en, "description_zh": zh, "site_zh": site_zh, "en": f"the {en} {site_en}", "zh": site_zh + zh, "en_short": f"the {en}", "zh_short": zh, "plural": plural_of(en),
            "drafted": draft, "fixes": fixes, "relief_300m_m": target.get("relief_300m_m"),
            "preview_top": preview(folder, target["preview_top"]), "preview_takeoff": preview(folder, target["preview_takeoff"])}


def preview(folder, relative):
    """Desert v3 previews sit in previews_v3/ although the configs say previews/."""
    for candidate in (folder / relative, folder / relative.replace("previews/", "previews_v3/", 1)):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(folder / relative)


def distractor_rows(map_id, config):
    rows = []
    for o in config.get("class_like_v2_objects", []):
        if o["counted"]:
            continue
        rows.append({"map": map_id, "key": "v2:" + o["name"], "cls": o["cls"], "role": "distractor", "split": SPLIT[map_id[6]], "center": ue_xy((o["x"], o["y"])),
                     "footprint_m": [float(o["w"]), float(o["l"])], "width_m": round(math.sqrt(o["w"] * o["l"]), 2), "source": "class_like_v2_objects, counted=false",
                     "reasons": o.get("reasons", [])})
    return rows


def house_rows(map_id, config_path):
    """Buildings that read as small houses from above. The desert v3 configs do not list v2 objects, so those come from the v2 config."""
    source = config_path.parent / "scene_config.json"
    objects = json.load(open(source, encoding="utf-8")).get("objects", [])
    return [{"map": map_id, "key": "house:" + o["name"], "cls": "small_houses", "role": "distractor", "split": SPLIT[map_id[6]], "center": ue_xy(o["location"]),
             "footprint_m": None, "width_m": HOUSE_WIDTH_M, "source": f"{source.name} building {o['name']} (decision of 11/05: houses count as small_houses)"}
            for o in objects if o.get("kind") == "building" and o["name"].startswith(HOUSE_PREFIXES)]


def build():
    rows = []
    for map_id, (_, config_path, _) in maps().items():
        config = json.load(open(config_path, encoding="utf-8"))
        rows += [target_row(map_id, config_path, t) for t in config["targets"]]
        rows += distractor_rows(map_id, config) + house_rows(map_id, config_path)
    targets = [r for r in rows if r["role"] == "target"]
    seen = {r["cls"] for r in targets if r["split"] == "train"}
    dev = {r["cls"] for r in targets if r["split"] == "dev"}
    for r in targets:  # the flag in the configs must agree with section 17.2
        expected = r["cls"] not in seen if r["split"] == "dev" else r["cls"] not in seen | dev if r["split"] == "test" else False
        if r["unseen_class"] != expected:
            raise ValueError(f"{r['key']}: unseen_class {r['unseen_class']} but the class table says {expected}")
    for r in targets:
        if COMPASS_EN.search(r["en"]) or COMPASS_ZH.search(r["zh"]):
            raise ValueError(f"{r['key']}: compass word in the name")
    keys = [(r["map"], r["key"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate keys")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="configs/v3s2_objects.json")
    args = p.parse_args()
    rows = build()
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    targets = [r for r in rows if r["role"] == "target"]
    print(f"{len(targets)} targets, {len(rows) - len(targets)} distractors; drafted {sum(r['drafted'] for r in targets)}, fixed {sum(bool(r['fixes']) for r in targets)}, plural {sum(r['plural'] for r in targets)}")


if __name__ == "__main__":
    main()
