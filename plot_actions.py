"""Plot five SR3M expert trajectories using the existing HAD adapter's actions.

Reads the published split and source captures. It never changes captures, split
files, weights, or training state. Printed checks cover all trajectory metadata;
PNG/PDF pages show five deterministic train examples selected by geometry only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT.parent / "SR3M"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "vln_benchmark"))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch

from vln_benchmark.dataset import iter_split
from vln_benchmark.unified.contract import adjacent_action, local_state, wrap
from vln_benchmark.unified.data import CaptureDataset
from vln_benchmark.unified.had import HadConfig, tensorize_payloads, validate_payload
from vln_benchmark.unified.tokenizer import ChineseTokenizer

SPLITS = ("train", "val_seen", "val_unseen", "test")
EXPECTED = dict(zip(SPLITS, (1027, 184, 199, 196)))
CATEGORIES = ("airfields", "integrated_base", "forests", "mountain", "desert")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quaternion_basis(quaternions):
    """Independent source-quaternion basis: columns are forward, right, up."""
    x, y, z, w = np.asarray(quaternions, dtype=np.float64).T
    return np.stack((
        np.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)), axis=-1),
        np.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), axis=-1),
        np.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), axis=-1),
    ), axis=-1)


def reconstruct(position, yaw, actions):
    recovered_yaw = yaw[0] + np.r_[0., np.cumsum(actions[:, 3])]
    c, s = np.cos(recovered_yaw[:-1]), np.sin(recovered_yaw[:-1])
    world_delta = np.column_stack((c*actions[:, 0]-s*actions[:, 1],
                                   s*actions[:, 0]+c*actions[:, 1], actions[:, 2]))
    recovered = position[0] + np.vstack((np.zeros(3), np.cumsum(world_delta, axis=0)))
    return recovered, recovered_yaw, world_delta


def collect(root):
    indices = root / "benchmark_split_v1"
    rows, selected, instructions = [], {}, defaultdict(list)
    split_info, language, latin = {}, Counter(), Counter()
    maximum = Counter()
    cameras = Counter()
    wrapped_transitions = 0
    for split in SPLITS:
        count = frames = 0
        for record, episode in iter_split(root, indices, split):
            rows.append(record)
            count += 1
            steps = episode["steps"]
            frames += len(steps)
            text = episode["instruction"]
            require(text == record["instruction"], "Instruction differs from frozen index")
            require(episode.get("source_plan", {}).get("instruction", text) == text,
                    "Source-plan instruction differs from capture")
            require(all(abs(step["rotation_deg"][key]) < 1e-6
                        for step in steps for key in ("roll", "pitch")), "Nonzero roll/pitch")
            source_ids = [step.get("source_pose_index") for step in steps]
            require(all(a is None or b is None or b == a+1
                        for a, b in zip(source_ids, source_ids[1:])), "Nonadjacent capture indices")
            instructions[split].append(text)
            language["with_han"] += bool(re.search(r"[\u3400-\u9fff]", text))
            language["with_ascii_letters"] += bool(re.search(r"[A-Za-z]", text))
            language["without_han"] += not bool(re.search(r"[\u3400-\u9fff]", text))
            language["empty"] += not bool(text.strip())
            latin.update(re.findall(r"[A-Za-z]+(?:[0-9_]+[A-Za-z0-9_]*)?", text))
            p = np.array([[step["position_m"][k] for k in "xyz"] for step in steps])
            yaw = np.radians([step["rotation_deg"]["yaw"] for step in steps])
            q = np.array([[step["quaternion"][k] for k in "xyzw"] for step in steps])
            basis = quaternion_basis(q)
            require(np.allclose(basis[:, :, 0], np.column_stack((np.cos(yaw), np.sin(yaw),
                        np.zeros(len(yaw)))), atol=1e-8), "Quaternion forward axis disagrees with yaw")
            actions = np.array([adjacent_action(a, b, c, d)
                                for a, b, c, d in zip(p, yaw, p[1:], yaw[1:])])
            independent = np.einsum("nwi,nw->ni", basis[:-1], np.diff(p, axis=0))
            error = float(np.max(np.abs(actions[:, :3]-independent)))
            maximum["quaternion_projection_error_m"] = max(maximum["quaternion_projection_error_m"], error)
            require(error < 1e-7, "Action disagrees with independent quaternion projection")
            recovered, recovered_yaw, world_delta = reconstruct(p, yaw, actions)
            pe = float(np.linalg.norm(recovered-p, axis=1).max())
            ye = float(np.abs(np.arctan2(np.sin(recovered_yaw-yaw), np.cos(recovered_yaw-yaw))).max())
            maximum["integrated_position_error_m"] = max(maximum["integrated_position_error_m"], pe)
            maximum["integrated_yaw_error_rad"] = max(maximum["integrated_yaw_error_rad"], ye)
            require(pe < 1e-7 and ye < 1e-8, "Integrated actions do not recover source trajectory")
            crossings = int(np.sum(np.abs(np.diff(yaw)) > math.pi))
            wrapped_transitions += crossings
            for step in steps:
                camera_record = step.get("cameras") or {}
                if not camera_record:
                    cameras["frames_without_calibration"] += 1
                    continue
                cameras["frames_with_calibration"] += 1
                for view, pitch in (("front", 0.), ("down", -90.)):
                    camera = camera_record.get(view)
                    require(camera is not None, "Incomplete recorded front/down calibration")
                    rotation = camera["rotation_deg"]
                    delta = abs(math.degrees(wrap(math.radians(rotation["yaw"]-step["rotation_deg"]["yaw"]))))
                    maximum["camera_yaw_difference_deg"] = max(maximum["camera_yaw_difference_deg"], delta)
                    require(abs(rotation["pitch"]-pitch) < .001 and abs(rotation["roll"]) < .001
                            and delta < .001, "Recorded camera orientation disagrees with view label")
                    cp = np.array(camera["position_ue_cm"])/100
                    up = np.array([step["position_m"][k] for k in "xyz"])
                    ce = float(np.linalg.norm(cp-up))
                    maximum["camera_position_difference_m"] = max(maximum["camera_position_difference_m"], ce)
                    require(ce < .01, "Recorded camera and UAV positions disagree")
            if split == "train" and record["category"] in CATEGORIES:
                category = record["category"]
                turn = float(np.abs(actions[:, 3]).sum())
                score = (float(np.ptp(p[:, 2])), turn) if category == "mountain" else (crossings, turn)
                if category not in selected or score > selected[category]["score"]:
                    selected[category] = dict(record=record, p=p, yaw=yaw, actions=actions,
                        recovered=recovered, recovered_yaw=recovered_yaw, world_delta=world_delta,
                        score=score, wrap_crossings=crossings, max_position_error_m=pe)
        require(count == EXPECTED[split], f"Frozen {split} count changed")
        split_info[split] = dict(routes=count, frames=frames, transitions=frames-count)
    groups = defaultdict(set)
    ids = set()
    for r in rows:
        require(r["episode_id"] not in ids, "Episode appears in multiple splits")
        ids.add(r["episode_id"])
        groups[r["route_group_id"]].add(r["split"])
    require(all(len(s) == 1 for s in groups.values()), "Related route group crosses splits")
    tokenizer = ChineseTokenizer.load(SOURCE / "vln_benchmark/unified/artifacts/prepared_v2/tokenizer.json")
    fitted = ChineseTokenizer.fit(instructions["train"], max_length=tokenizer.max_length)
    require(fitted.vocab == tokenizer.vocab, "Prepared vocabulary differs from frozen train")
    summary = dict(splits=split_info, unique_routes=len(ids), route_groups=len(groups),
        language=dict(language), latin_tokens=latin.most_common(20),
        tokenizer={s:tokenizer.describe(instructions[s]) for s in SPLITS},
        vocab_size=tokenizer.vocab_size, max_tokens=tokenizer.max_length,
        geometry_maxima=dict(maximum), cameras=dict(cameras), yaw_wrap_transitions=wrapped_transitions)
    return [selected[c] for c in CATEGORIES], summary, tokenizer


def check_actual_input(root, chosen, tokenizer):
    selected = {r["record"]["episode_id"]:r for r in chosen}
    dataset = CaptureDataset(root, root/"benchmark_split_v1", "train", "dual_relative_v1",
                             selected=set(selected), include_endpoints=True)
    representatives = []
    checked = endpoints = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        record = selected[sample.metadata["episode_id"]]
        t = sample.metadata["t"]
        if sample.target.endpoint:
            endpoints += 1
            require(sample.target.action_m_rad is None and not sample.target.transition_valid,
                    "Endpoint incorrectly has a synthetic action")
        else:
            require(np.allclose(sample.target.action_m_rad, record["actions"][t], atol=1e-12),
                    "Actual dataloader target differs from plotted action")
        state, angle = local_state(record["p"][t], record["yaw"][t], record["p"][0], record["yaw"][0])
        payload = sample.observation.for_profile("dual_relative_v1")
        require(np.allclose(payload["position_start_m"], state, atol=1e-10)
                and abs(wrap(payload["yaw_start_rad"]-angle)) < 1e-10, "State assembly mismatch")
        require(set(payload) == {"instruction", "images", "position_start_m", "yaw_start_rad"},
                "Unexpected supervision in model input")
        checked += 1
        if t in (0, len(record["p"])-2):
            representatives.append(payload)
    torch.set_num_threads(2)
    config = HadConfig(profile="dual_relative_v1")
    batch = tensorize_payloads(representatives, tokenizer, config)
    require(batch["front"].shape == batch["down"].shape == (10, 3, 224, 224), "Image shape mismatch")
    require(batch["tokens"].shape == (10, 256) and batch["state"].shape == (10, 5), "Text/state shape mismatch")
    for i, payload in enumerate(representatives):
        expected = [v/100 for v in payload["position_start_m"]] + [math.sin(payload["yaw_start_rad"]), math.cos(payload["yaw_start_rad"])]
        require(np.allclose(batch["state"][i].numpy(), expected, atol=1e-6), "State tensor normalization mismatch")
        for view in ("front", "down"):
            with Image.open(payload["images"][view]) as image:
                pixels = np.asarray(image.convert("RGB").resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)/255
            expected_image = ((pixels-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)).transpose(2,0,1)
            require(np.allclose(batch[view][i].numpy(), expected_image, atol=2e-6), "Image channels/normalization mismatch")
    front_config = HadConfig(profile="front_relative_v1")
    front_payloads = [{**p,"images":{"front":p["images"]["front"]}} for p in representatives]
    front_batch = tensorize_payloads(front_payloads,tokenizer,front_config)
    require("down" not in front_batch and torch.equal(front_batch["front"],batch["front"]), "Front input differs between profiles")
    rejected = 0
    for key in ("target", "future_pose", "endpoint", "route_id"):
        try:
            validate_payload({**representatives[0],key:None},config)
        except ValueError:
            rejected += 1
    require(rejected == 4, "Supervision field accepted as model input")
    return dict(actual_dataloader_samples_checked=checked, endpoints_checked=endpoints,
                tensorized_observations=len(representatives), decoded_images=20,
                image_shape=list(batch["front"].shape), state_shape=list(batch["state"].shape),
                tokens_shape=list(batch["tokens"].shape), forbidden_fields_rejected=rejected)


def draw_xy(ax, item):
    p, recovered = item["p"]-item["p"][0], item["recovered"]-item["p"][0]
    ax.plot(p[:,0],p[:,1],color="#2363aa",lw=2,label="Recorded trajectory")
    ax.plot(recovered[:,0],recovered[:,1],"--",color="#e28b20",lw=1.4,label="Integrated actions")
    indices = np.unique(np.linspace(0,len(p)-2,min(18,len(p)-1)).astype(int))
    delta = item["world_delta"][indices]
    ax.quiver(p[indices,0],p[indices,1],delta[:,0],delta[:,1],angles="xy",scale_units="xy",scale=1,color="#21262d",width=.004)
    ax.scatter(*p[0,:2],s=32,color="#29966f",zorder=4,label="Start")
    ax.scatter(*p[-1,:2],s=35,color="#c24646",marker="x",zorder=4,label="End")
    ax.set(xlabel="World X - start X (m)",ylabel="World Y - start Y (m)")
    ax.set_aspect("equal",adjustable="datalim")


def draw_height(ax,item):
    ax.plot(item["p"][:,2]-item["p"][0,2],color="#2363aa",lw=2,label="Recorded Z")
    ax.plot(item["recovered"][:,2]-item["p"][0,2],"--",color="#e28b20",lw=1.4,label="Integrated dz")
    ax.set(xlabel="Observation index",ylabel="World Z - start Z (m)")
    if np.ptp(item["p"][:,2]) < 1e-6:
        ax.set_ylim(-.1,.1)
        ax.text(.03,.08,"Constant height (variation < 1e-6 m)",transform=ax.transAxes,fontsize=8)


def draw_actions(ax,item,include_yaw=False):
    for column,label,color in ((0,"dx / forward","#2363aa"),(1,"dy / right","#29966f"),(2,"dz / up","#9a62a8")):
        ax.plot(item["actions"][:,column],label=label,color=color,lw=1.2)
    ax.set(xlabel="Transition index t -> t+1",ylabel="Local displacement (m)")
    if include_yaw:
        right=ax.twinx()
        right.plot(np.degrees(item["actions"][:,3]),color="#d37820",lw=1.,alpha=.8,label="dyaw (deg)")
        right.set_ylabel("dyaw (deg)",color="#b56c1d")
        right.tick_params(axis="y",labelcolor="#b56c1d")
    return ax


def render(chosen,out):
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    overview,axes=plt.subplots(5,3,figsize=(18,17),layout="constrained")
    for row,item in enumerate(chosen):
        draw_xy(axes[row,0],item);draw_height(axes[row,1],item);draw_actions(axes[row,2],item,True)
        axes[row,0].set_title(f"{row+1}. {item['record']['category']} | {len(item['p'])-1} actions",loc="left",fontweight="bold")
        axes[row,1].set_title(f"Position reconstruction max error: {item['max_position_error_m']:.2e} m",loc="left")
        axes[row,2].set_title(f"Local actions | {item['wrap_crossings']} yaw wrap crossings",loc="left")
        for ax in axes[row]:ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8);axes[0,2].legend(fontsize=8,loc="lower left")
    overview.suptitle("SR3M -> HAD: expert actions reconstruct all five source trajectories\nBlue: recorded poses; dashed orange: integrated actions. These are labels, not model predictions.",fontsize=15)
    overview.savefig(out/"actions_5_trajectories.png",dpi=180)
    plt.close(overview)
    with PdfPages(out/"actions_5_trajectories.pdf") as pdf:
        for number,item in enumerate(chosen,1):
            fig,ax=plt.subplots(2,2,figsize=(13,10),layout="constrained")
            draw_xy(ax[0,0],item);draw_height(ax[0,1],item);draw_actions(ax[1,0],item)
            ax[1,1].plot(np.degrees(np.unwrap(item["yaw"])),color="#2363aa",lw=2,label="Recorded yaw (unwrapped)")
            ax[1,1].plot(np.degrees(item["recovered_yaw"]),"--",color="#e28b20",lw=1.4,label="Integrated dyaw")
            ax[1,1].set(xlabel="Observation index",ylabel="Yaw (deg, unwrapped)")
            side=ax[1,1].twinx();side.plot(np.degrees(item["actions"][:,3]),color="#9a62a8",alpha=.6,lw=1)
            side.set_ylabel("Action dyaw (deg)",color="#9a62a8")
            for pane in ax.flat:pane.grid(alpha=.18);pane.legend(fontsize=8)
            route=item["record"]["route_id"]
            fig.suptitle(f"{number}. {item['record']['category']} | {route}\nExpert labels, current-yaw forward/right/up, m/rad; train split",fontsize=12)
            pdf.savefig(fig);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("/mnt/f/WSL_ColdStorage/datasets/DatasetCaptureBatches/workbench"))
    parser.add_argument("--out",type=Path,default=PROJECT/"figures")
    args=parser.parse_args()
    chosen,summary,tokenizer=collect(args.root)
    summary["actual_input"]=check_actual_input(args.root,chosen,tokenizer)
    summary["selected"]=[dict(episode_id=r["record"]["episode_id"],category=r["record"]["category"],
        transitions=len(r["actions"]),wrap_crossings=r["wrap_crossings"],
        vertical_range_m=float(np.ptp(r["p"][:,2])),max_position_error_m=r["max_position_error_m"]) for r in chosen]
    render(chosen,args.out)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
