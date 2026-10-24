"""Plot five SR3M expert trajectories using the existing HAD adapter's actions.

Reads the published split and source captures. It never changes captures, split
files, weights, or training state. Printed checks cover all trajectory metadata;
PNG/PDF pages show five deterministic train examples selected by geometry only.

Source: UE left-handed world, +Z up; engine cm, captured position_m meters,
captured yaw degrees (+X toward +Y). Actions: yaw-only forward/right/up,
meters and radians. AirSim/Cosys API uses NED meters and yaw angles in degrees.
The short final geometric transition is not a STOP rule; STOP is independently
labelled on the terminal 10m suffix and those actions are masked in training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sr3m.data import INDEX_DIR, iter_split, read_capture, split_hashes
from sr3m.geometry import adjacent_action, local_state, wrap, fru_to_world, check_climb
from experiments.protocol import PROJECT, machine_paths, stop_suffix, PreparedSplit, read_json
from had.model import image_transform
from had.tokenizer import Tokenizer

SPLITS = ("train", "val_seen", "val_unseen", "test")
EXPECTED = dict(zip(SPLITS, (1028, 184, 247, 147)))  # raw benchmark_split_v2
CATEGORIES = ("airfields", "integrated_base", "forests", "mountain", "prairie")


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
    world_delta = fru_to_world(actions[:, :3], recovered_yaw[:-1])
    recovered = position[0] + np.vstack((np.zeros(3), np.cumsum(world_delta, axis=0)))
    return recovered, recovered_yaw, world_delta


def collect_prepared(root, prepared):
    """Check the five selected raw captures, without reopening every published route."""
    ds=PreparedSplit(prepared,"train")
    index=ds.protocol["protocol"]["index"]
    require(split_hashes(root/index)==ds.protocol["source_split"],"Published indices changed")
    records={r["episode_id"]:r for r in map(json.loads,(root/index/"train.jsonl").read_text().splitlines())}
    chosen=[];checked=0
    for category in CATEGORIES:
        candidates=[]
        for r in ds.routes:
            if r["category"]!=category: continue
            sl=slice(r["offset"],r["offset"]+r["count"]);yaw=ds.arrays["yaw"][sl]
            turn=np.abs(np.arctan2(np.sin(np.diff(yaw)),np.cos(np.diff(yaw)))).sum()
            crossings=int((np.abs(np.diff(yaw))>np.pi).sum())
            score=(np.ptp(ds.arrays["position"][sl,2]),turn) if category=="mountain" else (crossings,turn)
            candidates.append((score,r))
        _,r=max(candidates,key=lambda v:v[0]);record=records[r["episode_id"]]
        episode,digest=read_capture(root/record["trajectory_relpath"])
        require(digest==record["trajectory_sha256"],"Selected raw capture changed")
        steps=episode["steps"];p=np.array([[s["position_m"][k] for k in "xyz"] for s in steps]);yaw=np.radians([s["rotation_deg"]["yaw"] for s in steps])
        q=np.array([[s["quaternion"][k] for k in "xyzw"] for s in steps]);basis=quaternion_basis(q)
        actions=np.array([adjacent_action(p[t],yaw[t],p[t+1],yaw[t+1]) for t in range(len(p)-1)])
        independent=np.einsum("nwi,nw->ni",basis[:-1],np.diff(p,axis=0))
        require(np.allclose(actions[:,:3],independent,atol=1e-8),"Independent quaternion projection mismatch")
        recovered,recovered_yaw,delta=reconstruct(p,yaw,actions)
        pe=float(np.linalg.norm(recovered-p,axis=1).max())
        require(pe<1e-7 and np.max(np.abs(np.arctan2(np.sin(recovered_yaw-yaw),np.cos(recovered_yaw-yaw))))<1e-8,"Source reconstruction mismatch")
        start=r["offset"];sl=slice(start,start+r["count"])
        require(np.allclose(ds.arrays["action"][sl][:-1],actions,atol=1e-6),"Prepared actions differ")
        require(np.array_equal(ds.arrays["stop"][sl],stop_suffix(p)[0]),"Prepared STOP suffix differs")
        for t in (0,len(p)-2,len(p)-1):
            payload=ds.payload(start+t,("front","down"),root)
            state,angle=local_state(p[t],yaw[t],p[0],yaw[0])
            expected=[*(np.asarray(state)/100),math.sin(angle),math.cos(angle)]
            require(set(payload)=={"instruction","images","state"} and np.allclose(payload["state"],expected,atol=1e-6),"Actual policy input mismatch")
            for path in payload["images"].values():
                with Image.open(path) as image: require(image_transform()(image.convert("RGB")).shape==(3,224,224),"Bad image tensor")
            checked+=1
        chosen.append(dict(record=record,p=p,yaw=yaw,actions=actions,recovered=recovered,recovered_yaw=recovered_yaw,
                           world_delta=delta,wrap_crossings=int((np.abs(np.diff(yaw))>np.pi).sum()),max_position_error_m=pe,stop_target=ds.arrays["stop"][sl]))
    climb=chosen[CATEGORIES.index("mountain")]["actions"]
    check_climb(climb[climb[:,2]>1e-6][0])
    summary={"scope":"five_geometry_selected_train_routes; full split/terrain preparation already completed",
             "prepared":read_json(Path(prepared)/"preparation.json"),"actual_input":{"observations":checked,"images":2*checked,"state_dim":5}}
    return chosen,summary,Tokenizer.load(Path(prepared)/"tokenizer.json")


def collect(root):
    indices = root / INDEX_DIR
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
                        score=score, wrap_crossings=crossings, max_position_error_m=pe, stop_target=stop_suffix(p)[0])
        require(count == EXPECTED[split], f"Frozen {split} count changed")
        split_info[split] = dict(routes=count, frames=frames, transitions=frames-count)
    groups = defaultdict(set)
    ids = set()
    for r in rows:
        require(r["episode_id"] not in ids, "Episode appears in multiple splits")
        ids.add(r["episode_id"])
        groups[r["route_group_id"]].add(r["split"])
    require(all(len(s) == 1 for s in groups.values()), "Related route group crosses splits")
    tokenizer = Tokenizer.fit(instructions["train"], max_length=256)
    climb = selected["mountain"]["actions"]
    positive = climb[climb[:, 2] > 1e-6]
    require(len(positive) > 0, "Selected mountain route contains no climb for sign check")
    check_climb(positive[0])
    summary = dict(splits=split_info, unique_routes=len(ids), route_groups=len(groups),
        language=dict(language), latin_tokens=latin.most_common(20),
        tokenizer={s:tokenizer.describe(instructions[s]) for s in SPLITS},
        vocab_size=tokenizer.vocab_size, max_tokens=tokenizer.max_length,
        geometry_maxima=dict(maximum), cameras=dict(cameras), yaw_wrap_transitions=wrapped_transitions)
    return [selected[c] for c in CATEGORIES], summary, tokenizer


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
    start = int(np.flatnonzero(item["stop_target"])[0])
    ax.axvspan(start, len(item["actions"]), alpha=.10, color="#ce5749", label="STOP suffix: action loss masked")
    ax.axvline(len(item["actions"]), color="#ce5749", ls=":", lw=1, label="Final observation: STOP, no action")
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
    paths=machine_paths()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=paths["data_root"])
    parser.add_argument("--out",type=Path,default=PROJECT/"figures")
    parser.add_argument("--prepared",type=Path,default=paths["out_dir"]/"prepared")
    parser.add_argument("--full-audit",action="store_true",help="Reopen all source trajectories, including historical test metadata")
    args=parser.parse_args()
    if args.full_audit:
        chosen,summary,tokenizer=collect(args.root)
    else:
        chosen,summary,tokenizer=collect_prepared(args.root,args.prepared)
    summary["selected"]=[dict(episode_id=r["record"]["episode_id"],category=r["record"]["category"],
        transitions=len(r["actions"]),wrap_crossings=r["wrap_crossings"],
        vertical_range_m=float(np.ptp(r["p"][:,2])),max_position_error_m=r["max_position_error_m"]) for r in chosen]
    render(chosen,args.out)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
