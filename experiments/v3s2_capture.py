"""Bulk render of dataset v3 stage 2 (protocol sections 14 and 17.5) with K's backend, after G1 and the preview check passed.

Per map, the plan's candidates in order: the UE checks of section 14 (start-to-target sight line, swept collision of every step,
UE ground clearance >= 2 m); a route that fails is skipped with its reason and the next one is taken, until the map's quota is met.
Every step is rendered (front and down, RGB) in the layout of the stage 1 pilot, so the stage 1 preparation code can read it.
Splits: training maps by accepted rank, every tenth to val (section 14); development maps "dev"; test maps "test".
Settings: STAGE2_CONSOLE (VSM off, shadow bias 20 / 4.5), the memory restart policy (commit 80 %, at most 512 renders per process),
the capture settings of the import check; the same as the capture side of G1 except that memory is read every 16 renders, not
after each (11/05: the reading cost 0.17-0.4 s a render here; restart timing does not change pixels, which G1 showed). Resumes per route. Close the GUI editor first. Writes only under --out.
"""
import argparse
import shutil
from pathlib import Path

from experiments.protocol import code_version, read_json, write_json
from experiments.v3_capture import check, render_route
from experiments.v3_ue_check import CAPTURE, PROFILE, STAGE2_CONSOLE, maps
from experiments.v3s2_stage0 import DEV, TEST, TRAIN

VAL_EVERY = 10
MEMORY_CHECK_EVERY = 16


def split_of(map_id, accepted_rank):
    if map_id in DEV:
        return "dev"
    if map_id in TEST:
        return "test"
    return "val" if accepted_rank % VAL_EVERY == VAL_EVERY - 1 else "train"


def render(args):
    from sim_k.backend import MemoryPolicy, UEBackend
    import numpy as np
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    plan_copy = root / "plan.json"
    if not plan_copy.exists():
        shutil.copy(args.plan, plan_copy)  # the data folder carries the plan it was rendered from
    planned = read_json(plan_copy)
    code = code_version()
    if code.get("dirty"):
        raise RuntimeError("Commit first: a render is tied to a clean commit")
    for map_id in args.maps:
        entry = planned["maps"][map_id]
        quota = entry["quota"]
        target = root / "routes" / map_id
        target.mkdir(parents=True, exist_ok=True)
        log_path = target / "checks.json"
        log = read_json(log_path) if log_path.exists() else {}
        accepted = [k for k, v in log.items() if v == "accepted"]
        if len(accepted) >= quota:
            print(map_id, "complete", flush=True)
            continue
        with UEBackend(target / "session", PROFILE, maps()[map_id][0], CAPTURE, policy=MemoryPolicy(check_every=MEMORY_CHECK_EVERY), console=STAGE2_CONSOLE) as backend:
            for route in entry["candidates"]:
                if len(accepted) >= quota:
                    break
                if route["id"] in log and log[route["id"]] != "accepted":
                    continue
                if route["id"] in accepted and (target / route["id"] / "route.json").exists():
                    continue
                arrays = {**route, "positions": np.asarray(route["positions"], dtype=float), "yaws": np.asarray(route["yaws"], dtype=float)}
                if route["id"] not in log:
                    reason = check(backend, arrays)
                    if reason:
                        log[route["id"]] = reason
                        write_json(log_path, log)
                        print(route["id"], "skipped:", reason, flush=True)
                        continue
                folder = target / route["id"]
                rank = accepted.index(route["id"]) if route["id"] in accepted else len(accepted)
                steps = list(range(len(route["positions"])))
                images = render_route(backend, arrays, folder, steps)
                write_json(folder / "route.json", {**route, "split": split_of(map_id, rank), "rank": rank, "rendered_steps": steps, "images": images, "map_package": maps()[map_id][0],
                                                   "profile": PROFILE, "capture": CAPTURE, "console": STAGE2_CONSOLE, "plan_seed": planned["seed"], "design": planned["design"], "code": code})
                if route["id"] not in accepted:
                    accepted.append(route["id"])
                log[route["id"]] = "accepted"
                write_json(log_path, log)
                print(route["id"], "rendered", len(accepted), "/", quota, flush=True)
            exits = backend.unexpected_exits
        write_json(target / "session_summary.json", {"accepted": len(accepted), "renders": backend.renders, "starts": backend.starts, "unexpected_exits": exits, "code": code})
        print(map_id, "accepted", len(accepted), "renders", backend.renders, "starts", backend.starts, "unexpected exits", exits, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", default="runs/v3/stage2/plan.json")
    p.add_argument("--out", default="/mnt/f/WSL_ColdStorage/datasets/sr3m_v3")
    p.add_argument("--maps", nargs="+", default=list(TRAIN + DEV + TEST))
    render(p.parse_args())
