"""Run the authorized offline experiment to completion, resuming existing stages."""
import argparse
import fcntl
from pathlib import Path
import os
import subprocess
import sys
import time

from experiment import PROTOCOL, read_json, write_json
from source import PROJECT


def run(args):
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=(out/".pipeline.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepared=out/"prepared"; cache=Path(args.cache).resolve()
    status={"pid":os.getpid(),"started":time.strftime("%Y-%m-%d %H:%M:%S %z"),"status":"running"}
    environment=dict(os.environ,OPENBLAS_NUM_THREADS="4",OMP_NUM_THREADS="4",PYTHONDONTWRITEBYTECODE="1")
    def stage(name,script,arguments):
        status.update(stage=name,status="running"); write_json(out/"pipeline.json",status)
        print(f"stage {name}",flush=True)
        with (out/f"{name}.log").open("a",encoding="utf8") as log:
            subprocess.run([sys.executable,"-u","-B",str(PROJECT/script),*map(str,arguments)],cwd=PROJECT,
                           env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        if not (prepared/"protocol.json").exists():
            stage("prepare","prepare.py",["--root",args.root,"--terrain",args.terrain,"--out",prepared])
        elif read_json(prepared/"protocol.json")["protocol"]!=PROTOCOL:
            raise ValueError("Existing preparation uses another protocol")
        stage("overfit","overfit.py",["--prepared",prepared,"--root",args.root,"--out",out/"overfit","--device",args.device])
        stage("cache","feature_cache.py",["--prepared",prepared,"--root",args.root,"--out",cache,"--device",args.device])
        stage("train","train.py",["--prepared",prepared,"--cache",cache,"--out",out/"models","--device",args.device])
        stage("evaluate","evaluate.py",["--prepared",prepared,"--cache",cache,"--runs",out/"models", "--out",out/"val_unseen","--device",args.device])
        stage("probe","probe.py",["--prepared",prepared,"--cache",cache,"--out",out/"probe"])
        stage("summarize","summarize.py",["--results",out/"val_unseen","--probe",out/"probe","--out",out/"decision"])
        status.update(status="complete",stage="complete",report=str(out/"decision/decision.md"))
    except Exception as error:
        status.update(status="failed",error=f"{type(error).__name__}: {error}")
        write_json(out/"pipeline.json",status); raise
    write_json(out/"pipeline.json",status)


if __name__=="__main__":
    from experiment import DEFAULT_ROOT,DEFAULT_TERRAIN
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default=str(DEFAULT_ROOT));p.add_argument("--terrain",default=str(DEFAULT_TERRAIN))
    p.add_argument("--out",default="runs/terrain_v1");p.add_argument("--cache",default="/home/qlj/sr3mxhad_runs/terrain_v1/features")
    p.add_argument("--device",default="cuda");run(p.parse_args())
