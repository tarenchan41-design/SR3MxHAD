"""Run the authorized offline experiment to completion, resuming existing stages."""
import argparse
import fcntl
from pathlib import Path
import os
import subprocess
import sys
import time

from experiments.protocol import PROJECT, PROTOCOL, machine_paths, read_json, write_json


def run(args):
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=(out/".pipeline.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepared=out/"prepared"; cache=Path(args.cache).resolve()
    status={"pid":os.getpid(),"started":time.strftime("%Y-%m-%d %H:%M:%S %z"),"status":"running"}
    environment=dict(os.environ,OPENBLAS_NUM_THREADS="4",OMP_NUM_THREADS="4",PYTHONDONTWRITEBYTECODE="1")
    def stage(name,module,arguments):
        status.update(stage=name,status="running"); write_json(out/"pipeline.json",status)
        print(f"stage {name}",flush=True)
        with (out/f"{name}.log").open("a",encoding="utf8") as log:
            subprocess.run([sys.executable,"-u","-B","-m",f"experiments.{module}",*map(str,arguments)],cwd=PROJECT,
                           env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        if not (prepared/"protocol.json").exists():
            stage("prepare","prepare",["--root",args.root,"--terrain",args.terrain,"--desert-terrain",args.desert_terrain,"--out",prepared])
        elif read_json(prepared/"protocol.json")["protocol"]!=PROTOCOL:
            raise ValueError("Existing preparation uses another protocol")
        stage("overfit","overfit",["--prepared",prepared,"--root",args.root,"--out",out/"overfit","--device",args.device])
        stage("cache","cache",["--prepared",prepared,"--root",args.root,"--out",cache,"--device",args.device])
        stage("train","train",["--prepared",prepared,"--cache",cache,"--out",out/"models","--device",args.device])
        stage("evaluate","evaluate",["--prepared",prepared,"--cache",cache,"--runs",out/"models", "--out",out/"val_unseen","--device",args.device])
        stage("probe","probe",["--prepared",prepared,"--cache",cache,"--out",out/"probe"])
        stage("summarize","summarize",["--results",out/"val_unseen","--probe",out/"probe","--out",out/"decision"])
        status.update(status="complete",stage="complete",report=str(out/"decision/decision.md"))
    except Exception as error:
        status.update(status="failed",error=f"{type(error).__name__}: {error}")
        write_json(out/"pipeline.json",status); raise
    write_json(out/"pipeline.json",status)


if __name__=="__main__":
    paths=machine_paths()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default=str(paths["data_root"]));p.add_argument("--terrain",default=str(paths["terrain_dir"]))
    p.add_argument("--desert-terrain",default=str(paths["desert_terrain_dir"]))
    p.add_argument("--out",default=str(paths["out_dir"]));p.add_argument("--cache",default=str(paths["cache_dir"]))
    p.add_argument("--device",default="cuda");run(p.parse_args())
