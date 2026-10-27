"""Ten-route engineering gate; no pilot weights are reused by formal runs."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from experiments.protocol import PROTOCOL, PreparedSplit, read_json, write_json
from experiments.cache import Images
from had.model import FEATURE_DIM, TerrainHAD
from had.tokenizer import Tokenizer
from experiments.train import batch_inputs, loss_for_batch, predict, token_rows


def subset(ds):
    selected = []
    for layer in ("low", "medium", "high"):
        for follow in (False, True):
            selected.append(next(r for r in ds.routes if r["category"] == "mountain" and r["height_layer"] == layer and r["terrain_following"] == follow))
    for category in ("airfields", "forests", "industry", "prairie"):
        selected.append(next(r for r in ds.routes if r["category"] == category))
    indices = np.concatenate([np.arange(r["offset"],r["offset"]+r["count"]) for r in selected])
    pilot = copy.copy(ds); pilot.routes = copy.deepcopy(selected)
    offset = 0
    for r in pilot.routes: r["offset"], offset = offset, offset+r["count"]
    pilot.arrays = {k:v[indices] for k,v in ds.arrays.items()}; pilot.n = len(indices)
    pilot.route_index = np.repeat(np.arange(10),[r["count"] for r in pilot.routes])
    return pilot


def run(args):
    torch.set_num_threads(4); torch.manual_seed(7)
    ds = subset(PreparedSplit(args.prepared,"train")); out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    identity={"prepared":ds.protocol,"episodes":[r["episode_id"] for r in ds.routes]}
    if (out/"overfit.json").exists():
        old=read_json(out/"overfit.json")
        if old.get("identity")!=identity: raise ValueError("Overfit result belongs to different data")
        if old.get("passed"):
            print("ten-route overfit already passed",flush=True); return
    tokenizer = Tokenizer.load(Path(args.prepared)/"tokenizer.json"); tokens = token_rows(ds,tokenizer)
    model = TerrainHAD("B",tokenizer).to(args.device)
    feature_path = out/"pilot_features.npz"
    if feature_path.exists():
        with np.load(feature_path,allow_pickle=False) as z:
            if json.loads(str(z["identity"]))!=identity: raise ValueError("Pilot feature identity differs")
            features = {v:z[v] for v in ("front","down")}
    else:
        # Stored like the formal cache (PROTOCOL feature_dtype), so the gate trains on what formal runs see.
        features = {v:np.empty((ds.n,FEATURE_DIM),dtype=PROTOCOL["feature_dtype"]) for v in ("front","down")}; cursor=0
        loader = DataLoader(Images(ds,args.root),batch_size=32,num_workers=4,prefetch_factor=1)
        with torch.inference_mode():
            for images in loader:
                n=len(images); values=model.backbone(images.flatten(0,1).to(args.device)).reshape(n,2,FEATURE_DIM).cpu().numpy()
                for j,v in enumerate(("front","down")): features[v][cursor:cursor+n]=values[:,j]
                cursor+=n
        np.savez(feature_path,**features,identity=np.array(json.dumps(identity)))
    model.eval()
    one=Images(ds,args.root)[0]
    with torch.inference_mode():
        cached=model(*batch_inputs(ds,features,tokens,np.array([0]),model.views,args.device))
        state0=torch.from_numpy(ds.arrays["state"][0:1]).to(args.device)
        direct=model.from_images({v:one[j:j+1].to(args.device) for j,v in enumerate(model.views)},tokens[0:1].to(args.device),state0)
        # Same RGB path, features rounded to the cache dtype: must match the cached path.
        rounded={v:model.backbone(one[j:j+1].to(args.device)).to(getattr(torch,PROTOCOL["feature_dtype"])).float() for j,v in enumerate(model.views)}
        stored=model(rounded,tokens[0:1].to(args.device),state0)
        cache_error=max(float((cached[k]-stored[k]).abs().max()) for k in cached)
        storage_error=max(float((cached[k]-direct[k]).abs().max()) for k in cached)
    # fp32 batch-size noise (~3e-6 at the output) flips float16 rounding of boundary elements,
    # which alone gives ~3e-5 (measured 2026-10-27); a pipeline error would be orders larger.
    if cache_error>1e-4: raise ValueError(f"RGB/cache prediction mismatch: {cache_error}")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-3,weight_decay=0.)
    statistics=read_json(Path(args.prepared)/"statistics.json"); rng=np.random.default_rng(7); started=time.monotonic(); result={}
    # Regularization is disabled only for the memorization check, never for A/B/C.
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout): m.p=0.
    for update in range(1,args.max_updates+1):
        model.train(); ids=rng.choice(ds.n,32,replace=False); optimizer.zero_grad(set_to_none=True)
        output=model(*batch_inputs(ds,features,tokens,ids,model.views,args.device))
        loss=loss_for_batch(output,torch.from_numpy(ds.arrays["action"][ids]).to(args.device),
                            torch.from_numpy(ds.arrays["action_valid"][ids]).to(args.device),torch.from_numpy(ds.arrays["stop"][ids]).to(args.device),statistics)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True); optimizer.step()
        if update%250==0 or update==args.max_updates:
            prediction,probability=predict(model,ds,features,tokens,args.device)
            mask=ds.arrays["action_valid"]; error=prediction[mask]-ds.arrays["action"][mask]
            error[:,3]=np.arctan2(np.sin(error[:,3]),np.cos(error[:,3]))
            mse=float(np.mean((error/np.asarray(statistics["scale"]))**2))
            target=ds.arrays["stop"]; best_f1=0.
            for threshold in np.linspace(.01,.99,99):
                pred=probability>=threshold; tp=(pred&target).sum(); fp=(pred&~target).sum(); fn=(~pred&target).sum()
                best_f1=max(best_f1,float(2*tp/max(2*tp+fp+fn,1)))
            result={"passed":mse<=.05 and best_f1>=.9,"updates":update,"normalized_action_mse":mse,"stop_best_train_f1":best_f1,
                    "episodes":[r["episode_id"] for r in ds.routes],"engineering_only":True,"weights_reused_by_formal_runs":False,
                    "seconds_training":time.monotonic()-started,"identity":identity,"rgb_cache_prediction_max_error":cache_error,"fp32_vs_stored_prediction_max_error":storage_error}
            print({k:result[k] for k in ("updates","normalized_action_mse","stop_best_train_f1","passed","seconds_training")},flush=True)
            if result["passed"]: break
    write_json(out/"overfit.json",result)
    if not result["passed"]: raise SystemExit("Ten-route memorization failed; formal runs were not started")
    feature_path.unlink(missing_ok=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--max-updates",type=int,default=10000)
    run(p.parse_args())
