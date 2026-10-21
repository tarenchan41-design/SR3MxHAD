from argparse import Namespace
from types import SimpleNamespace
import numpy as np
import pytest

from experiment import PROTOCOL,read_json,write_json
from metrics import evaluate
from summarize import run,stable_support


def test_frozen_nine_run_reporting_recovers_known_cohort_interaction(tmp_path):
    routes=[];offset=0
    specs=[(True,"low")]*17+[(True,"medium")]*13+[(True,"high")]*7+[(False,"low")]*7+[(False,"medium")]*4+[(False,None)]*151
    for i,(follow,layer) in enumerate(specs):
        routes.append({"episode_id":str(i),"category":"mountain" if layer else "airfields","terrain_following":follow,
                       "height_layer":layer,"map_package":"/Game/Mountain" if layer else "/Game/Air","offset":offset,"count":3})
        offset+=3
    target=np.tile([[2.,0.,0.,.1],[2.,0.,0.,0.],[np.nan]*4],(199,1))
    for r in routes:
        if r["height_layer"]: target[r["offset"],2]=.4 if r["terrain_following"] else .1
    ds=SimpleNamespace(n=offset,routes=routes,split="val_unseen",protocol={"protocol":PROTOCOL},route_index=np.repeat(np.arange(199),3),
        arrays={"action":target,"action_valid":np.tile([True,False,False],199),"stop":np.tile([False,True,True],199),
                "distance_goal":np.tile([30.,8.,0.],199),"slope":np.tile([1.,3.,8.],199)})
    stats={"scale":[1.,1.,1.,1.],"slope_bins_deg":[2.,5.]}
    results=tmp_path/"results";results.mkdir()
    for name,action in (("fixed_forward",[2.,0.,0.,0.]),("train_mean",[2.,0.,.08,0.])):
        write_json(results/f"{name}.json",evaluate(ds,np.broadcast_to(action,(ds.n,4)),stats))
    for variant in "ABC":
        for seed in PROTOCOL["seeds"]:
            pred=np.nan_to_num(target.copy())
            for r in routes:
                if r["height_layer"]:
                    dz={"A":(.1,.02),"B":(.04,.018),"C":(.15,.03)}[variant][0 if r["terrain_following"] else 1]
                else: dz=.01
                pred[r["offset"],2]+=dz*(1+(seed-43)*.01)
                pred[r["offset"],3]+={"A":.03,"B":.02,"C":.04}[variant]
            report=evaluate(ds,pred,stats,np.tile([.1,.8,.9],199),.5)
            report.update(variant=variant,seed=seed,selection={"split":"val_seen"})
            write_json(results/f"{variant}_seed{seed}.json",report)
    probe=tmp_path/"probe";probe.mkdir()
    write_json(probe/"probe.json",{"splits":{"val_unseen":{name:{"mae":.5,"train_mean_mae":1.,"r2":.5} for name in ("agl_m","slope_deg")}}})
    run(Namespace(results=str(results),probe=str(probe),out=str(tmp_path/"decision")))
    summary=read_json(tmp_path/"decision/summary.json")
    assert summary["interaction"]["agl_matched_D_m"]["mean"]==pytest.approx(.058)
    assert summary["baseline_gates"]["A"]["passed"] and summary["baseline_gates"]["B"]["passed"]
    assert "首轮支持" in summary["decision"]
    text=(tmp_path/"decision/decision.md").read_text()
    assert "7/0" in text and "NA" in text and "STOP" in text and "相对" in text


def test_unfinished_nine_runs_cannot_open_held_out_data(tmp_path,monkeypatch):
    import evaluate as module
    def forbidden(*args,**kwargs): raise AssertionError("held-out data opened before selection")
    monkeypatch.setattr(module,"PreparedSplit",forbidden)
    with pytest.raises(ValueError,match="All nine"):
        module.evaluate_runs(Namespace(runs=str(tmp_path)))


def test_a_single_large_seed_cannot_establish_the_hypothesis():
    d=[.001,.001,1.]
    interaction={"agl_matched_D_m":{"by_seed":d,"mean":float(np.mean(d)),"seed_std":float(np.std(d,ddof=1)),"route_95ci":[.1,.5]},
                 "follow_gain_m":{"by_seed":[.1,.1,1.1],"route_95ci":[.1,.5]},"raw_D_m":{"mean":.3}}
    assert not stable_support(interaction)
