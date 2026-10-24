from argparse import Namespace
from types import SimpleNamespace
import numpy as np
import pytest

from experiments.protocol import PROTOCOL,read_json,write_json
from experiments.metrics import evaluate
from experiments.summarize import positive,run


def test_frozen_nine_run_reporting_recovers_known_cohort_interaction(tmp_path):
    routes=[];offset=0
    terrain=[(True,"low")]*17+[(True,"medium")]*13+[(True,"high")]*7+[(False,"low")]*7+[(False,"medium")]*4
    specs=[("mountain",*x) for x in terrain]+[("desert",*x) for x in terrain]+[("airfields",False,None)]*151
    n=len(specs); assert n==247
    for i,(category,follow,layer) in enumerate(specs):
        routes.append({"episode_id":str(i),"category":category,"terrain_following":follow,
                       "height_layer":layer,"map_package":f"/Game/{category}","offset":offset,"count":3})
        offset+=3
    target=np.tile([[2.,0.,0.,.1],[2.,0.,0.,0.],[np.nan]*4],(n,1))
    for r in routes:
        if r["height_layer"] and r["terrain_following"]: target[r["offset"]:r["offset"]+2,2]=.4
    ds=SimpleNamespace(n=offset,routes=routes,split="val_unseen",protocol={"protocol":PROTOCOL},route_index=np.repeat(np.arange(n),3),
        arrays={"action":target,"action_valid":np.tile([True,True,False],n),"stop":np.tile([False,False,True],n),
                "distance_goal":np.tile([30.,15.,0.],n),"slope":np.tile([1.,8.,8.],n)})
    stats={"scale":[1.,1.,1.,1.],"slope_bins_deg":[2.,5.]}
    results=tmp_path/"results";results.mkdir()
    for name,action in (("fixed_forward",[2.,0.,0.,0.]),("train_mean",[2.,0.,.08,0.])):
        write_json(results/f"{name}.json",evaluate(ds,np.broadcast_to(action,(ds.n,4)),stats,follow_slope_edges=[2.,5.]))
    for variant in "ABC":
        for seed in PROTOCOL["seeds"]:
            pred=np.nan_to_num(target.copy())
            for r in routes:
                # dz error at the low-slope step, then at the high-slope step.
                if r["height_layer"] and r["terrain_following"]: dz={"A":(.1,.1),"B":(.08,.03),"C":(.15,.15)}[variant]
                elif r["height_layer"]: dz={"A":(.02,.02),"B":(.018,.018),"C":(.03,.03)}[variant]
                else: dz=(.01,.01)
                for step in range(2): pred[r["offset"]+step,2]+=dz[step]*(1+(seed-43)*.01)
                pred[r["offset"],3]+={"A":.03,"B":.02,"C":.04}[variant]
            report=evaluate(ds,pred,stats,np.tile([.1,.2,.9],n),.5,follow_slope_edges=[2.,5.])
            report.update(variant=variant,seed=seed,selection={"split":"val_seen"})
            write_json(results/f"{variant}_seed{seed}.json",report)
    probe=tmp_path/"probe";probe.mkdir()
    write_json(probe/"probe.json",{"splits":{"val_unseen":{name:{"mae":.5,"train_mean_mae":1.,"r2":.5} for name in ("agl_m","slope_deg")}}})
    run(Namespace(results=str(results),probe=str(probe),out=str(tmp_path/"decision")))
    summary=read_json(tmp_path/"decision/summary.json")
    assert summary["baseline_gates"]["A"]["passed"] and summary["baseline_gates"]["B"]["passed"]
    h=summary["hypotheses"]
    assert h["H1_follow_relative_dz_gain"]["mean"]==pytest.approx(.45)
    assert h["H2_low_slope_gain"]["mean"]==pytest.approx(.2) and h["H2_high_slope_gain"]["mean"]==pytest.approx(.7)
    assert h["control_dz_harm_m"]["mean"]==pytest.approx(-.002)
    assert summary["H1"] and summary["H2"] and summary["no_harm"]
    assert "核心假设获得支持" in summary["decision"]
    assert summary["interaction_descriptive"]["agl_matched_D_m"]["mean"]==pytest.approx(.043)
    for category in ("mountain","desert"):
        assert summary["hypotheses_by_category_descriptive"][category]["H1_follow_relative_dz_gain"]["mean"]==pytest.approx(.45)
    text=(tmp_path/"decision/decision.md").read_text()
    assert "34/14、26/8、14/0" in text and "mountain 37/11" in text and "desert 37/11" in text
    assert "NA" in text and "STOP" in text and "H1" in text and "仅作描述" in text and "高 AGL 无对照" in text


def test_unfinished_nine_runs_cannot_open_held_out_data(tmp_path,monkeypatch):
    import experiments.evaluate as module
    def forbidden(*args,**kwargs): raise AssertionError("held-out data opened before selection")
    monkeypatch.setattr(module,"PreparedSplit",forbidden)
    with pytest.raises(ValueError,match="All nine"):
        module.evaluate_runs(Namespace(runs=str(tmp_path)))


def test_a_single_large_seed_cannot_establish_a_hypothesis():
    assert not positive({"by_seed":[-.01,.01,1.],"route_95ci":[.1,.5]})
    assert not positive({"by_seed":[.1,.1,.1],"route_95ci":[-.01,.5]})
    assert not positive({"by_seed":[.1,None,.1],"route_95ci":[.1,.5]})
    assert positive({"by_seed":[.1,.1,.1],"route_95ci":[.01,.5]})
