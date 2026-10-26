import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from sr3m.geometry import action_target_ned, check_climb, fru_to_world, ned_to_ue_world_m, ue_world_m_to_ned
from experiments.protocol import PROTOCOL, stop_suffix
from had.model import TerrainHAD
from had.tokenizer import Tokenizer
from experiments.metrics import calibrate_stop, evaluate, stop_metrics
from experiments.summarize import gain
from experiments.train import loss_for_batch
from experiments.train import save_torch


def fixture_data():
    routes = [{"episode_id": str(i), "category": "mountain", "terrain_following": bool(i%2),
               "height_layer": "low" if i < 2 else "medium", "map_package": "/Game/Map",
               "offset": i*3, "count": 3} for i in range(4)]
    action = np.tile([[2., 0., .2, .1], [2., 0., .1, 0.], [np.nan]*4], (4, 1))
    return SimpleNamespace(n=12, routes=routes, split="val_seen", route_index=np.repeat(np.arange(4), 3),
        arrays={"action": action, "action_valid": np.tile([True, False, False], 4),
                "stop": np.tile([False, True, True], 4), "distance_goal": np.tile([30., 8., 0.], 4),
                "slope": np.tile([1., 3., 8.], 4)})


STATS = {"scale": [1., 1., 1., 1.], "turn_fraction": .5, "slope_bins_deg": [2., 5.]}


def test_terminal_suffix_preserves_return_start_and_masks_tail():
    p = [[0,0,0], [0,0,12], [0,0,4], [0,0,0]]
    stop, distance = stop_suffix(p)
    assert stop.tolist() == [False, False, True, True]
    assert distance[-1] == 0
    with pytest.raises(ValueError): stop_suffix([[0,0,0], [1,0,0]])


def test_frames_units_origins_and_translate_then_rotate_target():
    assert np.allclose(fru_to_world([2,0,3], math.pi/2), [0,2,3])
    assert np.allclose(ue_world_m_to_ned([10,20,30], [5,5,5]), [5,15,-25])
    assert np.allclose(ned_to_ue_world_m([5,15,-25], [5,5,5]), [10,20,30])
    target, yaw = action_target_ned([10,10,-10], math.pi/2, [2,0,3,-math.pi/2])
    assert np.allclose(target, [10,12,-13]) and abs(yaw) < 1e-10
    check_climb([2,0,1,0])


def test_stop_positive_has_zero_action_gradient_even_with_nan_label():
    action = torch.randn(3,4,requires_grad=True); stop = torch.randn(3,requires_grad=True)
    loss = loss_for_batch({"action": action, "stop_logit": stop}, torch.full((3,4),float("nan")),
                          torch.zeros(3,dtype=torch.bool), torch.ones(3,dtype=torch.bool), STATS)
    loss.backward()
    assert torch.equal(action.grad, torch.zeros_like(action)) and torch.isfinite(stop.grad).all()
    assert torch.count_nonzero(stop.grad) == 3


def test_action_metrics_ignore_stop_prediction_and_positive_tail_labels():
    ds = fixture_data(); pred = np.nan_to_num(ds.arrays["action"])
    result = evaluate(ds, pred, STATS)
    pred[~ds.arrays["action_valid"]] = 100000
    other = evaluate(ds, pred, STATS, np.ones(ds.n), .5)
    assert other["selection_score"] == result["selection_score"]
    assert result["selection_score"] == pytest.approx(0.,abs=1e-12)
    assert other["terrain"]["control/high"]["dz_mae_m"] is None
    assert other["terrain"]["control/high"]["routes"] == 0
    assert other["terrain_by_category"]["mountain"]["follow/all"]["routes"] == 2
    assert other["terrain_by_category"]["desert"]["follow/all"]["routes"] == 0


def test_selection_penalizes_straight_collapse_even_with_identical_translation():
    ds = fixture_data(); correct = np.nan_to_num(ds.arrays["action"]); straight = correct.copy(); straight[:,3] = 0
    a, b = evaluate(ds, correct, STATS), evaluate(ds, straight, STATS)
    assert a["overall"]["translation_m"] == b["overall"]["translation_m"]
    assert a["selection_score"] < b["selection_score"]
    straight[:,3]=2*np.pi
    wrapped=evaluate(ds,straight,STATS)
    assert wrapped["overall"]["predicted_yaw_lt_1deg_fraction"]==1.
    assert wrapped["overall"]["turn_sign_accuracy"]==0.


def test_stop_calibration_matches_exhaustive_threshold_search_and_rejects_unseen():
    ds = fixture_data(); p = np.array([.2,.8,.95,.6,.8,.9,.1,.7,.9,.85,.8,.95])
    ds.arrays["distance_goal"] = np.tile([30.,15.,0.],4)
    actual = calibrate_stop(ds, p)
    options = [stop_metrics(ds,p,t) for t in sorted(set(p),reverse=True)]
    expected = max(options, key=lambda r:(r["first_stop_within_20m"],r["threshold"]))
    assert actual["threshold"] == expected["threshold"] == .9
    assert actual["first_stop_within_20m"] == 4 and actual["first_stop_outside_20m"] == 0 and actual["never_triggered"] == 0
    # A higher threshold leaves routes untriggered; a lower one stops route 3 at 30m.
    assert stop_metrics(ds,p,.95)["never_triggered"] == 2 and stop_metrics(ds,p,.85)["first_stop_outside_20m"] == 1
    ds.split = "val_unseen"
    with pytest.raises(ValueError): calibrate_stop(ds,p)


def test_common_seed_initialization_and_down_only_does_not_require_front():
    torch.set_num_threads(2)
    tokenizer = Tokenizer.fit(["follow the terrain"], max_length=8)
    initial = None
    for variant in ("A","B","C"):
        torch.manual_seed(42)
        model = TerrainHAD(variant,tokenizer,cached_only=True,small=True).eval()
        state = model.head_state()
        if initial is None: initial = state
        else: assert all(torch.equal(state[k],initial[k]) for k in state)
        features = {v:torch.randn(2,2048) for v in model.views}
        out = model(features,torch.tensor([[2,3,4,0,0,0,0,0]]*2),torch.zeros(2,5))
        assert out["action"].shape == (2,4)
        if variant == "C":
            with pytest.raises(ValueError): model({"front":torch.zeros(2,2048)},torch.ones(2,8,dtype=torch.long),torch.zeros(2,5))


def test_depth_variant_shares_seeded_init_and_is_concat_projection():
    torch.set_num_threads(2)
    tokenizer = Tokenizer.fit(["hold altitude over the dunes"], max_length=8)
    torch.manual_seed(42); base = TerrainHAD("B", tokenizer, cached_only=True, small=True).eval()
    torch.manual_seed(42); depth = TerrainHAD("BD", tokenizer, cached_only=True, small=True).eval()
    shared = base.head_state()
    assert all(torch.equal(shared[k], depth.head_state()[k]) for k in shared)
    assert set(depth.inputs) == {"front", "down", "front_depth", "down_depth"} and depth.views == base.views
    x = {v: torch.randn(3, 2048) for v in ("front", "down")}; d = {f"{v}_depth": torch.randn(3, 256) for v in ("front", "down")}
    tokens, state = torch.tensor([[2, 3, 4, 0, 0, 0, 0, 0]] * 3), torch.zeros(3, 5)
    with torch.no_grad():
        for layer in depth.depth_proj.values(): layer.weight.zero_(); layer.bias.zero_()
        assert torch.allclose(depth({**x, **d}, tokens, state)["action"], base(x, tokens, state)["action"])
        # Equals a single linear projection of [cnn; depth].
        linear, proj = depth.front_encoder.proj[0], depth.depth_proj["front"]
        proj.weight.normal_(); proj.bias.normal_()
        joint = torch.nn.Linear(2048 + 256, linear.out_features)
        joint.weight.copy_(torch.cat((linear.weight, proj.weight), 1)); joint.bias.copy_(linear.bias + proj.bias)
        assert torch.allclose(depth.encode("front", {**x, **d}),
                              depth.front_encoder.proj[2](depth.front_encoder.proj[1](joint(torch.cat((x["front"], d["front_depth"]), 1)))), atol=1e-5)
    with pytest.raises(ValueError):
        depth(x, tokens, state)


def test_raw_image_and_cached_forward_match_and_projection_is_trainable():
    torch.set_num_threads(2)
    tokenizer = Tokenizer.fit(["fly right"],max_length=4)
    model = TerrainHAD("C",tokenizer,cached_only=True,small=True).eval()
    images = {"down":torch.randn(2,3,32,32)}; tokens = torch.ones(2,4,dtype=torch.long); state = torch.zeros(2,5)
    with torch.no_grad():
        cached = {"down":model.down_encoder.cnn(images["down"])}
        a,b = model(cached,tokens,state), model.from_images(images,tokens,state)
    assert all(torch.allclose(a[k],b[k],atol=1e-6) for k in a)
    assert not any(p.requires_grad for p in model.down_encoder.cnn.parameters())
    assert all(p.requires_grad for p in model.down_encoder.proj.parameters())


def test_agl_matched_interaction_has_no_invented_high_control():
    ds=fixture_data(); a={}; b={}
    for r in ds.routes:
        a[r["episode_id"]]=dict(r,dz_mae_m=.4 if r["terrain_following"] else .1)
        b[r["episode_id"]]=dict(r,dz_mae_m=.2 if r["terrain_following"] else .09)
    result=gain(a,b)
    assert result["agl_matched_D_m"] == pytest.approx(.19)


def test_checkpoint_safe_reload_preserves_vocab_contract_and_outputs(tmp_path):
    torch.set_num_threads(2)
    tok=Tokenizer.fit(["fly over terrain"],max_length=4)
    model=TerrainHAD("B",tok,cached_only=True,small=True).eval()
    features={v:torch.randn(2,2048) for v in model.views};tokens=torch.ones(2,4,dtype=torch.long);state=torch.zeros(2,5)
    with torch.no_grad(): expected=model(features,tokens,state)
    save_torch(tmp_path/"model.pt",{"head":model.head_state(),"tokenizer":tok.to_dict(),"protocol":PROTOCOL,"torch":str(torch.__version__)})
    restored=torch.load(tmp_path/"model.pt",map_location="cpu",weights_only=True)
    other=TerrainHAD("B",Tokenizer.from_dict(restored["tokenizer"]),cached_only=True,small=True).eval()
    other.load_head_state(restored["head"])
    with torch.no_grad(): actual=other(features,tokens,state)
    assert all(torch.equal(expected[k],actual[k]) for k in expected)
    assert restored["protocol"]["precision"]=="float32_ieee_no_tf32" and not torch.backends.cudnn.allow_tf32


def test_desert_relief_is_reproducible_and_strictly_increasing():
    from sr3m.desert_terrain import heightfields,mesh_slopes
    axis,z=heightfields(size=33)
    assert np.array_equal(z,heightfields(size=33)[1])
    assert np.allclose(z.std(axis=(1,2)),[0,10,25,50])
    slopes=[]
    for field in z:
        gx,gy=np.gradient(field,axis[1]-axis[0]);slopes.append(np.arctan(np.hypot(gx,gy)).max())
    assert np.all(np.diff(slopes)>0)
    assert np.allclose(mesh_slopes(np.array([[0.,0.],[0.,1.]]),1.),45.)


def test_epoch_checkpoint_resume_matches_uninterrupted_training(tmp_path,monkeypatch):
    import experiments.train as module
    from argparse import Namespace
    torch.set_num_threads(2)
    train_ds=fixture_data();train_ds.split="train";train_ds.protocol={"protocol":PROTOCOL}
    train_ds.arrays["action"]=train_ds.arrays["action"].astype("float32")
    train_ds.arrays["state"]=np.zeros((train_ds.n,5),dtype="float32")
    for r in train_ds.routes:r["instruction"]="follow terrain"
    seen=SimpleNamespace(**vars(train_ds));seen.split="val_seen"
    tokenizer=Tokenizer.fit(["follow terrain"],max_length=4)
    token=module.token_rows(train_ds,tokenizer)
    tokens={"train":token,"val_seen":token}
    x={v:np.random.default_rng(11).normal(size=(train_ds.n,2048)).astype("float32") for v in ("front","down")}
    data={"train":train_ds,"val_seen":seen};features={"train":x,"val_seen":x}
    monkeypatch.setitem(PROTOCOL,"optimizer",dict(PROTOCOL["optimizer"],max_epochs=2,min_epochs=1,batch_size=4))
    monkeypatch.setattr(module,"TerrainHAD",lambda variant,tok,cached_only:TerrainHAD(variant,tok,cached_only=cached_only,small=True))
    args=Namespace(out=str(tmp_path/"full"),device="cpu",code_version={"commit":"test","dirty":False})
    module.train_run(args,"B",42,data,features,tokenizer,tokens,STATS)
    original=module.save_torch
    def interrupt_after_epoch(path,value):
        original(path,value)
        if path.name=="last.pt" and value["epoch"]==1:raise RuntimeError("simulated interruption")
    monkeypatch.setattr(module,"save_torch",interrupt_after_epoch)
    args.out=str(tmp_path/"resumed")
    with pytest.raises(RuntimeError,match="simulated interruption"):
        module.train_run(args,"B",42,data,features,tokenizer,tokens,STATS)
    monkeypatch.setattr(module,"save_torch",original)
    module.train_run(args,"B",42,data,features,tokenizer,tokens,STATS)
    first=torch.load(tmp_path/"full/B_seed42/last.pt",weights_only=True)
    resumed=torch.load(tmp_path/"resumed/B_seed42/last.pt",weights_only=True)
    assert all(torch.equal(value,resumed["head"][key]) for key,value in first["head"].items())
    assert first["best_score"]==resumed["best_score"] and first["epoch"]==resumed["epoch"]==2

