"""The optional K restart policy, without UE: the default stays a restart every 32 renders."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sim_k import backend as k
from sim_k.backend import MemoryPolicy, UEBackend

GB = 2 ** 30


def memory(commit=50., private=8.):
    return {"commit_percent": commit, "private_bytes": private * GB, "working_set_bytes": private * GB}


class Scripted(UEBackend):
    """UEBackend with the server replaced: render writes tiny PNGs, memory follows a script, nothing starts UE."""

    def __init__(self, tmp, policy=None, restart_every=32, commits=None):
        super().__init__(tmp / "session", "desert", "/Game/Map", {}, restart_every=restart_every, policy=policy)
        self.calls, self.commits, self.closes = [], list(commits or []), 0
        self.images.mkdir(parents=True)

    def start(self):
        self.process, self.since_start, self.memory = object(), 0, None
        self.starts += 1
        self.session.mkdir(parents=True, exist_ok=True)

    def call(self, op, timeout=300., **args):
        if self.process is None:
            self.start()
        self.calls.append(op)
        if op == "render":
            Image.new("RGB", (2, 2)).save(self.images / f"{args['name']}_front.png")
            return {"files": {"front": f"{args['name']}_front.png"}}
        if op == "memory":
            return memory(self.commits.pop(0) if self.commits else 50.)
        return {"seconds": 0.}

    def close(self):
        if self.process is not None:
            self.closes += 1
        self.process = None


def render(backend, n):
    for _ in range(n):
        backend.render(np.zeros(3), 0.)


def test_default_restarts_every_32_and_never_asks_for_memory(tmp_path, monkeypatch):
    monkeypatch.delenv("SR3MXHAD_K_MEMORY_POLICY", raising=False)
    b = Scripted(tmp_path)
    assert b.policy is None
    render(b, 100)
    assert b.starts == 4 and b.closes == 3
    assert set(b.calls) == {"render"}


def test_environment_turns_the_policy_on_only_when_set(monkeypatch):
    monkeypatch.delenv("SR3MXHAD_K_MEMORY_POLICY", raising=False)
    assert k.policy_from_environment() is None
    monkeypatch.setenv("SR3MXHAD_K_MEMORY_POLICY", "")
    assert k.policy_from_environment() == MemoryPolicy()
    monkeypatch.setenv("SR3MXHAD_K_MEMORY_POLICY", json.dumps({"commit_percent": 75, "gc_every": 16}))
    assert k.policy_from_environment() == MemoryPolicy(commit_percent=75, gc_every=16)


def test_policy_restart_rules():
    p = MemoryPolicy(commit_percent=80., private_gb=20., max_renders=100)
    assert not p.restart_due(10, None)
    assert not p.restart_due(10, memory(79.9, 19.))
    assert p.restart_due(10, memory(80., 8.))
    assert p.restart_due(10, memory(50., 20.))
    assert p.restart_due(100, None)
    assert not MemoryPolicy(private_gb=0.).restart_due(10, memory(50., 1000.))


def test_policy_gc_and_check_schedule():
    p = MemoryPolicy(gc_every=16, check_every=8)
    assert [s for s in range(0, 50) if p.collect_due(s)] == [16, 32, 48]
    assert [s for s in range(0, 25) if p.check_due(s)] == [8, 16, 24]
    assert not any(MemoryPolicy(gc_every=0).collect_due(s) for s in range(100))


def test_policy_ignores_restart_every_and_restarts_on_commit(tmp_path):
    # Readings at renders 8, 16, 24: the third crosses 80 %, so the process ends before render 25.
    b = Scripted(tmp_path, policy=MemoryPolicy(check_every=8), restart_every=4, commits=[60., 70., 81.])
    render(b, 30)
    assert b.starts == 2 and b.closes == 1
    assert b.calls.count("memory") == 3
    assert b.since_start == 6 and b.memory is None  # a new process starts without a reading
    assert len(b.memory_log) == 3 and (tmp_path / "session/memory.jsonl").read_text().count("\n") == 3


def test_policy_max_renders_and_gc(tmp_path):
    b = Scripted(tmp_path, policy=MemoryPolicy(max_renders=40, gc_every=16, check_every=1000))
    render(b, 100)
    assert b.starts == 3  # 40 + 40 + 20
    assert b.calls.count("collect_garbage") == 2 * 2 + 1
    assert "memory" not in b.calls


def test_server_only_adds_operations():
    """The new server operations sit beside render/sweep/ground, which keep their code path."""
    source = (Path(k.__file__).parent / "ue_server.py").read_text(encoding="utf-8")
    for name in ("def memory(self)", "def collect_garbage(self)", "def console(self, command)", "def render(self, position_cm, yaw_deg, directory, name)"):
        assert name in source


def test_startup_console_is_off_unless_asked(monkeypatch):
    from sim_k.backend import UEBackend, console_from_environment
    monkeypatch.delenv("SR3MXHAD_K_CONSOLE", raising=False)
    assert console_from_environment() == []
    backend = UEBackend("/tmp/k-test-session", next(iter(__import__("sim_k.backend", fromlist=["PROJECTS"]).PROJECTS)), "/Game/Map", {})
    assert backend.console == []
    monkeypatch.setenv("SR3MXHAD_K_CONSOLE", '["r.Shadow.Virtual.Enable 0"]')
    assert console_from_environment() == ["r.Shadow.Virtual.Enable 0"]
