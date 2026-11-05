"""WSL side of the kinematic loop. Rendering lives in a restartable UnrealEditor-Cmd; all maths stays here."""
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "E:/Program Files/EpicGame/UE_5.8/Engine/Binaries/Win64/UnrealEditor-Cmd.exe"
TOOLS = "E:/UnrealProjects/firstProject/Automation/ue_scripts"
# The projects the dataset was captured in, by run-context profile. Loaded read-only: nothing here saves a package.
PROJECTS = {"mountain_base": "E:/UnrealProjects/mountain_base/military_mountains/military_mountains.uproject",
            "forests_new": "E:/UnrealProjects/forests_new/military_forests/military_forests.uproject",
            "desert": "E:/UnrealProjects/desert/desert.uproject",
            # Stage 2 (protocol section 17): v2 desert levels, L_DES_*_v3 and the rebuilt mountain L_MTR_A-D in one project.
            "desert_mountain": "E:/UnrealProjects/DesertMountain/DesertMountain.uproject"}
SWEEP = {"interval_cm": 50., "radius_cm": 100.}  # The capture tool's collision check.
MAX_UNEXPECTED_EXITS = 5


class ServerExited(RuntimeError):
    pass


def windows_path(path):
    # wslpath keeps a relative path relative, and UE resolves those against its own binaries folder.
    return subprocess.check_output(["wslpath", "-m", str(Path(path).resolve())], text=True).strip()


def powershell(command, **kwargs):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True, **kwargs)


def editors_running(attempts=5):
    """Number of Unreal editor processes. PowerShell from WSL now and then prints nothing; ask again before giving up."""
    for attempt in range(attempts):
        output = powershell("(Get-Process UnrealEditor* -ErrorAction SilentlyContinue | Measure-Object).Count", timeout=60).stdout
        counts = [int(line) for line in output.splitlines() if line.strip().isdigit()]
        if counts:
            return counts[0]
        time.sleep(5. * (attempt + 1))
    raise RuntimeError(f"Could not count Unreal editor processes; PowerShell printed {output!r}")


@dataclass(frozen=True)
class MemoryPolicy:
    """Optional restart policy (off by default; the default stays a restart every 32 renders, as stage 1 and earlier results).

    The server reads its own memory every check_every renders. The editor restarts before the next render once the Windows
    commit charge reaches commit_percent (the capture workbench stops at 80 %), the process's private bytes reach
    private_gb, or max_renders renders ran in the process. gc_every > 0 runs UObject garbage collection every that many renders.
    """
    commit_percent: float = 80.
    private_gb: float = 0.  # 0: no limit on the process itself
    max_renders: int = 512
    gc_every: int = 0  # 11/03: GC every 16 renders did not slow the growth at all
    check_every: int = 1  # 11/03: one render near the top added 2.4 GB (5 % commit)

    def collect_due(self, since_start):
        return self.gc_every > 0 and since_start > 0 and since_start % self.gc_every == 0

    def check_due(self, since_start):
        return since_start > 0 and since_start % self.check_every == 0

    def restart_due(self, since_start, memory):
        """memory: the server's last reading in this process, or None."""
        if since_start >= self.max_renders:
            return True
        if memory is None:
            return False
        return memory["commit_percent"] >= self.commit_percent or (self.private_gb > 0 and memory["private_bytes"] >= self.private_gb * 2 ** 30)


def policy_from_environment():
    """SR3MXHAD_K_MEMORY_POLICY='{}' (defaults) or a JSON object of MemoryPolicy fields turns the memory policy on; unset: off."""
    value = os.environ.get("SR3MXHAD_K_MEMORY_POLICY")
    return None if value is None else MemoryPolicy(**json.loads(value or "{}"))


def console_from_environment():
    """SR3MXHAD_K_CONSOLE='["r.Shadow.Virtual.Enable 0"]': console commands the server runs after loading the map, before any capture. Unset: none."""
    value = os.environ.get("SR3MXHAD_K_CONSOLE")
    return [] if not value else [str(c) for c in json.loads(value)]


class UEBackend:
    """render/sweep/ground in world metres and radians. Restarts the editor before its memory runs out.

    Default: a restart every restart_every renders. With policy (a MemoryPolicy, or SR3MXHAD_K_MEMORY_POLICY set) the restart
    follows the policy instead and restart_every is ignored.
    """

    def __init__(self, session, profile, map_package, capture, restart_every=32, keep_images=False, start_timeout=420., policy=None, console=None):
        if profile not in PROJECTS:
            raise ValueError(f"No project known for profile {profile}")
        self.session, self.profile, self.map_package = Path(session), profile, map_package
        self.capture, self.restart_every, self.keep_images, self.start_timeout = capture, restart_every, keep_images, start_timeout
        self.policy = policy if policy is not None else policy_from_environment()
        self.console = list(console) if console is not None else console_from_environment()  # changes pixels; off by default
        self.process, self.number, self.renders, self.since_start, self.starts, self.unexpected_exits = None, 0, 0, 0, 0, 0
        self.memory, self.memory_log = None, []
        self.images = self.session / "images"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def start(self):
        deadline = time.monotonic() + 120.
        while editors_running():
            if time.monotonic() > deadline:
                raise RuntimeError("An Unreal editor is running; close it first: K needs the GPU and the project to itself")
            time.sleep(2.)
        attempt = self.session / f"process-{self.starts:04d}"
        shutil.rmtree(attempt, ignore_errors=True)
        attempt.mkdir(parents=True)
        self.images.mkdir(parents=True, exist_ok=True)
        config = {"tools_dir": TOOLS, "map_package": self.map_package, "capture": self.capture, "idle_exit_s": 300.}
        if self.policy is not None:
            config["memory_policy"] = asdict(self.policy)  # Only recorded; the server does not read it.
        if self.console:
            config["console"] = self.console
        (attempt / "config.json").write_text(json.dumps(config), encoding="utf-8")
        arguments = [PROJECTS[self.profile], "-Unattended", "-NoSound", "-NoSplash", "-NoSourceControl", "-RenderOffscreen", "-AllowCommandletRendering", "-stdout", "-FullStdOutLogOutput",
                     "-ExecutePythonScript=" + windows_path(ROOT / "sim_k/ue_server.py"), "-KSession=" + windows_path(attempt), "-AbsLog=" + windows_path(attempt / "Unreal.log")]
        command = f"& '{ENGINE}' " + " ".join(f"'{a}'" for a in arguments) + "; exit $LASTEXITCODE"
        with open(attempt / "stdout.log", "wb") as log:
            self.process = subprocess.Popen(["powershell.exe", "-NoProfile", "-Command", command], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        self.attempt, self.number, self.since_start, self.memory = attempt, 0, 0, None
        self.starts += 1
        deadline = time.monotonic() + self.start_timeout
        while not (attempt / "ready.json").exists():
            if (attempt / "failed.json").exists():
                raise RuntimeError("UE server failed: " + (attempt / "failed.json").read_text(encoding="utf-8"))
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"UE server did not start; see {attempt / 'Unreal.log'}")
            time.sleep(.2)

    def call(self, op, timeout=300., **args):
        """Every operation is stateless, so a server that died is simply restarted and asked again."""
        while True:
            if self.process is None:
                self.start()
            try:
                return self.exchange(op, timeout, args)
            except ServerExited:
                self.unexpected_exits += 1
                self.process = None
                if self.unexpected_exits > MAX_UNEXPECTED_EXITS:
                    raise

    def exchange(self, op, timeout, args):
        self.number += 1
        request, reply = self.attempt / f"request-{self.number:06d}.json", self.attempt / f"reply-{self.number:06d}.json"
        temporary = request.with_name(request.name + ".tmp")
        temporary.write_text(json.dumps({"op": op, "args": args}), encoding="utf-8")
        os.replace(temporary, request)
        deadline = time.monotonic() + timeout
        while not reply.exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise ServerExited(f"UE server stopped answering {op} (exit code {self.process.poll()}); see {self.attempt / 'stdout.log'}")
            time.sleep(.005)
        while True:
            try:  # Same file-sharing race as on the server side.
                answer = json.loads(reply.read_text(encoding="utf-8"))
                break
            except (OSError, ValueError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(.01)
        if not answer["ok"]:
            raise RuntimeError(answer["error"])
        return answer

    def restart_due(self):
        if self.policy is None:
            return self.since_start >= self.restart_every
        return self.policy.restart_due(self.since_start, self.memory)

    def after_render(self):
        """Memory policy only: garbage collection and the memory reading the next restart decision uses."""
        if self.policy.collect_due(self.since_start):
            self.call("collect_garbage")
        if self.policy.check_due(self.since_start):
            started = time.monotonic()
            self.memory = self.call("memory")
            self.memory_log.append({"renders": self.renders, "process": self.starts, "since_start": self.since_start, "time": time.time(), "check_s": time.monotonic() - started,
                                    **{k: self.memory[k] for k in ("private_bytes", "working_set_bytes", "commit_percent")}})
            with open(self.session / "memory.jsonl", "a", encoding="utf-8") as log:
                log.write(json.dumps(self.memory_log[-1]) + "\n")

    def render(self, position_m, yaw_rad):
        """{view: RGB image} at the pose, or kept paths with keep_images."""
        if self.process is not None and self.restart_due():
            self.close()
        name = f"render_{self.renders:06d}"
        answer = self.call("render", position_cm=[float(p) * 100. for p in position_m], yaw_deg=math.degrees(yaw_rad), directory=windows_path(self.images), name=name)
        self.renders += 1
        self.since_start += 1
        if self.policy is not None:
            self.after_render()
        images = {}
        for view, file in answer["files"].items():
            with Image.open(self.images / file) as image:
                images[view] = image.convert("RGB")
            if not self.keep_images:
                (self.images / file).unlink(missing_ok=True)
        return images

    def sweep(self, start_m, end_m):
        return self.call("sweep", start_cm=[float(p) * 100. for p in start_m], end_cm=[float(p) * 100. for p in end_m], **SWEEP)

    def ground_z_m(self, position_m):
        z = self.call("ground", position_cm=[float(p) * 100. for p in position_m])["ground_z_cm"]
        return None if z is None else z / 100.

    def close(self):
        if self.process is None:
            return
        try:
            self.exchange("stop", 60., {})
            self.process.wait(timeout=120.)
        except (RuntimeError, subprocess.TimeoutExpired):
            pass  # The server exits by itself when idle; start() waits for that.
        self.process = None
