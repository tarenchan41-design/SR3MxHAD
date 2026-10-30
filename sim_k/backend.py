"""WSL side of the kinematic loop. Rendering lives in a restartable UnrealEditor-Cmd; all maths stays here."""
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
            "forests_new": "E:/UnrealProjects/forests_new/military_forests/military_forests.uproject"}
SWEEP = {"interval_cm": 50., "radius_cm": 100.}  # The capture tool's collision check.
MAX_UNEXPECTED_EXITS = 5


class ServerExited(RuntimeError):
    pass


def windows_path(path):
    # wslpath keeps a relative path relative, and UE resolves those against its own binaries folder.
    return subprocess.check_output(["wslpath", "-m", str(Path(path).resolve())], text=True).strip()


def powershell(command, **kwargs):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True, **kwargs)


def editors_running():
    output = powershell("(Get-Process UnrealEditor* -ErrorAction SilentlyContinue | Measure-Object).Count", timeout=60).stdout
    return int(next(line for line in output.splitlines() if line.strip().isdigit()))


class UEBackend:
    """render/sweep/ground in world metres and radians. Restarts the editor before its memory runs out."""

    def __init__(self, session, profile, map_package, capture, restart_every=32, keep_images=False, start_timeout=420.):
        if profile not in PROJECTS:
            raise ValueError(f"No project known for profile {profile}")
        self.session, self.profile, self.map_package = Path(session), profile, map_package
        self.capture, self.restart_every, self.keep_images, self.start_timeout = capture, restart_every, keep_images, start_timeout
        self.process, self.number, self.renders, self.since_start, self.starts, self.unexpected_exits = None, 0, 0, 0, 0, 0
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
        (attempt / "config.json").write_text(json.dumps({"tools_dir": TOOLS, "map_package": self.map_package, "capture": self.capture, "idle_exit_s": 300.}), encoding="utf-8")
        arguments = [PROJECTS[self.profile], "-Unattended", "-NoSound", "-NoSplash", "-NoSourceControl", "-RenderOffscreen", "-AllowCommandletRendering", "-stdout", "-FullStdOutLogOutput",
                     "-ExecutePythonScript=" + windows_path(ROOT / "sim_k/ue_server.py"), "-KSession=" + windows_path(attempt), "-AbsLog=" + windows_path(attempt / "Unreal.log")]
        command = f"& '{ENGINE}' " + " ".join(f"'{a}'" for a in arguments) + "; exit $LASTEXITCODE"
        with open(attempt / "stdout.log", "wb") as log:
            self.process = subprocess.Popen(["powershell.exe", "-NoProfile", "-Command", command], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        self.attempt, self.number, self.since_start = attempt, 0, 0
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

    def render(self, position_m, yaw_rad):
        """{view: RGB image} at the pose, or kept paths with keep_images."""
        if self.process is not None and self.since_start >= self.restart_every:
            self.close()
        name = f"render_{self.renders:06d}"
        answer = self.call("render", position_cm=[float(p) * 100. for p in position_m], yaw_deg=math.degrees(yaw_rad), directory=windows_path(self.images), name=name)
        self.renders += 1
        self.since_start += 1
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
