"""Runs inside UnrealEditor-Cmd; answers render, sweep and ground requests over atomic files.

Everything that affects pixels repeats UE5-VLN工具整合/ue_bridge/entry.py and capture_trajectory.CaptureRig, imported
rather than copied. Python blocks the game thread as the dataset capture did, which is why these images match it and
why a GUI editor or PIE does not. Never saves a package.
"""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import time
import traceback

import unreal


def command_line_value(name):
    return next(a.split("=", 1)[1] for a in unreal.SystemLibrary.get_command_line().split() if a.startswith(f"-{name}="))


SESSION = Path(command_line_value("KSession"))
CONFIG = json.loads((SESSION / "config.json").read_text(encoding="utf-8"))
sys.path.insert(0, CONFIG["tools_dir"])
import capture_trajectory as core  # noqa: E402


def write_atomic(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


class _ProcessMemory(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]


class _Performance(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
        "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
        "KernelNonpaged", "PageSize")] + [(n, wintypes.DWORD) for n in ("HandleCount", "ProcessCount", "ThreadCount")]


def memory_now():
    """This process's private bytes and working set, and the Windows commit charge, in bytes. Read only; no pixels change."""
    psapi, kernel32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    process = _ProcessMemory(cb=ctypes.sizeof(_ProcessMemory))
    if not psapi.GetProcessMemoryInfo(wintypes.HANDLE(kernel32.GetCurrentProcess()), ctypes.byref(process), process.cb):
        raise OSError("GetProcessMemoryInfo failed")
    system = _Performance(cb=ctypes.sizeof(_Performance))
    if not psapi.GetPerformanceInfo(ctypes.byref(system), system.cb):
        raise OSError("GetPerformanceInfo failed")
    page = system.PageSize
    return {"private_bytes": process.PrivateUsage, "working_set_bytes": process.WorkingSetSize, "commit_total_bytes": system.CommitTotal * page,
            "commit_limit_bytes": system.CommitLimit * page, "commit_percent": 100. * system.CommitTotal / system.CommitLimit,
            "physical_available_bytes": system.PhysicalAvailable * page, "physical_total_bytes": system.PhysicalTotal * page}


def load_map(package):
    """As entry.inspect_scene: load the level and, on a World Partition map, every actor."""
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(package):
        raise RuntimeError(f"could not load {package}")
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    if not world or str(world.get_path_name()).split(".")[0] != package:
        raise RuntimeError("loaded world differs from the requested map")
    if world.get_world_settings().get_editor_property("world_partition") is not None:
        descriptors = unreal.WorldPartitionBlueprintLibrary.get_actor_descs()
        if descriptors is None:
            raise RuntimeError("World Partition actor listing failed")
        if descriptors:
            unreal.WorldPartitionBlueprintLibrary.load_actors([d.guid for d in descriptors])
    return world


class Server:
    def __init__(self):
        self.capture = CONFIG["capture"]
        self.world = load_map(CONFIG["map_package"])
        core._finish_loading_before_screenshot("k-map-loaded")
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        self.rig = core.CaptureRig(self.world, actors, {"bias": self.capture["exposure_bias"]}, tuple(self.capture["resolution"]))
        for _, component, _, _ in self.rig.cameras:
            component.set_editor_properties({"fov_angle": self.capture["fov_deg"], "always_persist_rendering_state": self.capture["persistent_state"]})
        self.rig.verify_exposure_profiles()
        self.primed = False

    def prime(self, location, yaw):
        self.rig.set_pose(location, yaw)
        self.rig.prewarm(1)
        core._finish_loading_before_screenshot("k-after-prime")
        self.rig.prewarm(max(0, self.capture["warmup_frames"] - 1))
        self.primed = True

    def render(self, position_cm, yaw_deg, directory, name):
        location = unreal.Vector(*position_cm)
        if not self.primed:
            self.prime(location, yaw_deg)
        self.rig.set_pose(location, yaw_deg)
        time.sleep(self.capture["settle_ms"] / 1000)
        files = {}
        for view, component, rgb, _ in self.rig.cameras:
            if view not in self.capture["views"]:
                continue
            component.set_editor_properties({"texture_target": rgb, "capture_source": unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR})
            for _ in range(self.capture["capture_passes"]):
                component.capture_scene()
            files[view] = f"{name}_{view}.png"
            unreal.RenderingLibrary.export_render_target(self.world, rgb, directory, files[view])
        return {"files": files}

    def sweep(self, start_cm, end_cm, interval_cm, radius_cm):
        """The capture tool's collision check: sphere traces along the segment on the Visibility channel."""
        start, end = [unreal.Vector(*p) for p in (start_cm, end_cm)]
        length = max((end - start).length(), 1e-6)
        pieces = max(1, int(length // interval_cm) + (1 if length % interval_cm else 0))
        points = [start + (end - start) * (i / pieces) for i in range(pieces + 1)]
        for a, b in zip(points, points[1:]):
            hit = unreal.SystemLibrary.sphere_trace_single(self.world, a, b, radius_cm, unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, [], unreal.DrawDebugTrace.NONE)
            if hit is not None:
                values = hit.to_tuple()
                return {"hit": True, "actor": str(values[9].get_name()) if len(values) > 9 and values[9] else None}
        return {"hit": False}

    def ground(self, position_cm):
        x, y, z = position_cm
        hit = unreal.SystemLibrary.line_trace_single(self.world, unreal.Vector(x, y, z + 200000.), unreal.Vector(x, y, z - 500000.),
                                                     unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, False, [], unreal.DrawDebugTrace.NONE, True)
        return {"ground_z_cm": float(hit.to_tuple()[5].z) if hit is not None else None}

    def memory(self):
        return memory_now()

    def collect_garbage(self):
        """UObject garbage collection between renders (optional restart policy). Returns the time it took."""
        started = time.monotonic()
        unreal.SystemLibrary.collect_garbage()
        return {"seconds": time.monotonic() - started}

    def console(self, command):
        """One console command on the loaded world, for the streaming clean-up experiments. Does not change project settings."""
        started = time.monotonic()
        unreal.SystemLibrary.execute_console_command(self.world, command)
        return {"seconds": time.monotonic() - started}

    def barrier(self):
        """The render-readiness barrier the capture tool uses at prime (flushes loading, streaming and rendering); for the leak experiments."""
        return {"seconds": core._finish_loading_before_screenshot("k-barrier")}

    def close(self):
        rig, self.rig = self.rig, None
        if rig:
            rig.destroy()


def main():
    server, number, last = None, 1, time.monotonic()
    try:
        server = Server()
        write_atomic(SESSION / "ready.json", {"pid": os.getpid()})
        while True:
            request = SESSION / f"request-{number:06d}.json"
            if not request.exists():
                # A client that died must not leave the GPU held.
                if time.monotonic() - last > CONFIG["idle_exit_s"]:
                    break
                time.sleep(.01)
                continue
            # The file crosses from WSL to Windows, which can briefly refuse access just after it appears.
            try:
                message = json.loads(request.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                time.sleep(.01)
                continue
            if message["op"] == "stop":
                write_atomic(SESSION / f"reply-{number:06d}.json", {"ok": True})
                break
            try:
                reply = {"ok": True, **getattr(server, message["op"])(**message["args"])}
            except Exception:
                reply = {"ok": False, "error": traceback.format_exc()}
            write_atomic(SESSION / f"reply-{number:06d}.json", reply)
            number, last = number + 1, time.monotonic()
    except Exception:
        write_atomic(SESSION / "failed.json", {"error": traceback.format_exc()})
    finally:
        if server:
            try:
                server.close()
            except Exception:
                pass


main()
