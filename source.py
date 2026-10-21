"""Locate the existing SR3M/HAD implementation without copying its source tree."""
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
SR3M = Path(os.environ.get("SR3M_SOURCE", PROJECT.parent / "SR3M")).resolve()
if not (SR3M / "vln_benchmark/unified/had.py").is_file():
    raise RuntimeError("Set SR3M_SOURCE to the existing SR3M checkout")
sys.path[:0] = [str(SR3M), str(SR3M / "vln_benchmark")]

