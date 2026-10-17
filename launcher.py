#!/usr/bin/env python3
"""
launcher.py -- sets everything up and starts Tryptic. start.bat runs this; you never need to.

first run : makes a private python environment (labenv), installs PyTorch (the GPU build if it finds an
            NVIDIA card), opencv, gradio, pillow and ffmpeg, then opens the UI. the big download only happens once.
every run : quickly checks that everything still works, repairs what doesn't, then opens the UI.

    start.bat repair       wipe the environment and rebuild it from scratch (handled by start.bat)
    --skip-install       (debugging) don't check/install packages, just start the app
only the standard library is used here, so it runs before anything is installed.
"""

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = HERE / "labenv"
IS_WIN = os.name == "nt"
ENV_PY = ENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
GPU_MARKER = ENV / ".gpu_tried"          # so we only try the GPU install once, not on every launch

CUDA_BUILDS = ["cu128", "cu126", "cu124", "cu121"]   # newest first; older ones need older drivers
OTHER_PACKAGES = {"cv2": "opencv-python", "numpy": "numpy", "gradio": "gradio>=4.40,<7", "imageio_ffmpeg": "imageio-ffmpeg", "PIL": "pillow"}


def say(msg=""):
    print(msg, flush=True)


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], **kw)


def quiet(cmd):
    return run(cmd, capture_output=True, text=True)


def pip(*args, loud=True):
    cmd = [sys.executable, "-m", "pip", *args]
    return run(cmd) if loud else quiet(cmd)


def has_nvidia():
    for c in ("nvidia-smi", r"C:\Windows\System32\nvidia-smi.exe",
              r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if shutil.which(c) or Path(c).is_file():
            return True
    return False


def torch_state():
    """None if torch/torchvision won't import, else (version, gpu_visible)."""
    r = quiet([sys.executable, "-c",
               "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"])
    if r.returncode != 0 or not r.stdout.strip():
        return None
    ver, cuda = r.stdout.strip().splitlines()[-1].split()[-2:]
    return ver, cuda == "True"


def install_torch(nvidia):
    pip("install", "--upgrade", "pip", loud=False)
    pip("uninstall", "-y", "torch", "torchvision", loud=False)
    if nvidia:
        for build in CUDA_BUILDS:
            say(f"\ninstalling PyTorch ({build}). big download (~2.5 GB), but it only happens once...")
            r = pip("install", "torch", "torchvision", "--index-url", f"https://download.pytorch.org/whl/{build}")
            if r.returncode != 0:
                say(f"{build} wouldn't install, trying the next one...")
                continue
            st = torch_state()
            if st and st[1]:
                return
            say(f"{build} installed but your GPU isn't visible to it, trying an older build...")
    if torch_state() is None:
        say("\ninstalling PyTorch (CPU build, which is slow but works)...")
        pip("install", "torch", "torchvision")


def ensure_packages():
    nvidia = has_nvidia()
    st = torch_state()
    if st is None:
        say("PyTorch isn't working in this environment yet. installing it now...")
        install_torch(nvidia)
        if nvidia:
            GPU_MARKER.write_text("1")
    elif nvidia and not st[1] and not GPU_MARKER.exists():
        say(f"PyTorch {st[0]} can't see your NVIDIA GPU (it's a CPU-only build). installing the GPU build...")
        install_torch(nvidia)
        GPU_MARKER.write_text("1")

    st = torch_state()
    if st is None:
        say("\nPyTorch still won't load. copy everything in this window into an issue on the GitHub page.")
        sys.exit(1)

    missing = [pkg for mod, pkg in OTHER_PACKAGES.items() if quiet([sys.executable, "-c", f"import {mod}"]).returncode != 0]
    if missing:
        say(f"installing {', '.join(missing)}...")
        if pip("install", *missing).returncode != 0:
            say("\nsome packages failed to install. check your internet connection and run start.bat again.")
            sys.exit(1)

    st = torch_state()
    say(f"\nPyTorch {st[0]}  |  GPU: {'yes' if st[1] else 'NO, running on CPU (slow)'}")
    if not st[1] and nvidia:
        say("your NVIDIA GPU isn't usable by PyTorch. updating your NVIDIA driver (nvidia.com/drivers) usually fixes it.")


def main():
    inside_env = Path(sys.prefix).resolve() == ENV.resolve()
    if not inside_env:
        if ENV_PY.exists() and quiet([ENV_PY, "-c", "pass"]).returncode != 0:
            say("the old environment is broken. rebuilding it...")
            shutil.rmtree(ENV, ignore_errors=True)
        if not ENV_PY.exists():
            if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
                say(f"this python is {sys.version.split()[0]}, and PyTorch needs 3.10 to 3.13. install Python 3.12 from python.org.")
                sys.exit(1)
            say("first run: creating a private python environment...")
            venv.create(ENV, with_pip=True)
        sys.exit(run([ENV_PY, __file__, *sys.argv[1:]]).returncode)

    if "--skip-install" not in sys.argv:
        ensure_packages()
    say("\nstarting the UI. it opens in your browser. close this window to quit.\n")
    sys.exit(run([sys.executable, str(HERE / "app.py")], cwd=HERE).returncode)


if __name__ == "__main__":
    main()
