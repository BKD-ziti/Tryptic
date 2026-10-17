#!/usr/bin/env python3
"""
Datamosher by BKDziti  (mosh.py)
Rethinking food, media, and interaction, online and in person.

A local datamosh tool. One file, no pip installs. Needs python 3.8+ and ffmpeg
(Start Mosh.bat fetches both on Windows if you don't have them).

    python mosh.py                      opens the ui in your browser
    python mosh.py clip.mp4             headless, writes mosh_out/clip_mosh_<seed>.mp4
    python mosh.py clip.mp4 --second other.mp4     transition mosh: clip -> other

how it works (so you can trust it):
  1. your clip is re-encoded as a bare MPEG-4 part 2 stream with only one keyframe
     (plus keyframes at the jump landings we choose). P-frames only store "how pixels move".
  2. we edit the stream bytes directly: delete the landing keyframes (the old image
     is now dragged around by the new motion) and loop P-frames (smear/bloom).
  3. ffmpeg decodes the broken stream and encodes a normal mp4 (GPU if it can).

the mosh itself is byte editing, so it's near-instant. the gpu only helps with the
final mp4 encode.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import media  # shared frame-rate probe (never guesses)

HERE = Path(__file__).resolve().parent
WORK = HERE / "mosh_work"
UPLOADS = WORK / "uploads"
OUT = HERE / "mosh_out"
NOWIN = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

DEFAULTS = dict(
    orient="source",      # source | vertical | horizontal
    res=0,                # 0 = same as the clip, else short side in px: 2160 (4K), 1440, 1080, 720, 480
    size_w=0,             # optional exact size in px (needs size_w and/or size_h), overrides orient/res
    size_h=0,
    fit="crop",           # crop (fill the frame) | pad (bars) when the shape changes
    fps=0,                # 0 = same frame rate as the clip
    color_out="keep",     # keep (match the clip, HDR stays HDR) | sdr (tone-map HDR to standard bt709)
    quality=3,            # mpeg4 qscale, 1 (clean) .. 31 (blocky, smearier)
    preview=0,            # seconds, 0 = whole clip
    jumps=3,              # random jumps (single clip mode)
    jump_min=2.0,         # seconds skipped per jump
    jump_max=6.0,
    manual_jumps="",      # "4>10, 22>30"  overrides random jumps (10>4 jumps back)
    jump_dir="forward",   # forward | either  (either = random backward jumps too)
    bloom_after_jump=True,
    bloom_hold=15,        # frames the smear lasts
    bloom_run=2,          # how many P-frames get looped
    smear_var=0.0,        # 0..1 random variation of each smear's length
    bloom_extra=0,        # extra random smears
    manual_bloom="",      # "7.5, 12"
    audio="follow",       # follow | none
    encoder="auto",       # auto = gpu when it works | gpu = gpu or fail | cpu
    out_cq=18,            # final mp4 quality, lower = better/bigger
    hwdecode=True,        # decode the source clip on the gpu (NVDEC) when possible
    motion_range=0,       # how far (px) a block may drag; 0 = no limit, small = tight
    mv4=False,            # 4 motion vectors per block (more detailed drag)
    seed=0,               # 0 = random
    b_start=0.0,          # second clip: start offset
    a_end=0.0,            # second clip mode: cut the first clip here (0 = play it all)
)


# ----------------------------------------------------------------------------
# ffmpeg plumbing
# ----------------------------------------------------------------------------

def find_ffmpeg(hint=None):
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    cands = []

    def add(p):
        p = Path(str(p).strip().strip('"'))
        cands.extend([p, p / exe, p / "bin" / exe])

    if hint:
        add(hint)
    add(HERE / "ffmpeg")
    if os.environ.get("MOSH_FFMPEG"):
        add(os.environ["MOSH_FFMPEG"])
    w = shutil.which("ffmpeg")
    if w:
        cands.append(Path(w))
    cands.append(HERE / exe)
    if os.name == "nt":
        la = os.environ.get("LOCALAPPDATA", "")
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        cands += [
            Path(r"C:\ffmpeg\bin") / exe,
            Path(pf) / "ffmpeg" / "bin" / exe,
            Path(la) / "Microsoft" / "WinGet" / "Links" / exe,
            Path.home() / "ffmpeg" / "bin" / exe,
        ]
    for c in cands:
        try:
            if c.is_file():
                return str(c)
        except OSError:
            pass
    return None


def run_ff(cmd, on_frame=None):
    """run ffmpeg, stream progress, return (returncode, tail of error text)"""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors="replace", **NOWIN)
    tail = []
    noise = ("fps=", "stream_", "bitrate=", "total_size=", "out_time", "dup_frames",
             "drop_frames", "speed=", "progress=")
    for line in p.stdout:
        line = line.strip()
        if line.startswith("frame="):
            if on_frame:
                try:
                    on_frame(int(line.split("=", 1)[1].strip()))
                except ValueError:
                    pass
        elif line and not line.startswith(noise):
            tail.append(line)
            tail = tail[-30:]
    p.wait()
    return p.returncode, "\n".join(tail)


def probe(ff, path):
    r = subprocess.run([ff, "-hide_banner", "-i", str(path)], capture_output=True,
                       text=True, errors="replace", **NOWIN)
    t = r.stderr
    info = {"duration": 0.0, "w": 0, "h": 0, "fps": 0.0, "audio": False, "rot": 0}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", t)
    if m:
        info["duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    for line in t.splitlines():
        if "Video:" in line and not info["w"]:
            m = re.search(r"(?<!\w)(\d{2,5})x(\d{2,5})(?=[\s,\[])", line)
            if m:
                info["w"], info["h"] = int(m[1]), int(m[2])
            m = re.search(r"(\d+(?:\.\d+)?)\s*fps", line)
            if m:
                info["fps"] = float(m[1])
        if "Audio:" in line:
            info["audio"] = True
    # phones store vertical video sideways + a rotation flag. instead of trusting how the flag
    # is written (it changes between ffmpeg versions), decode one frame and read the size
    # ffmpeg actually produces. that already includes rotation.
    coded = (info["w"], info["h"])
    m = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", t) or re.search(r"rotate\s*:\s*(-?\d+)", t)
    if m:
        info["rot"] = int(round(float(m[1])))
    if info["w"]:
        r2 = subprocess.run([ff, "-hide_banner", "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
                             "-f", "null", "-"], capture_output=True, text=True, errors="replace", **NOWIN)
        out = r2.stderr
        k = out.find("Output #0")
        got = False
        if k >= 0:
            for line in out[k:].splitlines():
                if "Video:" in line:
                    mm = re.search(r"(?<!\w)(\d{2,5})x(\d{2,5})(?=[\s,\[])", line)
                    if mm:
                        info["w"], info["h"] = int(mm[1]), int(mm[2])
                        got = True
                        ms = re.search(r"SAR (\d+):(\d+)", line)
                        if ms and int(ms[2]) and int(ms[1]) != int(ms[2]):
                            info["w"] = int(round(info["w"] * int(ms[1]) / int(ms[2])))
                    break
        if not got and abs(info["rot"]) % 180 == 90:   # fallback: trust the flag
            info["w"], info["h"] = info["h"], info["w"]
    info["coded"] = coded
    # the frame rate comes from media.probe_rate: header + measured timestamps, never a guess
    info["fps_exact"], info["vfr"], info["fps_how"] = None, False, ""
    try:
        ra = media.probe_rate(ff, path)
        info["fps_exact"], info["vfr"], info["fps_how"] = ra["fps"], ra["vfr"], ra["how"]
        info["fps"] = float(ra["fps"])
    except Exception:
        pass
    return info


def ff_has_encoder(ff, name):
    r = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True,
                       text=True, errors="replace", **NOWIN)
    return name in r.stdout


_NVENC = {}


def nvenc_check(ff):
    """does this ffmpeg + this machine actually encode on the NVIDIA gpu? returns (ok, message)"""
    if ff in _NVENC:
        return _NVENC[ff]
    if not ff_has_encoder(ff, "h264_nvenc"):
        res = (False, "this ffmpeg build has no NVENC")
    else:
        r = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=256x256:d=0.2", "-frames:v", "1", "-c:v", "h264_nvenc",
                            "-f", "null", "-"], capture_output=True, text=True, errors="replace", **NOWIN)
        if r.returncode == 0:
            res = (True, "ok")
        else:
            lines = [x.strip() for x in (r.stderr or "").splitlines() if x.strip()]
            key = [x for x in lines if re.search(r"nvenc|nvcuda|cuda|driver|capable", x, re.I)]
            msg = (key or lines or ["nvenc test failed"])[0]
            res = (False, re.sub(r"^\[[^\]]*\]\s*", "", msg))
    _NVENC[ff] = res
    return res


FF_URLS = [
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
]


def download_ffmpeg(say=print, urls=None):
    """grab a windows ffmpeg build (has NVENC) into ./ffmpeg/bin. returns True on success."""
    import urllib.request
    import zipfile
    dest = HERE / "ffmpeg" / "bin"
    dest.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=True)
    z = WORK / "ffmpeg_download.zip"
    ok = False
    for url in (urls or FF_URLS):
        try:
            say(f"  downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (mosh)"})
            with urllib.request.urlopen(req, timeout=60) as r, open(z, "wb") as fh:
                total = int(r.headers.get("Content-Length") or 0)
                got, last = 0, -1
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
                    got += len(b)
                    shown = int(got * 100 / total) if total else got >> 20
                    if shown != last:
                        last = shown
                        print(f"\r  {shown}{'%' if total else ' MB'}   ", end="", flush=True)
            print()
            ok = True
            break
        except Exception as e:  # noqa
            print()
            say(f"  that one failed ({e})")
            z.unlink(missing_ok=True)
    if not ok:
        return False
    try:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                base = name.rsplit("/", 1)[-1].lower()
                if base in ("ffmpeg.exe", "ffprobe.exe", "ffmpeg", "ffprobe"):
                    with zf.open(name) as src, open(dest / base, "wb") as out:
                        shutil.copyfileobj(src, out)
                    if not base.endswith(".exe"):
                        os.chmod(dest / base, 0o755)
    finally:
        z.unlink(missing_ok=True)
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    return (dest / exe).is_file()


def ensure_ffmpeg(hint=None, say=print):
    ff = find_ffmpeg(hint)
    if os.name == "nt":
        local = HERE / "ffmpeg" / "bin" / "ffmpeg.exe"
        is_local = bool(ff) and Path(ff).resolve() == local.resolve()
        if (ff is None or not ff_has_encoder(ff, "h264_nvenc")) and not is_local:
            why = "not found" if ff is None else "found, but it has no NVENC (gpu encoding)"
            say(f"  [..] ffmpeg {why}. getting a build with gpu support, one time (about 90 MB)")
            if download_ffmpeg(say):
                ff = find_ffmpeg(hint) or ff
    return ff


# ----------------------------------------------------------------------------
# mpeg-4 part 2 elementary stream surgery
# ----------------------------------------------------------------------------

def parse_chunks(data):
    """split a raw .m4v into frames. returns [(type 'I'/'P'/'B'/'S', bytes)].
    headers (VOS/VOL/GOP/user data) stay attached to the frame that follows them."""
    codes = []
    i = 0
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i < 0 or i + 3 >= len(data):
            break
        codes.append((i, data[i + 3]))
        i += 3
    starts = []
    pending = None
    for pos, code in codes:
        if code == 0xB6:  # VOP
            vt = data[pos + 4] >> 6 if pos + 4 < len(data) else 0
            starts.append((pending if pending is not None else pos, "IPBS"[vt]))
            pending = None
        elif pending is None:
            pending = pos
    chunks = []
    for k, (st, t) in enumerate(starts):
        en = starts[k + 1][0] if k + 1 < len(starts) else len(data)
        if k == 0:
            st = 0
        chunks.append((t, data[st:en]))
    return chunks


# ----------------------------------------------------------------------------
# planning
# ----------------------------------------------------------------------------

def parse_pairs(txt, fps, nframes, log):
    out = []
    for part in re.split(r"[;,\n]+", txt or ""):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:>|->|to)\s*(\d+(?:\.\d+)?)", part)
        if not m:
            log(f"skipping jump '{part}' (write it like 4>10)")
            continue
        f, l = int(round(float(m[1]) * fps)), int(round(float(m[2]) * fps))
        if not (0 < f < nframes - 2 and 0 < l < nframes - 2 and f != l):
            log(f"skipping jump '{part}' (both times must be inside the clip)")
            continue
        out.append((f, l))
    return out


def parse_times(txt, fps):
    out = []
    for part in re.split(r"[;,\s]+", txt or ""):
        if not part:
            continue
        try:
            out.append(int(round(float(part) * fps)))
        except ValueError:
            pass
    return out


def auto_jumps(n_frames, fps, n, jmin, jmax, rng, either=False):
    jumps = []
    cursor = 0
    min_seg = int(round(fps))
    jmin_f, jmax_f = max(2, int(jmin * fps)), max(3, int(jmax * fps))
    if jmax_f < jmin_f:
        jmin_f, jmax_f = jmax_f, jmin_f
    for k in range(n):
        left = n - k
        skip = rng.randint(jmin_f, jmax_f)
        back = either and rng.random() < 0.5
        lo = max(cursor + min_seg, skip + 2 if back else 0)
        hi = n_frames - (0 if back else skip) - min_seg * left
        if hi < lo:
            if back:  # no room to go back here, try forward instead
                back = False
                lo = cursor + min_seg
                hi = n_frames - skip - min_seg * left
            if hi < lo:
                break
        window_hi = min(hi, lo + max(0, (hi - lo) // left))
        f = rng.randint(lo, max(lo, window_hi))
        l = f - skip if back else f + skip
        jumps.append((f, l))
        cursor = l
    return jumps


def apply_bloom(out, at, run_len, hold):
    i = at
    while i < len(out) and i < at + 6 and out[i][0] != "P":
        i += 1
    if i >= len(out) or out[i][0] != "P":
        return False
    run = []
    j = i
    while j < len(out) and len(run) < max(1, run_len) and out[j][0] == "P":
        run.append(out[j])
        j += 1
    n = min(hold, len(out) - i)
    out[i:i + n] = (run * (n // len(run) + 1))[:n]
    return True


# ----------------------------------------------------------------------------
# the actual mosh
# ----------------------------------------------------------------------------

def fps_frac(f):
    """a clean frame rate: whole numbers stay whole, 29.97/59.94 snap to the real NTSC fractions."""
    f = float(f)
    if abs(f - round(f)) < 0.005:
        return Fraction(max(1, int(round(f))), 1)
    n = round(f * 1.001)
    if abs(f - n * 1000 / 1001) < 0.01:
        return Fraction(n * 1000, 1001)
    return Fraction(f).limit_denominator(1000)


def fps_arg(fr):
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


def _even(x):
    return max(2, int(round(x)) // 2 * 2)


def pick_size(w, h, orient="source", res=0, fit="crop", cw=0, ch=0):
    """output size for a clip that plays as w x h. returns (W, H, reshaped).
    orient: source | vertical | horizontal.  res: short side in px (2160 = 4K, 1080 = 1080p), 0 = same as the clip.
    cw/ch: optional exact size, wins over everything else."""
    clip_ar = w / h
    cw, ch = int(cw or 0), int(ch or 0)
    if cw or ch:
        if cw and ch:
            W, H = cw, ch
        elif cw:
            W, H = cw, cw / clip_ar
        else:
            W, H = ch * clip_ar, ch
        W, H = _even(W), _even(H)
        return W, H, abs(W / H - clip_ar) > 0.01 * clip_ar
    res = int(res or 0)
    clip_orient = "vertical" if h > w else ("horizontal" if w > h else "square")
    if orient not in ("vertical", "horizontal") or orient == clip_orient:
        ar, reshaped = clip_ar, False
    else:
        ar, reshaped = (16 / 9 if orient == "horizontal" else 9 / 16), True
    if res:
        W, H = (res * ar, res) if ar >= 1 else (res, res / ar)
    elif not reshaped:
        W, H = w, h
    else:
        wider = clip_ar > ar
        if fit == "pad":
            W, H = (w, w / ar) if wider else (h * ar, h)
        else:
            W, H = (h * ar, h) if wider else (w, w / ar)
    return _even(W), _even(H), reshaped


def encode_m4v(ff, src, dst, W, H, fps, q, key_times, t_limit, ss, fit, log, on_frame, opt, decode=""):
    """source -> bare mpeg-4 part 2 stream. returns True if the source was decoded on the gpu.
    decode: the color plan's first filters (tags the frames with the clip's real color, tone-maps if asked).
    the yuv values go through untouched (only the bit depth drops), and the final encode tags them again."""
    if fit == "pad":
        scale = (f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos:out_range=tv,"
                 f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    elif fit == "crop":
        scale = f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos:out_range=tv,crop={W}:{H},setsar=1"
    else:
        scale = f"scale={W}:{H}:flags=lanczos:out_range=tv,setsar=1"
    vf = (f"{decode}," if decode else "") + f"fps={fps},{scale},format=yuv420p"

    def build(hw):
        c = [ff, "-y", "-hide_banner", "-loglevel", "error", "-progress", "pipe:1", "-nostats"]
        if hw:
            c += ["-hwaccel", "cuda"]
        if ss:
            c += ["-ss", f"{ss:.3f}"]
        c += ["-i", str(src), "-an", "-vf", vf, "-c:v", "mpeg4", "-q:v", str(q),
              "-g", "100000", "-bf", "0", "-sc_threshold", "1000000000"]
        if opt.get("range"):
            c += ["-me_range", str(int(opt["range"]))]
        if opt.get("mv4"):
            c += ["-flags", "+mv4"]
        if key_times:
            c += ["-force_key_frames", ",".join(f"{t:.4f}" for t in sorted(key_times))]
        if t_limit:
            c += ["-t", f"{t_limit:.3f}"]
        return c + ["-f", "m4v", str(dst)]

    tries = [True, False] if opt.get("hw") else [False]
    msg = ""
    for hw in tries:
        rc, msg = run_ff(build(hw), on_frame)
        if rc == 0 and dst.exists() and dst.stat().st_size > 0:
            return hw
        if hw:
            log("gpu decode didn't work on this file, using cpu for the source decode")
    raise RuntimeError("ffmpeg couldn't prep the clip:\n" + msg)


def run_mosh(ff, s, a_path, out_path, log, prog, b_path=None):
    s = {**DEFAULTS, **s}
    seed = int(s["seed"] or 0) or random.randrange(1, 1_000_000)
    rng = random.Random(seed)
    q = min(31, max(1, int(s["quality"])))
    two = bool(b_path)
    mode = s["encoder"]
    log(f"seed {seed}")

    gpu_ok, gpu_msg = nvenc_check(ff)
    if mode == "gpu" and not gpu_ok:
        raise RuntimeError("gpu encode isn't available: " + gpu_msg +
                           "\n(update your NVIDIA driver, or set final encode to cpu)")
    opt = dict(range=max(0, min(1023, int(s["motion_range"] or 0))), mv4=bool(s["mv4"]),
               hw=bool(s["hwdecode"]) and gpu_ok and mode != "cpu")

    ia = probe(ff, a_path)
    if not ia["w"]:
        raise RuntimeError("ffmpeg couldn't read that file. is it a video?")
    cplan = media.color_plan(ff, a_path, s.get("color_out", "keep"), log=log)
    want_fps = float(s["fps"] or 0)
    if want_fps > 0:
        fr = fps_frac(want_fps)
    elif ia.get("fps_exact"):
        fr = ia["fps_exact"]          # the clip's own exact rate (e.g. 30000/1001)
    else:
        raise RuntimeError("couldn't read this clip's frame rate, so it won't guess. "
                           "type the rate into the frame rate box and run again.")
    fps = float(fr)           # used for all frame/time maths
    fps_s = fps_arg(fr)       # what ffmpeg gets
    fitmode = s["fit"] if s["fit"] in ("crop", "pad") else "crop"
    W, H, reshaped = pick_size(ia["w"], ia["h"], s["orient"], s["res"], fitmode, s["size_w"], s["size_h"])
    if ia.get("vfr"):
        log(f"this clip is variable frame rate (its header says {ia.get('fps_how')}); using the average, {fps:.3f} fps")
    log(f"output size {W}x{H} at {fps:.3f} fps (clip: {ia['w']}x{ia['h']}" +
        ((f", stored sideways as {ia['coded'][0]}x{ia['coded'][1]}" if abs(ia["rot"]) % 180 == 90
          else f", stored as {ia['coded'][0]}x{ia['coded'][1]}") if ia["coded"] != (ia["w"], ia["h"]) else "") +
        f", {ia['fps']:g} fps)")

    preview = float(s["preview"] or 0)
    a_len = ia["duration"] or 0
    if two and float(s["a_end"] or 0) > 0:
        a_len = min(a_len, float(s["a_end"])) if a_len else float(s["a_end"])
    if preview > 0:
        a_len = min(a_len, preview) if a_len else preview
    n_est = int(a_len * fps)

    # --- plan jumps (single clip mode)
    jumps = []
    if two:
        if s["manual_jumps"].strip() or int(s["jumps"]) > 0:
            log("second clip is set, so jumps are skipped (transition mode)")
    elif s["manual_jumps"].strip():
        jumps = parse_pairs(s["manual_jumps"], fps, n_est, log)
    elif int(s["jumps"]) > 0:
        jumps = auto_jumps(n_est, fps, int(s["jumps"]), float(s["jump_min"]), float(s["jump_max"]),
                           rng, s["jump_dir"] == "either")
    if jumps and all(l > f for f, l in jumps):
        jumps.sort()
    clean, prev_land = [], 0
    for f, l in jumps:
        if f > prev_land + 2 and l != f:
            clean.append((f, l))
            prev_land = l
        else:
            log(f"dropped overlapping jump {f / fps:.1f}>{l / fps:.1f}")
    jumps = clean
    keys = sorted({(l - 0.5) / fps for _, l in jumps})
    if jumps:
        log("jumps: " + ", ".join(f"{f / fps:.1f}s>{l / fps:.1f}s" for f, l in jumps))

    WORK.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="mosh_", dir=WORK))
    try:
        # --- encode A (and B)
        end_a = 0.3 if not two else 0.15
        prog(0.0, "prepping clip")
        a_m4v = tmp / "a.m4v"
        used_hw = encode_m4v(ff, a_path, a_m4v, W, H, fps_s, q, keys,
                             a_len if (preview or (two and s["a_end"])) else 0,
                             0, fitmode if reshaped else "plain", log,
                             lambda n: prog(min(end_a, end_a * n / max(1, n_est))), opt, cplan["decode"])
        log("source decode: " + ("GPU (NVDEC)" if used_hw else "cpu"))
        A = parse_chunks(a_m4v.read_bytes())
        if len(A) < 3:
            raise RuntimeError("clip is too short to mosh")
        log(f"clip: {len(A)} frames, {sum(1 for t, _ in A if t == 'I')} keyframes")

        out, marks, aud = [], [], []
        ib = None
        if not two:
            ikeys = [i for i, (t, _) in enumerate(A) if t == "I"]
            pos, astart = 0, 0.0
            for f, l in jumps:
                li = min((i for i in ikeys if i > 0 and abs(i - l) <= 3), key=lambda i: abs(i - l), default=None)
                if li is None or li + 1 >= len(A) or f >= len(A) or f <= pos:
                    log(f"couldn't land jump {f / fps:.1f}>{l / fps:.1f}, skipped")
                    continue
                out += A[pos:f]
                aud.append((1, astart, f / fps))
                marks.append(len(out))
                pos, astart = li + 1, (li + 1) / fps
            out += A[pos:]
            aud.append((1, astart, None))
        else:
            out = list(A)
            aud.append((1, 0.0, len(A) / fps))
            remain = None
            if preview > 0:
                remain = preview - len(A) / fps
            if remain is not None and remain <= 0.2:
                log("preview ends before the second clip starts")
            else:
                ib = probe(ff, b_path)
                if not ib["w"]:
                    raise RuntimeError("couldn't read the second clip")
                b_m4v = tmp / "b.m4v"
                bs = float(s["b_start"] or 0)
                bplan = media.color_plan(ff, b_path, s.get("color_out", "keep"))
                if bplan["out"] != cplan["out"]:
                    log("note: the second clip's color (" + media.describe_color(bplan["src"]) + ") doesn't match the "
                        "first one's, so it will look off after the melt. shoot both the same way (both HDR or both SDR).")
                encode_m4v(ff, b_path, b_m4v, W, H, fps_s, q, [], remain or 0, bs, fitmode, log,
                           lambda n: prog(end_a + 0.15 * min(1.0, n / max(1, n_est))), opt, bplan["decode"])
                B = parse_chunks(b_m4v.read_bytes())
                if len(B) < 3:
                    raise RuntimeError("second clip is too short to mosh")
                marks.append(len(out))
                out += B[1:]   # B[0] is its keyframe: gone
                aud.append((2, bs + 1.0 / fps, None))
                log(f"second clip: {len(B)} frames, its opening keyframe removed")

        stray = sum(1 for c in out[1:] if c[0] == "I")
        if stray:
            out = out[:1] + [c for c in out[1:] if c[0] != "I"]
            log(f"removed {stray} extra keyframe(s) so the mosh doesn't reset")

        # --- smears
        extra = parse_times(s["manual_bloom"], fps)
        hold = max(1, int(s["bloom_hold"]))
        var = min(1.0, max(0.0, float(s["smear_var"] or 0)))
        spacing = int(hold * (1 + var)) + 2
        for _ in range(int(s["bloom_extra"])):
            lo, hi = int(round(fps)), len(out) - spacing - int(round(fps))
            if hi > lo:
                extra.append(rng.randint(lo, hi))
        pts = set(extra)
        if s["bloom_after_jump"]:
            pts |= set(marks)
        kept, last = [], -10 ** 9
        for p in sorted(pts):
            if p - last >= spacing and 0 < p < len(out):
                kept.append(p)
                last = p
        done = 0
        for p in sorted(kept, reverse=True):
            h = max(1, int(round(hold * (1 + rng.uniform(-var, var))))) if var else hold
            if apply_bloom(out, p, int(s["bloom_run"]), h):
                done += 1
        log(f"{done} smear(s) applied")

        mosh = tmp / "mosh.m4v"
        mosh.write_bytes(b"".join(c[1] for c in out))
        total = len(out)
        log(f"final: {total} frames ({total / fps:.1f}s)")

        # --- decode the broken stream + encode the real mp4
        use_audio = s["audio"] == "follow" and ia["audio"] and (not two or (ib and ib["audio"]))
        if s["audio"] == "follow" and not use_audio:
            log("no usable audio, going silent")
        if mode == "cpu":
            enc_order = ["cpu"]
        elif mode == "gpu":
            enc_order = ["gpu"]
        else:
            enc_order = (["gpu"] if gpu_ok else []) + ["cpu"]
        if mode == "auto" and not gpu_ok:
            log("gpu encode unavailable (" + gpu_msg + "), using cpu")
        cq = str(min(40, max(1, int(s["out_cq"]))))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for kind in enc_order:
            cmd = [ff, "-y", "-hide_banner", "-loglevel", "fatal", "-progress", "pipe:1", "-nostats",
                   "-framerate", fps_s, "-f", "m4v", "-i", str(mosh)]
            if use_audio:
                cmd += ["-i", str(a_path)]
                if two and ib:
                    cmd += ["-i", str(b_path)]
                parts, labels = [], []
                for k, (inp, st, en) in enumerate(aud):
                    trim = f"atrim=start={st:.4f}" + (f":end={en:.4f}" if en is not None else "")
                    parts.append(f"[{inp}:a]{trim},asetpts=PTS-STARTPTS[a{k}]")
                    labels.append(f"[a{k}]")
                fc = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(aud)}:v=0:a=1[aout]"
                cmd += ["-filter_complex", fc, "-map", "0:v:0", "-map", "[aout]",
                        "-c:a", "aac", "-b:a", "192k", "-shortest"]
            else:
                cmd += ["-map", "0:v:0", "-an"]
            # the edited stream's own timing is meaningless after cuts, so stamp frames at a steady rate
            cmd += ["-vf", f"setpts=N*{fr.denominator}/({fr.numerator}*TB)," + media.yuv_to_out(cplan, kind == "gpu"),
                    "-r", fps_s]
            cmd += media.codec_args(cplan, kind == "gpu", cq) + ["-movflags", "+faststart", str(out_path)]
            enc_name = ("GPU (NVENC" + (" HEVC 10-bit)" if cplan["hdr"] else ")")) if kind == "gpu" else \
                ("cpu (x265 10-bit)" if cplan["hdr"] else "cpu (x264)")
            prog(0.3, "encoding on the " + enc_name)
            rc, msg = run_ff(cmd, lambda n: prog(0.3 + 0.7 * min(1.0, n / max(1, total))))
            if rc == 0 and out_path.exists() and out_path.stat().st_size > 0:
                log("final encode: " + enc_name)
                try:
                    fo = probe(ff, out_path)
                    log(f"result file: {fo['w']}x{fo['h']} at {fo['fps']:g} fps"
                        + ("  (same as the clip)" if (fo["w"], fo["h"]) == (ia["w"], ia["h"]) else ""))
                except Exception:  # noqa
                    pass
                prog(1.0, "done")
                return {"seed": seed, "frames": total, "gpu": kind == "gpu", "color": cplan}
            log(f"{kind} encode failed" + (f": {msg}" if msg else ""))
        raise RuntimeError("final encode failed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------
# local web ui
# ----------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Datamosher by BKDziti</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%231c2038'/%3E%3Crect x='5' y='9' width='14' height='14' fill='%23b8287f'/%3E%3Crect x='11' y='9' width='6' height='14' fill='%232a8f62' opacity='.85'/%3E%3Crect x='17' y='9' width='10' height='14' fill='%232a8f62' opacity='.45'/%3E%3C/svg%3E">
<style>
:root{
  --paper:#e8e9f0; --panel:#f5f5f9; --ink:#1c2038; --soft:#585d7a; --line:#c6c9da;
  --mag:#b8287f; --grn:#1f7a53; --field:#ffffff;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark){
  :root{ --paper:#14172b; --panel:#1b1f39; --ink:#e7e8f3; --soft:#9a9fc2; --line:#31365a;
         --mag:#e2579f; --grn:#4cc790; --field:#11142a; }
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--paper);color:var(--ink);
  font:15px/1.5 "Segoe UI Variable Text","Segoe UI",system-ui,sans-serif;}
header{max-width:1180px;margin:0 auto;padding:28px 24px 4px;display:flex;flex-wrap:wrap;
  align-items:flex-end;justify-content:space-between;gap:12px}
h1{margin:0;font:700 76px/0.95 "Bahnschrift","DIN Alternate","Arial Narrow",system-ui,sans-serif;
  letter-spacing:-0.02em;
  text-shadow:
    4px 0 0 color-mix(in srgb,var(--mag) 75%,transparent),
    8px 0 0 color-mix(in srgb,var(--mag) 50%,transparent),
    12px 0 0 color-mix(in srgb,var(--grn) 38%,transparent),
    16px 0 0 color-mix(in srgb,var(--grn) 22%,transparent),
    20px 0 0 color-mix(in srgb,var(--grn) 10%,transparent);}
.by{margin:8px 0 0;font:600 20px/1.2 "Bahnschrift","DIN Alternate","Segoe UI",system-ui,sans-serif;color:var(--mag)}
.sub{color:var(--soft);margin:10px 0 0;max-width:60ch}
footer{max-width:1180px;margin:0 auto;padding:18px 24px 40px;border-top:1px solid var(--line);color:var(--soft);font-size:13.5px}
footer b{color:var(--ink)}
footer a{color:var(--mag);font-weight:600;text-decoration:none}
footer a:hover{text-decoration:underline}
#gpu{font-weight:600;padding:6px 12px;border:1px solid var(--line);border-radius:5px;background:var(--panel);max-width:46ch;font-size:14px}
#gpu.ok{border-color:var(--grn);color:var(--grn)}
#gpu.bad{border-color:var(--mag);color:var(--mag)}
main{max-width:1180px;margin:0 auto;padding:20px 24px 60px;display:grid;
  grid-template-columns:minmax(320px,450px) 1fr;gap:28px;align-items:start}
@media (max-width:900px){ main{grid-template-columns:1fr} h1{font-size:56px} }
fieldset{border:1px solid var(--line);background:var(--panel);border-radius:6px;
  margin:0 0 18px;padding:14px 16px 6px}
legend{font-weight:700;padding:0 6px;font-size:16px}
.row{margin:0 0 14px}
.row label{display:block;font-weight:600;margin-bottom:3px}
.hint{display:block;color:var(--soft);font-size:13px;margin-top:2px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}
input[type=number],input[type=text],select{width:100%;padding:7px 9px;border:1px solid var(--line);
  border-radius:4px;background:var(--field);color:var(--ink);font:inherit}
input[type=range]{width:100%;accent-color:var(--mag)}
input[type=checkbox]{accent-color:var(--mag);margin-right:7px}
.check label{font-weight:600;display:flex;align-items:center}
:focus-visible{outline:2px solid var(--mag);outline-offset:2px}
.drop{display:block;border:1.5px dashed var(--line);border-radius:6px;padding:14px;text-align:center;
  background:var(--field);cursor:pointer;font-weight:400 !important}
.drop.over{border-color:var(--mag);background:color-mix(in srgb,var(--mag) 8%,var(--field))}
.drop b{display:block}
.drop span{color:var(--soft);font-size:13px;word-break:break-all}
.drop input{display:none}
button{font:inherit;font-weight:700;border:0;border-radius:5px;padding:12px 22px;cursor:pointer;
  background:var(--mag);color:#fff;width:100%;font-size:17px}
button:disabled{opacity:.55;cursor:default}
button.ghost{background:transparent;color:var(--ink);border:1px solid var(--line);width:auto;
  font-size:14px;padding:6px 12px;font-weight:600}
button.ghost:hover{border-color:var(--mag);color:var(--mag)}
.seg{display:flex;flex-wrap:wrap;gap:6px}
.seg button{width:auto;background:var(--field);color:var(--ink);border:1px solid var(--line);
  font-size:14px;font-weight:600;padding:7px 13px;border-radius:5px}
.seg button:hover{border-color:var(--mag)}
.seg button.on{background:var(--mag);border-color:var(--mag);color:#fff}
.presets{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}
.side{position:sticky;top:16px}
.screen{background:#000;border-radius:6px;aspect-ratio:16/9;display:grid;place-items:center;overflow:hidden}
.screen video{width:100%;height:100%;object-fit:contain;background:#000}
.screen p{color:#8c90ad;margin:0;padding:20px;text-align:center}
.bar{height:8px;background:var(--line);border-radius:4px;margin:16px 0 6px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--mag),var(--grn));transition:width .25s}
#stage{font-weight:600;min-height:1.5em}
#result{margin:8px 0}
#result a{color:var(--mag);font-weight:600}
.acts{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
#hist{display:flex;flex-direction:column;gap:6px;margin-top:14px}
#hist button{width:100%;text-align:left;background:var(--panel);color:var(--ink);border:1px solid var(--line);
  font-size:13.5px;font-weight:500;padding:7px 10px}
#hist button.on{border-color:var(--mag)}
pre{background:var(--field);border:1px solid var(--line);border-radius:6px;padding:10px 12px;
  font:12.5px/1.5 Consolas,"Cascadia Mono",monospace;max-height:230px;overflow:auto;white-space:pre-wrap;margin:10px 0 0}
details summary{cursor:pointer;font-weight:600;margin-bottom:10px}
.warn{border-color:var(--mag)}
</style>
</head>
<body>
<header>
  <div>
    <h1>datamosher</h1>
    <p class="by">by BKDziti</p>
    <p class="sub">datamosh video on your own machine. your gpu, your footage, nothing uploaded anywhere.</p>
  </div>
  <div id="gpu">checking gpu...</div>
</header>
<main>
<div>
  <fieldset id="ffbox" class="warn" hidden>
    <legend>ffmpeg</legend>
    <div class="row">
      <label for="ffpath">where is ffmpeg?</label>
      <input type="text" id="ffpath" placeholder="C:\ffmpeg\bin\ffmpeg.exe  (or just the folder)">
      <span class="hint">i couldn't find it on my own. point me at it once.</span>
    </div>
    <div class="row"><button id="ffset" type="button" style="font-size:15px;padding:9px">use this ffmpeg</button></div>
  </fieldset>

  <fieldset>
    <legend>clips</legend>
    <div class="row">
      <label class="drop" id="drop-a"><b>drop a video here or click to pick</b><span id="st-a">nothing loaded</span>
        <input type="file" id="file-a" accept="video/*,.mkv,.avi,.mov,.webm"></label>
    </div>
    <details>
      <summary>second clip (transition mosh)</summary>
      <div class="row">
        <label class="drop" id="drop-b"><b>drop the clip you want to melt into</b><span id="st-b">optional</span>
          <input type="file" id="file-b" accept="video/*,.mkv,.avi,.mov,.webm"></label>
        <span class="hint">the first clip plays, then the second clip's opening keyframe is deleted so it drags the first one's pixels. random jumps are off in this mode.</span>
      </div>
      <div class="pair">
        <div class="row"><label for="a_end">cut first clip at (s)</label><input type="number" id="a_end" value="0" min="0" step="0.5"><span class="hint">0 plays it all</span></div>
        <div class="row"><label for="b_start">second clip starts at (s)</label><input type="number" id="b_start" value="0" min="0" step="0.5"></div>
      </div>
    </details>
  </fieldset>

  <fieldset>
    <legend>size</legend>
    <div class="row">
      <label>shape</label>
      <div class="seg" data-for="orient">
        <button type="button" data-v="source" class="on">same as the clip</button>
        <button type="button" data-v="vertical">vertical</button>
        <button type="button" data-v="horizontal">horizontal</button>
      </div>
      <input type="hidden" id="orient" value="source">
    </div>
    <div class="row">
      <label>resolution</label>
      <div class="seg" data-for="res">
        <button type="button" data-v="0" class="on">same as the clip</button>
        <button type="button" data-v="2160">4K</button>
        <button type="button" data-v="1440">1440p</button>
        <button type="button" data-v="1080">1080p</button>
        <button type="button" data-v="720">720p</button>
        <button type="button" data-v="480">480p</button>
      </div>
      <input type="hidden" id="res" value="0">
    </div>
    <div class="row" id="fitrow" hidden>
      <label for="fit">the shape is changing, so</label>
      <select id="fit"><option value="crop">crop the edges to fill the frame</option><option value="pad">keep everything and add bars</option></select>
    </div>
    <div class="row"><b id="outsize">load a clip to see the output size</b></div>
    <details>
      <summary>set an exact size instead (optional)</summary>
      <div class="pair">
        <div class="row"><label for="size_w">width (px)</label><input type="number" id="size_w" min="0" step="2" placeholder="auto"></div>
        <div class="row"><label for="size_h">height (px)</label><input type="number" id="size_h" min="0" step="2" placeholder="auto"></div>
      </div>
      <span class="hint" style="margin:-6px 0 12px">fill both to force that exact size and ignore the shape and resolution above. fill just one and the other follows the clip's shape. clear them to go back.</span>
    </details>
  </fieldset>

  <fieldset>
    <legend>the look</legend>
    <div class="presets" id="presets">
      <button type="button" class="ghost" data-p="classic">classic melt</button>
      <button type="button" class="ghost" data-p="heavy">heavy smear</button>
      <button type="button" class="ghost" data-p="stutter">glitch stutter</button>
      <button type="button" class="ghost" data-p="gentle">gentle drift</button>
      <button type="button" class="ghost" data-p="reset">reset all</button>
    </div>
    <span class="hint" style="margin:-6px 0 14px">presets just fill in the boxes below. tweak anything after.</span>

    <div class="row">
      <label for="jumps">jumps</label>
      <input type="number" id="jumps" value="3" min="0" max="20">
      <span class="hint">the video leaps and its keyframe is deleted, so the old picture gets dragged by the new motion.</span>
    </div>
    <div class="pair">
      <div class="row"><label for="jump_min">jump size, min (s)</label><input type="number" id="jump_min" value="2" min="0.2" step="0.5"></div>
      <div class="row"><label for="jump_max">jump size, max (s)</label><input type="number" id="jump_max" value="6" min="0.2" step="0.5"></div>
    </div>
    <div class="row"><label for="jump_dir">jump direction</label>
      <select id="jump_dir"><option value="forward">forward only</option><option value="either">forward or back</option></select>
      <span class="hint">going back replays footage, so the clip gets longer.</span></div>
    <div class="row">
      <label for="manual_jumps">or place jumps yourself</label>
      <input type="text" id="manual_jumps" placeholder="4>10, 22>30, 15>6">
      <span class="hint">from second &gt; to second, in order. 15&gt;6 jumps back. fills in over the random ones.</span>
    </div>

    <div class="row check"><label><input type="checkbox" id="bloom_after_jump" checked>smear after every jump</label></div>
    <div class="pair">
      <div class="row"><label for="bloom_hold">smear length (frames)</label><input type="number" id="bloom_hold" value="15" min="1" max="600"></div>
      <div class="row"><label for="bloom_run">motion loop (frames)</label><input type="number" id="bloom_run" value="2" min="1" max="30"></div>
    </div>
    <span class="hint" style="margin:-6px 0 14px">smear length is how long pixels keep dragging. motion loop 1 is a clean drag, higher wobbles.</span>
    <div class="row">
      <label for="smear_var">smear variation <span id="svv">0</span>%</label>
      <input type="range" id="smear_var" min="0" max="100" value="0">
      <span class="hint">0 makes every smear the same length. higher makes them uneven.</span>
    </div>
    <div class="pair">
      <div class="row"><label for="bloom_extra">extra random smears</label><input type="number" id="bloom_extra" value="0" min="0" max="30"></div>
      <div class="row"><label for="manual_bloom">smears at (s)</label><input type="text" id="manual_bloom" placeholder="7.5, 12"></div>
    </div>

    <div class="row">
      <label for="quality">blockiness <span id="qv">3</span></label>
      <input type="range" id="quality" min="1" max="31" value="3">
      <span class="hint">low is crisp. high gives chunky blocks and heavier smear.</span>
    </div>
    <div class="row">
      <label for="motion_range">motion reach (px)</label>
      <input type="number" id="motion_range" value="0" min="0" max="1023">
      <span class="hint">how far a block can drag. 0 is no limit. try 8 to 24 for tight, jittery motion.</span>
    </div>
    <div class="row check"><label><input type="checkbox" id="mv4">detailed motion (4 vectors per block)</label>
      <span class="hint">finer, more shredded drag.</span></div>
    <div class="row"><label for="seed">seed</label><input type="number" id="seed" value="0" min="0">
      <span class="hint">0 rolls a new one every time. keep a seed to repeat a look while you tweak other things.</span></div>
  </fieldset>

  <fieldset>
    <legend>output</legend>
    <div class="row"><label for="fps">frame rate (fps)</label><input type="number" id="fps" value="0" min="0" max="240" step="any"><span class="hint">0 = same as the clip</span></div>
    <div class="pair">
      <div class="row"><label for="preview">preview first (s)</label><input type="number" id="preview" value="0" min="0" step="1"><span class="hint">0 does the whole clip</span></div>
      <div class="row"><label for="audio">audio</label><select id="audio"><option value="follow">follow the cuts</option><option value="none">none</option></select></div>
    </div>
    <div class="row"><label for="encoder">final encode</label>
      <select id="encoder">
        <option value="auto">gpu if it works, else cpu</option>
        <option value="gpu">gpu only (stop if it can't)</option>
        <option value="cpu">cpu</option>
      </select></div>
    <div class="pair">
      <div class="row"><label for="out_cq">output quality</label><input type="number" id="out_cq" value="18" min="1" max="40"><span class="hint">lower is cleaner and bigger</span></div>
      <div class="row check" style="align-self:center"><label><input type="checkbox" id="hwdecode" checked>decode source on gpu</label></div>
    </div>
  </fieldset>

  <button id="go" type="button">mosh it</button>
</div>

<div class="side">
  <div class="screen" id="screen"><p>your result plays here</p></div>
  <div class="bar"><i id="bar"></i></div>
  <div id="stage">&nbsp;</div>
  <div id="result"></div>
  <div class="acts" id="acts" hidden>
    <button type="button" class="ghost" id="keepseed">keep this seed</button>
    <button type="button" class="ghost" id="openfolder">open output folder</button>
  </div>
  <div id="hist"></div>
  <pre id="log" hidden></pre>
</div>
</main>
<footer>
  <b>Datamosher by BKDziti.</b> Rethinking food, media, and interaction, online and in person.<br>
  <a href="https://bkdziti.com" target="_blank" rel="noopener">bkdziti.com</a>
</footer>

<script>
const $ = id => document.getElementById(id);
const num = id => Number($(id).value);
const clips = {a:null, b:null};
let busy = false, lastSeed = 0;
const hist = [];

const FIELDS = ['orient','res','fit','size_w','size_h','fps','quality','preview','jumps','jump_min','jump_max','manual_jumps','jump_dir',
  'bloom_after_jump','bloom_hold','bloom_run','smear_var','bloom_extra','manual_bloom','motion_range','mv4',
  'seed','audio','encoder','out_cq','hwdecode','a_end','b_start'];
const DEFAULTS = {};
for (const id of FIELDS){ const e = $(id); DEFAULTS[id] = e.type === 'checkbox' ? e.checked : e.value; }

function setVals(v){
  for (const [k, val] of Object.entries(v)){
    const e = $(k); if (!e) continue;
    if (e.type === 'checkbox') e.checked = !!val;
    else if (e.tagName === 'SELECT' && ![...e.options].some(o => o.value === String(val))) continue;
    else e.value = val;
  }
  syncLabels(); save();
}
function syncSeg(){
  document.querySelectorAll('.seg').forEach(g => {
    const v = $(g.dataset.for).value;
    g.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  });
}
document.querySelectorAll('.seg').forEach(g => g.addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  $(g.dataset.for).value = b.dataset.v;
  syncSeg(); save(); refreshSize();
}));
function syncLabels(){ $('qv').textContent = $('quality').value; $('svv').textContent = $('smear_var').value; syncSeg(); }
function save(){
  const o = {};
  for (const id of FIELDS){ const e = $(id); o[id] = e.type === 'checkbox' ? e.checked : e.value; }
  try{ localStorage.setItem('mosh.settings.v4', JSON.stringify(o)); }catch(e){}
}
function load(){
  try{ const o = JSON.parse(localStorage.getItem('mosh.settings.v4') || 'null'); if (o) setVals(o); }catch(e){}
  syncLabels();
}
for (const id of FIELDS){ $(id).addEventListener('input', () => { syncLabels(); save(); }); }

const PRESETS = {
  classic: {jumps:3, jump_min:2, jump_max:6, jump_dir:'forward', manual_jumps:'', bloom_after_jump:true, bloom_hold:20, bloom_run:2,
            smear_var:0, bloom_extra:0, manual_bloom:'', quality:4, motion_range:0, mv4:false},
  heavy:   {jumps:4, jump_min:1, jump_max:4, jump_dir:'forward', manual_jumps:'', bloom_after_jump:true, bloom_hold:50, bloom_run:1,
            smear_var:30, bloom_extra:3, manual_bloom:'', quality:12, motion_range:0, mv4:false},
  stutter: {jumps:3, jump_min:1, jump_max:3, jump_dir:'either', manual_jumps:'', bloom_after_jump:true, bloom_hold:9, bloom_run:3,
            smear_var:60, bloom_extra:10, manual_bloom:'', quality:8, motion_range:16, mv4:true},
  gentle:  {jumps:2, jump_min:3, jump_max:8, jump_dir:'forward', manual_jumps:'', bloom_after_jump:true, bloom_hold:10, bloom_run:1,
            smear_var:0, bloom_extra:0, manual_bloom:'', quality:2, motion_range:0, mv4:false}
};
$('presets').addEventListener('click', e => {
  const p = e.target.dataset && e.target.dataset.p; if (!p) return;
  if (p === 'reset') setVals(DEFAULTS); else setVals(PRESETS[p]);
});

async function api(path, opt){
  const r = await fetch(path, opt);
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

async function upload(slot, file){
  const st = $('st-'+slot);
  st.textContent = 'loading ' + file.name + '...';
  try{
    const j = await api('/api/upload?name=' + encodeURIComponent(file.name), {method:'POST', body:file});
    clips[slot] = j;
    const shape = j.w < j.h ? 'vertical' : (j.w > j.h ? 'horizontal' : 'square');
    st.textContent = `${j.name}:  ${j.w}x${j.h}, ${shape}, ${+j.fps.toFixed(2)} fps, ${j.duration.toFixed(1)}s${j.audio ? '' : ', no audio'}`;
    if (slot === 'a'){
      const first = document.querySelector('[data-for="orient"] [data-v="source"]');
      first.textContent = 'same as the clip (' + shape + ')';
    }
    if (slot === 'a') refreshSize();
  }catch(e){
    clips[slot] = null;
    st.textContent = 'could not load: ' + e.message;
  }
}
for (const slot of ['a','b']){
  const drop = $('drop-'+slot), inp = $('file-'+slot);
  inp.addEventListener('change', () => inp.files[0] && upload(slot, inp.files[0]));
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('over');
    if (e.dataTransfer.files[0]) upload(slot, e.dataTransfer.files[0]);
  });
}

function setStage(t){ $('stage').textContent = t; }

let sizeTimer = null;
function refreshSize(){
  clearTimeout(sizeTimer);
  sizeTimer = setTimeout(async () => {
    if (!clips.a) return;
    try{
      const j = await api('/api/size', {method:'POST', body:JSON.stringify({
        id: clips.a.id, orient: $('orient').value, res: num('res'), fit: $('fit').value,
        size_w: num('size_w'), size_h: num('size_h'), fps: num('fps')})});
      const shape = j.w < j.h ? 'vertical' : (j.w > j.h ? 'horizontal' : 'square');
      const same = j.w === j.src_w && j.h === j.src_h;
      let note = ', same size as the clip';
      if (!same){
        const px = j.w * j.h, src = j.src_w * j.src_h;
        note = px > src * 1.02 ? ', bigger than the clip, so it gets upscaled' : (px < src * 0.98 ? ', smaller than the clip' : '');
      }
      $('fitrow').hidden = !j.reshaped;
      $('outsize').textContent = `output: ${j.w}x${j.h} (${shape}) at ${+j.fps.toFixed(2)} fps${note}`;
    }catch(e){ $('outsize').textContent = e.message; }
  }, 150);
}
for (const id of ['fit','size_w','size_h','fps']) $(id).addEventListener('input', refreshSize);

function showGpu(g){
  const el = $('gpu');
  if (g && g.ok){ el.className = 'ok'; el.textContent = 'gpu encoding ready (NVENC)'; }
  else { el.className = 'bad'; el.textContent = 'gpu encoding unavailable: ' + ((g && g.msg) || 'unknown') + '. using cpu until fixed.'; }
}
async function checkInfo(){
  try{
    const j = await api('/api/info');
    $('ffbox').hidden = !!j.ffmpeg;
    showGpu(j.gpu);
  }catch(e){ setStage(e.message); }
}
$('ffset').addEventListener('click', async () => {
  try{
    const j = await api('/api/ffmpeg', {method:'POST', body:JSON.stringify({path:$('ffpath').value})});
    $('ffbox').hidden = !!j.ffmpeg; showGpu(j.gpu);
    setStage(j.ffmpeg ? 'ffmpeg found' : "still can't find ffmpeg there");
  }catch(e){ setStage(e.message); }
});
$('openfolder').addEventListener('click', () => api('/api/open', {method:'POST', body:'{}'}).catch(e => setStage(e.message)));
$('keepseed').addEventListener('click', () => { if (lastSeed){ $('seed').value = lastSeed; save(); setStage('seed ' + lastSeed + ' locked in'); } });

async function go(){
  if (busy) return;
  if (!clips.a){ setStage('pick a clip first'); return; }
  busy = true; $('go').disabled = true;
  $('result').innerHTML = ''; $('acts').hidden = true; $('log').hidden = false; $('log').textContent = '';
  $('bar').style.width = '0%'; setStage('starting...');
  const body = {
    a: clips.a.id, b: clips.b ? clips.b.id : null,
    orient: $('orient').value, res: num('res'), fit: $('fit').value, size_w: num('size_w'), size_h: num('size_h'),
    fps: num('fps'), quality: num('quality'), preview: num('preview'),
    jumps: num('jumps'), jump_min: num('jump_min'), jump_max: num('jump_max'),
    manual_jumps: $('manual_jumps').value, jump_dir: $('jump_dir').value,
    bloom_after_jump: $('bloom_after_jump').checked, bloom_hold: num('bloom_hold'), bloom_run: num('bloom_run'),
    smear_var: num('smear_var') / 100, bloom_extra: num('bloom_extra'), manual_bloom: $('manual_bloom').value,
    motion_range: num('motion_range'), mv4: $('mv4').checked,
    audio: $('audio').value, encoder: $('encoder').value, out_cq: num('out_cq'), hwdecode: $('hwdecode').checked,
    seed: num('seed'), b_start: num('b_start'), a_end: num('a_end')
  };
  try{
    const {job} = await api('/api/run', {method:'POST', body:JSON.stringify(body)});
    await poll(job, body);
  }catch(e){ setStage('error: ' + e.message); }
  busy = false; $('go').disabled = false;
}
$('go').addEventListener('click', go);

function play(file){
  $('screen').innerHTML = '<video controls loop autoplay src="/out/' + encodeURIComponent(file) + '"></video>';
  document.querySelectorAll('#hist button').forEach(b => b.classList.toggle('on', b.dataset.f === file));
}
function renderHist(){
  const h = $('hist'); h.innerHTML = '';
  for (const it of hist.slice().reverse()){
    const b = document.createElement('button');
    b.dataset.f = it.file; b.textContent = it.label;
    b.addEventListener('click', () => { play(it.file); $('result').innerHTML = 'seed ' + it.seed; lastSeed = it.seed; $('acts').hidden = false; });
    h.appendChild(b);
  }
}

async function poll(job, body){
  while (true){
    const j = await api('/api/job/' + job);
    $('bar').style.width = Math.round(j.progress * 100) + '%';
    setStage(j.stage || '');
    const lg = $('log'); lg.textContent = j.log.join('\n'); lg.scrollTop = lg.scrollHeight;
    if (j.status === 'error'){ setStage('failed. see the log below'); return; }
    if (j.status === 'done'){
      lastSeed = j.seed;
      const url = '/out/' + encodeURIComponent(j.file);
      $('result').innerHTML = 'saved to <b>' + j.path + '</b><br>final encode on the <b>' + (j.gpu ? 'GPU' : 'cpu') +
        '</b>. seed ' + j.seed + '<br><a href="' + url + '?dl=1" download>download</a>';
      $('acts').hidden = false;
      hist.push({file:j.file, seed:j.seed, label:'seed ' + j.seed + ', blockiness ' + body.quality + ', smear ' + body.bloom_hold + ', ' + body.jumps + ' jumps'});
      renderHist(); play(j.file);
      setStage('done');
      return;
    }
    await new Promise(r => setTimeout(r, 450));
  }
}
load();
checkInfo();
</script>
</body>
</html>
"""

JOBS = {}
CLIPS = {}
FF = {"path": None}


def gpu_info():
    if not FF["path"]:
        return {"ok": False, "msg": "no ffmpeg yet"}
    ok, msg = nvenc_check(FF["path"])
    return {"ok": ok, "msg": msg}


def open_folder(path):
    path = str(path)
    if os.name == "nt":
        os.startfile(path)  # noqa
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def safe_name(n):
    n = os.path.basename(n.replace("\\", "/"))
    n = re.sub(r"[^\w.\- ]+", "_", n).strip() or "clip"
    return n[:120]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        if u.path == "/api/info":
            return self._send(200, {"ffmpeg": FF["path"], "gpu": gpu_info()})
        if u.path.startswith("/api/job/"):
            j = JOBS.get(u.path.rsplit("/", 1)[1])
            if not j:
                return self._send(404, {"error": "no such job"})
            return self._send(200, j)
        if u.path.startswith("/out/"):
            f = OUT / unquote(u.path[5:])
            try:
                f = f.resolve()
                if OUT.resolve() not in f.parents or not f.is_file():
                    raise FileNotFoundError
            except (OSError, FileNotFoundError):
                return self._send(404, {"error": "not found"})
            return self._serve_file(f, "dl" in parse_qs(u.query))
        self._send(404, {"error": "not found"})

    def _serve_file(self, f, download):
        size = f.stat().st_size
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            try:
                if a:
                    start = int(a)
                    end = int(b) if b else size - 1
                elif b:
                    start = max(0, size - int(b))
                code = 206
            except ValueError:
                pass
        end = min(end, size - 1)
        self.send_response(code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{f.name}"')
        self.end_headers()
        try:
            with open(f, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/upload":
                return self._upload(parse_qs(u.query).get("name", ["clip"])[0])
            data = json.loads(self._body() or b"{}")
            if u.path == "/api/ffmpeg":
                FF["path"] = find_ffmpeg(data.get("path")) or FF["path"]
                return self._send(200, {"ffmpeg": FF["path"], "gpu": gpu_info()})
            if u.path == "/api/size":
                c = CLIPS.get(data.get("id"))
                if not c:
                    return self._send(200, {"error": "no clip"})
                i = c["info"]
                w, h, reshaped = pick_size(i["w"], i["h"], data.get("orient", "source"), data.get("res", 0),
                                           data.get("fit", "crop"), data.get("size_w", 0), data.get("size_h", 0))
                want = float(data.get("fps") or 0)
                eff = float(fps_frac(want)) if want > 0 else float(i.get("fps_exact") or i["fps"] or 0)
                return self._send(200, {"w": w, "h": h, "src_w": i["w"], "src_h": i["h"], "reshaped": reshaped,
                                        "fps": eff, "src_fps": i["fps"]})
            if u.path == "/api/open":
                OUT.mkdir(exist_ok=True)
                open_folder(OUT)
                return self._send(200, {"ok": True})
            if u.path == "/api/run":
                return self._run(data)
        except Exception as e:  # noqa
            return self._send(200, {"error": str(e)})
        self._send(404, {"error": "not found"})

    def _upload(self, name):
        if not FF["path"]:
            self._body()
            return self._send(200, {"error": "ffmpeg not found yet"})
        UPLOADS.mkdir(parents=True, exist_ok=True)
        cid = uuid.uuid4().hex[:10]
        name = safe_name(name)
        dst = UPLOADS / f"{cid}_{name}"
        left = int(self.headers.get("Content-Length") or 0)
        with open(dst, "wb") as fh:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                fh.write(chunk)
                left -= len(chunk)
        info = probe(FF["path"], dst)
        if not info["w"]:
            dst.unlink(missing_ok=True)
            return self._send(200, {"error": "that doesn't look like a video"})
        CLIPS[cid] = {"path": dst, "name": name, "info": info}
        self._send(200, {"id": cid, "name": name, **info})

    def _run(self, data):
        if not FF["path"]:
            return self._send(200, {"error": "ffmpeg not found yet"})
        a = CLIPS.get(data.get("a"))
        b = CLIPS.get(data.get("b")) if data.get("b") else None
        if not a:
            return self._send(200, {"error": "load a clip first"})
        jid = uuid.uuid4().hex[:8]
        job = {"status": "running", "progress": 0.0, "stage": "starting", "log": []}
        JOBS[jid] = job
        settings = {k: data[k] for k in DEFAULTS if k in data}
        stem = Path(a["name"]).stem
        seed_hint = int(settings.get("seed") or 0)

        def work():
            try:
                def log(m):
                    job["log"].append(m)

                def prog(p, stage=None):
                    job["progress"] = p
                    if stage:
                        job["stage"] = stage

                tag = seed_hint or random.randrange(1, 1_000_000)
                settings["seed"] = tag
                out = OUT / f"{stem}_mosh_{tag}.mp4"
                res = run_mosh(FF["path"], settings, a["path"], out, log, prog, b["path"] if b else None)
                job.update(status="done", file=out.name, path=str(out), seed=res["seed"],
                           gpu=res.get("gpu", False), progress=1.0)
            except Exception as e:  # noqa
                job["log"].append(str(e))
                job["status"] = "error"

        threading.Thread(target=work, daemon=True).start()
        self._send(200, {"job": jid})


def serve(port, open_browser=True):
    OUT.mkdir(exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    srv = None
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    if not srv:
        sys.exit("couldn't open a local port")
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print()
    print(f"  Datamosher by BKDziti is running at {url}")
    print("  leave this window open while you use it. close it (or ctrl+c) to quit.")
    print()
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("bye")


# ----------------------------------------------------------------------------
# cli
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Datamosher by BKDziti: a local datamosh tool. run with no args for the ui.")
    ap.add_argument("input", nargs="?", help="video to mosh (headless mode)")
    ap.add_argument("-o", "--output")
    ap.add_argument("--second", help="second clip for a transition mosh")
    ap.add_argument("--ffmpeg", help="path to ffmpeg(.exe) or its folder")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--check", action="store_true", help="just check python/ffmpeg/gpu and exit")
    d = DEFAULTS
    ap.add_argument("--jumps", type=int, default=d["jumps"])
    ap.add_argument("--jump-min", type=float, default=d["jump_min"])
    ap.add_argument("--jump-max", type=float, default=d["jump_max"])
    ap.add_argument("--manual-jumps", default=d["manual_jumps"], help='like "4>10, 22>30"')
    ap.add_argument("--orient", choices=["source", "vertical", "horizontal"], default=d["orient"])
    ap.add_argument("--res", type=int, default=d["res"], help="short side: 2160 (4K), 1440, 1080, 720, 480. 0 = same as the clip")
    ap.add_argument("--size", default="", help="exact size like 1080x1920 (wins over --orient/--res)")
    ap.add_argument("--fit", choices=["crop", "pad"], default=d["fit"])
    ap.add_argument("--jump-dir", choices=["forward", "either"], default=d["jump_dir"])
    ap.add_argument("--smear-var", type=float, default=d["smear_var"], help="0..1")
    ap.add_argument("--motion-range", type=int, default=d["motion_range"])
    ap.add_argument("--mv4", action="store_true")
    ap.add_argument("--out-cq", type=int, default=d["out_cq"])
    ap.add_argument("--no-hwdecode", action="store_true")
    ap.add_argument("--no-bloom-after-jump", action="store_true")
    ap.add_argument("--bloom-hold", type=int, default=d["bloom_hold"])
    ap.add_argument("--bloom-run", type=int, default=d["bloom_run"])
    ap.add_argument("--bloom-extra", type=int, default=d["bloom_extra"])
    ap.add_argument("--manual-bloom", default=d["manual_bloom"])
    ap.add_argument("--fps", type=float, default=d["fps"], help="0 = same as the clip")
    ap.add_argument("--quality", type=int, default=d["quality"])
    ap.add_argument("--preview", type=float, default=d["preview"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audio", choices=["follow", "none"], default=d["audio"])
    ap.add_argument("--encoder", choices=["auto", "gpu", "cpu"], default=d["encoder"])
    ap.add_argument("--b-start", type=float, default=0.0)
    ap.add_argument("--a-end", type=float, default=0.0)
    ap.add_argument("--color", choices=["keep", "sdr"], default="keep",
                    help="keep = same color as the clip (HDR stays HDR). sdr = tone-map HDR to standard bt709")
    a = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:  # noqa
            pass
    print(f"  [ok] python {sys.version.split()[0]}")
    FF["path"] = ensure_ffmpeg(a.ffmpeg)
    if FF["path"]:
        print(f"  [ok] ffmpeg  {FF['path']}")
        ok, msg = nvenc_check(FF["path"])
        if ok:
            print("  [ok] gpu encoding (NVENC) works")
        else:
            print(f"  [!!] gpu encoding unavailable: {msg}")
            print("       update your NVIDIA driver (nvidia.com/drivers). until then mosh uses the cpu.")
    else:
        print("  [!!] no ffmpeg. install one (or pass --ffmpeg) and run again.")
    if a.check:
        return
    if not a.input:
        return serve(a.port, not a.no_browser)
    if not FF["path"]:
        sys.exit("can't find ffmpeg. pass --ffmpeg C:\\path\\to\\ffmpeg.exe or set MOSH_FFMPEG")

    src = Path(a.input)
    size_w = size_h = 0
    if a.size:
        m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", a.size)
        if not m:
            sys.exit("--size should look like 1080x1920")
        size_w, size_h = int(m[1]), int(m[2])
    settings = dict(
        jumps=a.jumps, jump_min=a.jump_min, jump_max=a.jump_max, manual_jumps=a.manual_jumps,
        bloom_after_jump=not a.no_bloom_after_jump, bloom_hold=a.bloom_hold, bloom_run=a.bloom_run,
        bloom_extra=a.bloom_extra, manual_bloom=a.manual_bloom,
        orient=a.orient, res=a.res, size_w=size_w, size_h=size_h, fps=a.fps, quality=a.quality, preview=a.preview, seed=a.seed,
        audio=a.audio, encoder=a.encoder, b_start=a.b_start, a_end=a.a_end,
        fit=a.fit, jump_dir=a.jump_dir, smear_var=a.smear_var, motion_range=a.motion_range, mv4=a.mv4,
        out_cq=a.out_cq, hwdecode=not a.no_hwdecode, color_out=a.color,
    )
    seed = a.seed or random.randrange(1, 1_000_000)
    settings["seed"] = seed
    out = Path(a.output) if a.output else OUT / f"{src.stem}_mosh_{seed}.mp4"
    t0 = time.time()
    last = [-1]

    def prog(p, stage=None):
        pct = int(p * 100)
        if pct != last[0] and pct % 5 == 0:
            last[0] = pct
            print(f"\r{pct:3d}%  {stage or ''}      ", end="", flush=True)

    run_mosh(FF["path"], settings, src, out, lambda m: print("\n" + m), prog, a.second)
    print(f"\ndone in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
