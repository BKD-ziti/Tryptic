#!/usr/bin/env python3
"""
mosh.py -- the datamosh engine behind Tryptic (app.py), plus a command-line version.

    python mosh.py clip.mp4             writes mosh_out/clip_mosh_<seed>.mp4
    python mosh.py clip.mp4 --second other.mp4     transition mosh: clip -> other

for the full app, double-click start.bat.

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
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import media  # shared frame-rate probe (never guesses)

HERE = Path(__file__).resolve().parent
WORK = HERE / "mosh_work"
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


FF = {"path": None}


# ----------------------------------------------------------------------------
# cli
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="datamosh from the command line (the full app is start.bat).")
    ap.add_argument("input", nargs="?", help="video to mosh (headless mode)")
    ap.add_argument("-o", "--output")
    ap.add_argument("--second", help="second clip for a transition mosh")
    ap.add_argument("--ffmpeg", help="path to ffmpeg(.exe) or its folder")
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
        sys.exit("give it a video to mosh (python mosh.py clip.mp4). for the full app, double-click start.bat.")
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
