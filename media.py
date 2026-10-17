"""
media.py -- the one place that decides what a video's frame rate is.

why this exists: every engine (deepdream, datamosh, ascii) used to find the frame rate its own way, and each
way could quietly fail and fall back to a guess (24, 30...). here there is no guessing:

  1. ask ffprobe (or `ffmpeg -i` when there's no ffprobe) for the header rate
  2. read the timestamps of the first few hundred packets and measure the real rate
     (a phone clip or a screen recording is often *variable* frame rate, and its header says "60" or "30"
      while the footage really averages 26 or 29.6)
  3. if neither works, raise an error. never invent a number.

rates are exact fractions (30000/1001, not 29.97003) all the way to ffmpeg, so a 29.97 clip comes out 29.97.
"""
import json
import os
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

NOWIN = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

# the rates real cameras and editors actually use
STANDARD = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
            Fraction(48), Fraction(50), Fraction(60000, 1001), Fraction(60), Fraction(120000, 1001),
            Fraction(120), Fraction(240)]


def snap(f, tol=0.0005):
    """a clean exact rate. within 0.05% of a standard rate -> that exact rate, otherwise the closest simple fraction.
    (tight on purpose: 24 and 23.976 are only 0.1% apart and they must never be mixed up.)"""
    f = float(f)
    for s in STANDARD:
        if abs(f - float(s)) / float(s) <= tol:
            return s
    return Fraction(f).limit_denominator(1000)


def typed_rate(f):
    """a rate somebody typed in. 24 means exactly 24. 23.976 / 29.97 / 59.94 mean the NTSC rates (24000/1001...)."""
    f = float(f)
    if abs(f - round(f)) < 1e-6:
        return Fraction(int(round(f)))
    return snap(f)


def fps_arg(fr):
    """what to hand ffmpeg: '30' or '30000/1001'."""
    fr = Fraction(fr)
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


def fps_text(fr):
    f = float(fr)
    return f"{f:.3f}".rstrip("0").rstrip(".") + " fps"


def _rate(s):
    """'30000/1001' or '29.97' -> Fraction, or None when it's missing / 0/0 / absurd (90000 is a timebase, not fps)."""
    if s in (None, "", "N/A", "0/0"):
        return None
    try:
        r = Fraction(str(s))
    except (ValueError, ZeroDivisionError):
        return None
    return r if Fraction(1, 2) < r <= 1000 else None


def ffprobe_for(ff):
    exe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    if ff:
        side = Path(ff).with_name(exe)
        if side.is_file():
            return str(side)
    return shutil.which("ffprobe")


def _run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout, **NOWIN)


def _header_rate(ff, fp, path):
    """(avg, r) as Fractions or None, from ffprobe; from the `ffmpeg -i` text if there's no ffprobe."""
    if fp:
        try:
            r = _run([fp, "-v", "error", "-select_streams", "v:0", "-show_entries",
                      "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(path)])
            st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
            return _rate(st.get("avg_frame_rate")), _rate(st.get("r_frame_rate")), "ffprobe"
        except Exception:
            pass
    try:
        t = _run([ff, "-hide_banner", "-i", str(path)]).stderr
        line = next((l for l in t.splitlines() if "Video:" in l), "")
        fps = re.search(r"(\d+(?:\.\d+)?)\s*fps", line)
        tbr = re.search(r"(\d+(?:\.\d+)?)(k?)\s*tbr", line)
        avg = _rate(fps[1]) if fps else None
        r = _rate(float(tbr[1]) * (1000 if tbr[2] else 1)) if tbr else None
        return avg, r, "ffmpeg -i text"
    except Exception:
        return None, None, "none"


def _measure(fp, path, n=600):
    """real rate from the timestamps of the first n video packets: (median_rate, mean_rate, count) or None.
    only reads the start of the file, so it's quick on any size."""
    if not fp:
        return None
    try:
        r = _run([fp, "-v", "error", "-select_streams", "v:0", "-read_intervals", f"%+#{n}",
                  "-show_entries", "packet=pts_time,dts_time", "-of", "csv=p=0", str(path)], timeout=90)
    except Exception:
        return None
    ts = []
    for line in r.stdout.splitlines():
        for part in line.split(","):
            try:
                ts.append(float(part))
                break
            except ValueError:
                continue
    ts = sorted(set(ts))
    if len(ts) < 30:
        return None
    d = sorted(b - a for a, b in zip(ts, ts[1:]))
    med = d[len(d) // 2]
    mean = (ts[-1] - ts[0]) / (len(ts) - 1)
    if med <= 0 or mean <= 0:
        return None
    return 1 / med, 1 / mean, len(ts)


def probe_rate(ff, path):
    """the clip's frame rate. returns dict(fps=Fraction, how=str, vfr=bool, header=float|None).
    raises RuntimeError if it can't be worked out (so nothing downstream ever runs on a guessed rate)."""
    fp = ffprobe_for(ff)
    avg, r, src = _header_rate(ff, fp, path)
    header = avg or r
    m = _measure(fp, path)
    if m:
        med_rate, mean_rate, _ = m
        steady = abs(med_rate - mean_rate) / mean_rate < 0.03
        if steady:
            if header and abs(float(header) - mean_rate) / mean_rate < 0.01:
                return dict(fps=snap(header), how=f"{src} header, confirmed by timestamps", vfr=False,
                            header=float(header))
            return dict(fps=snap(mean_rate), how="measured from timestamps", vfr=False,
                        header=float(header) if header else None)
        return dict(fps=Fraction(round(mean_rate * 1000), 1000), how="measured average (variable frame rate)", vfr=True,
                    header=float(header) if header else None)
    if header:
        return dict(fps=snap(header), how=f"{src} header", vfr=False, header=float(header))
    raise RuntimeError("couldn't work out this clip's frame rate. type one into the frame rate box and try again.")


def verify_output(ff, path, want, log=print):
    """after a render: does the finished file really play at `want`? logs one line either way. returns True/False."""
    try:
        got = probe_rate(ff, path)["fps"]
    except Exception as e:
        log(f"couldn't re-check the output's frame rate ({e})")
        return False
    ok = abs(float(got) - float(want)) / float(want) < 0.002
    log(("frame rate check: " + fps_text(got) + " (as asked)") if ok else
        f"frame rate check: the file came out {fps_text(got)} but {fps_text(want)} was asked for")
    return ok


def thumbnail(ff, path, at=0.0, width=640, plan=None):
    """one frame of the clip as an RGB numpy array (for the preview), or None. with a color plan it's decoded
    with the right color math and HDR is squeezed into something that reads right on a normal screen."""
    import numpy as np
    try:
        vf = (plan["decode"] + "," if plan else "") + f"scale={width}:-2,format=rgb24"
        r = subprocess.run([ff, "-v", "error", "-ss", f"{max(0.0, at):.3f}", "-i", str(path), "-frames:v", "1",
                            "-vf", vf, "-f", "image2pipe", "-vcodec", "png", "-"],
                           capture_output=True, timeout=60, **NOWIN)
        if not r.stdout:
            return None
        import cv2
        img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
        return None if img is None else preview_rgb(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), plan)
    except Exception:
        return None


# ==================================================================================================
# color -- the other thing every engine has to agree on.
#
# why this exists: phones (iPhones especially) shoot HDR by default: 10-bit HEVC, bt2020 color, HLG or PQ
# brightness curve. the engines used to read those pixels as if they were plain SDR video, convert RGB back to
# YUV with the old bt601 matrix, and write the file with no color tags at all. a player then guesses "bt709 SDR",
# and the HDR picture shows up flat, grey and desaturated.
#
# the rule now:
#   1. probe what the clip really is (matrix, primaries, transfer curve, range, bit depth)
#   2. tag the decoded frames with exactly that, so every YUV<->RGB conversion uses the right math
#   3. write the output with the same color and the same tags. HDR stays HDR (10-bit HEVC, which phones play
#      as HDR), SDR stays SDR. or, if asked, tone-map HDR down to standard bt709 SDR.
# ==================================================================================================

COLOR_MODES = {"keep": "match the clip", "sdr": "standard SDR (tone-mapped)"}
HDR_TRC = {"arib-std-b67": "HLG", "smpte2084": "PQ"}
_SWS = {"bt709": "bt709", "bt2020nc": "bt2020", "bt2020c": "bt2020", "smpte170m": "smpte170m",
        "bt470bg": "bt470", "fcc": "fcc", "smpte240m": "smpte240m"}
_UNKNOWN = (None, "", "unknown", "unspecified", "reserved")
_CACHE = {}


def _ff_list(ff, what):
    key = (ff, what)
    if key not in _CACHE:
        try:
            _CACHE[key] = _run([ff, "-hide_banner", f"-{what}"]).stdout
        except Exception:
            _CACHE[key] = ""
    return _CACHE[key]


def has_encoder(ff, name):
    return re.search(rf"^\s*V\S*\s+{re.escape(name)}\s", _ff_list(ff, "encoders"), re.M) is not None


def has_filter(ff, name):
    return re.search(rf"^\s*\S+\s+{re.escape(name)}\s", _ff_list(ff, "filters"), re.M) is not None


def probe_color(ff, path):
    """what color the clip is really in. dict(matrix, primaries, trc, range, pix_fmt, bits, hdr, dovi, tagged)."""
    fp = ffprobe_for(ff)
    st, dovi = {}, False
    if fp:
        try:
            r = _run([fp, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(path)])
            st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
            dovi = any("DOVI" in str(s.get("side_data_type", "")) for s in st.get("side_data_list") or [])
        except Exception:
            st = {}
    if not st:                                          # no ffprobe: read the `ffmpeg -i` line instead
        try:
            t = _run([ff, "-hide_banner", "-i", str(path)]).stderr
            line = next((l for l in t.splitlines() if "Video:" in l), "")
            m = re.search(r"(yuv\w+|nv12|p010\w*|rgb\w+|gbr\w+)\((tv|pc)(?:, ([\w-]+)/([\w-]+)/([\w-]+)|, ([\w-]+))?", line)
            if m:
                st = dict(pix_fmt=m[1], color_range=m[2], color_space=m[3] or m[6], color_primaries=m[4] or m[6],
                          color_transfer=m[5] or m[6])
            sz = re.search(r"(?<!\w)(\d{2,5})x(\d{2,5})(?=[\s,\[])", line)
            if sz:
                st["width"], st["height"] = int(sz[1]), int(sz[2])
            dovi = "DOVI" in t
        except Exception:
            pass
    g = lambda k: None if st.get(k) in _UNKNOWN else st.get(k)
    matrix, prim, trc, rng = g("color_space"), g("color_primaries"), g("color_transfer"), g("color_range")
    tagged = bool(matrix or prim or trc)
    pix = st.get("pix_fmt") or "yuv420p"
    bits = 10 if ("10" in pix or "p010" in pix) else (12 if "12" in pix else 8)
    hd = max(int(st.get("width") or 0), int(st.get("height") or 0)) >= 1280 or min(
        int(st.get("width") or 0), int(st.get("height") or 0)) >= 720
    # fill the gaps the way players do
    if not matrix:
        matrix = "bt2020nc" if prim == "bt2020" else ("bt709" if (hd or prim == "bt709") else "smpte170m")
    if not prim:
        prim = {"bt2020nc": "bt2020", "bt2020c": "bt2020", "bt709": "bt709", "bt470bg": "bt470bg"}.get(matrix, "smpte170m")
    if not trc:
        trc = "bt709" if prim in ("bt709", "bt2020") else "smpte170m"
    if matrix not in _SWS:                              # rgb / ycgco / something exotic: treat as bt709
        matrix = "bt709"
    rng = "pc" if rng in ("pc", "full", "jpeg") else "tv"
    return dict(matrix=matrix, primaries=prim, trc=trc, range=rng, pix_fmt=pix, bits=bits,
                hdr=HDR_TRC.get(trc), dovi=dovi, tagged=tagged)


def describe_color(c):
    if c["hdr"]:
        s = f"HDR ({c['hdr']}{', Dolby Vision' if c.get('dovi') else ''}), {c['primaries']}, {c['bits']}-bit"
    else:
        s = f"SDR, {c['matrix']}, {c['bits']}-bit"
    return s + ("" if c.get("tagged", True) else " (untagged, assumed)")


def _setparams(c):
    return (f"setparams=range={c['range']}:color_primaries={c['primaries']}:color_trc={c['trc']}"
            f":colorspace={c['matrix']}")


SDR709 = dict(matrix="bt709", primaries="bt709", trc="bt709", range="tv")


def color_plan(ff, path, mode="keep", log=None):
    """decide the color path for one render. returns dict:
        src     what the clip is (probe_color)
        out     matrix/primaries/trc/range the output is written in
        hdr     True when the output is HDR (10-bit HEVC)
        decode  filters to put first in every decode chain (tags the frames, tone-maps if asked)
        sig     short string that changes whenever the pixel values the engines see would change (frame caches)
        note    one line for the log"""
    mode = mode if mode in COLOR_MODES else "keep"
    c = probe_color(ff, path)
    src_tag = _setparams(c)
    can_hevc = has_encoder(ff, "libx265") or has_encoder(ff, "hevc_nvenc")
    can_tm = has_filter(ff, "zscale") and has_filter(ff, "tonemap")
    want_sdr = bool(c["hdr"]) and (mode == "sdr" or not can_hevc)
    if want_sdr and not can_tm:
        want_sdr = False
        why = "this ffmpeg can't tone-map (no zscale), so it stays HDR"
    elif want_sdr and mode != "sdr":
        why = "no HEVC encoder in this ffmpeg, so it was tone-mapped to SDR"
    else:
        why = ""
    if want_sdr:
        decode = (src_tag + ",zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                  "tonemap=tonemap=mobius:param=0.3:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
                  + _setparams(SDR709))
        out, hdr = dict(SDR709), False
        note = f"color: clip is {describe_color(c)} -> tone-mapped to standard SDR (bt709)"
    else:
        decode = src_tag
        out = dict(matrix=c["matrix"], primaries=c["primaries"], trc=c["trc"], range="tv")
        hdr = bool(c["hdr"])
        note = f"color: clip is {describe_color(c)} -> output kept the same" + (" (10-bit HEVC, plays as HDR)" if hdr else "")
    if why:
        note += f"  ({why})"
    sig = f"c2-{'sdr' if want_sdr else 'keep'}-{c['matrix']}-{c['primaries']}-{c['trc']}-{c['range']}"
    plan = dict(src=c, out=out, hdr=hdr, decode=decode, sig=sig, note=note, tonemapped=want_sdr)
    if log:
        log(note)
    return plan


def out_tags(plan):
    """encoder flags so the file says what color it is."""
    o = plan["out"]
    return ["-colorspace", o["matrix"], "-color_primaries", o["primaries"], "-color_trc", o["trc"],
            "-color_range", o["range"]]


def out_pix(plan, gpu):
    return ("p010le" if gpu else "yuv420p10le") if plan["hdr"] else "yuv420p"


def rgb_to_out(plan, gpu):
    """filters that take RGB frames (from numpy / png) to the output's YUV with the right matrix, then tag them.
    (left alone, ffmpeg converts RGB->YUV with the bt601 matrix and writes no tags: that's the old color shift.)"""
    o = plan["out"]
    return (f"format=rgb24,scale=out_color_matrix={_SWS.get(o['matrix'], 'bt709')}:out_range={o['range']},"
            f"format={out_pix(plan, gpu)},{_setparams(o)}")


def yuv_to_out(plan, gpu):
    """filters for a YUV stream already in the output's color (datamosh): just bit depth + tags."""
    return f"{_setparams(plan['out'])},format={out_pix(plan, gpu)}"


def codec_args(plan, gpu, q):
    """video encoder args (without -pix_fmt; the filter chain sets the format). q = crf / cq number."""
    q = str(q)
    if plan["hdr"]:
        a = (["-c:v", "hevc_nvenc", "-profile:v", "main10", "-preset", "p5", "-rc", "vbr", "-cq", q, "-b:v", "0"]
             if gpu else ["-c:v", "libx265", "-preset", "fast", "-crf", q, "-x265-params", "log-level=error"])
        a += ["-tag:v", "hvc1"]                         # so apple devices recognize it
    else:
        a = (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", q, "-b:v", "0"]
             if gpu else ["-c:v", "libx264", "-preset", "veryfast", "-crf", q])
    return a + ["-pix_fmt", out_pix(plan, gpu)] + out_tags(plan)


def decoded_size(ff, path):
    """(w, h) of a decoded frame, rotation included."""
    import numpy as np
    import cv2
    r = subprocess.run([ff, "-v", "error", "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
                        "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, timeout=120, **NOWIN)
    img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_UNCHANGED) if r.stdout else None
    if img is None:
        raise RuntimeError("couldn't decode a frame from that video")
    return img.shape[1], img.shape[0]


def read_frames(ff, path, plan, start=0.0, fps=None, length=0.0, size=None, pix="bgr24"):
    """generator of uint8 frames (H, W, 3) decoded by ffmpeg with the plan's color handling.
    fps: output rate (frames are picked by timestamp, like every other engine). size: (w, h) if already known."""
    import numpy as np
    w, h = size or decoded_size(ff, path)
    vf = [plan["decode"]] + ([f"fps={fps_arg(fps)}"] if fps else []) + [f"format={pix}"]
    cmd = [ff, "-v", "error"] + (["-ss", f"{start:.3f}"] if start > 0 else [])
    cmd += ["-i", str(path), "-map", "0:v:0", "-an"] + (["-t", f"{length:.3f}"] if length > 0 else [])
    cmd += ["-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", pix, "-"]
    n = w * h * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=n * 2, **NOWIN)
    try:
        while True:
            raw = p.stdout.read(n)
            if len(raw) < n:
                break
            yield np.frombuffer(bytearray(raw), np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def grab_frame(ff, path, plan, at=0.0, pix="bgr24"):
    """one frame at `at` seconds (falls back to the first frame)."""
    for t in ([at, 0.0] if at > 0 else [0.0]):
        for f in read_frames(ff, path, plan, start=t, pix=pix):
            return f.copy()
    raise RuntimeError("couldn't read a frame from that video")


# ---- previews: an HDR frame shown as-is on a normal screen looks flat. this is a quick look-alike for the UI only.
def preview_rgb(img, plan, bgr=False):
    """uint8 image in the plan's output color -> roughly what an HDR screen shows, squeezed into SDR. UI only."""
    if not plan or not plan.get("hdr") or plan["out"]["trc"] != "arib-std-b67":
        return img
    import numpy as np
    x = img[..., ::-1] if bgr else img
    e = x.astype(np.float32) / 255.0
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    lin = np.where(e <= 0.5, (e * e) / 3.0, (np.exp((e - c) / a) + b) / 12.0)          # HLG -> scene light
    lin = np.power(np.maximum(lin, 0), 1.2) / 0.203                                       # OOTF, diffuse white = 1
    m = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]], np.float32)
    lin = np.maximum(lin @ m.T, 0)                                                       # bt2020 -> bt709
    lin = np.where(lin <= 0.8, lin, 0.8 + 0.2 * (1 - np.exp(-(lin - 0.8) / 0.2)))         # soft highlight shoulder
    out = np.clip(np.power(lin, 1 / 2.4), 0, 1)
    out = (out * 255 + 0.5).astype(np.uint8)
    return out[..., ::-1].copy() if bgr else out


def color_check(ff, path, plan, log=print):
    """after a render: is the file tagged the way it should be? one log line."""
    try:
        c = probe_color(ff, path)
    except Exception as e:
        log(f"couldn't re-check the output's color ({e})")
        return False
    o = plan["out"]
    ok = c["tagged"] and (c["matrix"], c["primaries"], c["trc"]) == (o["matrix"], o["primaries"], o["trc"])
    log(("color check: " + describe_color(c) + (" (as planned)" if ok else "")) if ok else
        f"color check: the file says {describe_color(c)}, expected {o['matrix']}/{o['primaries']}/{o['trc']}")
    return ok
