#!/usr/bin/env python3
"""
app.py -- Tryptic by BKDziti. one local UI for three tools: DeepDream, Datamosh, ASCII art.

drop in any video (any extension, any size), pick what to do to it, set the look, run.
everything happens on your own machine. nothing is uploaded anywhere.

size and frame rate work the same for all three: the finished video matches the clip's own size, shape and
exact frame rate unless you change them in the "size and speed" tab. launch it with start.bat (or: python app.py).
"""

import inspect
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

import asciicore
import dreamcore as core
import media
import mosh

HERE = Path(__file__).resolve().parent
PRESET_DIR = HERE / "presets"
OUT_DIR = HERE / "output"
TITLE = "Tryptic by BKDziti"
MAXL = 16                                    # most layers any dream network here has (vgg19)
SLIDERS = [f"w{i}" for i in range(MAXL)]

MODES = [("DeepDream", "dream"), ("Datamosh", "mosh"), ("ASCII art", "ascii")]
SHAPES = [("same as the clip", "source"), ("vertical", "vertical"), ("horizontal", "horizontal")]
FIT_CHOICES = [("crop the edges to fill the frame", "crop"), ("keep everything and add bars", "pad")]
RES_CHOICES = [("same as the clip", 0), ("4K", 2160), ("1440p", 1440), ("1080p", 1080), ("720p", 720), ("480p", 480)]

SHARED = ["shape", "fit", "res", "size_w", "size_h", "fps", "start", "length"]
DREAM = (["model"] + SLIDERS + ["focus", "channels_n", "channel_seed", "channels", "steps", "lr", "octaves",
                                 "octave_scale", "jitter", "smooth", "dream_res", "small_out", "blend", "flow",
                                 "follow_pct", "cut", "color_keep", "strength"])
MOSH = ["jumps", "jump_min", "jump_max", "jump_dir", "manual_jumps", "bloom_after_jump", "bloom_hold", "bloom_run",
        "smear_var", "bloom_extra", "manual_bloom", "quality", "motion_range", "mv4", "seed", "audio", "encoder",
        "out_cq", "hwdecode", "a_end", "b_start"]
ASCII = ["cell", "ramp", "custom", "color", "mono", "boost", "bg", "invert", "contrast", "brightness", "gamma"]
COLOR = ["color_out"]
ORDER = SHARED + DREAM + MOSH + ASCII + COLOR
COLOR_CHOICES = [("match the clip (iPhone HDR stays HDR)", "keep"),
                 ("standard SDR (tone-mapped, for older screens / some sites)", "sdr")]

MOSH_PRESETS = {
    "classic melt":  dict(jumps=3, jump_min=2, jump_max=6, jump_dir="forward", manual_jumps="", bloom_after_jump=True,
                          bloom_hold=20, bloom_run=2, smear_var=0, bloom_extra=0, manual_bloom="", quality=4,
                          motion_range=0, mv4=False),
    "heavy smear":   dict(jumps=4, jump_min=1, jump_max=4, jump_dir="forward", manual_jumps="", bloom_after_jump=True,
                          bloom_hold=50, bloom_run=1, smear_var=30, bloom_extra=3, manual_bloom="", quality=12,
                          motion_range=0, mv4=False),
    "glitch stutter": dict(jumps=3, jump_min=1, jump_max=3, jump_dir="either", manual_jumps="", bloom_after_jump=True,
                           bloom_hold=9, bloom_run=3, smear_var=60, bloom_extra=10, manual_bloom="", quality=8,
                           motion_range=16, mv4=True),
    "gentle drift":  dict(jumps=2, jump_min=3, jump_max=8, jump_dir="forward", manual_jumps="", bloom_after_jump=True,
                          bloom_hold=10, bloom_run=1, smear_var=0, bloom_extra=0, manual_bloom="", quality=2,
                          motion_range=0, mv4=False),
}

FF = {"path": None}                          # the ffmpeg everything uses (set in startup)


# ---- BKDziti branding --------------------------------------------------------------------

BRAND_CSS = """
:root, .dark, .gradio-container {
  --bkd-paper:#14172b; --bkd-panel:#1b1f39; --bkd-field:#11142a; --bkd-ink:#e7e8f3;
  --bkd-soft:#9a9fc2; --bkd-line:#31365a; --bkd-mag:#e2579f; --bkd-grn:#4cc790;
  --body-background-fill:var(--bkd-paper);
  --background-fill-primary:var(--bkd-panel);
  --background-fill-secondary:var(--bkd-paper);
  --block-background-fill:var(--bkd-panel);
  --panel-background-fill:var(--bkd-panel);
  --block-border-color:var(--bkd-line);
  --border-color-primary:var(--bkd-line);
  --input-background-fill:var(--bkd-field);
  --body-text-color:var(--bkd-ink);
  --body-text-color-subdued:var(--bkd-soft);
  --block-title-text-color:var(--bkd-ink);
  --block-label-text-color:var(--bkd-soft);
  --color-accent:var(--bkd-mag);
  --slider-color:var(--bkd-mag);
  --checkbox-background-color-selected:var(--bkd-mag);
  --checkbox-border-color-selected:var(--bkd-mag);
  --button-primary-background-fill:var(--bkd-mag);
  --button-primary-background-fill-hover:#ee6fae;
  --button-primary-border-color:var(--bkd-mag);
  --button-primary-text-color:#ffffff;
  --link-text-color:var(--bkd-grn);
}
footer, .gradio-container footer { display:none !important; }
.gradio-container .bkd-head { padding:14px 2px 6px; }
.gradio-container .bkd-head h1 {
  margin:0; border:0; color:var(--bkd-ink);
  font:700 64px/0.95 "Bahnschrift","DIN Alternate","Arial Narrow",system-ui,sans-serif; letter-spacing:-0.02em;
  text-shadow:4px 0 0 color-mix(in srgb,var(--bkd-mag) 75%,transparent),
              8px 0 0 color-mix(in srgb,var(--bkd-mag) 50%,transparent),
              12px 0 0 color-mix(in srgb,var(--bkd-grn) 38%,transparent),
              16px 0 0 color-mix(in srgb,var(--bkd-grn) 22%,transparent);
}
.gradio-container .bkd-by { margin:14px 0 0; font-weight:600; letter-spacing:0.04em; color:var(--bkd-soft); }
.gradio-container .bkd-by b { color:var(--bkd-mag); }
.gradio-container .bkd-sub { margin:6px 0 0; max-width:66ch; color:var(--bkd-soft); }
.gradio-container .bkd-foot { margin:18px 0 6px; padding-top:12px; border-top:1px solid var(--bkd-line);
  color:var(--bkd-soft); font-size:13px; text-align:center; }
.gradio-container .bkd-foot b { color:var(--bkd-ink); }
"""

HEADER_HTML = """
<div class="bkd-head">
  <h1>Tryptic</h1>
  <p class="bkd-by">by <b>BKDziti</b></p>
  <p class="bkd-sub">drop in a video, pick what happens to it: <b>DeepDream</b>, <b>Datamosh</b>, or <b>ASCII art</b>.
  the result keeps the clip's own size, shape and exact frame rate unless you change them.
  everything runs on your machine. nothing is uploaded.</p>
</div>
"""

FOOTER_HTML = """
<div class="bkd-foot"><b>Tryptic</b> by <b>BKDziti</b> &middot; rethinking food, media, and interaction, online and in person.</div>
"""

STYLE = dict(
    theme=gr.themes.Base(font=[gr.themes.Font("Segoe UI"), "system-ui", "sans-serif"],
                         font_mono=[gr.themes.Font("Cascadia Mono"), "Consolas", "monospace"]),
    css=BRAND_CSS,
    js="() => { document.body.classList.add('dark'); }",
)
_STYLE_IN_LAUNCH = "css" in inspect.signature(gr.Blocks.launch).parameters


def _favicon():
    p = HERE / ".favicon.png"
    try:
        if not p.exists():
            img = np.zeros((64, 64, 3), np.uint8)
            img[:] = (43, 23, 20)
            cv2.circle(img, (26, 32), 17, (159, 87, 226), 7, cv2.LINE_AA)
            cv2.circle(img, (38, 32), 17, (144, 199, 76), 7, cv2.LINE_AA)
            media.imwrite(p, img)
        return str(p)
    except Exception:
        return None


def _no_download():
    """no download button on the result player: it may be the small preview, not the real file."""
    params = inspect.signature(gr.Video.__init__).parameters
    if "buttons" in params:
        return {"buttons": []}
    if "show_download_button" in params:
        return {"show_download_button": False}
    return {}


def _launch_kwargs():
    params = inspect.signature(gr.Blocks.launch).parameters
    kw = dict(STYLE) if _STYLE_IN_LAUNCH else {}
    if "footer_links" in params:
        kw["footer_links"] = []
    elif "show_api" in params:
        kw["show_api"] = False
    fav = _favicon()
    if fav and "favicon_path" in params:
        kw["favicon_path"] = fav
    return kw


# ---- small helpers -------------------------------------------------------------------------

def _friendly(e):
    if isinstance(e, (MemoryError, ValueError, RuntimeError)):
        return str(e)
    return f"{type(e).__name__}: {e}"


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _path(f):
    if isinstance(f, dict):
        f = f.get("path") or (f.get("video") or {}).get("path")
    return str(f) if f else None


def pick_source(file, typed):
    """a typed path wins (no upload copy at all), else the dropped file."""
    t = (typed or "").strip().strip('"')
    if t:
        if not Path(t).is_file():
            raise gr.Error(f"can't find a file at: {t}")
        return t
    p = _path(file)
    if not p:
        raise gr.Error("drop a video in first")
    return p


def clip_of(state):
    w, h, num, den = state["w"], state["h"], state["num"], state["den"]
    return w, h, Fraction(num, den)


def clip_orient(w, h):
    return "vertical" if h > w else ("horizontal" if w > h else "square")


def mode_name(m):
    return dict((v, k) for k, v in MODES)[m]


# ---- turning the controls into each engine's settings -----------------------------------------

def dream_params(d, clip):
    """d: dict of control values. the dream engine's Params, with the shared size/speed controls mapped in."""
    w, h, _ = clip_of(clip)
    shape = d["shape"]
    if shape == "source" or shape == clip_orient(w, h):
        aspect = "original"                       # same shape as the clip: leave it alone (matches datamosh and ascii)
    else:
        aspect = "9:16" if shape == "vertical" else "16:9"
    names = core.layer_names(d["model"])
    lw = {}
    for i in range(min(MAXL, len(names))):
        v = d.get(f"w{i}")
        if v and float(v) > 0:
            lw[names[i]] = float(v)
    base = {k: d[k] for k in DREAM if k not in SLIDERS and k != "small_out"}
    base.update(layers=core.format_layers(lw), aspect=aspect, fit=d["fit"],
                res=(-1 if d["small_out"] else int(d["res"] or 0)),
                size_w=int(d["size_w"] or 0), size_h=int(d["size_h"] or 0),
                fps=float(d["fps"] or 0), start=float(d["start"] or 0), length=float(d["length"] or 0))
    return core.params_from_dict(base)


def mosh_settings(d):
    s = dict(mosh.DEFAULTS)
    s.update(orient=d["shape"], res=int(d["res"] or 0), size_w=int(d["size_w"] or 0), size_h=int(d["size_h"] or 0),
             fit=d["fit"], fps=float(d["fps"] or 0), preview=float(d["length"] or 0),
             color_out=d.get("color_out") or "keep")
    ints = ("jumps", "bloom_hold", "bloom_run", "bloom_extra", "quality", "motion_range", "out_cq", "seed")
    for k in MOSH:
        v = d[k]
        s[k] = int(v or 0) if k in ints else (float(v or 0) if k in ("jump_min", "jump_max", "a_end", "b_start") else v)
    s["smear_var"] = float(d["smear_var"] or 0) / 100.0
    s["manual_jumps"] = (d["manual_jumps"] or "").strip()
    s["manual_bloom"] = (d["manual_bloom"] or "").strip()
    return s


def ascii_params(d):
    x = {k: d[k] for k in ASCII}
    x.update(start=float(d["start"] or 0), length=float(d["length"] or 0))
    return asciicore.params_from_dict(x)


def output_plan(mode, d, clip):
    """(W, H, exact_rate, notes). one answer for the live readout and for the real render."""
    w, h, src_rate = clip_of(clip)
    want = float(d["fps"] or 0)
    if mode == "dream":
        P = dream_params(d, clip)
        pl = core.plan_size(w, h, P)
        (W, H) = pl["out"]
        rate = core.out_rate(P, src_rate)
        extra = f"dreaming at {pl['dream'][0]}x{pl['dream'][1]}" + (
            f", then upscaled {W / pl['dream'][0]:.1f}x" if W > pl["dream"][0] * 1.02 else "")
        capped = 0 < want and want >= float(src_rate) - 0.01 and abs(want - float(src_rate)) > 0.01
        return W, H, rate, extra + ("  ·  dream can't go above the clip's rate" if capped else "")
    W, H, _ = mosh.pick_size(w, h, d["shape"], int(d["res"] or 0), d["fit"], int(d["size_w"] or 0), int(d["size_h"] or 0))
    rate = media.typed_rate(want) if want > 0 else src_rate
    return W, H, rate, ""


# ---- handlers ----------------------------------------------------------------------------

def on_clip(file, typed):
    """probe whatever was dropped. returns (info line, thumbnail, clip state)."""
    try:
        src = pick_source(file, typed)
    except gr.Error:
        return "", None, None
    if not FF["path"]:
        return "ffmpeg isn't ready yet. restart the app.", None, None
    try:
        i = mosh.probe(FF["path"], src)
        if not i["w"]:
            return "that doesn't look like a video ffmpeg can read.", None, None
        if not i.get("fps_exact"):
            return "couldn't read this clip's frame rate. type one into the frame rate box (it won't guess).", None, None
        fr = Fraction(i["fps_exact"])
        state = dict(w=i["w"], h=i["h"], num=fr.numerator, den=fr.denominator, vfr=bool(i["vfr"]),
                     duration=float(i["duration"] or 0), audio=bool(i["audio"]), path=src)
        line = f"{i['w']}x{i['h']} ({clip_orient(i['w'], i['h'])})  ·  {media.fps_text(fr)}  ·  {i['duration']:.1f} s"
        if i["vfr"]:
            line += f"  ·  variable frame rate, using its average ({i['fps_how']})"
        line += "" if i["audio"] else "  ·  no audio"
        try:
            cplan = media.color_plan(FF["path"], src, "keep")
            line += "  ·  " + media.describe_color(cplan["src"])
        except Exception:
            cplan = None
        thumb = media.thumbnail(FF["path"], src, at=0.0, plan=cplan)
        return line, thumb, state
    except Exception as e:
        return _friendly(e), None, None


def on_size(mode, clip, *vals):
    if not clip:
        return "load a clip to see what you'll get"
    d = dict(zip(SHARED + DREAM, vals))
    try:
        W, H, rate, extra = output_plan(mode, d, clip)
    except Exception as e:
        return _friendly(e)
    w, h, src_rate = clip_of(clip)
    same = "same size as the clip" if (W, H) == (w, h) else ("bigger than the clip, upscaled" if W * H > w * h * 1.02
                                                           else "different from the clip")
    rtxt = media.fps_text(rate) + (" (same as the clip)" if abs(float(rate) - float(src_rate)) < 1e-6 else "")
    return f"output: {W}x{H} ({core.shape_of(W, H)}), {same}  ·  {rtxt}" + (f"\n{extra}" if extra else "")


# dream helpers (same behavior as DeepDreamer)
def slider_state(model, lw):
    names = core.layer_names(model)
    out = []
    for i in range(MAXL):
        if i < len(names):
            out.append(dict(label=names[i], visible=True, value=float(lw.get(names[i], 0.0))))
        else:
            out.append(dict(label=f"layer {i}", visible=False, value=0.0))
    return out


def dream_preset_values(preset):
    P = core.params_from_dict(preset)
    d = asdict(P)
    st = slider_state(P.model, core.parse_layers(P.layers))
    out = []
    for n in DREAM:
        if n in SLIDERS:
            out.append(gr.update(**st[int(n[1:])]))
        elif n == "small_out":
            out.append(P.res < 0)
        else:
            out.append(d[n])
    return out


def on_dream_preset(name):
    return dream_preset_values(core.all_presets(PRESET_DIR)[name])


def on_model(model):
    return [gr.update(**s) for s in slider_state(model, {core.DEFAULT_LAYER[model]: 1.0})]


def on_dream_save(name, *vals):
    name = re.sub(r"[^\w\- ]", "", name or "").strip()
    if not name:
        raise gr.Error("type a name for the preset first")
    d = dict(zip(DREAM, vals))
    names = core.layer_names(d["model"])
    lw = {names[i]: float(d[f"w{i}"]) for i in range(min(MAXL, len(names))) if d.get(f"w{i}") and float(d[f"w{i}"]) > 0}
    keep = {k: d[k] for k in DREAM if k not in SLIDERS and k != "small_out"}
    keep["layers"] = core.format_layers(lw)
    PRESET_DIR.mkdir(exist_ok=True)
    (PRESET_DIR / f"{name}.json").write_text(json.dumps(asdict(core.params_from_dict(keep)), indent=2))
    return gr.update(choices=list(core.all_presets(PRESET_DIR)), value=name)


def on_mosh_preset(name):
    p = MOSH_PRESETS.get(name)
    return [gr.skip() if (p is None or k not in p) else p[k] for k in MOSH]


def _vals(vals):
    return dict(zip(ORDER, vals))


def on_still(mode, clip, file, typed, *vals):
    if not clip:
        raise gr.Error("drop a video in first")
    src = clip["path"]
    d = _vals(vals)
    try:
        if mode == "dream":
            P = dream_params(d, clip)
            small, out, cplan = core.dream_still(src, P, core.get_dreamer(P.model), color_mode=d["color_out"], want_plan=True)
            return (media.preview_rgb(_rgb(np.hstack([small, out])), cplan),
                    f"still at {P.start:g}s  ·  left: original  ·  right: dream")
        if mode == "ascii":
            W, H, _, _ = output_plan("ascii", d, clip)
            P = ascii_params(d)
            fit = d["fit"] if d["fit"] in ("crop", "pad") else "crop"
            a, b = asciicore.still(FF["path"], src, P, W, H, fit, color_mode=d["color_out"])
            return np.hstack([a, b]), f"still at {P.start:g}s  ·  left: original  ·  right: ascii"
        raise gr.Error("datamosh has no single-frame preview. set length to 3 and render a short one.")
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(_friendly(e))


RUN_LOCK = threading.Lock()                   # one render at a time (manual or queued): they'd fight over the GPU
MOSH_BG = {"t": None}                         # the datamosh thread. it can't be interrupted, so a stopped one finishes


def mosh_still_running():
    t = MOSH_BG["t"]
    return t is not None and t.is_alive()


def run_render(mode, clip, d, b_path, say, cancelled=lambda: False):
    """the one render path, shared by the render button and the queue.
    a generator: yields ("frame", preview_rgb_or_None, status_text) while it works and ("done", path, status) last.
    cancelled() is checked between frames; returning True stops the render cleanly (ffmpeg gets killed)."""
    src = clip["path"]
    W, H, rate, _ = output_plan(mode, d, clip)
    w, h, src_rate = clip_of(clip)
    say(f"clip: {w}x{h} at {media.fps_text(src_rate)}" + ("  (variable frame rate, using the average)" if clip["vfr"] else ""))
    say(f"asked for: {W}x{H} at {media.fps_text(rate)}" + ("  (same as the clip)" if (W, H, rate) == (w, h, src_rate) else ""))
    OUT_DIR.mkdir(exist_ok=True)
    yield "frame", None, f"starting {mode_name(mode)}…"

    final, final_rate, cplan = None, rate, None

    if mode == "dream":
        P = dream_params(d, clip)
        mb = core.weights_download(P.model) if core.torch is not None else None
        if mb is not None:
            msg = f"downloading the {P.model} network, one time only" + (f" (about {mb:.0f} MB)" if mb else "") + "…"
            say(msg)
            yield "frame", None, msg
        else:
            yield "frame", None, "loading network…"
        dreamer = core.get_dreamer(P.model)
        gen = core.iter_render(src, P, dreamer, out_dir=OUT_DIR, color_mode=d["color_out"], log=say)
        try:
            for ev in gen:
                if cancelled():
                    raise RuntimeError("cancelled")
                if ev["kind"] == "frame":
                    tot = ev["total"] or "?"
                    eta = f"  ·  ~{ev['eta'] / 60:.1f} min left" if ev["eta"] is not None else ""
                    yield "frame", _rgb(ev["image"]), f"frame {ev['idx']}/{tot}  ·  {ev['per']:.1f}s per frame{eta}"
                else:
                    final, final_rate, cplan = ev["path"], ev["fps"], ev.get("color")
        finally:
            gen.close()
        say("dreamed on the " + ("GPU" if (core.torch is not None and core.torch.cuda.is_available()) else "CPU (slow)"))

    elif mode == "ascii":
        P = ascii_params(d)
        fit = d["fit"] if d["fit"] in ("crop", "pad") else "crop"
        avail = max(0.0, clip["duration"] - P.start) if clip["duration"] else 0
        secs = min(x for x in (avail or 1e9, P.length or 1e9))
        total = int(secs * float(rate)) if secs < 1e8 else None
        ok_enc, msg = mosh.nvenc_check(FF["path"])
        gen = asciicore.iter_render(FF["path"], src, P, W, H, rate, fit, OUT_DIR, use_gpu_encode=ok_enc,
                                    nvenc_ok=ok_enc, total_hint=total, color_mode=d["color_out"], log=say)
        try:
            for ev in gen:
                if cancelled():
                    raise RuntimeError("cancelled")
                if ev["kind"] == "frame":
                    tot = ev["total"] or "?"
                    eta = f"  ·  ~{ev['eta'] / 60:.1f} min left" if ev["eta"] is not None else ""
                    yield "frame", ev["image"], f"frame {ev['idx']}/{tot}  ·  {ev['per'] * 1000:.0f} ms per frame{eta}"
                else:
                    final, final_rate, cplan = ev["path"], ev["fps"], ev.get("color")
                    hdr = bool((ev.get("color") or {}).get("hdr"))
                    enc = (("GPU (NVENC HEVC 10-bit)" if hdr else "GPU (NVENC)") if ev["gpu_encode"]
                           else ("CPU (x265 10-bit)" if hdr else "CPU (x264)"))
                    say(f"characters drawn on: {ev['device']}  ·  encoded on the {enc}")
                    if ev.get("note"):
                        say("note: " + ev["note"])
        finally:
            gen.close()

    else:  # mosh (runs in a thread; it can't be interrupted mid-way, so a cancel lands when it finishes)
        s = mosh_settings(d)
        tag = int(s["seed"] or 0) or random.randrange(1, 1_000_000)
        s["seed"] = tag
        out = OUT_DIR / f"{Path(src).stem}_mosh_{tag}.mp4"
        state = {"p": 0.0, "stage": "prepping clip", "err": None, "done": False, "res": None}

        def prog(p, stage=None):
            state["p"] = p
            if stage:
                state["stage"] = stage

        def work():
            try:
                state["res"] = mosh.run_mosh(FF["path"], s, src, out, say, prog, b_path)
            except Exception as e:  # noqa
                state["err"] = e
            state["done"] = True

        t = threading.Thread(target=work, daemon=True)
        MOSH_BG["t"] = t                          # if the render is stopped, this keeps going until it's done
        t.start()
        while not state["done"]:
            time.sleep(0.4)
            yield "frame", None, f"{state['stage']}  ·  {state['p'] * 100:.0f}%"
        if state["err"]:
            raise state["err"]
        final = out
        cplan = (state.get("res") or {}).get("color")
        say(f"seed {tag}  (put it in the seed box to repeat this exact look)")

    if final is None:
        raise RuntimeError("nothing was made")
    media.verify_output(FF["path"], final, final_rate, say)
    if cplan:
        media.color_check(FF["path"], final, cplan, say)
    yield "done", str(final), f"done  ·  saved to {final}"


def on_render(mode, clip, file, typed, file_b, *vals):
    """a generator: streams (preview image, status, result video path, log text)."""
    if not clip:
        raise gr.Error("drop a video in first")
    if mosh_still_running() and not RUN_LOCK.locked():
        raise gr.Error("a datamosh you stopped is still finishing in the background (it can't be cut off mid-way). "
                       "give it a few seconds and press render again.")
    if not RUN_LOCK.acquire(blocking=False):
        raise gr.Error("the render queue is using the GPU right now. add this to the queue instead, "
                       "or stop the queue first.")
    log_lines = []

    def say(m):
        log_lines.append(m)

    def logtxt():
        return "\n".join(log_lines[-14:])

    try:
        for kind, a, status in run_render(mode, clip, _vals(vals), _path(file_b), say):
            if kind == "frame":
                yield a, status, None, logtxt()
            else:
                shown = a
                if media.needs_player_preview(FF["path"], a):
                    yield None, "making a quick preview for the player…", None, logtxt()
                    shown = media.player_preview(FF["path"], a, OUT_DIR, log=say)
                    if shown != a:
                        say("the player shows a small standard-color preview so any browser can play it. "
                            "the real file (full size, original color) is in the output folder.")
                yield None, status, shown, logtxt()
    except gr.Error:
        raise
    except Exception as e:
        yield None, f"stopped: {_friendly(e)}", None, logtxt()
    finally:
        RUN_LOCK.release()


# ---- render queue ------------------------------------------------------------------------------
# set jobs up, press start, walk away. one job at a time, in order. a failed job is logged and skipped.
# the queue lives as long as the app is open (closing the app clears it).

AFTER_CHOICES = [("do nothing", "nothing"), ("put the PC to sleep", "sleep"), ("shut the PC down", "shutdown")]
GRACE_S = 60                                   # seconds to change your mind before sleep / shutdown


class RenderQueue:
    def __init__(self):
        self.jobs = []
        self.lock = threading.Lock()
        self.worker = None
        self.stop_after_current = False
        self.cancel_current = False
        self.cancel_after = False
        self.after = "nothing"
        self.status = "queue is empty"
        self.preview = None
        self._next = 1

    # -- editing
    def add(self, mode, clip, d, b_path):
        W, H, rate, _ = output_plan(mode, d, clip)
        extra = ""
        if mode == "dream":
            extra = f" · {d['model']}"
        elif mode == "mosh":
            extra = f" · {d['jumps']} jumps" + (" · + 2nd clip" if b_path else "")
        elif mode == "ascii":
            extra = f" · {d['ramp']}, {int(d['cell'])}px"
        with self.lock:
            job = dict(id=self._next, mode=mode, clip=dict(clip), d=dict(d), b_path=b_path,
                       what=f"{mode_name(mode)}{extra}", out=f"{W}x{H} · {media.fps_text(rate)}"
                       + (" · SDR" if d.get("color_out") == "sdr" else ""),
                       status="waiting", result="", log=[], started=None, ended=None)
            self._next += 1
            self.jobs.append(job)
        return job["id"]

    def remove(self, num):
        with self.lock:
            before = len(self.jobs)
            self.jobs = [j for j in self.jobs if not (j["id"] == num and j["status"] != "running")]
            return before != len(self.jobs)

    def clear_finished(self):
        with self.lock:
            self.jobs = [j for j in self.jobs if j["status"] in ("waiting", "running")]

    def retry_failed(self):
        with self.lock:
            for j in self.jobs:
                if j["status"] in ("failed", "cancelled"):
                    j.update(status="waiting", result="", log=[])

    def running(self):
        return self.worker is not None and self.worker.is_alive()

    # -- running
    def start(self, after):
        self.after = after or "nothing"
        self.stop_after_current = self.cancel_current = self.cancel_after = False
        if not self.running():
            self.status = "queue started"
        if self.running():
            return "already running (the 'when it's all done' choice was updated)"
        if not any(j["status"] == "waiting" for j in self.jobs):
            return "nothing waiting in the queue"
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        return "queue started"

    def _run(self):
        _keep_awake(True)
        ran = 0
        try:
            while not self.stop_after_current:
                with self.lock:
                    job = next((j for j in self.jobs if j["status"] == "waiting"), None)
                if job is None:
                    break
                if RUN_LOCK.locked():
                    self.status = f"job {job['id']} is waiting for the render that's going now to finish"
                with RUN_LOCK:                   # waits politely if a manual render is going
                    if mosh_still_running():     # a stopped datamosh is still finishing
                        self.status = f"job {job['id']} is waiting for a stopped datamosh to finish"
                        MOSH_BG["t"].join()
                    self._run_job(job)
                ran += 1
        finally:
            _keep_awake(False)
        left = sum(1 for j in self.jobs if j["status"] == "waiting")
        done = sum(1 for j in self.jobs if j["status"] == "done")
        bad = sum(1 for j in self.jobs if j["status"] == "failed")
        cut = sum(1 for j in self.jobs if j["status"] == "cancelled")
        summary = (f"queue finished: {done} done" + (f", {bad} failed" if bad else "") + (f", {cut} cancelled" if cut else "")
                   + (f", {left} still waiting" if left else ""))
        self.status = summary
        action = self.after
        if self.stop_after_current or ran == 0 or action not in ("sleep", "shutdown"):
            return
        for left_s in range(GRACE_S, 0, -1):
            if self.cancel_after:
                self.status = summary + f"  ·  {'sleep' if action == 'sleep' else 'shutdown'} cancelled"
                return
            self.status = summary + f"  ·  {'sleeping' if action == 'sleep' else 'shutting down'} in {left_s}s (press 'cancel sleep / shutdown' to stop it)"
            time.sleep(1)
        self.status = summary + f"  ·  {'going to sleep' if action == 'sleep' else 'shutting down'} now"
        _power_action(action)

    def _run_job(self, job):
        job.update(status="running", started=time.time(), log=[])
        self.cancel_current = False
        say = job["log"].append
        try:
            if not Path(job["clip"]["path"]).is_file():
                raise RuntimeError("the clip isn't there anymore (was it moved or deleted?)")
            for kind, a, status in run_render(job["mode"], job["clip"], job["d"], job["b_path"], say,
                                              cancelled=lambda: self.cancel_current):
                self.status = f"job {job['id']} ({job['what']}): {status}"
                if kind == "frame" and a is not None:
                    self.preview = _small(a)
                elif kind == "done":
                    job["result"] = a
            if self.cancel_current and job["mode"] == "mosh":
                say("cancelled, but datamosh had already finished, so the file was kept")
            job["status"] = "done"
        except Exception as e:  # noqa
            job["status"] = "cancelled" if (self.cancel_current or str(e) == "cancelled") else "failed"
            job["result"] = "" if job["status"] == "cancelled" else _friendly(e)
            say(f"{job['status']}: {_friendly(e)}")
        finally:
            job["ended"] = time.time()

    # -- display
    def table(self):
        rows = []
        for j in self.jobs:
            took = ""
            if j["started"]:
                secs = (j["ended"] or time.time()) - j["started"]
                took = f"{secs / 60:.1f} min" if secs >= 60 else f"{secs:.0f} s"
            res = Path(j["result"]).name if (j["status"] == "done" and j["result"]) else j["result"]
            rows.append([j["id"], j["what"], Path(j["clip"]["path"]).name, j["out"], j["status"], took, res])
        return rows

    def log_text(self):
        j = next((j for j in self.jobs if j["status"] == "running"), None) or \
            next((j for j in reversed(self.jobs) if j["status"] != "waiting"), None)
        if not j:
            return ""
        return f"job {j['id']} · {j['what']}\n" + "\n".join(j["log"][-14:])


def _small(img, side=640):
    """a light copy for the queue preview (it's re-sent to the browser every couple of seconds)."""
    h, w = img.shape[:2]
    k = side / max(h, w)
    return cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA) if k < 1 else img


QUEUE = RenderQueue()
QUEUE_HEADERS = ["#", "what", "clip", "output", "status", "time", "result"]


def _keep_awake(on):
    """windows: stop the PC from going to sleep on its own while the queue runs (the screen can still turn off)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def _power_action(what):
    try:
        if os.name == "nt":
            if what == "shutdown":
                subprocess.run(["shutdown", "/s", "/t", "0"], **media.NOWIN)
            elif what == "sleep":
                subprocess.run(["powershell", "-NoProfile", "-Command",
                                "Add-Type -AssemblyName System.Windows.Forms;"
                                "[System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)"], **media.NOWIN)
        elif sys.platform == "darwin":
            subprocess.run(["pmset", "sleepnow"] if what == "sleep" else ["osascript", "-e", 'tell app "System Events" to shut down'])
        else:
            subprocess.run(["systemctl", "suspend" if what == "sleep" else "poweroff"])
    except Exception as e:  # noqa
        QUEUE.status += f"  ·  couldn't {what}: {e}"


def queue_view():
    import pandas as pd
    rows = QUEUE.table()
    waiting = sum(1 for r in rows if r[4] == "waiting")
    if QUEUE.running() or QUEUE.status.startswith("queue finished"):
        head = QUEUE.status
    elif waiting:
        head = f"{waiting} job(s) waiting. press 'start queue' and walk away."
    elif rows:
        head = "nothing waiting."
    else:
        head = "queue is empty. set a job up on the left and press 'add to queue'."
    return pd.DataFrame(rows, columns=QUEUE_HEADERS), head, QUEUE.log_text(), QUEUE.preview


def on_queue_add(mode, clip, file, typed, file_b, *vals):
    if not clip:
        raise gr.Error("drop a video in first")
    d = _vals(vals)
    try:
        n = QUEUE.add(mode, clip, d, _path(file_b))
    except Exception as e:
        raise gr.Error(_friendly(e))
    gr.Info(f"added job {n} to the queue")
    return queue_view()


def on_queue_start(after):
    msg = QUEUE.start(after)
    gr.Info(msg)
    return queue_view()


def on_queue_stop_after():
    if QUEUE.running():
        QUEUE.stop_after_current = True
        gr.Info("the queue will stop after the job that's running now")
    return queue_view()


def on_queue_cancel_current():
    if QUEUE.running():
        QUEUE.cancel_current = True
        gr.Info("cancelling the running job (datamosh can't be stopped mid-way; it'll finish first)")
    return queue_view()


def on_queue_cancel_after():
    QUEUE.cancel_after = True
    QUEUE.after = "nothing"
    return gr.update(value="nothing"), *queue_view()


def on_queue_remove(num):
    try:
        n = int(num or 0)
    except (TypeError, ValueError):
        n = 0
    if not QUEUE.remove(n):
        gr.Warning(f"no waiting job #{n} to remove (a running job can't be removed; cancel it instead)")
    return queue_view()


def on_queue_clear():
    QUEUE.clear_finished()
    return queue_view()


def on_queue_retry():
    QUEUE.retry_failed()
    return queue_view()


def on_open_folder():
    OUT_DIR.mkdir(exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(OUT_DIR)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(OUT_DIR)])
    else:
        subprocess.run(["xdg-open", str(OUT_DIR)])


# ---- the UI ------------------------------------------------------------------------------

def build_ui():
    init = core.params_from_dict(core.PRESETS["classic"])
    st = slider_state(init.model, core.parse_layers(init.layers))
    A = asciicore.AsciiParams()
    C = {}

    with gr.Blocks(title=TITLE, **({} if _STYLE_IN_LAUNCH else STYLE)) as demo:
        gr.HTML(HEADER_HTML)
        clip = gr.State(None)
        with gr.Row():
            with gr.Column(scale=6):
                file = gr.File(label="your video (any format, any size)", file_count="single", type="filepath")
                with gr.Accordion("or point at a file already on this computer (skips the upload copy, best for huge files)", open=False):
                    typed = gr.Textbox(label="full path", placeholder=r"C:\Users\you\Videos\clip.mov", max_lines=1)
                thumb = gr.Image(label="first frame", interactive=False, height=200)
                info = gr.Textbox(label="video info", interactive=False, max_lines=2)
                size_info = gr.Textbox(label="what you'll get", interactive=False, lines=2)
                mode = gr.Radio(MODES, value="dream", label="what should happen to it?")

                # ---- DeepDream -------------------------------------------------------------
                with gr.Column(visible=True) as dream_col:
                    with gr.Row():
                        preset = gr.Dropdown(choices=list(core.all_presets(PRESET_DIR)), value="classic", label="deepdream preset", scale=3)
                        save_name = gr.Textbox(label="save current settings as…", scale=3)
                        save_btn = gr.Button("save", scale=1)
                    with gr.Tabs():
                        with gr.Tab("what it dreams"):
                            C["model"] = gr.Dropdown(choices=list(core.LAYERS), value=init.model, label="network",
                                                     info="googlenet = the classic look. vgg = painterly. inception_v3 = newer, different shapes. resnet50 = weird grids.")
                            gr.Markdown("**layer weights** (0 = off). shallow layers = patterns & textures, deep layers = parts & objects (eyes, fur, animals). mix a few.")
                            for i, s in enumerate(st):
                                C[f"w{i}"] = gr.Slider(0, 2, step=0.1, value=s["value"], label=s["label"], visible=s["visible"])
                            C["focus"] = gr.Slider(0, 8, step=0.5, value=init.focus, label="focus",
                                                   info="0 = classic: every spot gets dreamed equally (wallpaper tiles). higher = it piles onto spots where it already sees something -> fewer, bigger, more recognizable things.")
                            with gr.Row():
                                C["channels_n"] = gr.Number(value=init.channels_n, precision=0, label="random channel subset (0 = all)",
                                                            info="dream only N random features out of the layer: a 'theme'.")
                                C["channel_seed"] = gr.Number(value=init.channel_seed, precision=0, label="theme seed",
                                                              info="change it to get a different theme.")
                            C["channels"] = gr.Textbox(value=init.channels, label="or pick channel numbers yourself",
                                                       info="e.g. 12, 40, 97. overrides the random subset.")
                        with gr.Tab("shape of the dream"):
                            C["steps"] = gr.Slider(1, 40, step=1, value=init.steps, label="steps per zoom level", info="more = stronger dream.")
                            C["lr"] = gr.Slider(0.005, 0.2, step=0.005, value=init.lr, label="step size", info="more = stronger and noisier.")
                            C["octaves"] = gr.Slider(1, 8, step=1, value=init.octaves, label="zoom levels (octaves)",
                                                     info="more = bigger structures, because it starts dreaming on a tiny version of the frame.")
                            C["octave_scale"] = gr.Slider(1.1, 2.0, step=0.05, value=init.octave_scale, label="zoom jump",
                                                          info="bigger = the smallest level is tinier = huger shapes.")
                            C["jitter"] = gr.Slider(0, 64, step=1, value=init.jitter, label="jitter", info="random shake each step. hides seams between tiles.")
                            C["smooth"] = gr.Slider(0, 3, step=0.1, value=init.smooth, label="smoothing",
                                                    info="blurs the nudges. higher = softer, bigger blobs, less noisy pattern.")
                            C["dream_res"] = gr.Slider(144, 1080, step=8, value=init.dream_res, label="dream resolution (short side, px)",
                                                       info="what the network actually sees. lower = chunkier, bigger-looking hallucinations and much faster. the gpu runs out of memory if this is too high.")
                            C["small_out"] = gr.Checkbox(value=False, label="deliver at the dream's own small size (fastest, but NOT the clip's size)")
                        with gr.Tab("motion"):
                            C["blend"] = gr.Slider(0, 0.95, step=0.05, value=init.blend, label="carry-over",
                                                   info="how much of the last frame's dream flows into the next. 0 = every frame separate (flickery). high = melty and flowing, but patterns pile up.")
                            C["flow"] = gr.Checkbox(value=init.flow, label="move the carried dream with the footage (optical flow)")
                            C["follow_pct"] = gr.Slider(10, 100, step=5, value=init.follow_pct, label="effort on later frames (% of steps)", info="lower = faster and calmer.")
                            C["cut"] = gr.Slider(5, 255, step=5, value=init.cut, label="scene-cut threshold",
                                                 info="bigger = needs a bigger jump between frames to count as a new scene. 255 = never reset.")
                            C["color_keep"] = gr.Slider(0, 1, step=0.05, value=init.color_keep, label="keep original colors",
                                                        info="pulls the colors back toward the source so it doesn't drift to neon.")
                            C["strength"] = gr.Slider(0, 1, step=0.05, value=init.strength, label="dream strength", info="mix between the original (0) and the full dream (1).")

                # ---- Datamosh --------------------------------------------------------------
                with gr.Column(visible=False) as mosh_col:
                    mosh_preset = gr.Dropdown(choices=list(MOSH_PRESETS), value=None, label="datamosh look (fills in the boxes below, tweak anything after)")
                    with gr.Tabs():
                        with gr.Tab("jumps and smears"):
                            C["jumps"] = gr.Number(value=mosh.DEFAULTS["jumps"], precision=0, label="jumps",
                                                   info="the video leaps and its keyframe is deleted, so the old picture gets dragged by the new motion.")
                            with gr.Row():
                                C["jump_min"] = gr.Number(value=mosh.DEFAULTS["jump_min"], label="jump size, min (s)")
                                C["jump_max"] = gr.Number(value=mosh.DEFAULTS["jump_max"], label="jump size, max (s)")
                            C["jump_dir"] = gr.Dropdown(choices=[("forward only", "forward"), ("forward or back", "either")], value="forward",
                                                        label="jump direction", info="going back replays footage, so the clip gets longer.")
                            C["manual_jumps"] = gr.Textbox(value="", label="or place jumps yourself", placeholder="4>10, 22>30, 15>6",
                                                           info="from second > to second, in order. 15>6 jumps back. fills in over the random ones.")
                            C["bloom_after_jump"] = gr.Checkbox(value=True, label="smear after every jump")
                            with gr.Row():
                                C["bloom_hold"] = gr.Number(value=mosh.DEFAULTS["bloom_hold"], precision=0, label="smear length (frames)")
                                C["bloom_run"] = gr.Number(value=mosh.DEFAULTS["bloom_run"], precision=0, label="motion loop (frames)",
                                                           info="1 is a clean drag, higher wobbles.")
                            C["smear_var"] = gr.Slider(0, 100, step=1, value=0, label="smear variation (%)", info="0 = every smear the same length. higher = uneven.")
                            with gr.Row():
                                C["bloom_extra"] = gr.Number(value=0, precision=0, label="extra random smears")
                                C["manual_bloom"] = gr.Textbox(value="", label="smears at (s)", placeholder="7.5, 12")
                        with gr.Tab("texture"):
                            C["quality"] = gr.Slider(1, 31, step=1, value=mosh.DEFAULTS["quality"], label="blockiness",
                                                     info="low is crisp. high gives chunky blocks and heavier smear.")
                            C["motion_range"] = gr.Number(value=0, precision=0, label="motion reach (px)",
                                                          info="how far a block can drag. 0 is no limit. try 8 to 24 for tight, jittery motion.")
                            C["mv4"] = gr.Checkbox(value=False, label="detailed motion (4 vectors per block)")
                            C["seed"] = gr.Number(value=0, precision=0, label="seed",
                                                  info="0 rolls a new one every time. keep a seed to repeat a look while you tweak other things.")
                        with gr.Tab("output"):
                            C["audio"] = gr.Dropdown(choices=[("follow the cuts", "follow"), ("none", "none")], value="follow", label="audio")
                            C["encoder"] = gr.Dropdown(choices=[("gpu if it works, else cpu", "auto"), ("gpu only (stop if it can't)", "gpu"), ("cpu", "cpu")],
                                                       value="auto", label="final encode")
                            with gr.Row():
                                C["out_cq"] = gr.Number(value=mosh.DEFAULTS["out_cq"], precision=0, label="output quality", info="lower is cleaner and bigger")
                                C["hwdecode"] = gr.Checkbox(value=True, label="decode source on gpu")
                        with gr.Tab("second clip (transition)"):
                            file_b = gr.File(label="the clip you want to melt into (optional)", file_count="single", type="filepath")
                            gr.Markdown("the first clip plays, then the second clip's opening keyframe is deleted so it drags the first one's pixels. random jumps are off in this mode.")
                            with gr.Row():
                                C["a_end"] = gr.Number(value=0, label="cut first clip at (s)", info="0 plays it all")
                                C["b_start"] = gr.Number(value=0, label="second clip starts at (s)")

                # ---- ASCII -----------------------------------------------------------------
                with gr.Column(visible=False) as ascii_col:
                    C["cell"] = gr.Slider(4, 40, step=1, value=A.cell, label="character size (px wide)",
                                          info="smaller = more detail, more characters. 6-10 reads like a photo, 16+ is chunky.")
                    C["ramp"] = gr.Dropdown(choices=[("short  . : - = + * # % @", "short"), ("detailed (70 characters)", "detailed"),
                                                     ("blocks  ░ ▒ ▓ █", "blocks"), ("binary  0 1", "binary"),
                                                     ("letters  . o x O 0 X #", "letters"), ("my own characters", "custom")],
                                            value=A.ramp, label="characters")
                    C["custom"] = gr.Textbox(value="", label="my own characters", info="any characters you like. they're sorted dark to bright by how much ink they use.")
                    with gr.Row():
                        C["color"] = gr.Dropdown(choices=list(asciicore.COLORS.items()), value=A.color, label="color")
                        C["mono"] = gr.ColorPicker(value=A.mono, label="the one color (when 'one color')")
                    C["boost"] = gr.Slider(0, 1, step=0.05, value=A.boost, label="color boost", info="brightens the clip's colors so dark scenes still read.")
                    C["bg"] = gr.Slider(0, 1, step=0.05, value=A.bg, label="picture behind the characters", info="0 = pure black. higher = the clip dimmed in behind the text.")
                    C["invert"] = gr.Checkbox(value=A.invert, label="invert (dark characters for bright areas)")
                    with gr.Row():
                        C["contrast"] = gr.Slider(0.3, 3, step=0.05, value=A.contrast, label="contrast")
                        C["brightness"] = gr.Slider(-0.5, 0.5, step=0.05, value=A.brightness, label="brightness")
                        C["gamma"] = gr.Slider(0.4, 2.5, step=0.05, value=A.gamma, label="gamma", info=">1 = fewer dense characters")

                # ---- shared: size, speed, clip ---------------------------------------------
                gr.Markdown("### size and speed  (same for all three)")
                with gr.Tabs():
                    with gr.Tab("size and speed"):
                        C["shape"] = gr.Radio(SHAPES, value="source", label="shape", info="same as the clip = the clip's own shape.")
                        C["res"] = gr.Dropdown(RES_CHOICES, value=0, label="resolution (short side)", info="same as the clip = the clip's own size.")
                        C["fit"] = gr.Dropdown(FIT_CHOICES, value="crop", label="when the shape changes", info="only matters if the shape is different from the clip's.")
                        with gr.Accordion("set an exact output size instead (optional)", open=False):
                            with gr.Row():
                                C["size_w"] = gr.Number(value=0, precision=0, label="width (px)")
                                C["size_h"] = gr.Number(value=0, precision=0, label="height (px)")
                            gr.Markdown("fill both to force that exact size, which ignores the shape and resolution above. fill just one and the other follows the clip's shape. 0 = auto.")
                        C["color_out"] = gr.Dropdown(COLOR_CHOICES, value="keep", label="color",
                                                     info="match the clip = the export has the same color as what you shot (HDR in, HDR out, so it looks the same on your phone). standard SDR = squeezes HDR down for screens and sites that don't do HDR.")
                        C["fps"] = gr.Number(value=0, label="frame rate (fps)",
                                             info="0 = exactly the clip's own rate. only type a number if you want a different one. deepdream can only go lower than the clip, not higher.")
                    with gr.Tab("clip"):
                        C["start"] = gr.Number(value=0, label="start at (seconds)", info="deepdream and ascii. datamosh always starts at the beginning.")
                        C["length"] = gr.Number(value=0, label="length (seconds, 0 = to the end)", info="use 3 or 4 while you're tuning.")

            with gr.Column(scale=6):
                with gr.Row():
                    still_btn = gr.Button("preview still")
                    render_btn = gr.Button("render video", variant="primary")
                    queue_btn = gr.Button("add to queue")
                    stop_btn = gr.Button("stop")
                    open_btn = gr.Button("open output folder")
                status = gr.Textbox(label="status", interactive=False, max_lines=2)
                image = gr.Image(label="preview", interactive=False)
                video_out = gr.Video(label="result (preview. the real file is in the output folder)", interactive=False,
                                     **_no_download())
                log = gr.Textbox(label="what it did", interactive=False, lines=8)

                # ---- render queue --------------------------------------------------------------
                gr.Markdown("### render queue\nset a job up on the left, press **add to queue**, change the settings, "
                            "add another. then **start queue** and walk away. jobs run one after another, a failed "
                            "one is skipped, and the PC is kept awake until it's done. closing the app clears the queue.")
                q_status = gr.Textbox(label="queue", interactive=False, lines=2)
                q_table = gr.Dataframe(headers=QUEUE_HEADERS, interactive=False, wrap=True)
                q_after = gr.Dropdown(AFTER_CHOICES, value="nothing", label="when it's all done",
                                      info=f"sleep and shut down wait {GRACE_S} seconds first, so you can cancel.")
                with gr.Row():
                    q_start = gr.Button("start queue", variant="primary")
                    q_stop = gr.Button("stop after this job")
                    q_cancel = gr.Button("cancel running job")
                with gr.Row():
                    q_num = gr.Number(value=0, precision=0, label="job #", scale=1)
                    q_remove = gr.Button("remove that job", scale=1)
                    q_retry = gr.Button("retry failed", scale=1)
                    q_clear = gr.Button("clear finished", scale=1)
                with gr.Row():
                    q_cancel_after = gr.Button("cancel sleep / shutdown")
                    q_refresh = gr.Button("refresh")
                q_preview = gr.Image(label="queue: latest frame", interactive=False, height=260)
                q_log = gr.Textbox(label="queue: what it did", interactive=False, lines=6)

        gr.HTML(FOOTER_HTML)

        controls = [C[n] for n in ORDER]
        dream_c = [C[n] for n in DREAM]
        mosh_c = [C[n] for n in MOSH]
        size_c = [C[n] for n in SHARED + DREAM]

        file.change(on_clip, [file, typed], [info, thumb, clip]).then(on_size, [mode, clip] + size_c, size_info)
        typed.submit(on_clip, [file, typed], [info, thumb, clip]).then(on_size, [mode, clip] + size_c, size_info)
        typed.blur(on_clip, [file, typed], [info, thumb, clip]).then(on_size, [mode, clip] + size_c, size_info)
        gr.on(triggers=[c.change for c in [mode] + [C[n] for n in SHARED] + [C["dream_res"], C["small_out"], C["model"]]],
              fn=on_size, inputs=[mode, clip] + size_c, outputs=size_info)
        mode.change(lambda m: (gr.update(visible=m == "dream"), gr.update(visible=m == "mosh"),
                               gr.update(visible=m == "ascii"), gr.update(visible=m != "mosh")),
                    mode, [dream_col, mosh_col, ascii_col, still_btn])

        preset.change(on_dream_preset, preset, dream_c)
        C["model"].input(on_model, C["model"], [C[n] for n in SLIDERS])
        save_btn.click(on_dream_save, [save_name] + dream_c, preset)
        mosh_preset.change(on_mosh_preset, mosh_preset, mosh_c)

        still_btn.click(on_still, [mode, clip, file, typed] + controls, [image, status])
        render_evt = render_btn.click(on_render, [mode, clip, file, typed, file_b] + controls, [image, status, video_out, log])
        stop_btn.click(None, None, None, cancels=[render_evt])

        q_outs = [q_table, q_status, q_log, q_preview]
        queue_btn.click(on_queue_add, [mode, clip, file, typed, file_b] + controls, q_outs)
        q_start.click(on_queue_start, q_after, q_outs)
        q_stop.click(on_queue_stop_after, None, q_outs)
        q_cancel.click(on_queue_cancel_current, None, q_outs)
        q_cancel_after.click(on_queue_cancel_after, None, [q_after] + q_outs)
        q_remove.click(on_queue_remove, q_num, q_outs)
        q_retry.click(on_queue_retry, None, q_outs)
        q_clear.click(on_queue_clear, None, q_outs)
        q_refresh.click(queue_view, None, q_outs)
        q_after.change(lambda a: setattr(QUEUE, "after", a or "nothing"), q_after, None)
        demo.load(queue_view, None, q_outs)
        if hasattr(gr, "Timer"):                 # live updates while it runs (gradio 4.40+)
            gr.Timer(2).tick(queue_view, None, q_outs, show_progress="hidden")
        open_btn.click(on_open_folder, None, None)
    demo._controls = C                      # (handy for tests)
    return demo


def open_in_default_browser(url):
    """hands the url straight to the OS, so it opens in whatever your default browser is."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(url)
        elif sys.platform == "darwin":
            subprocess.run(["open", url])
        else:
            subprocess.run(["xdg-open", url])
    except Exception:
        import webbrowser
        webbrowser.open(url)


def startup():
    """one ffmpeg for everything: yours if it works, otherwise a build with NVIDIA encoding, downloaded once."""
    ff = mosh.ensure_ffmpeg(None, say=print)
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = None
    if not ff:
        sys.exit("no ffmpeg, and couldn't download one. check your internet connection and start again.")
    FF["path"] = str(ff)
    os.environ["VIDEOLAB_FFMPEG"] = FF["path"]
    ok, msg = mosh.nvenc_check(FF["path"])
    print(f"  ffmpeg: {FF['path']}")
    print("  video encode: " + ("NVIDIA GPU (NVENC)" if ok else f"cpu ({msg})"))
    if core.torch is not None:
        print("  deepdream + ascii drawing: " + ("NVIDIA GPU (CUDA)" if core.torch.cuda.is_available() else "cpu (no CUDA GPU visible to PyTorch)"))
    else:
        print(f"  deepdream needs PyTorch ({core.TORCH_ERROR}). datamosh still works.")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    PRESET_DIR.mkdir(exist_ok=True)
    startup()
    demo = build_ui().queue()
    demo.launch(inbrowser=False, prevent_thread_lock=True, allowed_paths=[str(OUT_DIR)], **_launch_kwargs())
    open_in_default_browser(demo.local_url)
    demo.block_thread()
