"""
dreamcore.py -- the DeepDream engine behind app.py (the UI) and dream.py (the command line).

nothing in here draws any UI. it takes a Params object and a video and gives you dreamed frames.
"""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

import media  # shared frame-rate probe (never guesses)

TORCH_ERROR = None
try:  # torch is only needed once you actually dream; everything else works without it
    import torch
    import torch.nn.functional as F
    from torchvision import models
except Exception as _e:  # ImportError, or an OSError from a broken DLL
    torch = F = models = None
    TORCH_ERROR = f"{type(_e).__name__}: {_e}"


def _need_torch():
    if torch is None:
        raise RuntimeError(
            f"PyTorch couldn't be loaded ({TORCH_ERROR}). python in use: {sys.executable}. "
            "close this window and run start.bat again (or start.bat repair to rebuild the install).")


# --------------------------------------------------------------------------------------
# networks + the layers you can dream on
# --------------------------------------------------------------------------------------

def _vgg(counts, relu_idx):
    names, k = [], 0
    for block, n in enumerate(counts, 1):
        for i in range(1, n + 1):
            names.append((f"relu{block}_{i}", f"features.{relu_idx[k]}"))
            k += 1
    return names


LAYERS = {
    # the classic. same family as the original 2015 deepdream.
    "googlenet": [(n, n) for n in ["inception3a", "inception3b", "inception4a", "inception4b", "inception4c",
                                   "inception4d", "inception4e", "inception5a", "inception5b"]],
    # newer cousin. different vocabulary of shapes.
    "inception_v3": [(n, n) for n in ["Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c",
                                      "Mixed_6d", "Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c"]],
    # painterly, brushy, less "creature".
    "vgg16": _vgg([2, 2, 3, 3, 3], [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29]),
    "vgg19": _vgg([2, 2, 4, 4, 4], [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35]),
    # tends to dream in grids/checkerboards. weird on purpose.
    "resnet50": ([("layer1", "layer1")]
                 + [(f"layer2.{i}", f"layer2.{i}") for i in range(4)]
                 + [(f"layer3.{i}", f"layer3.{i}") for i in range(6)]
                 + [(f"layer4.{i}", f"layer4.{i}") for i in range(3)]),
}
DEFAULT_LAYER = {"googlenet": "inception4c", "inception_v3": "Mixed_6d", "vgg16": "relu4_3",
                 "vgg19": "relu4_4", "resnet50": "layer3.3"}
MIN_SIDE = {"inception_v3": 96}  # smallest image the network will accept per octave (default 64)


def layer_names(model):
    return [n for n, _ in LAYERS[model]]


# --------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------

@dataclass
class Params:
    # what it dreams
    model: str = "googlenet"
    layers: str = "inception4c=1"      # "name=weight, name=weight" (a bare name means weight 1)
    focus: float = 0.0                 # 0 = classic. higher = pile onto spots where it already sees something
    channels_n: int = 0                # 0 = all channels. N = dream only a random N (a "theme")
    channel_seed: int = 0
    channels: str = ""                 # explicit channel numbers, overrides the random subset
    # shape of the dream
    steps: int = 10
    lr: float = 0.04
    octaves: int = 4
    octave_scale: float = 1.4
    jitter: int = 16
    smooth: float = 0.0
    # size (see plan_size for how these combine)
    dream_res: int = 360               # short side of what the network sees, px. 360 on a 16:9 clip = 640x360
    aspect: str = "original"           # shape of the output. original = the clip's own
    fit: str = "crop"                  # crop | pad. only matters when the shape changes
    res: int = 0                       # finished video's short side, px. 0 = same as the clip, -1 = same as the dream
    size_w: int = 0                    # exact output size in px (0 = auto). both set = exact, one set = the other follows the shape
    size_h: int = 0
    fps: float = 0.0                   # output frame rate. 0 = match the source. lower = fewer frames to dream
    # motion
    blend: float = 0.55
    flow: bool = True
    follow_pct: int = 100
    cut: float = 30.0
    color_keep: float = 0.0
    strength: float = 1.0
    # clip
    start: float = 0.0
    length: float = 0.0                # seconds, 0 = to the end


def _migrate(d):
    """presets saved before the size rework stored 'width' and fit='letterbox'/'stretch'. translate them."""
    d = dict(d)
    if "dream_res" not in d and d.get("width"):
        ratio = ASPECTS.get(str(d.get("aspect", "original"))) or 16 / 9
        d["dream_res"] = int(round(float(d["width"]) / max(1.0, ratio) / 8)) * 8
    if str(d.get("fit", "crop")) not in FITS:
        d["fit"] = "pad" if str(d.get("fit")) == "letterbox" else "crop"
    return d


def params_from_dict(d):
    """build a Params from any dict, coercing types and ignoring unknown keys."""
    d = _migrate(d)
    base = Params()
    out = {}
    for f in fields(base):
        default = getattr(base, f.name)
        v = d.get(f.name, default)
        if v is None:
            v = default
        if isinstance(default, bool):
            v = bool(v)
        elif isinstance(default, int):
            v = int(round(float(v)))
        elif isinstance(default, float):
            v = float(v)
        else:
            v = str(v)
        out[f.name] = v
    return Params(**out)


def parse_layers(s):
    out = {}
    for part in str(s).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, w = part.partition("=")
        try:
            out[name.strip()] = float(w) if w.strip() else 1.0
        except ValueError:
            out[name.strip()] = 1.0
    return {k: v for k, v in out.items() if v > 0}


def format_layers(d):
    return ", ".join(f"{k}={v:g}" for k, v in d.items())


PRESETS = {
    "classic": dict(),
    # deeper layers + more zoom levels + 'focus' = fewer, bigger, more recognisable things instead of wallpaper
    "big-shapes": dict(layers="inception4d=1, inception4e=1", focus=3.0, octaves=6, octave_scale=1.6, steps=8,
                       lr=0.05, smooth=1.0, blend=0.4, color_keep=0.5, dream_res=288),
    "creatures": dict(layers="inception4d=0.6, inception4e=1", focus=4.0, octaves=5, octave_scale=1.5, steps=10,
                      lr=0.05, smooth=0.8, blend=0.45, color_keep=0.4),
    "ornament": dict(layers="inception3b=1, inception4a=1", octaves=4, octave_scale=1.4, steps=10, lr=0.05),
    "melty": dict(layers="inception4c=1, inception4d=0.5", blend=0.8, follow_pct=50, focus=1.5, octaves=5,
                  octave_scale=1.5, color_keep=0.6, smooth=0.5),
    "vgg-paint": dict(model="vgg16", layers="relu4_3=1, relu3_3=0.5", steps=12, lr=0.05, octaves=4,
                      octave_scale=1.6, smooth=1.0, blend=0.5, color_keep=0.3),
}


def all_presets(preset_dir=None):
    out = dict(PRESETS)
    if preset_dir and Path(preset_dir).is_dir():
        for p in sorted(Path(preset_dir).glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text())
            except Exception:
                pass
    return out


ASPECTS = {"original": None, "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4,
           "1:1": 1.0, "21:9": 21 / 9, "2.39:1": 2.39}
FITS = ["crop", "pad"]


# --------------------------------------------------------------------------------------
# the dreamer
# --------------------------------------------------------------------------------------

class _Stop(Exception):
    """raised from a hook so we don't run network layers deeper than we need."""


def _blur(t, sigma):
    """gaussian blur on a (1,C,H,W) tensor, separable, reflect-padded."""
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.arange(-r, r + 1, device=t.device, dtype=t.dtype)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    c = t.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), kx, groups=c)
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), ky, groups=c)
    return t


def _resize(t, size):
    try:
        return F.interpolate(t, size=size, mode="bilinear", align_corners=False, antialias=True)
    except (NotImplementedError, RuntimeError):
        return F.interpolate(t, size=size, mode="bilinear", align_corners=False)


class Dreamer:
    def __init__(self, model_name, device):
        _need_torch()
        if model_name not in LAYERS:
            raise ValueError(f"unknown network '{model_name}'. choose from: {', '.join(LAYERS)}")
        self.name, self.device = model_name, device
        self.model = models.get_model(model_name, weights="DEFAULT").eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.order = layer_names(model_name)
        self.acts, self.active, self.deepest = {}, set(), None
        for name, path in LAYERS[model_name]:
            self.model.get_submodule(path).register_forward_hook(self._hook(name))

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.mean, self.std = mean, std
        self.lo, self.hi = (0 - mean) / std, (1 - mean) / std
        self._chan_cache = {}

    def _hook(self, name):
        def hook(_module, _inp, out):
            if name in self.active:
                self.acts[name] = out
                if name == self.deepest:
                    raise _Stop
        return hook

    def _channels(self, name, c, P):
        if P.channels.strip():
            idx = sorted({int(t) % c for t in re.split(r"[,\s]+", P.channels.strip()) if t.lstrip("-").isdigit()})
            return torch.tensor(idx, device=self.device) if idx else None
        if 0 < P.channels_n < c:
            key = (name, P.channel_seed, P.channels_n)
            if key not in self._chan_cache:
                rng = np.random.default_rng(P.channel_seed + 1)
                self._chan_cache[key] = torch.tensor(rng.permutation(c)[:P.channels_n], device=self.device)
            return self._chan_cache[key]
        return None

    def _loss(self, x, lw, P):
        self.acts.clear()
        try:
            self.model(x)
        except _Stop:
            pass
        total = 0.0
        for name, w in lw.items():
            a = self.acts[name]
            sel = self._channels(name, a.shape[1], P)
            if sel is not None:
                a = a[:, sel]
            e = a.pow(2).mean(1).flatten(1)                      # energy at every spot: (1, positions)
            if P.focus > 0:                                      # rich get richer: spots that already light up
                ratio = e.detach() / (e.detach().max(1, keepdim=True).values + 1e-8)   # (0..1, 1 = strongest spot)
                e = e * (torch.softmax(P.focus * ratio, dim=1) * e.shape[1])           # bounded: peak gets ~e^focus x
            total = total + w * e.mean() / (e.detach().mean() + 1e-8)  # per-layer normalised, so weights mean something
        return total

    def _ascend(self, x, steps, lr, jitter, lw, P):
        for _ in range(steps):
            ox, oy = (int(v) for v in np.random.randint(-jitter, jitter + 1, 2))
            xs = torch.roll(x, (ox, oy), (2, 3)).detach().requires_grad_(True)
            self._loss(xs, lw, P).backward()
            g = xs.grad
            if P.smooth > 0:
                g = _blur(g, P.smooth)
            g = g / (g.abs().mean() + 1e-8)
            xs = xs.detach() + lr * g
            x = torch.roll(xs, (-ox, -oy), (2, 3))
            x = torch.max(torch.min(x, self.hi), self.lo)
        return x

    def dream(self, rgb01, P, steps=None):
        """rgb01: HxWx3 float32 in 0..1 -> same shape out."""
        lw = parse_layers(P.layers)
        bad = [k for k in lw if k not in self.order]
        if bad:
            raise ValueError(f"layer(s) {bad} don't exist on {self.name}. valid: {', '.join(self.order)}")
        if not lw:
            raise ValueError("no layers selected. give at least one layer a weight above 0.")
        self.active, self.deepest = set(lw), max(lw, key=self.order.index)
        steps = steps or P.steps
        try:
            x = torch.from_numpy(np.ascontiguousarray(rgb01)).permute(2, 0, 1)[None].to(self.device)
            x = (x - self.mean) / self.std
            H, W = x.shape[-2:]
            ms = MIN_SIDE.get(self.name, 64)

            sizes = []
            for i in reversed(range(max(1, P.octaves))):
                h, w = int(H / P.octave_scale ** i), int(W / P.octave_scale ** i)
                f = max(1.0, ms / max(1, min(h, w)))
                h, w = min(H, int(round(h * f))), min(W, int(round(w * f)))
                if not sizes or sizes[-1] != (h, w):
                    sizes.append((h, w))

            detail, out = None, x
            for h, w in sizes:
                base = _resize(x, (h, w))
                detail = torch.zeros_like(base) if detail is None else _resize(detail, (h, w))
                j = max(0, min(P.jitter, min(h, w) // 4))
                out = self._ascend(base + detail, steps, P.lr, j, lw, P)
                detail = out - base

            out = (out * self.std + self.mean).clamp(0, 1)
            return out[0].permute(1, 2, 0).detach().cpu().numpy()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise MemoryError("the gpu ran out of memory. lower the dream resolution (or the number of octaves) and try again.") from e
            raise


def pick_device():
    _need_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_DREAMERS = {}


def get_dreamer(model_name):
    """loads a network once and keeps it. only one network is kept in memory at a time."""
    dev = pick_device()
    key = (model_name, dev.type)
    if key not in _DREAMERS:
        _DREAMERS.clear()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        _DREAMERS[key] = Dreamer(model_name, dev)
    return _DREAMERS[key]


# --------------------------------------------------------------------------------------
# frames: fitting, motion, colour
# --------------------------------------------------------------------------------------

def _even(x):
    return max(2, int(round(x)) // 2 * 2)


def shape_of(w, h):
    return "vertical" if h > w else ("horizontal" if w > h else "square")


def out_fps(P, src_fps):
    """output frame rate: the box if it's lower than the source's, else the source's own."""
    return P.fps if 0 < P.fps < src_fps - 0.01 else src_fps


def plan_size(w, h, P):
    """every size decision, in one place. (w, h) is the clip as it actually plays.
    returns dict(out=(W, H), dream=(DW, DH), reshaped=bool, fit='crop' | 'pad').

      shape  : the 'aspect' dropdown (original = the clip's own). setting both exact-size boxes sets its own shape.
      output : the exact size if given, else 'res' = short side in px (0 = same as the clip, -1 = same as the dream).
      dream  : always the output's shape, with its short side = dream_res. dreaming small and delivering big is the point.
    all sizes are even numbers, because video encoders need that."""
    clip_ar = w / h
    cw, ch = max(0, int(P.size_w or 0)), max(0, int(P.size_h or 0))
    ar = cw / ch if (cw and ch) else (ASPECTS.get(P.aspect) or clip_ar)
    reshaped = abs(ar / clip_ar - 1) > 0.01
    fit = P.fit if (reshaped and P.fit in FITS) else "crop"

    short = max(64, int(P.dream_res))
    dw, dh = (short * ar, short) if ar >= 1 else (short, short / ar)
    dream = (_even(dw), _even(dh))

    res = int(P.res or 0)
    if cw or ch:
        W, H = (cw, ch) if (cw and ch) else ((cw, cw / ar) if cw else (ch * ar, ch))
        out = (_even(W), _even(H))
    elif res < 0:
        out = dream
    elif res > 0:
        out = (_even(res * ar), _even(res)) if ar >= 1 else (_even(res), _even(res / ar))
    elif not reshaped:
        out = (_even(w), _even(h))
    else:                                    # same size as the clip, but a new shape
        wider = clip_ar > ar
        if fit == "pad":                     # the whole clip fits inside, bars fill the rest
            W, H = (w, w / ar) if wider else (h * ar, h)
        else:                                # the biggest piece of the clip with that shape
            W, H = (h * ar, h) if wider else (w, w / ar)
        out = (_even(W), _even(H))
    return dict(out=out, dream=dream, reshaped=reshaped, fit=fit)


def fit_frame(frame, W, H, mode):
    """crop: fill the W x H canvas, trimming the edges. pad: shrink to fit inside it. the bars are added
    at the very end, so the network never dreams on black."""
    h, w = frame.shape[:2]
    interp = lambda nw, nh: cv2.INTER_AREA if nw * nh < w * h else cv2.INTER_CUBIC
    if mode == "pad":
        s = min(W / w, H / h)
        nw, nh = max(2, round(w * s)), max(2, round(h * s))
        return cv2.resize(frame, (nw, nh), interpolation=interp(nw, nh))
    s = max(W / w, H / h)                                    # crop to fill
    nw, nh = max(W, math.ceil(w * s - 1e-9)), max(H, math.ceil(h * s - 1e-9))
    r = cv2.resize(frame, (nw, nh), interpolation=interp(nw, nh))
    x0, y0 = (nw - W) // 2, (nh - H) // 2
    return np.ascontiguousarray(r[y0:y0 + H, x0:x0 + W])


def warp_previous(prev_dream, prev_gray, cur_gray):
    """move last frame's dream along with the motion in the footage (optical flow)."""
    flow = cv2.calcOpticalFlowFarneback(cur_gray, prev_gray, None, 0.5, 3, 25, 3, 5, 1.2, 0)
    h, w = cur_gray.shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mapx = (gx + flow[..., 0]).astype(np.float32)
    mapy = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(prev_dream, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def color_match(dream_bgr, ref_bgr, amount):
    """pull the dream's overall colour toward the source frame's (stops drift to neon)."""
    if amount <= 0:
        return dream_bgr
    d = cv2.cvtColor(dream_bgr, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    r = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    matched = (d - d.mean(0)) / (d.std(0) + 1e-6) * r.std(0) + r.mean(0)
    out = (d * (1 - amount) + matched * amount).reshape(dream_bgr.shape)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def to_rgb01(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_bgr8(rgb01):
    return cv2.cvtColor((rgb01 * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def _finish(dreamed, small, P):
    dreamed = color_match(dreamed, small, P.color_keep)
    if P.strength < 1:
        dreamed = cv2.addWeighted(dreamed, P.strength, small, 1 - P.strength, 0)
    return dreamed


def find_ffmpeg(hint="ffmpeg"):
    """--ffmpeg path, then PATH, then the copy bundled with the imageio-ffmpeg pip package."""
    env = os.environ.get("VIDEOLAB_FFMPEG")
    if env and Path(env).is_file():
        return env
    found = shutil.which(hint) or (hint if Path(hint).is_file() else None)
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# video in -> dreamed video out
# --------------------------------------------------------------------------------------

def _real_size(cap):
    """(width, height) of an actually decoded frame, then rewinds. phones store vertical video sideways plus a
    rotation flag, and the size properties don't always account for it, so the decoded frame is the truth."""
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ok, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    if ok and frame is not None:
        return frame.shape[1], frame.shape[0]
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def source_rate(src, cap=None):
    """the clip's exact frame rate as a Fraction. ffmpeg's view first (header + measured timestamps), opencv's only
    if there's no ffmpeg. raises instead of guessing: a wrong default here was silently changing output frame rates."""
    ff = find_ffmpeg()
    if ff:
        return media.probe_rate(ff, src)
    own = cap is None
    cap = cap or cv2.VideoCapture(str(src))
    try:
        f = cap.get(cv2.CAP_PROP_FPS)
    finally:
        if own:
            cap.release()
    if not f or not (0.5 < f <= 1000):
        raise RuntimeError("couldn't read this clip's frame rate (and there's no ffmpeg to ask). type one into the frame rate box.")
    return dict(fps=media.snap(f), how="opencv", vfr=False, header=f)


def out_rate(P, src_rate):
    """exact output rate: the box if it's lower than the source's, else the source's own exact rate."""
    src = float(src_rate)
    return media.typed_rate(P.fps) if 0 < P.fps < src - 0.01 else src_rate


def video_info(src):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"couldn't open {src}")
    ra = source_rate(src, cap)
    fps = float(ra["fps"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow, oh = _real_size(cap)
    cap.release()
    return dict(fps=fps, fps_exact=ra["fps"], vfr=ra["vfr"], fps_how=ra["how"], total=total,
                width=ow, height=oh, seconds=(total / fps if total else 0))


def param_hash(P, src, plan, color_sig=""):
    """names the frame cache. only what changes the dreamed frames counts. the output size and upscaling don't,
    so changing 'output resolution' on an interrupted render still picks up where it stopped."""
    d = asdict(P)
    for k in ("aspect", "fit", "res", "size_w", "size_h", "dream_res"):
        d.pop(k)
    d["_canvas"] = [*plan["dream"], plan["fit"]]
    d["_src"] = [Path(src).name, os.path.getsize(src)]
    d["_color"] = color_sig                  # frames dreamed under the old color handling are never reused
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:6]


def dream_still(src, P, dreamer, color_mode="keep", want_plan=False):
    """one frame (at P.start seconds), dreamed. returns (source_bgr, dream_bgr) (+ the color plan if want_plan).
    decoded by ffmpeg with the clip's real color math (opencv assumes bt601 SDR for everything)."""
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    cplan = media.color_plan(ff, src, color_mode)
    frame = media.grab_frame(ff, src, cplan, at=max(0.0, P.start))
    plan = plan_size(frame.shape[1], frame.shape[0], P)
    small = fit_frame(frame, *plan["dream"], plan["fit"])
    out = to_bgr8(dreamer.dream(to_rgb01(small), P))
    res = (small, _finish(out, small, P))
    return (*res, cplan) if want_plan else res


def iter_render(src, P, dreamer, out_dir="dream_output", ffmpeg_hint="ffmpeg", keep_frames=False, color_mode="keep",
                log=None):
    """generator. yields {'kind': 'frame', ...} per frame and {'kind': 'done', 'path': ...} at the end.
    finished frames are kept on disk, so re-running the same settings resumes where it stopped."""
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg(ffmpeg_hint)
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. install imageio-ffmpeg (pip install imageio-ffmpeg) or put ffmpeg on your PATH.")
    cplan = media.color_plan(ffmpeg, src, color_mode, log=log)
    cap = cv2.VideoCapture(str(src))                                 # only used for the frame count now
    if not cap.isOpened():
        raise RuntimeError(f"couldn't open {src}")
    reader = None
    try:
        ra = source_rate(src, cap)
        src_rate = ra["fps"]                                         # exact, e.g. 30000/1001
        src_fps = float(src_rate)
        out_r = out_rate(P, src_rate)                                # can't go above the source's rate
        fps = float(out_r)
        step = src_fps / fps                                         # source frames per output frame (>= 1)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        ow, oh = media.decoded_size(ffmpeg, src)
        plan = plan_size(ow, oh, P)
        (DW, DH), (W, H), fit = plan["dream"], plan["out"], plan["fit"]

        start_f = int(round(P.start * src_fps))
        avail = max(0, int((total - start_f) / step)) if total else None
        want = int(round(P.length * fps)) if P.length > 0 else None
        n = min(x for x in (avail, want) if x is not None) if (avail is not None or want is not None) else None

        stem = f"{src.stem}_{param_hash(P, src, plan, cplan['sig'])}"
        frames_dir = out_dir / f"_frames_{stem}"
        frames_dir.mkdir(exist_ok=True)
        dst = out_dir / f"{stem}_{min(W, H)}p.mp4"
        follow_steps = max(1, round(P.steps * P.follow_pct / 100))

        prev_dream = prev_gray = None
        content, idx, done, t0 = None, 0, 0, time.time()
        # ffmpeg decodes (right color math, picks frames at the output rate by timestamp)
        reader = media.read_frames(ffmpeg, src, cplan, start=P.start, fps=out_r if step > 1.0001 else None,
                                   length=P.length, size=(ow, oh))
        for frame in reader:
            if n is not None and idx >= n:
                break
            small = fit_frame(frame, DW, DH, fit)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            path = frames_dir / f"{idx:06d}.png"

            if path.exists():
                dreamed = cv2.imread(str(path))
            else:
                if prev_gray is not None and cv2.absdiff(gray, prev_gray).mean() > P.cut:
                    prev_dream = None                      # hard cut: don't drag the old scene along
                if prev_dream is None or P.blend <= 0:
                    init, steps = small, P.steps
                else:
                    carried = warp_previous(prev_dream, prev_gray, gray) if P.flow else prev_dream
                    init = cv2.addWeighted(carried, P.blend, small, 1 - P.blend, 0)
                    steps = follow_steps
                dreamed = _finish(to_bgr8(dreamer.dream(to_rgb01(init), P, steps=steps)), small, P)
                cv2.imwrite(str(path), dreamed)
                done += 1

            content = (dreamed.shape[1], dreamed.shape[0])
            prev_dream, prev_gray = dreamed, gray
            idx += 1
            per = (time.time() - t0) / done if done else 0.0
            left = per * (n - idx) if (n is not None and done) else None
            yield dict(kind="frame", idx=idx, total=n, image=media.preview_rgb(dreamed, cplan, bgr=True),
                       source=small, per=per, eta=left)
    finally:
        cap.release()
        if reader is not None:
            reader.close()

    if idx == 0:
        raise RuntimeError("couldn't read any frames from that video")

    vf = []
    if fit == "pad":                         # shrink to fit inside the final frame, then fill the rest with bars
        vf.append(f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
                  f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
    elif content != (W, H):                  # dreamed small, delivered at the output size
        vf.append(f"scale={W}:{H}:flags=lanczos,setsar=1")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", media.fps_arg(out_r), "-i", str(frames_dir / "%06d.png")]
    if P.start > 0:
        cmd += ["-ss", f"{P.start:.3f}"]
    if P.length > 0:
        cmd += ["-t", f"{P.length:.3f}"]
    cmd += ["-i", str(src), "-map", "0:v", "-map", "1:a?"]
    tries = ([True] if (cplan["hdr"] and media.has_encoder(ffmpeg, "hevc_nvenc")) else []) + [False]
    err = ""
    for gpu in tries:                        # HDR: try the gpu's HEVC encoder first (x265 on the cpu is slow)
        c = cmd + ["-vf", ",".join(vf + [media.rgb_to_out(cplan, gpu)])]
        c += media.codec_args(cplan, gpu, 17) + ["-r", media.fps_arg(out_r), "-c:a", "aac", "-shortest",
                                                 "-movflags", "+faststart", str(dst)]
        try:
            subprocess.run(c, check=True, capture_output=True, text=True, **media.NOWIN)
            err = ""
            break
        except subprocess.CalledProcessError as e:
            err = e.stderr.strip()[-400:]
    if err:
        raise RuntimeError(f"ffmpeg failed: {err}")
    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    yield dict(kind="done", path=dst, fps=out_r, color=cplan)
