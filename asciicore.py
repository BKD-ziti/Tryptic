"""
asciicore.py -- the ASCII art video engine.

pipeline, one frame at a time, nothing written to disk in between:
    ffmpeg decodes the clip -> raw frames at the exact output size and frame rate
    -> each frame becomes a grid of characters (on the GPU when torch + CUDA are there, numpy otherwise)
    -> raw frames are piped into ffmpeg, which encodes the mp4 (NVENC when it works) and carries the audio over.

the output is exactly the size and frame rate you ask for (default: the clip's own). the character grid
just fills that frame; any leftover few pixels at the edge become bars.
"""
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path

import numpy as np

import media

TORCH_ERROR = None
try:
    import torch
except Exception as _e:  # not installed, or a broken DLL. numpy takes over.
    torch = None
    TORCH_ERROR = f"{type(_e).__name__}: {_e}"

RAMPS = {
    "short":    " .:-=+*#%@",
    "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "blocks":   " \u2591\u2592\u2593\u2588",
    "binary":   " 01",
    "letters":  " .oxO0X#",
}
COLORS = {"the clip's colors": "original", "one color": "mono", "grayscale": "gray"}
FONT_FILES = [
    r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\lucon.ttf", r"C:\Windows\Fonts\cour.ttf",
    "/System/Library/Fonts/Menlo.ttc", "/Library/Fonts/Courier New.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


@dataclass
class AsciiParams:
    cell: int = 8                    # width of one character cell in output pixels (height follows the font's shape)
    ramp: str = "short"              # a key of RAMPS, or "custom"
    custom: str = ""                 # your own characters, used when ramp == "custom"
    color: str = "original"          # original | mono | gray
    mono: str = "#33ff66"            # the one color
    boost: float = 0.6               # how much to brighten the clip's colors so dark scenes still read
    bg: float = 0.0                  # 0 = black behind the characters, up to 1 = the clip's own picture dimmed in behind
    invert: bool = False
    contrast: float = 1.0
    brightness: float = 0.0          # -0.5 .. 0.5
    gamma: float = 1.0               # >1 = darker mids (fewer dense glyphs)
    start: float = 0.0
    length: float = 0.0              # seconds, 0 = to the end


def params_from_dict(d):
    base = AsciiParams()
    out = {}
    for f in fields(base):
        default = getattr(base, f.name)
        v = d.get(f.name, default)
        try:
            out[f.name] = type(default)(v) if not isinstance(default, bool) else bool(v)
        except (TypeError, ValueError):
            out[f.name] = default
    return AsciiParams(**out)


# ---- glyph atlas -----------------------------------------------------------------------------

def _font(cw, ch):
    from PIL import ImageFont
    for path in FONT_FILES:
        if not Path(path).is_file():
            continue
        try:
            size = ch
            while size > 4:
                f = ImageFont.truetype(path, size)
                l, t, r, b = f.getbbox("Mg")
                if f.getlength("M") <= cw and (b - t) <= ch:
                    return f
                size -= 1
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=max(6, ch - 2))
    except TypeError:
        return ImageFont.load_default()


def cell_size(cell):
    """(width, height) of a character cell. monospace fonts are about twice as tall as wide."""
    cw = max(3, int(cell))
    return cw, max(4, int(round(cw * 2.0)))


def build_atlas(chars, cw, ch):
    """(n, ch, cw) float32 masks, sorted from emptiest to fullest, so the brightness ramp is always in true ink order
    whatever font or characters are used."""
    from PIL import Image, ImageDraw
    font = _font(cw, ch)
    seen, masks = set(), []
    for c in " " + chars:
        if c in seen or c in "\r\n\t":
            continue
        seen.add(c)
        im = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(im).text((cw / 2, ch / 2), c, fill=255, font=font, anchor="mm")
        masks.append(np.asarray(im, np.float32) / 255.0)
    masks.sort(key=lambda m: float(m.mean()))
    a = np.stack(masks)
    top = a.max()
    return a / top if top > 0 else a


def _hex(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (0.2, 1.0, 0.4)


# ---- the renderer (torch on the gpu, numpy as the fallback) -------------------------------------

class Renderer:
    def __init__(self, P, W, H):
        self.P, self.W, self.H = P, W, H
        self.cw, self.ch = cell_size(P.cell)
        self.cols, self.rows = max(1, W // self.cw), max(1, H // self.ch)
        self.ox = (W - self.cols * self.cw) // 2          # leftover pixels split evenly into edge bars
        self.oy = (H - self.rows * self.ch) // 2
        chars = P.custom if (P.ramp == "custom" and P.custom.strip()) else RAMPS.get(P.ramp, RAMPS["short"])
        atlas = build_atlas(chars, self.cw, self.ch)
        self.n = len(atlas)
        self.mono = _hex(P.mono)
        self.atlas_np = atlas
        self.atlas = atlas
        self.note = ""
        self.use_torch = False
        self.dev = "cpu (numpy)"
        if torch is not None:
            try:
                self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.atlas_t = torch.from_numpy(atlas).to(self.dev)
                self.monot = torch.tensor(self.mono, dtype=torch.float32, device=self.dev)
                self.canvas = torch.zeros(H, W, 3, device=self.dev)      # made once; the edge bars stay black
                self.use_torch = True
            except Exception as e:                                       # a broken torch install shouldn't stop a render
                self.note = f"torch wouldn't start ({type(e).__name__}), using numpy"
                self.dev = "cpu (numpy)"

    @property
    def device_name(self):
        return str(self.dev)

    def _shape_luma(self, lum, xp):
        P = self.P
        l = (lum - 0.5) * P.contrast + 0.5 + P.brightness
        l = xp.clip(l, 0.0, 1.0) if xp is np else l.clamp(0.0, 1.0)
        if abs(P.gamma - 1.0) > 1e-3:
            l = l ** P.gamma
        return 1.0 - l if P.invert else l

    def render(self, frame):
        """frame: (H, W, 3) uint8 RGB numpy. returns (H, W, 3) uint8 numpy."""
        if self.use_torch:
            try:
                return self._render_torch(frame)
            except Exception as e:                      # e.g. gpu out of memory: finish the job on numpy instead of dying
                self.use_torch = False
                self.dev = "cpu (numpy)"
                self.note = f"gpu drawing failed ({type(e).__name__}: {str(e)[:80]}), finished on the cpu"
        return self._render_numpy(frame)

    # -- torch --
    def _render_torch(self, frame):
        P, cw, ch, rows, cols = self.P, self.cw, self.ch, self.rows, self.cols
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(frame)).to(self.dev).float().div_(255.0)    # H, W, 3
            x = x[self.oy:self.oy + rows * ch, self.ox:self.ox + cols * cw]
            cells = torch.nn.functional.avg_pool2d(x.permute(2, 0, 1)[None], (ch, cw))[0]   # 3, rows, cols
            lum = 0.2126 * cells[0] + 0.7152 * cells[1] + 0.0722 * cells[2]
            idx = (self._shape_luma(lum, torch) * (self.n - 1)).round().long()
            mask = self.atlas_t[idx].permute(0, 2, 1, 3).reshape(rows * ch, cols * cw)     # glyphs laid out on the canvas
            if P.color == "mono":
                col = self.monot.view(1, 1, 3).expand(rows, cols, 3)
            elif P.color == "gray":
                col = lum[..., None].expand(rows, cols, 3)
                col = col / col.amax(dim=-1, keepdim=True).clamp(min=1e-3) * (0.35 + 0.65 * lum[..., None])
            else:
                c = cells.permute(1, 2, 0)
                hue = c / c.amax(dim=-1, keepdim=True).clamp(min=1e-3)
                col = c * (1 - P.boost) + hue * P.boost
            colp = col.repeat_interleave(ch, 0).repeat_interleave(cw, 1)                  # rows*ch, cols*cw, 3
            out = mask[..., None] * colp
            if P.bg > 0:
                out = out + (cells.permute(1, 2, 0).repeat_interleave(ch, 0).repeat_interleave(cw, 1)) * (P.bg * 0.5)
            self.canvas[self.oy:self.oy + rows * ch, self.ox:self.ox + cols * cw] = out.clamp(0, 1)
            return (self.canvas * 255.0).round().to(torch.uint8).cpu().numpy()

    # -- numpy --
    def _render_numpy(self, frame):
        P, cw, ch, rows, cols = self.P, self.cw, self.ch, self.rows, self.cols
        x = frame[self.oy:self.oy + rows * ch, self.ox:self.ox + cols * cw].astype(np.float32) / 255.0
        cells = x.reshape(rows, ch, cols, cw, 3).mean(axis=(1, 3))                          # rows, cols, 3
        lum = 0.2126 * cells[..., 0] + 0.7152 * cells[..., 1] + 0.0722 * cells[..., 2]
        idx = np.rint(self._shape_luma(lum, np) * (self.n - 1)).astype(np.int64)
        mask = self.atlas[idx].transpose(0, 2, 1, 3).reshape(rows * ch, cols * cw)
        if P.color == "mono":
            col = np.broadcast_to(np.array(self.mono, np.float32), (rows, cols, 3))
        elif P.color == "gray":
            col = np.repeat(lum[..., None], 3, axis=-1)
            col = col / np.maximum(col.max(axis=-1, keepdims=True), 1e-3) * (0.35 + 0.65 * lum[..., None])
        else:
            hue = cells / np.maximum(cells.max(axis=-1, keepdims=True), 1e-3)
            col = cells * (1 - P.boost) + hue * P.boost
        colp = np.repeat(np.repeat(col, ch, axis=0), cw, axis=1)
        out = mask[..., None] * colp
        if P.bg > 0:
            out = out + np.repeat(np.repeat(cells, ch, axis=0), cw, axis=1) * (P.bg * 0.5)
        canvas = np.zeros((self.H, self.W, 3), np.float32)
        canvas[self.oy:self.oy + rows * ch, self.ox:self.ox + cols * cw] = np.clip(out, 0, 1)
        return np.rint(canvas * 255.0).astype(np.uint8)


# ---- video in -> ascii video out -----------------------------------------------------------------

def _scale_chain(W, H, fit):
    if fit == "pad":
        return (f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
    if fit == "crop":
        return f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1"
    return f"scale={W}:{H}:flags=lanczos,setsar=1"


def iter_render(ff, src, P, W, H, rate, fit, out_dir, use_gpu_encode=False, nvenc_ok=False, preview_every=6,
                total_hint=None, color_mode="keep", log=None):
    """generator. yields {'kind':'frame', idx, total, image(RGB), per, eta} and finally {'kind':'done', path, fps, device}.
    W, H and rate are already decided (media.py + the size planner), so the output is exactly that."""
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rate = Fraction(rate)
    r = media.fps_arg(rate)
    rend = Renderer(P, W, H)
    cplan = media.color_plan(ff, src, color_mode, log=log)
    sig = json.dumps([asdict(P), W, H, str(rate), fit, cplan["sig"]], sort_keys=True, default=str)
    dst = out_dir / f"{src.stem}_ascii_{W}x{H}_{hashlib.md5(sig.encode()).hexdigest()[:6]}.mp4"  # new settings, new file

    dec = [ff, "-v", "error"]
    if P.start > 0:
        dec += ["-ss", f"{P.start:.3f}"]
    dec += ["-i", str(src), "-map", "0:v:0", "-an"]
    if P.length > 0:
        dec += ["-t", f"{P.length:.3f}"]
    dec += ["-vf", f"{cplan['decode']},fps={r},{_scale_chain(W, H, fit)},format=rgb24",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    def enc_cmd(gpu):
        c = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", r, "-i", "pipe:0"]
        c += (["-ss", f"{P.start:.3f}"] if P.start > 0 else []) + (["-t", f"{P.length:.3f}"] if P.length > 0 else [])
        c += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?"]
        c += ["-vf", media.rgb_to_out(cplan, gpu)] + media.codec_args(cplan, gpu, 18 if gpu else 17)
        c += ["-r", r, "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(dst)]
        return c

    attempts = [True, False] if (use_gpu_encode and nvenc_ok) else [False]     # gpu encode first, cpu if it won't start
    frame_bytes = W * H * 3
    err_log = out_dir / "_ascii_ffmpeg_error.txt"
    last_err = ""
    for gpu in attempts:
        nd = subprocess.Popen(dec, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 2, **media.NOWIN)
        with open(err_log, "w") as ef:
            ne = subprocess.Popen(enc_cmd(gpu), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=ef, **media.NOWIN)
        idx, t0, broke, aborted = 0, time.time(), False, False
        try:
            while True:
                raw = nd.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(bytearray(raw), np.uint8).reshape(H, W, 3)
                out = rend.render(frame)
                try:
                    ne.stdin.write(out.tobytes())
                except (BrokenPipeError, OSError):
                    broke = True
                    break
                idx += 1
                if idx % preview_every == 1 or idx == 1:
                    per = (time.time() - t0) / idx
                    left = per * (total_hint - idx) if total_hint else None
                    yield dict(kind="frame", idx=idx, total=total_hint, image=media.preview_rgb(out, cplan), per=per, eta=left,
                               device=rend.device_name)
        except GeneratorExit:                                # stopped from outside (stop button / queue cancel)
            aborted = True
            raise
        finally:
            try:
                ne.stdin.close()
            except Exception:
                pass
            nd.stdout.close()
            nd.kill()
            nd.wait()
            ne.wait()
            if aborted:
                dst.unlink(missing_ok=True)                  # don't leave a half-finished file behind
                err_log.unlink(missing_ok=True)
        last_err = Path(err_log).read_text(errors="replace").strip()[-400:] if err_log.exists() else ""
        if broke or ne.returncode != 0 or idx == 0:
            if gpu:
                continue                                    # nvenc didn't work on this machine: do it again on the cpu
            raise RuntimeError("couldn't make the video" + (f": {last_err}" if last_err else
                                                              " (no frames could be read from that file)"))
        err_log.unlink(missing_ok=True)
        yield dict(kind="done", path=dst, fps=rate, device=rend.device_name, gpu_encode=gpu, frames=idx, note=rend.note,
                   color=cplan)
        return
    raise RuntimeError("couldn't make the video" + (f": {last_err}" if last_err else ""))


def still(ff, src, P, W, H, fit, color_mode="keep"):
    """one frame at P.start, rendered. returns (source_rgb, ascii_rgb), HDR squeezed for an ordinary screen."""
    cplan = media.color_plan(ff, src, color_mode)
    cmd = [ff, "-v", "error", "-ss", f"{max(0.0, P.start):.3f}", "-i", str(src), "-frames:v", "1", "-map", "0:v:0",
           "-vf", f"{cplan['decode']},{_scale_chain(W, H, fit)},format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = subprocess.run(cmd, capture_output=True, **media.NOWIN)
    if len(r.stdout) < W * H * 3:
        r = subprocess.run([c if c != f"{max(0.0, P.start):.3f}" else "0" for c in cmd], capture_output=True, **media.NOWIN)
    if len(r.stdout) < W * H * 3:
        raise RuntimeError("couldn't read a frame from that video")
    frame = np.frombuffer(r.stdout[:W * H * 3], np.uint8).reshape(H, W, 3).copy()
    return media.preview_rgb(frame, cplan), media.preview_rgb(Renderer(P, W, H).render(frame), cplan)
