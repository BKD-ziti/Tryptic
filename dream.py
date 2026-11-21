#!/usr/bin/env python3
"""
dream.py -- command-line version of the deepdream video tool (the UI is app.py, started by start.bat).

    python dream.py input.mp4                                  # -> output/input_xxxxxx_1080p.mp4
    python dream.py input.mp4 result.mp4 --preset big-shapes --length 4
    python dream.py input.mp4 still.png --preview               # one frame, original | dream
    python dream.py input.mp4 --layers "inception4d=1, inception4e=0.5" --focus 3 --octaves 6

every setting in the UI is a flag here (steps, lr, octaves, octave-scale, jitter, smooth, dream-res, aspect,
fit, res, size-w, size-h, blend, flow/no-flow, follow-pct, cut, color-keep, strength, start, length ...).

sizes work like this:
    --dream-res 360          what the network sees: short side in px (360 on a 16:9 clip = 640x360). small = fast
    --res 1080               finished video's short side. 0 = same as the clip (default), -1 = same as the dream
    --aspect 9:16            new shape. with --fit crop (fill the frame) or --fit pad (add bars)
    --size 1080x1920         exact output size. wins over --aspect and --res
presets: classic, big-shapes, creatures, ornament, melty, vgg-paint (plus anything saved from the UI).
"""

import argparse
import re
import shutil
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

import dreamcore as core

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="DeepDream for video files.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", help="optional: copy the result here (.mp4, or .png with --preview)")
    ap.add_argument("--preset", default="classic")
    ap.add_argument("--preview", action="store_true", help="dream one frame (at --start) and exit")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="path to ffmpeg if it's not found automatically")
    ap.add_argument("--out-dir", default=str(HERE / "output"))
    ap.add_argument("--color", choices=["keep", "sdr"], default="keep",
                    help="keep = same color as the clip (HDR stays HDR). sdr = tone-map HDR to standard bt709")
    ap.add_argument("--size", default="", help="exact output size like 1080x1920 (wins over --aspect/--res)")

    base = core.Params()
    for f in fields(base):
        flag, default = "--" + f.name.replace("_", "-"), getattr(base, f.name)
        if isinstance(default, bool):
            ap.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=None)
        else:
            ap.add_argument(flag, dest=f.name, type=type(default), default=None)
    args = ap.parse_args()

    presets = core.all_presets(HERE / "presets")
    if args.preset not in presets:
        sys.exit(f"unknown preset '{args.preset}'. choose from: {', '.join(presets)}")
    settings = dict(presets[args.preset])
    settings.update({f.name: getattr(args, f.name) for f in fields(base) if getattr(args, f.name) is not None})
    if args.size:
        m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", args.size)
        if not m:
            sys.exit("--size should look like 1080x1920")
        settings["size_w"], settings["size_h"] = int(m[1]), int(m[2])
    P = core.params_from_dict(settings)

    device = core.pick_device()
    print(f"device: {device} | network: {P.model} | layers: {P.layers} | preset: {args.preset}")
    if device.type == "cpu":
        print("heads up: no GPU found, so this will be slow. try --dream-res 270 --length 3 first.")
    info = core.video_info(args.input)
    plan = core.plan_size(info["width"], info["height"], P)
    print(f"clip {info['width']}x{info['height']} | dreaming at {plan['dream'][0]}x{plan['dream'][1]} "
          f"| output {plan['out'][0]}x{plan['out'][1]}")
    dreamer = core.get_dreamer(P.model)

    if args.preview:
        small, out = core.dream_still(args.input, P, dreamer, color_mode=args.color)
        dst = Path(args.output) if args.output else Path(args.out_dir) / (Path(args.input).stem + "_preview.png")
        dst = dst if dst.suffix.lower() == ".png" else dst.with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        core.media.imwrite(dst, np.hstack([small, out]))
        print(f"saved {dst} (left: original, right: dream)")
        return

    result = None
    for ev in core.iter_render(args.input, P, dreamer, out_dir=args.out_dir, ffmpeg_hint=args.ffmpeg,
                               keep_frames=args.keep_frames, color_mode=args.color, log=print):
        if ev["kind"] == "frame":
            if ev["idx"] == 1 or ev["idx"] % 5 == 0:
                eta = f"  ~{ev['eta'] / 60:.1f} min left" if ev["eta"] is not None else ""
                print(f"frame {ev['idx']}/{ev['total'] or '?'}  {ev['per']:.1f}s/frame{eta}", flush=True)
        else:
            result = ev["path"]
    print(f"done -> {result}")
    if args.output:
        shutil.copyfile(result, args.output)
        print(f"copied to {args.output}")


if __name__ == "__main__":
    main()
