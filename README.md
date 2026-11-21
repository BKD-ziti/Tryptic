# Tryptic

**by BKDziti**

Drop in a video, pick what happens to it, and take a trip into your own footage. There are three ways in:

- **DeepDream** lets a neural network hallucinate into every frame: patterns, eyes, fur, creatures, paint. Optical flow carries the dream along with the motion, so the dream moves with the footage instead of flickering on top of it.
- **Datamosh** deletes keyframes so the old picture gets dragged around by the new motion. You get melts, smears, stutters, and transitions that bleed one clip into the next.
- **ASCII art** rebuilds the clip out of characters, in its own colors, one color, or grayscale.

Everything runs on your own computer, and none of your video is ever uploaded. The finished video keeps your clip's size, shape, and exact frame rate. HDR footage (like an iPhone shoots) stays HDR, unless you change it.

<p align="center">


https://github.com/user-attachments/assets/f3736ced-61a9-407e-b4bd-f4faa78e5d1d


</p>

---

## What you need

| | |
|---|---|
| **System** | Windows 10 or 11 (64-bit) |
| **Python** | 3.10 to 3.14 from [python.org](https://www.python.org/downloads/windows/). 3.12 is recommended. When you install it, tick **"Add python.exe to PATH"**. |
| **Graphics** | An NVIDIA GPU is strongly recommended, and its driver should be up to date ([nvidia.com/drivers](https://www.nvidia.com/drivers)). Tryptic also runs without one, but DeepDream is a lot slower on the CPU. Datamosh and ASCII art are fine on any machine. |
| **Disk space** | About 5 GB for the app and its tools, plus room for your renders |
| **Internet** | Needed on the first run, which downloads PyTorch, ffmpeg, and the other packages. After that it works offline, except the first time you use each DeepDream network, which is a one-time download of 50 to 550 MB. |

Mac and Linux aren't supported yet.

## Install

1. Download Tryptic: click **Code → Download ZIP** on this page, then unzip it somewhere on your computer. (Or `git clone https://github.com/BKD-ziti/Tryptic.git`.)
2. Double-click **`start.bat`**. If Windows asks whether you want to run it, say yes.
3. The first run sets everything up in a private folder inside Tryptic (`labenv`). It picks the right PyTorch build for your GPU, which is a big download (about 2.5 GB), but it only happens once. When it's done, Tryptic opens in your web browser.

From then on, just double-click `start.bat`, and Tryptic starts in a few seconds. Close the black window to quit.

## Using it

1. **Drop a video in.** Any format and any size works. For really big files, open "point at a file already on this computer" and paste the path instead, which skips the upload copy.
2. **Pick DeepDream, Datamosh, or ASCII art**, then set the look. DeepDream and ASCII art have a **preview still** button, so you can tune the look on one frame first.
3. **Set the length to 3 or 4 seconds** while you're tuning, then set it back to 0 for the whole clip.
4. **Render video.** The result shows up in the player, and the real file is saved in the `output` folder (use the **open output folder** button).

Some other things worth knowing:

- **Size and speed** works the same for all three modes. By default you get the clip's own size, shape, and frame rate. You can switch the shape to vertical or horizontal, change the resolution, or type an exact size. DeepDream can lower the frame rate, but it can't raise it.
- **Color:** "match the clip" keeps HDR as HDR. "Standard SDR" tone-maps it down for screens and sites that don't handle HDR.
- **The player** in the app sometimes shows a small standard-color preview, so that every browser can play it. The file in `output` is always full size, in the original color.
- **The render queue** lets you set up a job, press **add to queue**, change the settings, and add another. Then press **start queue** and walk away. It can put the PC to sleep or shut it down when it's finished.
- **Datamosh seeds:** every datamosh logs its seed. Put the seed back in the seed box to repeat that exact look.
- **DeepDream presets:** save your own presets, and they appear in the dropdown next time.

## If something goes wrong

- **The window says "GPU: NO" on a computer with an NVIDIA card:** update your NVIDIA driver, then run `start.bat` again.
- **"The gpu ran out of memory":** lower the dream resolution or the number of zoom levels (octaves).
- **Something broke after an update, or the install got interrupted:** open a command prompt in the Tryptic folder and run `start.bat repair`. That rebuilds the private environment from scratch.
- **Still stuck?** [Open an issue](https://github.com/BKD-ziti/Tryptic/issues) and paste everything from the black window.

## Command line (optional)

Inside the environment you can also run each engine on its own:

```
labenv\Scripts\python.exe dream.py clip.mp4 --preset big-shapes --length 4
labenv\Scripts\python.exe mosh.py clip.mp4 --jumps 4 --bloom-hold 40
```

Run either one with `--help` to see every setting.

## Contact

- **Email:** [Contact@BKDziti.com](mailto:Contact@BKDziti.com)
- **Website:** [bkdziti.com](https://bkdziti.com)
- **Bugs and problems:** [open an issue](https://github.com/BKD-ziti/Tryptic/issues)

## License

Tryptic is **source-available, not open source**. You can download it, run it, and change it for your own personal, non-commercial use, and the videos you make with it are yours. Selling it, offering it as a paid product or an online service, or redistributing it needs written permission from BKDziti LLC. For commercial licensing, email [Contact@BKDziti.com](mailto:Contact@BKDziti.com). The full terms are in [LICENSE](LICENSE).

Tryptic downloads and uses third-party tools that have their own licenses: [FFmpeg](https://ffmpeg.org), [PyTorch and torchvision](https://pytorch.org), [Gradio](https://gradio.app), [OpenCV](https://opencv.org), and [Pillow](https://python-pillow.org).

---

**Tryptic** by **[BKDziti](https://bkdziti.com)** · rethinking food, media, and interaction, online and in person.
