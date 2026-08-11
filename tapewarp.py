#!/usr/bin/env python3
"""tapewarp — make any video look like it was shot on a 1990s camcorder tape.

Not a static grain overlay. Every frame gets a per-scanline, time-varying
treatment modeled on how VHS actually degrades:

  * wave warp + line jitter (per-scanline horizontal displacement)
  * tracking-error glitch bands that crawl up the frame
  * head-switching noise at the bottom edge, tear band at the top
  * chroma bleed / delay / drifting color bands (processed in YIQ space)
  * tape ghosting, dropout streaks, streaky luma grain
  * VCR OSD (PLAY / SP / date stamp) and tape-wow audio

Usage:
    python tapewarp.py input.mov
    python tapewarp.py input.mov -o out.mp4 --wear trashed --date "SEP.7 1998"

Requires: ffmpeg + ffprobe on PATH, numpy.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys

import numpy as np

# ---------------------------------------------------------------- presets

PRESETS = {
    # fresh tape, one careful owner
    "clean": dict(ghost=0.08, noise=0.022, band_i=0.05, band_q=0.08,
                  cast=0.018, glitch_every=8.0, glitch_amp=(14, 30),
                  dropout_p=0.2, jitter=0.25, contrast=1.06),
    # rented twice a week since 1992 (default)
    "worn": dict(ghost=0.13, noise=0.035, band_i=0.09, band_q=0.14,
                 cast=0.032, glitch_every=3.0, glitch_amp=(30, 55),
                 dropout_p=0.45, jitter=0.35, contrast=1.10),
    # found in a flooded basement
    "trashed": dict(ghost=0.18, noise=0.055, band_i=0.13, band_q=0.20,
                    cast=0.045, glitch_every=1.6, glitch_amp=(45, 80),
                    dropout_p=0.75, jitter=0.6, contrast=1.14),
}

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", path])
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = any(s["codec_type"] == "audio" for s in info["streams"])
    num, den = v["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    dur = float(info["format"].get("duration", 0)) or 1.0
    return int(v["width"]), int(v["height"]), fps, dur, a


def auto_date(path, style):
    """Date stamp from a YYYY-MM-DD filename, else file mtime."""
    m = re.search(r"(19|20)(\d{2})[-_.](\d{2})[-_.](\d{2})", os.path.basename(path))
    if m:
        year, mon, day = int(m.group(1) + m.group(2)), int(m.group(3)), int(m.group(4))
    else:
        d = datetime.date.fromtimestamp(os.path.getmtime(path))
        year, mon, day = d.year, d.month, d.day
    if style == "numeric":  # JVC/Panasonic-style " 6 26 2020"
        return f"{mon} {day} {year}"
    return f"{MONTHS[mon - 1]}.{day} {year}"


def auto_time(path):
    """Sony-style 'PM 1:03' from a ..._HH-MM-SS filename, else file mtime."""
    m = re.search(r"(19|20)\d{2}[-_.]\d{2}[-_.]\d{2}\D+(\d{2})[-_.](\d{2})",
                  os.path.basename(path))
    if m:
        hh, mm = int(m.group(2)), int(m.group(3))
    else:
        d = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        hh, mm = d.hour, d.minute
    ap = "AM" if hh < 12 else "PM"
    return f"{ap} {(hh - 1) % 12 + 1}:{mm:02d}"


def box_x(a, r):
    """Horizontal box blur of a 2-D array via cumsum, edge-padded."""
    if r <= 0:
        return a
    p = np.pad(a, ((0, 0), (r + 1, r)), mode="edge")
    c = np.cumsum(p, axis=1, dtype=np.float32)
    return (c[:, 2 * r + 1:] - c[:, :-(2 * r + 1)]) / (2 * r + 1)


def shift_lines(img, shifts, W):
    """Shift each scanline horizontally by a float per-line amount (wraps)."""
    H = img.shape[0]
    x = np.arange(W, dtype=np.float32)[None, :] - shifts[:, None]
    x0 = np.floor(x).astype(np.int32)
    frac = x - x0
    rows = np.arange(H)[:, None]
    return img[rows, x0 % W] * (1 - frac) + img[rows, (x0 + 1) % W] * frac


def main():
    ap = argparse.ArgumentParser(description="Make a video look like a 90s VHS tape.")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="output file (default: <input>_vhs.mp4)")
    ap.add_argument("--wear", choices=PRESETS, default="worn",
                    help="how beat-up the tape is (default: worn)")
    ap.add_argument("--dropouts", choices=["subtle", "classic", "off"], default="subtle",
                    help="white dropout streaks: subtle = faint smears like a VCR "
                         "with dropout compensation (default), classic = harsh "
                         "bright slashes, off = none")
    ap.add_argument("--ratio", choices=["source", "4:3", "4:3-box"], default="source",
                    help="4:3 crops to the era's camera framing; "
                         "4:3-box letterboxes instead (nothing lost)")
    ap.add_argument("--crop-bias", type=float, default=0.5,
                    help="where the 4:3 crop window sits: 0=left/top, "
                         "0.5=center (default), 1=right/bottom")
    ap.add_argument("--date", default="auto",
                    help='date stamp text, "auto" (from filename/mtime), or "none"')
    ap.add_argument("--time", default="none", dest="time_stamp",
                    help='time stamp text (e.g. "PM 1:03"), "auto", or "none"')
    ap.add_argument("--date-style", choices=["word", "numeric"], default="word",
                    help='"JUN.26 2020" (Sony) vs "6 26 2020" (JVC/Panasonic)')
    ap.add_argument("--stamp-pos", choices=["br", "bl", "bc", "tr", "tl"], default="br",
                    help="date/time corner; bottom-right is period-accurate (default)")
    ap.add_argument("--counter", action="store_true",
                    help="running VCR tape counter (top-right, under SP)")
    ap.add_argument("--wobble", type=float, default=1.0,
                    help="scale the per-line wave warp: 1.0 = full (default), "
                         "0.3 = gentle drift, 0 = dead steady. Edge tear and "
                         "head-switch noise are unaffected")
    ap.add_argument("--bleed", type=float, default=1.0,
                    help="scale chroma bleed and delay: 1.0 = full (default), "
                         "0.5 keeps colored text legible")
    ap.add_argument("--no-osd", action="store_true", help="skip PLAY/SP/date overlay")
    ap.add_argument("--no-audio-fx", action="store_true", help="pass audio through untouched")
    ap.add_argument("--seed", type=int, default=90, help="random seed (default: 90)")
    args = ap.parse_args()

    P = PRESETS[args.wear]
    rng = np.random.default_rng(args.seed)
    out_path = args.output or os.path.splitext(args.input)[0] + "_vhs.mp4"

    src_w, src_h, fps, dur, has_audio = probe(args.input)

    # process at VHS-ish resolution (~480 luma lines)
    if args.ratio == "4:3":
        # crop to the era's actual camera framing; --crop-bias slides the window
        out_w = min(src_w, src_h * 4 // 3) // 2 * 2
        out_h = min(src_h, src_w * 3 // 4) // 2 * 2
        bias = min(1.0, max(0.0, args.crop_bias))
        cx = int(bias * (src_w - out_w)) // 2 * 2
        cy = int(bias * (src_h - out_h)) // 2 * 2
        W, H = 640, 480
        dec_vf = f"crop={out_w}:{out_h}:{cx}:{cy},scale={W}:{H}:flags=bilinear"
    elif args.ratio == "4:3-box":
        # letterbox onto a 4:3 canvas — how widescreen actually landed on tape
        out_w = max(src_w, src_h * 4 // 3) // 2 * 2
        out_h = max(src_h, src_w * 3 // 4) // 2 * 2
        W, H = 640, 480
        dec_vf = (f"scale={W}:{H}:flags=bilinear:force_original_aspect_ratio=decrease,"
                  f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black")
    else:
        out_w, out_h = src_w, src_h
        H = 480 if src_w >= src_h else 640
        W = max(2, round(src_w / src_h * H / 2) * 2)
        dec_vf = f"scale={W}:{H}:flags=bilinear"

    # ---- tracking-glitch schedule
    events, t = [], rng.uniform(0.6, 1.6)
    while t < dur:
        f0 = int(t * fps)
        events.append((f0, int(rng.integers(4, 8)),
                       int(rng.integers(H // 8, H * 3 // 4)),
                       int(rng.integers(H // 9, H // 4)),
                       rng.uniform(*P["glitch_amp"])))
        t += rng.uniform(0.7, 1.3) * P["glitch_every"]

    # ---- drifting chroma-band track (long 1-D smooth noise, scrolled per frame)
    track_len = H * 8
    k = np.exp(-0.5 * np.linspace(-3, 3, 61) ** 2)
    k /= k.sum()
    band_i = np.convolve(rng.normal(0, 1, track_len).astype(np.float32), k, "same")
    band_q = np.convolve(rng.normal(0, 1, track_len).astype(np.float32), k, "same")

    bleed_r = int(round(8 * max(0.0, args.bleed)))
    bleed_dx = int(round(3 * max(0.0, args.bleed)))

    yn = np.linspace(0, 1, H, dtype=np.float32)
    xr = (np.arange(W, dtype=np.float32) - W / 2) / (W / 2)
    yr = (np.arange(H, dtype=np.float32) - H / 2) / (H / 2)
    vignette = 1.0 - 0.26 * (xr[None, :] ** 2 + yr[:, None] ** 2) ** 1.2

    # ---- decode / encode pipes
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", args.input, "-vf", dec_vf,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)

    font = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vcr_osd_mono.ttf")
    osd = []
    if not args.no_osd and os.path.exists(font):
        style = ("fontcolor=white@0.92:shadowx=2:shadowy=2:shadowcolor=black@0.45"
                 f":fontfile={font}")
        sz = H // 18
        mx, my = W // 19, H // 15
        # VCR playback overlay: PLAY top-left, tape speed top-right
        osd = [f"drawtext=text='PLAY':x={mx}:y={my}:fontsize={sz}:{style}",
               f"drawtext=text='SP':x=w-tw-{mx}:y={my}:fontsize={sz}:{style}"]
        # chunky pixel play-triangle (the OSD font has no glyph for it)
        tx, ty, u = mx + int(sz * 3.1), my + 2, max(2, sz // 6)
        for j, hh in enumerate((6, 4, 2)):
            osd.append(f"drawbox=x={tx + j * u}:y={ty + (6 - hh) * u // 2}"
                       f":w={u}:h={hh * u}:color=white@0.92:t=fill")
        if args.counter:  # running tape counter under SP
            cnt = (r"%{eif\:trunc(t/3600)\:d}\:%{eif\:mod(trunc(t/60)\,60)\:d\:2}"
                   r"\:%{eif\:mod(trunc(t)\,60)\:d\:2}")
            osd.append(f"drawtext=text='{cnt}':x=w-tw-{mx}:y={my + sz + 8}"
                       f":fontsize={sz}:{style}")
        # camera date/time stamp (burned in by 90s camcorders, bottom-right)
        lines = []
        if args.date.lower() != "none":
            lines.append(auto_date(args.input, args.date_style)
                         if args.date == "auto" else args.date)
        if args.time_stamp.lower() != "none":
            lines.append(auto_time(args.input)
                         if args.time_stamp == "auto" else args.time_stamp)
        lh = sz + 6
        for j, txt in enumerate(lines):
            xs = {"br": f"w-tw-{mx}", "bl": f"{mx}", "bc": "(w-tw)/2",
                  "tr": f"w-tw-{mx}", "tl": f"{mx}"}[args.stamp_pos]
            if args.stamp_pos in ("br", "bl", "bc"):
                y = H - H // 10 - (len(lines) - j - 1) * lh - sz
            else:  # below the VCR overlay row (and counter, if shown)
                y = my + sz + 8 + (sz + 8 if args.counter else 0) + j * lh
            txt = txt.replace(":", r"\:")  # colon is a filtergraph delimiter
            osd.append(f"drawtext=text='{txt}':x={xs}:y={y}:fontsize={sz - 2}:{style}")

    vf = ",".join(osd + [f"scale={out_w}:{out_h}:flags=bilinear", "format=yuv420p"])
    enc_cmd = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-"]
    if has_audio:
        enc_cmd += ["-i", args.input]
        if args.no_audio_fx:
            enc_cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
        else:
            enc_cmd += ["-f", "lavfi", "-i",
                        "anoisesrc=color=pink:sample_rate=44100:amplitude=0.06",
                        "-filter_complex",
                        "[1:a]highpass=f=80,lowpass=f=8500,vibrato=f=0.4:d=0.03,"
                        "volume=2.0[a0];[2:a]volume=0.5[n];"
                        "[a0][n]amix=inputs=2:duration=first,alimiter=limit=0.92[a]",
                        "-map", "0:v", "-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    enc_cmd += ["-vf", vf, "-c:v", "libx264", "-crf", "17", "-preset", "medium",
                "-shortest", "-y", out_path]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    frame_bytes = W * H * 3
    prev = None
    i = 0
    while True:
        buf = dec.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0
        t = i / fps
        if prev is None:
            prev = f

        # tape ghost: smear in a bit of the previous frame
        fr = (1 - P["ghost"]) * f + P["ghost"] * prev
        prev = f

        # RGB -> YIQ (NTSC's actual color space — chroma abuse happens here)
        r, g, b = fr[..., 0], fr[..., 1], fr[..., 2]
        Y = 0.299 * r + 0.587 * g + 0.114 * b
        I = 0.596 * r - 0.274 * g - 0.322 * b
        Q = 0.211 * r - 0.523 * g + 0.312 * b

        # luma: soften + ringing (wide unsharp overshoot)
        Y = box_x(Y, 1)
        Y = Y + 0.45 * (Y - box_x(Y, 4))

        # chroma: heavy horizontal bleed, delayed right and down a line
        I = np.roll(box_x(I, bleed_r), (1, bleed_dx), axis=(0, 1)) * 0.95
        Q = np.roll(box_x(Q, bleed_r), (1, bleed_dx), axis=(0, 1)) * 0.95

        # ---- per-line displacement field (the part that "moves") ----
        shifts = (1.7 * np.sin(2 * np.pi * (yn * 1.6 + t * 0.35))
                  + 0.7 * np.sin(2 * np.pi * (yn * 6.3 - t * 1.1))
                  + 0.5 * np.sin(2 * np.pi * t * 0.21)
                  + rng.normal(0, P["jitter"], H).astype(np.float32))
        bad = rng.random(H) < 0.02  # occasional torn lines
        shifts[bad] += rng.uniform(-3.5, 3.5, bad.sum())
        shifts += np.clip((yn - 0.86) / 0.14, 0, 1) * rng.normal(0, 2.2, H).astype(np.float32)
        shifts *= args.wobble  # tracking bands and edge tear are added after this

        noise_boost = np.zeros(H, dtype=np.float32)
        desat = np.ones(H, dtype=np.float32)
        lift = np.zeros(H, dtype=np.float32)

        # tracking-error bands crawling up the frame
        for (f0, ln, by, bh, amp) in events:
            if f0 <= i < f0 + ln:
                env = np.sin(np.pi * min(1, (i - f0) / ln + 0.15))
                y0 = by - (i - f0) * 5
                band = np.zeros(H, dtype=np.float32)
                lo, hi = max(0, y0), min(H, y0 + bh)
                if hi > lo:
                    band[lo:hi] = np.sin(np.linspace(0, np.pi, hi - lo)) ** 0.7
                shifts += band * env * (amp + rng.normal(0, 10, H).astype(np.float32) * band)
                noise_boost += band * env * 0.26
                desat -= band * env * 0.55
                lift += band * env * 0.10

        # drifting green/magenta chroma bands
        off = (i * 11) % (track_len - H)
        I += band_i[off:off + H, None] * P["band_i"]
        Q += band_q[off:off + H, None] * P["band_q"]

        # mangled tear band at the top edge
        tp = int(rng.integers(3, 7))
        shifts[:tp] += np.cumsum(rng.uniform(4, 20, tp))[::-1].astype(np.float32)
        noise_boost[:tp] += np.linspace(0.35, 0.08, tp)
        desat[:tp] *= 0.3

        # head-switching noise at the very bottom
        hs = max(6, H // 53)
        shifts[-hs:] += 12 + np.cumsum(rng.uniform(3, 16, hs)).astype(np.float32)
        noise_boost[-hs:] += np.linspace(0.05, 0.30, hs)
        desat[-hs:] *= 0.4

        Y = shift_lines(Y, shifts, W)
        cs = shifts + rng.normal(0, 0.5, H).astype(np.float32)  # chroma jitters extra
        I = shift_lines(I, cs, W)
        Q = shift_lines(Q, cs, W)

        desat = np.clip(desat, 0, 1)[:, None]
        I *= desat
        Q *= desat
        Y = np.clip(Y + lift[:, None], 0, 1.2)

        # dropout streaks — oxide shedding; a real VCR's dropout compensation
        # patched these with the previous scanline, so default to faint smears
        if args.dropouts != "off" and rng.random() < P["dropout_p"] * \
                (1.0 if args.dropouts == "classic" else 0.6):
            for _ in range(rng.integers(1, 3)):
                ry = int(rng.integers(0, H - 2))
                rx = int(rng.integers(0, max(1, W - 60)))
                ln = int(rng.integers(40, 220))
                seg = Y[ry:ry + 1, rx:rx + ln]
                a = np.sin(np.linspace(0, np.pi, seg.shape[1]))[None, :]  # feathered ends
                if args.dropouts == "classic":
                    lvl, mix = (0.95 if rng.random() < 0.8 else 0.05), 0.9
                else:
                    lvl, mix = 0.85, 0.35  # dim blend, reads as a smear not a slash
                Y[ry:ry + 1, rx:rx + ln] = seg * (1 - a * mix) + lvl * a * mix

        # noise: streaky luma grain + soft chroma noise
        Y += box_x(rng.normal(0, 1, (H, W)).astype(np.float32), 1) * (P["noise"] + noise_boost[:, None])
        I += box_x(rng.normal(0, 1, (H, W)).astype(np.float32), 6) * 0.015
        Q += box_x(rng.normal(0, 1, (H, W)).astype(np.float32), 6) * 0.015

        # interlace field flicker (subtle)
        Y[i % 2::2] *= 0.985

        # lo-fi grade: magenta cast, contrast, blown highlights
        Q += P["cast"]
        Y = (Y - 0.5) * P["contrast"] + 0.52

        R = Y + 0.956 * I + 0.621 * Q
        G = Y - 0.272 * I - 0.647 * Q
        B = Y - 1.106 * I + 1.703 * Q
        rgb = np.stack([R * 1.03, G * 0.99, B * 1.02], axis=-1)
        rgb = (rgb * 0.93 + 0.045) * vignette[..., None]  # lifted blacks
        enc.stdin.write((np.clip(rgb, 0, 1) * 255).astype(np.uint8).tobytes())

        i += 1
        if i % 60 == 0:
            print(f"\r  {i} frames ({i / fps:.1f}s)", end="", file=sys.stderr, flush=True)

    enc.stdin.close()
    dec.wait()
    if enc.wait() != 0:
        sys.exit("ffmpeg encode failed")
    print(f"\r  {i} frames — wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
