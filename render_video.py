#!/usr/bin/env python3
"""Turn the replay log of a painting run into an MP4 for X / LinkedIn.

    render_video.py OUT.mp4 RUN_DIR [RUN_DIR2] [options]

One run: the painting fills in with live instrumentation (decisions, cost, the
real elapsed clock, a progress bar and a speed badge). Two runs: side by side,
a race on ONE shared real-time clock. Both are sped up by the same factor, so
the comparison is honest; the one that finishes first stays finished with a
"done" badge while the other keeps painting.

While it paints, every number comes from the run's event log at the current
video time (the last event at or before it), never interpolated past real data;
the final hold shows live.js's final numbers (accuracy, decisions, time, cost).

Examples
  render_video.py jev.mp4 runs/bobross-blocks-512-2b721093
  render_video.py tall.mp4 runs/bobross-256-3a2ad587 --format portrait
  render_video.py race.mp4 runs/jev-run runs/laya-run \\
      --label "Jev · cloud API" --label2 "Laya · on my laptop"

Writes OUT.mp4 (H.264 High, yuv420p, CRF 18, constant frame rate, faststart,
silent AAC stereo 48 kHz track) and OUT_poster.png (the final frame).
"""

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    _RS = Image.Resampling
except AttributeError:  # Pillow < 9.1
    _RS = Image
NEAREST, BOX = _RS.NEAREST, _RS.BOX

FORMATS = {"square": (1080, 1080), "landscape": (1920, 1080), "portrait": (1080, 1350)}
FFMPEG_FALLBACK = "/opt/homebrew/bin/ffmpeg"
PULSE_PERIOD = 2.4      # seconds; the viewer's breathing status dot
FRAME_MAX = 24          # thickest wood frame, px


# ----------------------------------------------------------------------------- colour

def hex_rgb(value, default=None):
    if not isinstance(value, str):
        return default
    h = value.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        return default
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


def mix(a, b, t):
    """a over b at opacity t."""
    return tuple(int(round(a[i] * t + b[i] * (1 - t))) for i in range(3))


# viewer.html's palette: dark warm background, serif titles, mono numbers, wood frame
BG = hex_rgb("#17130F")
LINE = hex_rgb("#2E2720")
INK = hex_rgb("#EFE6D7")
INK2 = hex_rgb("#AC9F8C")
INK3 = hex_rgb("#7C7061")
ACCENT = hex_rgb("#D9A441")
ACCENT_TXT = hex_rgb("#F0D9A6")
GOOD = hex_rgb("#8FA96C")
CANVAS = hex_rgb("#F4EFE4")
WOOD_HI = hex_rgb("#7A5A3A")
WOOD = hex_rgb("#543C26")
WOOD_LO = hex_rgb("#3B2917")
ACCENT_BG = mix(ACCENT, BG, 0.10)
ACCENT_LINE = mix(ACCENT, BG, 0.30)
PILL_BG = mix(ACCENT, BG, 0.14)
PILL_LINE = mix(ACCENT, BG, 0.42)
GOOD_BG = mix(GOOD, BG, 0.16)
GOOD_LINE = mix(GOOD, BG, 0.50)
TRACK = mix((255, 255, 255), BG, 0.07)


# ----------------------------------------------------------------------------- fonts

FONT_FILES = {
    "serif": [("/System/Library/Fonts/NewYork.ttf", 0),
              ("/System/Library/Fonts/Supplemental/Iowan Old Style.ttc", 0),
              ("/System/Library/Fonts/Supplemental/Georgia.ttf", 0),
              ("/System/Library/Fonts/Palatino.ttc", 0),
              ("/System/Library/Fonts/Times.ttc", 0),
              ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 0)],
    "serif-italic": [("/System/Library/Fonts/NewYorkItalic.ttf", 0),
                     ("/System/Library/Fonts/Supplemental/Iowan Old Style.ttc", 2),
                     ("/System/Library/Fonts/Supplemental/Georgia Italic.ttf", 0),
                     ("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf", 0)],
    "sans": [("/System/Library/Fonts/SFNS.ttf", 0),
             ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
             ("/System/Library/Fonts/Helvetica.ttc", 0),
             ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0)],
    "mono": [("/System/Library/Fonts/SFNSMono.ttf", 0),
             ("/System/Library/Fonts/Menlo.ttc", 0),
             ("/System/Library/Fonts/Monaco.ttf", 0),
             ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0)],
}
WEIGHTS = {"regular": 400, "medium": 500, "semibold": 600}


class Fonts:
    """First font file that exists per kind; variable fonts get weight + optical size."""

    def __init__(self):
        self.cache = {}
        self.files = {kind: next(((p, i) for p, i in cands if os.path.exists(p)), None)
                      for kind, cands in FONT_FILES.items()}
        if self.files["serif-italic"] is None:
            self.files["serif-italic"] = self.files["serif"]

    def get(self, kind, size, weight="regular"):
        size = max(6, int(round(size)))
        key = (kind, size, weight)
        if key not in self.cache:
            self.cache[key] = self._load(kind, size, weight)
        return self.cache[key]

    def _load(self, kind, size, weight):
        spec = self.files.get(kind)
        if spec:
            try:
                font = ImageFont.truetype(spec[0], size, index=spec[1])
                self._vary(font, size, weight)
                return font
            except Exception:
                pass
        try:
            return ImageFont.load_default(size=size)
        except Exception:
            return ImageFont.load_default()

    @staticmethod
    def _vary(font, size, weight):
        try:
            axes = font.get_variation_axes()
        except Exception:
            return  # static font
        names = [a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"]) for a in axes]
        wght = WEIGHTS.get(weight, 400)
        try:
            if "Optical Size" in names:
                values = []
                for name, a in zip(names, axes):
                    v = a["default"]
                    if name == "Weight":
                        v = wght
                    elif name == "Optical Size":
                        v = size * 0.56   # seen small on a phone: favour a sturdier optical size
                    values.append(min(a["maximum"], max(a["minimum"], v)))
                font.set_variation_by_axes(values)
            else:
                font.set_variation_by_name(weight.capitalize())
        except Exception:
            pass


FONTS = Fonts()
TR_LABEL = 0.14   # letter spacing of the small uppercase labels, in em
TR_STATUS = 0.16
TR_PILL = 0.13


def cap_h(font):
    try:
        return -font.getbbox("H", anchor="ls")[1]
    except Exception:
        return int(getattr(font, "size", 12) * 0.7)


def text_width(font, s, track=0.0):
    if not s:
        return 0.0
    if not track:
        return font.getlength(s)
    return sum(font.getlength(ch) for ch in s) + track * (len(s) - 1)


def draw_text(d, x, y, s, font, fill, align="l", track=0.0):
    """Draw s with its baseline at y; x is the left, middle or right edge (align l/m/r)."""
    w = text_width(font, s, track)
    if align == "m":
        x -= w / 2
    elif align == "r":
        x -= w
    if not track:
        d.text((x, y), s, font=font, fill=fill, anchor="ls")
        return w
    for ch in s:
        d.text((x, y), ch, font=font, fill=fill, anchor="ls")
        x += font.getlength(ch) + track
    return w


def ellipsize(font, s, max_w, track=0.0):
    if text_width(font, s, track) <= max_w:
        return s
    while s and text_width(font, s.rstrip() + "…", track) > max_w:
        s = s[:-1]
    return s.rstrip() + "…" if s else ""


def fit_text(kind, px, weight, s, max_w, min_px, track_em=0.0):
    """Largest font from px down to min_px that fits max_w, else ellipsize at min_px."""
    px, min_px = int(round(px)), int(round(min(px, min_px)))
    for size in range(px, min_px - 1, -1):
        font = FONTS.get(kind, size, weight)
        if text_width(font, s, track_em * size) <= max_w:
            return font, s
    font = FONTS.get(kind, min_px, weight)
    return font, ellipsize(font, s, max_w, track_em * min_px)


def wrap(font, s, max_w):
    lines, cur = [], ""
    for word in s.split():
        trial = (cur + " " + word).strip()
        if not cur or text_width(font, trial) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def wrap_balanced(font, s, max_w):
    """Greedy wrap, then the narrowest width that keeps the same line count (no orphan words)."""
    lines = wrap(font, s, max_w)
    if len(lines) < 2:
        return lines
    lo, hi = max_w / len(lines), max_w
    for _ in range(18):
        mid = (lo + hi) / 2
        if len(wrap(font, s, mid)) <= len(lines):
            hi = mid
        else:
            lo = mid
    return wrap(font, s, hi)


# ----------------------------------------------------------------------------- shapes

def ss_draw(img, box, fn, ss=4):
    """Anti-aliased drawing: redraw the patch under box at ss x and area-average it back."""
    x0, y0 = max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1])))
    x1, y1 = min(img.width, int(math.ceil(box[2]))), min(img.height, int(math.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return
    patch = img.crop((x0, y0, x1, y1)).resize(((x1 - x0) * ss, (y1 - y0) * ss), NEAREST)
    fn(ImageDraw.Draw(patch), lambda x, y: ((x - x0) * ss, (y - y0) * ss), ss)
    img.paste(patch.resize((x1 - x0, y1 - y0), BOX), (x0, y0))


def rrect(img, rect, radius, fill=None, outline=None, width=0):
    x0, y0, x1, y1 = rect

    def fn(d, tr, ss):
        a, b = tr(x0, y0), tr(x1, y1)
        d.rounded_rectangle([a[0], a[1], b[0] - 1, b[1] - 1], radius=radius * ss, fill=fill,
                            outline=outline, width=int(round(width * ss)) if outline else 0)
    ss_draw(img, (x0 - 1, y0 - 1, x1 + 1, y1 + 1), fn)


def circle(img, cx, cy, r, fill):
    def fn(d, tr, ss):
        a, b = tr(cx - r, cy - r), tr(cx + r, cy + r)
        d.ellipse([a[0], a[1], b[0], b[1]], fill=fill)
    ss_draw(img, (cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1), fn)


def rounded_mask(w, h, r, ss=4):
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=r * ss, fill=255)
    return m.resize((w, h), BOX)


def drop_shadow(img, rect, dy, blur, spread, alpha, radius=5):
    """CSS-style box-shadow: offset dy, blur (sigma = blur / 2), spread, black at alpha."""
    x0, y0, x1, y1 = rect[0] - spread, rect[1] - spread + dy, rect[2] + spread, rect[3] + spread + dy
    pad = int(blur * 1.6) + 2
    bx0, by0 = max(0, x0 - pad), max(0, y0 - pad)
    bx1, by1 = min(img.width, x1 + pad), min(img.height, y1 + pad)
    if bx1 <= bx0 or by1 <= by0:
        return
    m = Image.new("L", (bx1 - bx0, by1 - by0), 0)
    ImageDraw.Draw(m).rounded_rectangle([x0 - bx0, y0 - by0, x1 - bx0 - 1, y1 - by0 - 1],
                                        radius=radius, fill=int(255 * alpha))
    m = m.filter(ImageFilter.GaussianBlur(blur / 2))
    img.paste((0, 0, 0), (bx0, by0, bx1, by1), m)


def draw_frame(img, fx0, fy0, fw, T, canvas_rgb):
    """The viewer's wood frame: 148deg gradient, bevel highlights, a dark hairline round the canvas."""
    rect = (fx0, fy0, fx0 + fw, fy0 + fw)
    drop_shadow(img, rect, dy=26, blur=52, spread=-22, alpha=0.85)
    drop_shadow(img, rect, dy=4, blur=14, spread=-6, alpha=0.60)
    ang = math.radians(148.0)
    dx, dy = math.sin(ang), -math.cos(ang)
    length = abs(fw * dx) + abs(fw * dy)
    c = np.arange(fw, dtype=np.float32) - (fw - 1) / 2
    p = (c[None, :] * dx + c[:, None] * dy) / length + 0.5
    stops = [0.0, 0.34, 0.66, 1.0]
    cols = [WOOD_HI, WOOD, WOOD_LO, WOOD]
    arr = np.stack([np.interp(p, stops, [col[i] for col in cols]) for i in range(3)], axis=-1)
    arr[0] = arr[0] * 0.84 + np.array([255, 236, 204]) * 0.16
    arr[-1] *= 0.55
    ring = np.zeros((fw, fw), bool)
    ring[T - 1:fw - T + 1, T - 1:fw - T + 1] = True
    ring[T:fw - T, T:fw - T] = False
    arr[ring] *= 0.45
    arr[T:fw - T, T:fw - T] = canvas_rgb
    wood = Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8))
    img.paste(wood, (fx0, fy0), rounded_mask(fw, fw, 5))


def make_pill(text, font, track, fg, fill, border, height, pad_x, dot=None, dot_r=5):
    """An opaque badge sprite (drawn over BG)."""
    tw = text_width(font, text, track)
    dot_w = 2 * dot_r + 11 if dot else 0
    w, h = int(math.ceil(2 * pad_x + dot_w + tw)), int(height)
    ss = 4
    big = Image.new("RGB", (w * ss, h * ss), BG)
    bd = ImageDraw.Draw(big)
    bd.rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=h * ss // 2, fill=fill,
                         outline=border, width=2 * ss if border else 0)
    if dot:
        cx, cy = (pad_x + dot_r) * ss, h * ss / 2
        bd.ellipse([cx - dot_r * ss, cy - dot_r * ss, cx + dot_r * ss, cy + dot_r * ss], fill=dot)
    im = big.resize((w, h), BOX)
    draw_text(ImageDraw.Draw(im), pad_x + dot_w, (h + cap_h(font)) / 2, text, font, fg, track=track)
    return im


# ----------------------------------------------------------------------------- formatting

def fmt_num(n):
    return f"{int(round(n)):,}"


def fmt_money(x):
    return f"${x:,.4f}"


def fmt_clock(s):
    """m:ss.s, truncated like a stopwatch (never shows a time not yet reached)."""
    tenths = int(math.floor(max(0.0, float(s)) * 10 + 1e-6))
    h, rem = divmod(tenths, 36000)
    m, rem = divmod(rem, 600)
    return f"{h}:{m:02d}:{rem / 10:04.1f}" if h else f"{m}:{rem / 10:04.1f}"


def fmt_pct(frac):
    return f"{math.floor(frac * 1000 + 1e-9) / 10:.1f}%"


def fmt_acc(a):
    s = f"{a * 100:.2f}"
    if s == "100.00" and a < 1:
        s = "99.99"
    return s + "%"


def fmt_speed(s):
    return str(int(round(s))) if abs(s - round(s)) < 1e-6 else f"{s:.1f}"


def nice_speed(t_max, target):
    """Speed-up so the longest run fits the target, rounded UP to 2 significant figures
    (so the badge states the exact factor used). Runs already shorter play at 1x."""
    if t_max <= target or t_max <= 0:
        return 1.0
    raw = t_max / target
    step = 10.0 ** (math.floor(math.log10(raw)) - 1)
    return round(math.ceil(raw / step - 1e-9) * step, 6)


# ----------------------------------------------------------------------------- run data

def load_js(path):
    """window.__x = {...};  ->  dict"""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("window."):
        text = text.split("=", 1)[1].strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    return json.loads(text)


def number(v, default=None):
    if isinstance(v, bool):
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


class Run:
    """One run directory: the event log decoded into flat arrays, plus the final numbers."""

    def __init__(self, run_dir):
        d = Path(run_dir).expanduser()
        if not d.is_dir():
            raise SystemExit(f"not a run directory: {d}")
        self.dir, self.name = d, d.name
        try:
            replay = load_js(d / "replay.js")
        except FileNotFoundError:
            raise SystemExit(f"{d}: no replay.js")
        except Exception as e:
            raise SystemExit(f"{d / 'replay.js'}: cannot parse ({e})")
        live = self._optional(d / "live.js", load_js)
        meta = self._optional(d / "run.json", lambda p: json.loads(Path(p).read_text(encoding="utf-8")))
        events = [e for e in (replay.get("events") or []) if isinstance(e, dict)]

        size = int(number(replay.get("size")) or number(live.get("size")) or number(meta.get("size")) or 0)
        if size <= 0:
            far = max((int(number(e.get("start"), 0)) + len(str(e.get("colours") or "")) // 6 for e in events),
                      default=1)
            size = max(1, math.isqrt(max(0, far - 1)) + 1)
        self.size, self.total = size, size * size
        self.canvas_rgb = hex_rgb(replay.get("canvas")) or hex_rgb(live.get("canvas")) or CANVAS

        # decode, stable-sorted by t (a repainted pixel: the later event wins)
        t_raw = np.array([number(e.get("t"), 0.0) for e in events], dtype=np.float64)
        order = np.argsort(t_raw, kind="stable")
        ts, starts, lens, offs, cost, qs, writes, chunks = [], [], [], [], [], [], [], []
        last_cost, last_q, n_written, off, self.has_q = 0.0, 0.0, 0, 0, False
        for k in order:
            e = events[k]
            c = number(e.get("cost"))
            if c is not None:
                last_cost = c
            q = number(e.get("questions"))
            if q is not None:
                last_q, self.has_q = q, True
            for start, raw in (self._segments(e) or [(0, b"")]):
                n = len(raw) // 3
                ts.append(t_raw[k]); starts.append(start); lens.append(n); offs.append(off)
                chunks.append(raw); off += n; n_written += n
                cost.append(last_cost); qs.append(last_q); writes.append(n_written)
        self.n = len(ts)
        self.ts = np.array(ts, dtype=np.float64)
        self.starts, self.lens, self.offs = starts, lens, offs
        self.colours = (np.frombuffer(b"".join(chunks), dtype=np.uint8).reshape(-1, 3)
                        if off else np.zeros((0, 3), np.uint8))
        # cumulative counters never go down on screen, even if a log has a blip
        self.cost = np.maximum.accumulate(np.array(cost, dtype=np.float64)) if self.n else np.zeros(0)
        self.q = np.maximum.accumulate(np.array(qs, dtype=np.float64)) if self.n else np.zeros(0)
        self.writes = np.array(writes, dtype=np.int64)
        self.t_end = float(self.ts[-1]) if self.n else 0.0

        # final numbers: live.js is authoritative, the event log is the fallback
        self.title = str(live.get("title") or meta.get("title") or "").strip()
        self.byline = str(live.get("byline") or "").strip()
        self.method = meta.get("method") if isinstance(meta.get("method"), str) else ""
        engine, model = meta.get("engine"), meta.get("model")
        self.painter = (engine.strip().capitalize() if isinstance(engine, str) and engine.strip() else
                        model.strip().rsplit("/", 1)[-1] if isinstance(model, str) and model.strip() else "")
        self.finished = bool(live.get("finished", True))
        fc = number(live.get("cost"))
        self.final_cost = fc if fc is not None else (float(self.cost[-1]) if self.n else 0.0)
        self.free = self.final_cost == 0 and not bool(np.any(self.cost > 0))
        if self.has_q:
            lq = number(live.get("questions"))
            self.final_decisions = int(lq if lq is not None else self.q[-1])
            self.decisions_label = "DECISIONS"
        else:  # older logs: no decision count, so count painted pixels and say so
            self.final_decisions = int(self.writes[-1]) if self.n else 0
            self.decisions_label = "PIXELS PAINTED"
        el = number(live.get("elapsed"))
        if el is None:
            el = number(meta.get("seconds"))
        self.took = el if el is not None and el + 0.05 >= self.t_end else self.t_end
        acc = number(live.get("accuracy"))
        if acc is None and "accuracy" not in live:
            acc = number(meta.get("accuracy"))
        self.accuracy = acc if acc is not None and 0 <= acc <= 1 else None
        self.palette = [c for c in (hex_rgb(p.get("hex")) for p in (live.get("palette") or [])
                                    if isinstance(p, dict)) if c]
        self.max_decisions = max(self.final_decisions,
                                 int(self.q.max()) if self.has_q and self.n else 0,
                                 int(self.writes[-1]) if not self.has_q and self.n else 0)
        self.max_cost = max(self.final_cost, float(self.cost.max()) if self.n else 0.0)

    @staticmethod
    def _optional(path, loader):
        if not Path(path).is_file():
            return {}
        try:
            data = loader(path)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"warning: ignoring {path}: {e}", file=sys.stderr)
            return {}

    def _segments(self, e):
        """(start, rgb bytes) runs for one event, clipped to the canvas; bad hex pixels are skipped."""
        col = e.get("colours")
        start = number(e.get("start"))
        if not isinstance(col, str) or start is None:
            return []
        start, m = int(start), len(col) // 6
        col = col[:m * 6]
        try:
            runs = [(start, bytes.fromhex(col))]
        except ValueError:
            runs, cur_s, cur = [], None, bytearray()
            for i in range(m):
                try:
                    px = bytes.fromhex(col[i * 6:i * 6 + 6])
                except ValueError:
                    px = None
                if px is None or len(px) != 3:
                    if cur:
                        runs.append((cur_s, bytes(cur)))
                    cur_s, cur = None, bytearray()
                    continue
                if cur_s is None:
                    cur_s = start + i
                cur += px
            if cur:
                runs.append((cur_s, bytes(cur)))
        out = []
        for s, raw in runs:
            n = len(raw) // 3
            if s < 0:
                raw, n, s = raw[-s * 3:], n + s, 0
            if s + n > self.total:
                n = self.total - s
                raw = raw[:max(0, n) * 3]
            if n > 0:
                out.append((s, raw))
        return out


class Player:
    """Applies a run's events incrementally to a numpy canvas."""

    def __init__(self, run):
        self.r = run
        self.flat = np.empty((run.total, 3), np.uint8)
        self.flat[:] = run.canvas_rgb
        self.cov = np.zeros(run.total, bool)
        self.idx, self.covered, self._img = 0, 0, None

    def seek(self, j):
        if j <= self.idx:
            return
        r, flat, cov = self.r, self.flat, self.cov
        for i in range(self.idx, j):
            n = r.lens[i]
            if n:
                s, o = r.starts[i], r.offs[i]
                flat[s:s + n] = r.colours[o:o + n]
                cov[s:s + n] = True
        self.idx = j
        self.covered = int(np.count_nonzero(cov))
        self._img = None

    @property
    def done(self):
        return self.idx >= self.r.n

    def stats(self, clock):
        r, i = self.r, self.idx - 1
        dec = (r.q[i] if r.has_q else r.writes[i]) if i >= 0 else 0
        cost = r.cost[i] if i >= 0 else 0.0
        elapsed = r.took if self.done else min(clock, r.t_end)
        return dec, cost, elapsed

    def image(self, D):
        if self._img is None:
            N = self.r.size
            im = Image.fromarray(self.flat.reshape(N, N, 3))
            if D != N:   # NEAREST keeps pixels crisp going up; area-average going down
                im = im.resize((D, D), NEAREST if D > N else BOX)
            self._img = im
        return self._img


def frame_thickness(D):
    return int(round(min(FRAME_MAX, max(12, D * 0.036))))


def fit_canvas(sizes, outer, fill=False):
    """Canvas display size D whose frame (D + 2 * thickness) fits in outer.

    One run: the largest integer multiple of its size (NEAREST keeps every pixel an even,
    crisp square); only a canvas bigger than the space is scaled down (area average).
    A race: both canvases share one size. A common integer multiple of both sizes is used
    when it reaches 80% of the space; otherwise (e.g. 256 vs 64 in a narrow column, where
    the only common multiple is 256) both fill the space. --fill always fills."""
    outer = max(1, int(outer))
    biggest = next((D for D in range(outer, 0, -1) if D + 2 * frame_thickness(D) <= outer), 1)
    if fill:
        return biggest
    lcm = 1
    for s in sizes:
        lcm = lcm * s // math.gcd(lcm, s)
    k = biggest // lcm
    if k >= 1 and (len(sizes) == 1 or k * lcm >= 0.8 * biggest):
        return k * lcm
    return biggest


# ----------------------------------------------------------------------------- scene

class Scene:
    def __init__(self, runs, args, speed):
        self.runs, self.speed, self.fill = runs, speed, bool(getattr(args, "fill", False))
        self.race = len(runs) > 1
        self.fmt = args.format or ("landscape" if self.race else "square")
        self.W, self.H = FORMATS[self.fmt]
        self.land = self.W > self.H
        labels, subs = [args.label, args.label2], [args.sub, args.sub2]
        self.cols = []
        for i, r in enumerate(runs):
            c = SimpleNamespace(run=r, i=i, status_mode=None)
            label, sub = (labels[i] or "").strip(), (subs[i] or "").strip()
            if self.race:
                c.label = label or " · ".join(x for x in (r.painter, f"{r.size} × {r.size}", r.method) if x)
                c.sub = sub
            else:
                c.label, c.sub = label, (sub if label else "")
            self.cols.append(c)
        r0 = runs[0]
        if args.title:
            self.title = args.title.strip()
        elif not self.race:
            self.title = r0.title or "A painting, decision by decision"
        elif r0.title and all(r.title == r0.title for r in runs):
            self.title = r0.title
        else:  # "Jev paints ..." + "Laya paints ..." -> "Jev vs Laya" (the labels say the rest)
            heads = [r.title.split()[0] for r in runs if r.title.split()]
            self.title = f"{heads[0]} vs {heads[1]}" if len(heads) == 2 and heads[0] != heads[1] \
                else "A painting race"
        self.line2 = ((args.sub or "").strip() if (args.sub and not args.label) else "") \
            or r0.byline or f"{r0.size} × {r0.size} pixels"
        sp = "REAL TIME" if speed == 1 else f"SHOWN AT {fmt_speed(speed)}× SPEED"
        self.speed_text = ("ONE SHARED CLOCK · " + sp) if self.race else sp
        self.note = (("Two real runs, replayed from their logs on one shared clock" if self.race
                      else "A real run, replayed from its log")
                     + (" · shown in real time" if speed == 1 else f" · shown at {fmt_speed(speed)}× speed"))
        self.players = [Player(r) for r in runs]
        self._pulse = {}
        self._sprites()
        if self.race and self.fmt == "portrait":
            self._layout_race_rows()
        elif self.race:
            self._layout_race_cols()
        elif self.land:
            self._layout_single_side()
        else:
            self._layout_single_stacked()
        self._check_bounds()
        self.base_paint = self._base("paint")

    # ---- sprites -------------------------------------------------------------

    def _sprites(self):
        pf = FONTS.get("sans", 18 if self.race else 17, "semibold")
        self.speed_pill = make_pill(self.speed_text, pf, TR_PILL * pf.size, ACCENT_TXT, PILL_BG, PILL_LINE,
                                    height=40, pad_x=18)
        bf = FONTS.get("sans", 17, "semibold")
        tr = TR_STATUS * bf.size
        self.done_pill = make_pill("DONE", bf, tr, GOOD, GOOD_BG, GOOD_LINE, height=36, pad_x=15, dot=GOOD)
        self.first_pill = make_pill("FINISHED FIRST", bf, tr, GOOD, GOOD_BG, GOOD_LINE, height=36, pad_x=15,
                                    dot=GOOD)
        self.stopped_pill = make_pill("STOPPED", bf, tr, INK2, mix(INK3, BG, 0.14), mix(INK3, BG, 0.5),
                                      height=36, pad_x=15, dot=INK3)
        self.status_f = FONTS.get("sans", 18, "semibold")
        self.dot_r = 6
        self.status_w = max(text_width(self.status_f, s, TR_STATUS * self.status_f.size)
                            for s in ("PAINTING", "FINISHED", "STOPPED")) + 2 * self.dot_r + 14
        if self.race:
            self.status_w = max(self.status_w, self.done_pill.width, self.first_pill.width,
                                self.stopped_pill.width)

    def pulse_sprite(self, vt):
        """The viewer's breathing dot: an RGBA sprite, composited over whatever is below."""
        n = 48
        k = int((vt % PULSE_PERIOD) / PULSE_PERIOD * n) % n
        if k not in self._pulse:
            phase, r, grow = k / n, self.dot_r, 10
            if phase < 0.7:
                q = phase / 0.7
                spread, alpha = grow * (1 - (1 - q) ** 2), 0.42 * (1 - q)
            else:
                q = (phase - 0.7) / 0.3
                spread, alpha = grow * (1 - q), 0.42 * q
            size, ss = 2 * (r + grow + 2), 4
            big = Image.new("RGBA", (size * ss, size * ss), ACCENT + (0,))
            bd = ImageDraw.Draw(big)
            c = size * ss / 2
            if alpha > 0.005:
                rr = (r + spread) * ss
                bd.ellipse([c - rr, c - rr, c + rr, c + rr], fill=ACCENT + (int(255 * alpha),))
            bd.ellipse([c - r * ss, c - r * ss, c + r * ss, c + r * ss], fill=ACCENT + (255,))
            self._pulse[k] = big.resize((size, size), BOX)
        return self._pulse[k]

    def _final_pill(self, c):
        r = c.run
        if not r.finished:
            return self.stopped_pill
        others = [o.run for o in self.cols if o is not c]
        if all(r.took < o.took - 0.05 or not o.finished for o in others):
            return self.first_pill
        return self.done_pill

    # ---- geometry helpers ----------------------------------------------------

    def _cells(self, r):
        """Live stat cells: (label, widest value string, kind)."""
        return [(r.decisions_label, fmt_num(r.max_decisions), "num"),
                ("COST", fmt_money(r.max_cost), "cost"),
                ("ELAPSED", fmt_clock(max(r.t_end, r.took)), "clock")]

    def _final_cells(self, r):
        cells = []
        if r.accuracy is not None:
            cells.append(("ACCURACY", fmt_acc(r.accuracy), "plain"))
        cells.append((r.decisions_label, fmt_num(r.final_decisions), "plain"))
        cells.append(("TOOK", fmt_clock(r.took), "plain"))
        cells.append(("COST", fmt_money(r.final_cost), "cost"))
        return cells

    @staticmethod
    def _value_w(vf, xf, s, kind, r):
        w = text_width(vf, s)
        if kind == "cost" and r.free:
            w += text_width(xf, " · free")
        return w

    def _fit_cells(self, width, specs, label_px, value_px, min_gap=28, min_scale=0.5):
        """Fonts so every column's cells fit side by side in width. Values keep their size
        as long as possible: the small labels shrink first (to 14 px), then the values."""
        s = 1.0
        while True:
            vf = FONTS.get("mono", value_px * s, "regular")
            xf = FONTS.get("sans", value_px * s * 0.46, "medium")
            for lp in range(int(round(label_px * min(1.0, s + 0.25))), 13, -1):
                lf = FONTS.get("sans", lp, "medium")
                widths = [[max(text_width(lf, lab, TR_LABEL * lp), self._value_w(vf, xf, v, kind, r))
                           for lab, v, kind in spec] for r, spec in specs]
                if all(sum(w) + min_gap * (len(w) - 1) <= width for w in widths):
                    return lf, vf, xf, widths, True
            if s <= min_scale + 1e-9:
                return lf, vf, xf, widths, False
            s = round(s - 0.05, 3)

    @staticmethod
    def _spread(width, widths):
        """x offsets for cells laid out space-between across width."""
        if len(widths) == 1:
            return [0]
        gap = (width - sum(widths)) / (len(widths) - 1)
        xs, x = [], 0.0
        for w in widths:
            xs.append(int(round(x)))
            x += w + gap
        return xs

    def _stats_row(self, width, cols, label_px, value_px, caption_px, pill):
        """Three stats side by side, a progress bar, a caption (and optionally the speed pill)."""
        specs = [(c.run, self._cells(c.run)) for c in cols]
        lf, vf, xf, widths, fits = self._fit_cells(width, specs, label_px, value_px)
        cf = FONTS.get("mono", caption_px, "regular")
        g = SimpleNamespace(kind="row", width=width, lf=lf, vf=vf, xf=xf, cf=cf, fits=fits)
        g.cells = [list(zip([lab for lab, _, _ in spec], self._spread(width, w), w, [k for _, _, k in spec]))
                   for (_, spec), w in zip(specs, widths)]
        g.label_base = cap_h(lf)
        g.value_base = g.label_base + round(0.36 * vf.size) + cap_h(vf)
        g.bar_y = g.value_base + round(0.44 * vf.size)
        g.bar_h = 8
        g.caption_base = g.bar_y + g.bar_h + 16 + cap_h(cf)
        g.height = g.caption_base + round(0.3 * cf.size)
        g.pill_y = None
        if pill is not None:
            g.pill_y = int(round(g.caption_base - cap_h(cf) / 2 - pill.height / 2))
            g.height = max(g.height, g.pill_y + pill.height)
        return g

    def _stats_stack(self, width, cols, label_px, value_px, caption_px, pill, gap_y=30):
        """The same stats stacked vertically (a side panel)."""
        specs = [(c.run, self._cells(c.run)) for c in cols]
        s = 1.0
        while True:  # the widest value must fit the panel
            lf = FONTS.get("sans", label_px * min(1.0, s + 0.2), "medium")
            vf = FONTS.get("mono", value_px * s, "regular")
            xf = FONTS.get("sans", value_px * s * 0.46, "medium")
            fits = all(self._value_w(vf, xf, v, k, r) <= width for r, spec in specs for _, v, k in spec)
            if fits or s <= 0.5:
                break
            s = round(s - 0.05, 3)
        cf = FONTS.get("mono", caption_px, "regular")
        g = SimpleNamespace(kind="stack", width=width, lf=lf, vf=vf, xf=xf, cf=cf, fits=fits)
        y, cells = 0, []
        for lab, _, kind in specs[0][1]:
            lb = y + cap_h(lf)
            vb = lb + round(0.36 * vf.size) + cap_h(vf)
            cells.append((0, kind, lb, vb))
            y = vb + round(0.22 * vf.size) + gap_y
        g.cells = [[(lab,) + cell for (lab, _, _), cell in zip(spec, cells)] for _, spec in specs]
        g.bar_y = y + 4
        g.bar_h = 8
        g.caption_base = g.bar_y + g.bar_h + 16 + cap_h(cf)
        g.height = g.caption_base + round(0.3 * cf.size)
        g.pill_y = None
        if pill is not None:
            g.pill_y = g.height + gap_y
            g.height = g.pill_y + pill.height
        return g

    def _card(self, width, cols, head_px, label_px, value_px, grid=False):
        """The final-hold plate: optional serif headline, then the final stats (row or 2x2 grid)."""
        pad = round((head_px or value_px * 0.6) * 0.8)
        inner = width - 2 * pad
        specs = [(c.run, self._final_cells(c.run)) for c in cols]
        if grid:
            lf, vf, xf, _, fits = self._fit_cells(inner / 2 - 14, [(r, [cell]) for r, spec in specs
                                                                  for cell in spec], label_px, value_px)
        else:
            lf, vf, xf, _, fits = self._fit_cells(inner, specs, label_px, value_px, min_gap=24, min_scale=0.7)
            if not fits:
                return self._card(width, cols, head_px, label_px, value_px, grid=True)
        g = SimpleNamespace(width=width, pad=pad, lf=lf, vf=vf, xf=xf, grid=grid, head=bool(head_px), fits=fits)
        top = pad
        if head_px:
            g.hf = FONTS.get("serif", head_px, "regular")
            g.head_base = pad + cap_h(g.hf)
            top = g.head_base + round(0.3 * head_px) + 20
        row_h = cap_h(lf) + round(0.36 * vf.size) + cap_h(vf)
        row_gap = round(0.3 * vf.size) + 22
        g.cells = []
        for r, spec in specs:
            placed = []
            if grid:
                for i, (lab, val, kind) in enumerate(spec):
                    lb = top + (i // 2) * (row_h + row_gap) + cap_h(lf)
                    placed.append((lab, val, kind, pad + (i % 2) * (inner // 2 + 7), lb, lb + row_h - cap_h(lf)))
            else:
                w = [max(text_width(lf, lab, TR_LABEL * lf.size), self._value_w(vf, xf, val, kind, r))
                     for lab, val, kind in spec]
                for (lab, val, kind), x in zip(spec, self._spread(inner, w)):
                    lb = top + cap_h(lf)
                    placed.append((lab, val, kind, pad + x, lb, lb + row_h - cap_h(lf)))
            g.cells.append(placed)
        rows = max(((len(spec) + 1) // 2 if grid else 1) for _, spec in specs)
        g.height = top + rows * row_h + (rows - 1) * row_gap + round(0.25 * vf.size) + pad
        return g

    def _label_geom(self, cols, lab_px, sub_px, status_line=False):
        """Label / sub (/ status line) above a run's frame or at the top of its panel."""
        lab_f, sub_f = FONTS.get("serif", lab_px), FONTS.get("sans", sub_px)
        g = SimpleNamespace(height=0, label_off=0, sub_off=0, status_cy=0, lab_px=lab_px, sub_px=sub_px,
                            lab_f=lab_f, sub_f=sub_f)
        has_label, has_sub = any(c.label for c in cols), any(c.sub for c in cols)
        y, tail = 0, 0
        if has_label:
            y = g.label_off = cap_h(lab_f)
            tail = round(0.3 * lab_px)
        if has_sub:
            y = g.sub_off = y + (tail + 12 if y else 0) + cap_h(sub_f)
            tail = round(0.3 * sub_px)
        if status_line:
            top = y + tail + 16 if y else 0
            g.status_cy = top + 18
            y, tail = top + 36, 0
        if y:
            g.height = y + tail + 22
        return g

    def _place_frame(self, c, fx0, fy0, D, T):
        c.D, c.T, c.fw = D, T, D + 2 * T
        c.fx0, c.fy0 = int(fx0), int(fy0)
        c.cx0, c.cy0 = c.fx0 + T, c.fy0 + T

    # ---- layouts -------------------------------------------------------------

    def _header_single(self, M, top, title_px, line2_px):
        W = self.W
        self.h_title = fit_text("serif", title_px, "regular", self.title, W - 2 * M, 30)
        self.h_title_base = top + cap_h(self.h_title[0])
        self.h_line2 = fit_text("sans", line2_px, "regular", self.line2, W - 2 * M - self.status_w - 36, 17)
        self.h_line2_base = self.h_title_base + round(title_px * 0.3) + 16 + cap_h(self.h_line2[0])
        self.h_status = (W - M, self.h_line2_base - cap_h(self.status_f) / 2)
        self.rule = (M, self.h_line2_base + round(line2_px * 0.3) + 24, W - M)
        self.M = M
        self.header_bottom = self.rule[1] + 2
        return self.header_bottom

    def _header_race(self, M, top, title_px):
        self.h_title = fit_text("serif", title_px, "regular", self.title, self.W - 2 * M, 30)
        self.h_title_base = top + cap_h(self.h_title[0])
        self.pill_xy = ((self.W - self.speed_pill.width) // 2, self.h_title_base + 24)
        self.header_bottom = self.pill_xy[1] + self.speed_pill.height
        return self.header_bottom

    def _layout_single_stacked(self):
        """square / portrait: header, framed canvas, stats row underneath."""
        W, H = self.W, self.H
        tall = H > W
        M = 60
        c = self.cols[0]
        body_top = self._header_single(M, 56 if tall else 54, 54 if tall else 50, 26 if tall else 24) \
            + (44 if tall else 34)
        body_bot = H - (60 if tall else 54)
        cw = W - 2 * M
        lr = self._label_geom([c], 36 if tall else 34, 23 if tall else 22)
        sg = self._stats_row(cw, [c], label_px=21 if tall else 20, value_px=58 if tall else 52,
                             caption_px=22 if tall else 21, pill=self.speed_pill)
        # most generous spacing first; take the first that still reaches the biggest canvas
        # (so a label row never costs the painting its integer scale if tighter spacing avoids it)
        head_px, card_px = (32, 52) if tall else (30, 46)
        options = []
        for gap, head, value_px in ((48 if tall else 36, True, card_px), (36 if tall else 30, True, card_px - 6),
                                    (30, False, card_px - 6), (24, False, card_px - 10)):
            cg = self._card(cw, [c], head_px=head_px if head else None, label_px=20 if tall else 19,
                            value_px=value_px)
            reserve = max(sg.height, cg.height)
            D = fit_canvas([c.run.size], min(body_bot - body_top - lr.height - gap - reserve, cw), self.fill)
            options.append((D, gap, cg, reserve))
        best = max(o[0] for o in options)
        D, gap, cg, reserve = next(o for o in options if o[0] == best)
        T = frame_thickness(D)
        group = lr.height + D + 2 * T + gap + reserve
        gy = body_top + max(0, (body_bot - body_top - group) // 2)
        self._place_frame(c, (W - D - 2 * T) // 2, gy + lr.height, D, T)
        c.lr, c.lx, c.ly, c.lw = lr, c.fx0, gy, c.fw
        c.sg, c.sx, c.sy = sg, M, c.fy0 + c.fw + gap
        c.cg, c.kx, c.ky = cg, M, c.sy

    def _layout_single_side(self):
        """landscape: header, framed canvas on the left, stats panel on the right."""
        W, H = self.W, self.H
        M = 80
        c = self.cols[0]
        body_top = self._header_single(M, 56, 56, 26) + 40
        body_bot = H - 60
        PW, GX = 540, 100
        lr = self._label_geom([c], 38, 24)
        sg = self._stats_stack(PW, [c], label_px=21, value_px=66, caption_px=21, pill=self.speed_pill)
        cg = self._card(PW, [c], head_px=34, label_px=20, value_px=56, grid=True)
        D = fit_canvas([c.run.size], min(body_bot - body_top, W - 2 * M - PW - GX), self.fill)
        T = frame_thickness(D)
        fw = D + 2 * T
        gx = (W - (fw + GX + PW)) // 2
        fy0 = body_top + (body_bot - body_top - fw) // 2
        self._place_frame(c, gx, fy0, D, T)
        px = gx + fw + GX
        panel_h = lr.height + max(sg.height, cg.height)
        py = min(max(body_top, fy0 + fw // 2 - (lr.height + sg.height) // 2), body_bot - panel_h)
        c.lr, c.lx, c.ly, c.lw = lr, px, py, PW
        c.sg, c.sx, c.sy = sg, px, py + lr.height
        c.cg, c.kx, c.ky = cg, px, py + lr.height

    def _layout_race_cols(self):
        """landscape / square race: two columns, each label row + frame + stats row."""
        W, H = self.W, self.H
        land = self.land
        M, G = (64, 110) if land else (40, 36)
        body_top = self._header_race(M, 46, 48) + (30 if land else 40)
        body_bot = H - 44
        col_max = (W - 2 * M - G) // 2
        lr = self._label_geom(self.cols, 34 if land else 30, 22 if land else 20)
        gap = 28
        outer = col_max
        card_px = 40 if land else 36
        for _ in range(12):  # shrink the card's numbers, then the canvas, until everything fits
            D = fit_canvas([c.run.size for c in self.cols], outer, self.fill)
            T = frame_thickness(D)
            fw = D + 2 * T
            sg = self._stats_row(fw, self.cols, label_px=18, value_px=44 if land else 40, caption_px=19,
                                 pill=None)
            for px in range(card_px, int(card_px * 0.75) - 1, -2):
                cg = self._card(fw, self.cols, head_px=None, label_px=17, value_px=px, grid=True)
                group = lr.height + fw + gap + max(sg.height, cg.height)
                over = group - (body_bot - body_top)
                if over <= 0:
                    break
            if over <= 0 or D <= 64:
                break
            outer = fw - over
        gy = body_top + max(0, (body_bot - body_top - group) // 2)
        x_left = (W - (2 * fw + G)) // 2
        for i, c in enumerate(self.cols):
            self._place_frame(c, x_left + i * (fw + G), gy + lr.height, D, T)
            c.lr, c.lx, c.ly, c.lw, c.status_mode = lr, c.fx0, gy, fw, "right"
            c.sg, c.sx, c.sy = sg, c.fx0, c.fy0 + fw + gap
            c.cg, c.kx, c.ky = cg, c.fx0, c.sy
        self.col_centres = [c.fx0 + fw // 2 for c in self.cols]

    def _layout_race_rows(self):
        """portrait race: two rows, each framed canvas + a panel (label, status, stats) beside it."""
        W, H = self.W, self.H
        M, GX, GY = 44, 40, 30
        body_top = self._header_race(M, 46, 46) + 26
        body_bot = H - 40
        row_h = (body_bot - body_top - GY) // 2
        lr = self._label_geom(self.cols, 32, 21, status_line=True)
        outer = min(row_h, W - 2 * M - GX - 340)
        for _ in range(12):
            D = fit_canvas([c.run.size for c in self.cols], outer, self.fill)
            T = frame_thickness(D)
            fw = D + 2 * T
            PW = W - 2 * M - GX - fw
            sg = self._stats_stack(PW, self.cols, label_px=17, value_px=42, caption_px=18, pill=None, gap_y=20)
            for px in range(38, 27, -2):
                cg = self._card(PW, self.cols, head_px=None, label_px=17, value_px=px, grid=True)
                panel_h = lr.height + max(sg.height, cg.height)
                over = max(fw, panel_h) - row_h
                if over <= 0:
                    break
            if over <= 0 or D <= 64:
                break
            outer = fw - over
        x0 = (W - (fw + GX + PW)) // 2
        for i, c in enumerate(self.cols):
            ry = body_top + i * (row_h + GY)
            fy0 = ry + (row_h - fw) // 2
            self._place_frame(c, x0, fy0, D, T)
            py = min(max(ry, fy0 + (fw - panel_h) // 2), ry + row_h - panel_h)
            px = x0 + fw + GX
            c.lr, c.lx, c.ly, c.lw, c.status_mode = lr, px, py, PW, "line"
            c.sg, c.sx, c.sy = sg, px, py + lr.height
            c.cg, c.kx, c.ky = cg, px, py + lr.height
        self.col_centres = [W // 4, 3 * W // 4]

    def _check_bounds(self):
        rects = []
        for c in self.cols:
            rects += [(f"run {c.i + 1} frame", (c.fx0, c.fy0, c.fx0 + c.fw, c.fy0 + c.fw)),
                      (f"run {c.i + 1} stats", (c.sx, c.sy, c.sx + c.sg.width, c.sy + c.sg.height)),
                      (f"run {c.i + 1} final card", (c.kx, c.ky, c.kx + c.cg.width, c.ky + c.cg.height))]
            if c.lr.height:
                rects.append((f"run {c.i + 1} label", (c.lx, c.ly, c.lx + c.lw, c.ly + c.lr.height)))
        for name, (x0, y0, x1, y1) in rects:
            if x0 < 0 or y0 < self.header_bottom or x1 > self.W or y1 > self.H:
                print(f"warning: {name} does not fit below the header in the {self.fmt} frame "
                      f"({x0},{y0})-({x1},{y1})", file=sys.stderr)
        for c in self.cols:
            if not c.sg.fits or not c.cg.fits:
                print(f"warning: run {c.i + 1}'s numbers are too wide for their space at the smallest "
                      f"type size; try another --format", file=sys.stderr)
        live = [r for r in rects if "final" not in r[0]]
        final = [r for r in rects if "stats" not in r[0]]
        for group in (live, final):
            for i, (na, a) in enumerate(group):
                for nb, b in group[i + 1:]:
                    if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                        print(f"warning: {na} overlaps {nb}", file=sys.stderr)

    # ---- static layers ---------------------------------------------------------

    def _base(self, mode):
        """Static layer. mode: "paint" (stat labels, bar track), "final" (finish plates), or
        "bare" (neither: the midpoint of the fade from the last painting frame to the final one)."""
        final = mode == "final"
        img = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(img)
        f, t = self.h_title
        if self.race:
            draw_text(d, self.W / 2, self.h_title_base, t, f, INK, align="m")
            img.paste(self.speed_pill, self.pill_xy)
        else:
            draw_text(d, self.M, self.h_title_base, t, f, INK)
            f, t = self.h_line2
            draw_text(d, self.M, self.h_line2_base, t, f, INK3)
            x0, y, x1 = self.rule
            d.rectangle([x0, y, x1 - 1, y + 1], fill=LINE)
            if final:
                r = self.cols[0].run
                self._indicator(img, d, *self.h_status, "FINISHED" if r.finished else "STOPPED",
                                GOOD if r.finished else INK3, align="r")
        for c in self.cols:
            self._label_row(d, c)
            draw_frame(img, c.fx0, c.fy0, c.fw, c.T, c.run.canvas_rgb)
            if final:
                if self.race:
                    self._race_status(img, d, c, self._final_pill(c))
                self._draw_card(img, d, c)
            elif mode == "bare":
                if self.race:
                    self._race_status(img, d, c, self.done_pill if c.run.finished else self.stopped_pill)
            else:
                self._draw_stats_static(img, d, c)
        return img

    def _label_sizes(self):
        """One label size and one sub size for every column (the largest that fits them all)."""
        if not hasattr(self, "_lab_sizes"):
            lab, sub = [], []
            for c in self.cols:
                room = c.lw - (self.status_w + 18 if c.status_mode == "right" else 0)
                if c.label:
                    lab.append(fit_text("serif", c.lr.lab_px, "regular", c.label, room, c.lr.lab_px * 0.66)[0].size)
                if c.sub:
                    sub.append(fit_text("sans", c.lr.sub_px, "regular", c.sub, c.lw, 17)[0].size)
            self._lab_sizes = (min(lab, default=0), min(sub, default=0))
        return self._lab_sizes

    def _label_row(self, d, c):
        lr = c.lr
        if not lr.height:
            return
        lab_px, sub_px = self._label_sizes()
        room = c.lw - (self.status_w + 18 if c.status_mode == "right" else 0)
        if c.label:
            f, t = fit_text("serif", lab_px, "regular", c.label, room, lab_px)
            draw_text(d, c.lx, c.ly + lr.label_off, t, f, INK)
        if c.sub:
            f, t = fit_text("sans", sub_px, "regular", c.sub, c.lw, sub_px)
            draw_text(d, c.lx, c.ly + lr.sub_off, t, f, INK3)

    def _indicator(self, img, d, x, cy, text, dot_colour, align="r", vt=None):
        """dot + tracked caps, e.g. the breathing "PAINTING" status; x is its right or left edge."""
        f = self.status_f
        tr = TR_STATUS * f.size
        width = 2 * self.dot_r + 14 + text_width(f, text, tr)
        x0 = x - width if align == "r" else x
        draw_text(d, x0 + 2 * self.dot_r + 14, cy + cap_h(f) / 2, text, f, INK2, track=tr)
        dx = x0 + self.dot_r
        if vt is None:
            circle(img, dx, cy, self.dot_r, dot_colour)
        else:
            sp = self.pulse_sprite(vt)
            img.paste(sp, (int(round(dx - sp.width / 2)), int(round(cy - sp.height / 2))), sp)

    def _race_status(self, img, d, c, pill, vt=None):
        """A race column's status: breathing PAINTING, or a DONE / FINISHED FIRST badge."""
        if c.status_mode == "right":
            x = c.lx + c.lw
            cy = c.ly + c.lr.label_off - cap_h(c.lr.lab_f) / 2 if c.lr.height else c.fy0 - 30
            if pill is None:
                self._indicator(img, d, x, cy, "PAINTING", ACCENT, align="r", vt=vt)
            else:
                img.paste(pill, (int(x - pill.width), int(round(cy - pill.height / 2))))
        else:
            cy = c.ly + c.lr.status_cy
            if pill is None:
                self._indicator(img, d, c.lx, cy, "PAINTING", ACCENT, align="l", vt=vt)
            else:
                img.paste(pill, (int(c.lx), int(round(cy - pill.height / 2))))

    def _bar_sprites(self, c):
        w, h = c.sg.width, c.sg.bar_h
        full = Image.new("RGB", (w, h), BG)
        rrect(full, (0, 0, w, h), h / 2, fill=ACCENT)
        capw = int(math.ceil(h / 2)) + 1
        cap = Image.new("RGB", (capw * 2, h), TRACK)
        rrect(cap, (0, 0, capw * 2, h), h / 2, fill=ACCENT)
        c.bar_full, c.bar_cap = full, cap.crop((capw, 0, capw * 2, h))

    def _draw_stats_static(self, img, d, c):
        g, x0, y0 = c.sg, c.sx, c.sy
        lf = g.lf
        for cell in g.cells[c.i if len(g.cells) > 1 else 0]:
            lab, x = cell[0], cell[1]
            lb = g.label_base if g.kind == "row" else cell[3]
            draw_text(d, x0 + x, y0 + lb, lab, lf, INK3, track=TR_LABEL * lf.size)
        rrect(img, (x0, y0 + g.bar_y, x0 + g.width, y0 + g.bar_y + g.bar_h), g.bar_h / 2, fill=TRACK)
        self._bar_sprites(c)
        if g.pill_y is not None:
            px = x0 + g.width - self.speed_pill.width if g.kind == "row" else x0
            img.paste(self.speed_pill, (int(px), int(y0 + g.pill_y)))

    def _value(self, d, x, base, text, kind, run, vf, xf, colour):
        w = draw_text(d, x, base, text, vf, colour)
        if kind == "cost" and run.free:
            draw_text(d, x + w, base, " · free", xf, GOOD)

    def _draw_card(self, img, d, c):
        g, x0, y0 = c.cg, c.kx, c.ky
        r = c.run
        rrect(img, (x0, y0, x0 + g.width, y0 + g.height), 8, fill=ACCENT_BG, outline=ACCENT_LINE, width=2)
        if g.head:
            head = "The painting is finished" if r.finished else "The run stopped before finishing"
            f, head = fit_text("serif", g.hf.size, "regular", head, g.width - 2 * g.pad, 18)
            draw_text(d, x0 + g.pad, y0 + g.head_base, head, f, INK)
        for lab, val, kind, x, lb, vb in g.cells[c.i if len(g.cells) > 1 else 0]:
            draw_text(d, x0 + x, y0 + lb, lab, g.lf, INK3, track=TR_LABEL * g.lf.size)
            self._value(d, x0 + x, y0 + vb, val, kind, r, g.vf, g.xf, ACCENT if kind == "cost" else INK)

    # ---- frames ------------------------------------------------------------------

    def paint_frame(self, frame_idx, clock, vt):
        img = self.base_paint.copy()
        d = ImageDraw.Draw(img)
        for c, pl, j in zip(self.cols, self.players, frame_idx):
            pl.seek(int(j))
            img.paste(pl.image(c.D), (c.cx0, c.cy0))
            self._draw_live(img, d, c, pl, clock)
            if self.race:
                ended = self.done_pill if c.run.finished else self.stopped_pill
                self._race_status(img, d, c, ended if pl.done else None, vt=vt)
        if not self.race:
            self._indicator(img, d, *self.h_status, "PAINTING", ACCENT, align="r", vt=vt)
        return img

    def _draw_live(self, img, d, c, pl, clock):
        g, x0, y0, r = c.sg, c.sx, c.sy, c.run
        dec, cost, elapsed = pl.stats(clock)
        cells = g.cells[c.i if len(g.cells) > 1 else 0]
        for cell, text in zip(cells, (fmt_num(dec), fmt_money(cost), fmt_clock(elapsed))):
            x, kind = cell[1], (cell[3] if g.kind == "row" else cell[2])
            vb = g.value_base if g.kind == "row" else cell[4]
            self._value(d, x0 + x, y0 + vb, text, kind, r, g.vf, g.xf, ACCENT if kind == "cost" else INK)
        frac = pl.covered / r.total if r.total else 0.0
        w = int(round(g.width * frac))
        by = y0 + g.bar_y
        if w >= g.width:
            img.paste(c.bar_full, (x0, by))
        elif w > c.bar_cap.width:
            img.paste(c.bar_full.crop((0, 0, w - c.bar_cap.width, g.bar_h)), (x0, by))
            img.paste(c.bar_cap, (x0 + w - c.bar_cap.width, by))
        elif w > 0:
            img.paste(c.bar_full.crop((0, 0, w, g.bar_h)), (x0, by))
        cb = y0 + g.caption_base
        cw = draw_text(d, x0, cb, f"{pl.covered:,} / {r.total:,} px", g.cf, INK2)
        if g.kind == "row" and not self.race:
            draw_text(d, x0 + cw, cb, " · " + fmt_pct(frac), g.cf, INK3)
        else:
            draw_text(d, x0 + g.width, cb, fmt_pct(frac), g.cf, INK3, align="r")

    def final_frame(self, mode="final"):
        img = self._base(mode)
        for c, pl in zip(self.cols, self.players):
            pl.seek(c.run.n)
            img.paste(pl.image(c.D), (c.cx0, c.cy0))
        return img

    def intro_frame(self):
        W, H = self.W, self.H
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        max_w = int(W * (0.70 if self.land else 0.84))
        size = 76 if self.land else 68
        while True:
            tf = FONTS.get("serif", size)
            lines = wrap_balanced(tf, self.title, max_w)
            if len(lines) <= 3 or size <= 36:
                break
            size -= 4
        lh = round(size * 1.16)
        blocks = []   # (height, draw(y_top), gap above)
        th = cap_h(tf) + (len(lines) - 1) * lh + round(size * 0.24)

        def title_block(y):
            for i, line in enumerate(lines):
                draw_text(d, W / 2, y + cap_h(tf) + i * lh, line, tf, INK, align="m")
        blocks.append((th, title_block, 0))
        blocks.append((3, lambda y: d.rectangle([W // 2 - 36, y, W // 2 + 35, y + 2], fill=ACCENT), 34))

        if self.race:
            lab_px, sub_px = (40, 24) if self.land else (34, 21)
            vs_f = FONTS.get("serif-italic", round(lab_px * 0.8))
            centres = self.col_centres
            room = centres[1] - centres[0] - text_width(vs_f, "vs") - 56
            lab_px = min(fit_text("serif", lab_px, "regular", c.label, room, 22)[0].size for c in self.cols)
            sub_px = min((fit_text("sans", sub_px, "regular", c.sub, room, 17)[0].size for c in self.cols if c.sub),
                         default=sub_px)
            fitted = [(fit_text("serif", lab_px, "regular", c.label, room, lab_px),
                       fit_text("sans", sub_px, "regular", c.sub, room, sub_px) if c.sub else None)
                      for c in self.cols]
            lab_cap = max(cap_h(f) for (f, _), _ in fitted)
            subs = [s for _, s in fitted if s]
            sub_step = round(lab_px * 0.3) + 16 + max((cap_h(s[0]) for s in subs), default=0)
            bh = lab_cap + (sub_step if subs else 0) + round(sub_px * 0.3)

            def labels_block(y):
                base = y + lab_cap
                for cx, ((f, t), sub) in zip(centres, fitted):
                    draw_text(d, cx, base, t, f, INK, align="m")
                    if sub:
                        draw_text(d, cx, base + sub_step, sub[1], sub[0], INK3, align="m")
                draw_text(d, W / 2, base, "vs", vs_f, INK3, align="m")
            blocks.append((bh, labels_block, 40))
        else:
            c = self.cols[0]
            if c.label:
                lf, lt = fit_text("serif", 40, "regular", c.label, max_w, 24)
                sf, st = fit_text("sans", 26, "regular", c.sub or self.line2, max_w, 17)
                step = round(40 * 0.3) + 16 + cap_h(sf)

                def label_block(y):
                    draw_text(d, W / 2, y + cap_h(lf), lt, lf, INK2, align="m")
                    draw_text(d, W / 2, y + cap_h(lf) + step, st, sf, INK3, align="m")
                blocks.append((cap_h(lf) + step + 8, label_block, 40))
            else:
                sf, st = fit_text("sans", 28, "regular", self.line2, max_w, 17)
                blocks.append((cap_h(sf) + 8, lambda y: draw_text(d, W / 2, y + cap_h(sf), st, sf, INK2,
                                                                  align="m"), 40))
        pal = self.runs[0].palette
        if pal and all(r.palette == pal for r in self.runs):
            chip, gapc = 26, 8
            n = min(len(pal), (max_w + gapc) // (chip + gapc))
            row_w = n * chip + (n - 1) * gapc

            def chips(y):
                x = (W - row_w) // 2
                for rgb in pal[:n]:
                    rrect(img, (x, y, x + chip, y + chip), 4, fill=rgb, outline=mix((0, 0, 0), rgb, 0.45),
                          width=1)
                    x += chip + gapc
            blocks.append((chip, chips, 46))
        total = sum(h + g for h, _, g in blocks)
        y = (H - total) // 2 - 16
        for h, fn, g in blocks:
            y += g
            fn(y)
            y += h
        nf, nt = fit_text("sans", 22, "regular", self.note, W - 120, 16)
        draw_text(d, W / 2, H - 66, nt, nf, INK3, align="m")
        return img


# ----------------------------------------------------------------------------- encode

class Encoder:
    def __init__(self, out, W, H, fps, seconds):
        ff = shutil.which("ffmpeg") or FFMPEG_FALLBACK
        if not os.path.exists(ff):
            raise SystemExit("ffmpeg not found (looked on PATH and at /opt/homebrew/bin/ffmpeg)")
        self.out = Path(out)
        self.tmp = self.out.with_name(f".{self.out.stem}.rendering.mp4")
        self.err = tempfile.TemporaryFile()
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "pipe:0",
               "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
               "-map", "0:v:0", "-map", "1:a:0", "-t", f"{seconds:.6f}",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd,format=yuv420p,"
                      "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
               "-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", "-g", str(fps * 2), "-fps_mode", "cfr", "-r", str(fps),
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
               "-movflags", "+faststart", "-f", "mp4", str(self.tmp)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self.err)

    def write(self, data):
        try:
            self.p.stdin.write(data)
        except (BrokenPipeError, OSError):
            self._fail()

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        if self.p.wait() != 0:
            self._fail()
        os.replace(self.tmp, self.out)

    def abort(self):
        """Stop ffmpeg and remove the partial file (on an error or Ctrl-C)."""
        try:
            self.p.kill()
            self.p.wait()
        except OSError:
            pass
        try:
            self.tmp.unlink()
        except OSError:
            pass

    def _fail(self):
        self.p.kill()
        self.err.seek(0)
        msg = self.err.read().decode(errors="replace").strip()
        try:
            self.tmp.unlink()
        except OSError:
            pass
        raise SystemExit(f"ffmpeg failed:\n{msg[-4000:]}")


def fade_frames(a_img, mid_img, b_img, n):
    """n frames fading a -> mid -> b (smoothstep each half), as raw RGB bytes. Going through a
    midpoint keeps two different layouts from being overlaid on top of each other."""
    A, M, B = (np.asarray(im, dtype=np.float32) for im in (a_img, mid_img, b_img))
    for i in range(n):
        t = (i + 1) / (n + 1) * 2
        src, dst, t = (A, M, t) if t < 1 else (M, B, t - 1)
        t = t * t * (3 - 2 * t)
        yield (src + (dst - src) * t + 0.5).astype(np.uint8).tobytes()


# ----------------------------------------------------------------------------- main

def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="render_video.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Turn a painting run's replay log into an MP4 for X / LinkedIn (plus OUT_poster.png).",
        epilog="examples:" + __doc__.split("Examples", 1)[1].split("Writes", 1)[0].rstrip())
    ap.add_argument("out", metavar="OUT.mp4")
    ap.add_argument("runs", nargs="+", metavar="RUN_DIR", help="one run, or two for a side-by-side race")
    ap.add_argument("--label", help='name of run 1, e.g. "Jev · cloud API"')
    ap.add_argument("--label2", help='name of run 2, e.g. "Laya · on my laptop"')
    ap.add_argument("--sub", help="small line under label 1")
    ap.add_argument("--sub2", help="small line under label 2")
    ap.add_argument("--title", help="title for the intro card and header (default: the run's title)")
    ap.add_argument("--format", choices=sorted(FORMATS),
                    help="square 1080x1080 | landscape 1920x1080 | portrait 1080x1350 "
                         "(default: square for one run, landscape for two)")
    ap.add_argument("--duration", type=float, default=20.0,
                    help="target seconds for the painting phase (default 20); runs shorter than this play at 1x")
    ap.add_argument("--intro", type=float, default=2.0, help="title card seconds (default 2)")
    ap.add_argument("--hold", type=float, default=4.0, help="final frame hold with final stats, seconds (default 4)")
    ap.add_argument("--fps", type=int, default=30, help="frames per second (default 30)")
    ap.add_argument("--fill", action="store_true",
                    help="let the canvas fill its space at a non-integer NEAREST scale (default: integer "
                         "scales only, so every painted pixel stays an even, crisp square)")
    args = ap.parse_args(argv)
    if len(args.runs) > 2:
        ap.error("give one run directory, or two for a race")
    if args.duration <= 0 or args.intro < 0 or args.hold < 0 or not 1 <= args.fps <= 120:
        ap.error("--duration must be > 0, --intro/--hold >= 0, --fps between 1 and 120")
    if len(args.runs) == 1 and (args.label2 or args.sub2):
        print("warning: --label2/--sub2 ignored with a single run", file=sys.stderr)
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    runs = [Run(p) for p in args.runs]
    for r in runs:
        if not r.n:
            print(f"warning: {r.dir} has no replay events", file=sys.stderr)
    fps = args.fps
    t_max = max(r.t_end for r in runs)
    speed = nice_speed(t_max, args.duration)
    scene = Scene(runs, args, speed)

    intro_n = int(round(args.intro * fps))
    paint_n = int(math.ceil(t_max / speed * fps - 1e-9)) + 1
    hold_n = int(round(args.hold * fps))
    total_n = intro_n + paint_n + hold_n
    seconds = total_n / fps
    clocks = np.minimum(np.arange(paint_n) * (speed / fps), t_max)
    idx = np.stack([np.searchsorted(r.ts, clocks, side="right") for r in runs], axis=1)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    poster = out.with_name(out.stem + "_poster.png")
    W, H = scene.W, scene.H
    speed_s = "real time" if speed == 1 else f"{fmt_speed(speed)}x"
    for r in runs:
        print(f"{r.name}: {r.size}x{r.size}, {r.n:,} events, {r.t_end:.2f}s of painting", file=sys.stderr)
    print(f"{scene.fmt} {W}x{H} @ {fps}fps: intro {intro_n / fps:.2f}s + painting {paint_n / fps:.2f}s "
          f"at {speed_s} + hold {hold_n / fps:.2f}s = {seconds:.2f}s", file=sys.stderr)
    if seconds > 140:
        print("warning: longer than 2:20, X's limit for most accounts", file=sys.stderr)
    elif seconds > 60:
        print("warning: longer than 60 s", file=sys.stderr)

    enc = Encoder(out, W, H, fps, seconds)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # clean up on kill, like Ctrl-C
    try:
        final = render(scene, enc, idx, clocks, fps, intro_n, paint_n, hold_n, total_n, args, started)
    except BaseException:
        enc.abort()
        raise
    enc.close()
    final.save(poster)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({mb:.1f} MB, {seconds:.2f}s) and {poster.name} in {time.time() - started:.1f}s",
          file=sys.stderr)


def render(scene, enc, idx, clocks, fps, intro_n, paint_n, hold_n, total_n, args, started):
    """Emit every frame to the encoder; returns the final frame (the poster)."""
    W, H = scene.W, scene.H
    written, last_report = 0, 0.0

    def emit(data):
        nonlocal written, last_report
        enc.write(data)
        written += 1
        if written / total_n - last_report >= 0.1 or written == total_n:
            last_report = written / total_n
            print(f"  {written}/{total_n} frames ({time.time() - started:.1f}s)", file=sys.stderr)

    first = scene.paint_frame(idx[0], clocks[0], vt=intro_n / fps)
    if intro_n:
        intro = scene.intro_frame()
        fade_n = min(intro_n, int(round(min(0.5, args.intro * 0.3) * fps)))
        data = intro.tobytes()
        for _ in range(intro_n - fade_n):
            emit(data)
        for data in fade_frames(intro, Image.new("RGB", (W, H), BG), first, fade_n):
            emit(data)
    frame = first
    for k in range(paint_n):
        if k:
            frame = scene.paint_frame(idx[k], clocks[k], vt=(intro_n + k) / fps)
        emit(frame.tobytes())
    final = scene.final_frame()
    if hold_n:
        fade_n = min(hold_n - 1, int(round(min(0.6, args.hold * 0.25) * fps)))
        for data in fade_frames(frame, scene.final_frame("bare"), final, max(0, fade_n)):
            emit(data)
        data = final.tobytes()
        for _ in range(hold_n - max(0, fade_n)):
            emit(data)
    return final


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("interrupted; nothing written")
