#!/usr/bin/env python3
"""Jev Paints: every pixel's colour is decided by TypeSafe's Jev decisions model.

Each pixel is either three `score` decisions (amount of red, green, blue) or one
`choice` among the hex codes of a 16-colour palette, batched into requests of up
to --max-pixels pixels. Raw answers are cached per chunk, so an interrupted run
resumes without re-paying and images can be re-decoded offline.

  python3 jev_paint.py --scene quadrants --size 32 --assert-accuracy 0.98
  python3 jev_paint.py --scene hi --size 64 --layout bands --mode palette
  python3 jev_paint.py --brief briefs/hi_pure.txt --size 256
"""
import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

URL = os.environ.get("JEV_API_URL", "https://openrouter.ai/api/alpha/decisions")  # override to point at a stub for offline testing
MODEL = "typesafe/jev-1.13"
ROOT = os.path.dirname(os.path.abspath(__file__))

ANCHORS = (0, 64, 128, 192, 255)
PRICE_PER_TOKEN = 0.042e-6
TOKENS_PER_REQUEST = 370  # measured with probe_api.py, excluding the state itself

MAX_ATTEMPTS = 6
UNPAINTED = (128, 128, 128)
TOLERANCE = 32  # half an anchor step; a channel within this of the truth counts as right

# The 16 basic web colours: the hex codes a palette-mode pixel chooses between.
PALETTE = {
    "black": "#000000", "white": "#FFFFFF", "red": "#FF0000", "lime": "#00FF00",
    "blue": "#0000FF", "yellow": "#FFFF00", "cyan": "#00FFFF", "magenta": "#FF00FF",
    "silver": "#C0C0C0", "gray": "#808080", "maroon": "#800000", "olive": "#808000",
    "green": "#008000", "purple": "#800080", "teal": "#008080", "navy": "#000080",
}


def hex_to_rgb(code):
    return tuple(int(code[i:i + 2], 16) for i in (1, 3, 5))


# Scenes are rectangles on a 256 grid (inclusive ranges), so the same scene yields
# both the brief Jev reads and the ground truth its painting is scored against.
SCENES = {
    "quadrants": {
        "background": "black",
        "shapes": [
            ("Top-left quadrant", (0, 127), (0, 127), "red"),
            ("Top-right quadrant", (0, 127), (128, 255), "lime"),
            ("Bottom-left quadrant", (128, 255), (0, 127), "blue"),
            ("Bottom-right quadrant", (128, 255), (128, 255), "white"),
        ],
    },
    "hi": {
        "background": "white",
        "shapes": [
            ("Left bar of the letter H", (64, 191), (40, 71), "red"),
            ("Right bar of the letter H", (64, 191), (104, 135), "red"),
            ("Crossbar of the letter H", (112, 143), (72, 103), "red"),
            ("Top serif of the letter I", (64, 87), (160, 223), "red"),
            ("Stem of the letter I", (88, 167), (176, 207), "red"),
            ("Bottom serif of the letter I", (168, 191), (160, 223), "red"),
            ("Underline beneath the word", (208, 223), (40, 223), "blue"),
        ],
    },
}


class FatalAPIError(Exception):
    """The API rejected us in a way retrying will not fix."""


class ChannelMode:
    """Three score decisions per pixel: the amount of each colour channel."""

    questions_per_pixel = 3
    tokens_per_question = 66
    max_pixels = 256  # 768 questions per request, verified with probe_api.py

    def __init__(self, abstract):
        # Abstract drops the colour words, which Jev otherwise reads as "is this pixel red?"
        self.abstract = abstract
        self.channels = ("R", "G", "B") if abstract else ("red", "green", "blue")

    def intro(self):
        if self.abstract:
            return ("Every pixel's colour is three channel values, R, G and B, each from 0 to 255. "
                    "Each question names one pixel and one channel; answer with that channel's "
                    "value for that pixel in the painting described below.")
        return ("Each question names one pixel and one colour channel; answer with how much of "
                "that colour the pixel needs so the finished image matches the painting described "
                "below. 0 means none of that colour, 255 means full intensity.")

    def describe(self, colour):
        values = ", ".join(f"{ch}={v}" for ch, v in zip(self.channels, hex_to_rgb(PALETTE[colour])))
        return values if self.abstract else f"pure {colour} ({values})"

    def questions(self, row, col):
        criteria = [str(a) for a in ANCHORS]
        for ch in self.channels:
            text = (f"Value of the {ch} channel for the pixel at row {row}, column {col}." if self.abstract
                    else f"Amount of {ch} in the pixel at row {row}, column {col}.")
            yield f"p{row}_{col}_{ch[0].lower()}", {"type": "score", "instructions": text, "criteria": criteria}

    def decode(self, answers, row, col, decode):
        decoded = [self.decode_channel(answers.get(f"p{row}_{col}_{ch[0].lower()}"), decode) for ch in self.channels]
        if any(value is None for value, _ in decoded):
            return None, 0.0
        return tuple(v for v, _ in decoded), sum(c for _, c in decoded) / len(decoded)

    @staticmethod
    def decode_channel(answer, decode):
        if not isinstance(answer, dict):
            return None, 0.0
        probs = answer.get("probabilities") or {}
        weights = [float(probs.get(str(i), 0)) for i in range(len(ANCHORS))]
        total = sum(weights)
        if total > 0:
            if decode == "argmax":
                value = ANCHORS[max(range(len(ANCHORS)), key=weights.__getitem__)]
            else:
                value = sum(w * a for w, a in zip(weights, ANCHORS)) / total
        elif isinstance(answer.get("score"), (int, float)):
            value = answer["score"] / (len(ANCHORS) - 1) * 255
        else:
            return None, 0.0
        return max(0, min(255, round(value))), float(answer.get("confidence", 0))


class PaletteMode:
    """One choice decision per pixel: which hex code of the palette it is."""

    questions_per_pixel = 1
    max_pixels = 128  # each answer carries a distribution over the palette; 256 per request overflows Jev's output limit

    def __init__(self, palette=PALETTE, row_paints=None):
        self.palette = palette
        self.codes = set(palette.values())
        # Repeating the whole palette for every pixel is ~90% of a painting's cost. row_paints
        # offers each pixel only the paints present in its row (a list of names per row).
        self.row_paints = row_paints
        options = len(palette) if row_paints is None else sum(map(len, row_paints)) / len(row_paints)
        self.tokens_per_question = 38 + 18.6 * options  # measured: 355 with 17 two-word paints

    def intro(self):
        return ("Each question names one pixel; answer with the hex code of the colour that pixel "
                "has in the painting described below.")

    def describe(self, colour):
        return f"{colour} ({self.palette[colour]})"

    def questions(self, row, col):
        yield f"p{row}_{col}", {
            "type": "choice",
            "instructions": f"Colour of the pixel at row {row}, column {col}.",
            "criteria": {self.palette[name]: name
                         for name in (self.palette if self.row_paints is None else self.row_paints[row])},
        }

    def decode(self, answers, row, col, decode):
        answer = answers.get(f"p{row}_{col}")
        if not isinstance(answer, dict):
            return None, 0.0
        probs = {k: float(v) for k, v in (answer.get("probabilities") or {}).items() if k in self.codes}
        total = sum(probs.values())
        if decode == "mean" and total > 0:
            mix = [sum(p * hex_to_rgb(code)[i] for code, p in probs.items()) / total for i in range(3)]
            rgb = tuple(round(v) for v in mix)
        elif answer.get("choice") in self.codes:
            rgb = hex_to_rgb(answer["choice"])
        else:
            return None, 0.0
        return rgb, float(answer.get("confidence", 0))


MODES = {"named": lambda: ChannelMode(False), "abstract": lambda: ChannelMode(True), "palette": PaletteMode}


def load_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("OPENROUTER_API_KEY not found in env or .env")


def scaled_shapes(scene, size):
    shapes = []
    for label, (r0, r1), (c0, c1), colour in scene["shapes"]:
        span = lambda lo, hi: (lo * size // 256, (hi + 1) * size // 256 - 1)
        shapes.append((label, span(r0, r1), span(c0, c1), colour))
    return shapes


def scene_colours(scene, size):
    """Colour name and whether a shape covers it, per pixel in raster order. Later shapes win."""
    grid = [(scene["background"], False)] * (size * size)
    for _, (r0, r1), (c0, c1), colour in scaled_shapes(scene, size):
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                grid[row * size + col] = (colour, True)
    return grid


def rects_brief(scene, size, mode):
    """Shapes listed over a background: compact, but background pixels need a 'not inside anything' inference."""
    lines = [
        f"The background is {mode.describe(scene['background'])}.",
        "The painting is made of solid, axis-aligned rectangles drawn on the background. "
        "Row and column ranges are inclusive. Any pixel that is not inside a rectangle "
        "shows the background.",
    ]
    for label, (r0, r1), (c0, c1), colour in scaled_shapes(scene, size):
        lines.append(f"- {label}: rows {r0}-{r1}, columns {c0}-{c1}, {mode.describe(colour)}.")
    return "\n".join(lines)


def bands_brief(scene, size, mode):
    """The scene as a partition: every pixel sits in exactly one row band and one column span."""
    grid = scene_colours(scene, size)
    bands = []
    for row in range(size):
        spans = []
        for col in range(size):
            colour = grid[row * size + col][0]
            if spans and spans[-1][2] == colour:
                spans[-1][1] = col
            else:
                spans.append([col, col, colour])
        if bands and bands[-1][2] == spans:
            bands[-1][1] = row
        else:
            bands.append([row, row, spans])
    lines = [
        "The painting is described band by band. Each band is a range of rows, and inside a band "
        "each range of columns has one colour. Ranges are inclusive, and every pixel belongs to "
        "exactly one band and one column range."
    ]
    for r0, r1, spans in bands:
        parts = "; ".join(f"columns {c0}-{c1} are {mode.describe(colour)}" for c0, c1, colour in spans)
        lines.append(f"- Rows {r0}-{r1}: {parts}.")
    return "\n".join(lines)


LAYOUTS = {"rects": rects_brief, "bands": bands_brief}


def build_state(size, brief, mode):
    return (
        f"You are painting a {size}x{size} pixel image, one pixel at a time. Rows are "
        f"numbered 0 to {size - 1} from top to bottom. Columns are numbered 0 to "
        f"{size - 1} from left to right. {mode.intro()}\n\nTHE PAINTING:\n" + brief.strip()
    )


class Painter:
    def __init__(self, run_dir, state, size, mode, max_pixels, workers, max_cost):
        self.run_dir = run_dir
        # One state for every request, or a function of the chunk index when each request
        # only needs the part of the painting its own pixels fall in.
        self.state_for = state if callable(state) else (lambda index: state)
        self.size = size
        self.mode = mode
        self.workers = workers
        self.max_cost = max_cost
        pixels = [(row, col) for row in range(size) for col in range(size)]
        self.chunks = [pixels[i:i + max_pixels] for i in range(0, len(pixels), max_pixels)]
        self.answers = {}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "requests": 0}
        self.cached_chunks = 0
        self.failed_chunks = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.local = threading.local()
        self.key = load_key()
        os.makedirs(os.path.join(run_dir, "chunks"), exist_ok=True)

    def estimate(self):
        todo = [i for i in range(len(self.chunks)) if not os.path.exists(self.chunk_path(i))]
        questions = sum(len(self.chunks[i]) for i in todo) * self.mode.questions_per_pixel
        tokens = (sum(TOKENS_PER_REQUEST + len(self.state_for(i)) // 4 for i in todo)
                  + questions * self.mode.tokens_per_question)
        return len(todo), questions, tokens * PRICE_PER_TOKEN

    def chunk_path(self, index):
        return os.path.join(self.run_dir, "chunks", f"chunk_{index:05d}.json")

    def session(self):
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def fetch_chunk(self, index):
        questions = {key: q for row, col in self.chunks[index] for key, q in self.mode.questions(row, col)}
        return self.ask(f"chunk_{index:05d}", self.state_for(index), questions)

    def ask(self, name, state, questions):
        """Answers to one request: from cache if present, else from Jev (then cached)."""
        path = os.path.join(self.run_dir, "chunks", f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)["answers"], True
        payload = json.dumps({"model": MODEL, "state": state, "questions": questions})
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "Jev Paints",
        }
        best, err = {}, "not attempted"
        for attempt in range(MAX_ATTEMPTS):
            if self.stop.is_set():
                return best, False
            try:
                r = self.session().post(URL, headers=headers, data=payload, timeout=180)
            except requests.RequestException as e:
                err = repr(e)
            else:
                if r.status_code == 200:
                    body = r.json()
                    answers = body.get("answers") or {}
                    self.record_usage(body.get("usage") or {})
                    if len(answers) > len(best):
                        best = answers
                    if all(k in answers for k in questions):
                        tmp = path + ".tmp"
                        with open(tmp, "w") as f:
                            json.dump({"answers": answers, "usage": body.get("usage")}, f)
                        os.replace(tmp, path)
                        return answers, False
                    err = f"only {len(answers)}/{len(questions)} answers"
                elif r.status_code == 429 or r.status_code >= 500:
                    err = f"HTTP {r.status_code}: {r.text[:200]}"
                else:
                    raise FatalAPIError(f"HTTP {r.status_code}: {r.text[:500]}")
            time.sleep(min(60, 2 ** attempt) + random.random())
        # Incomplete requests are used for this render but never cached, so a re-run retries them.
        with self.lock:
            self.failed_chunks.append((name, err))
        return best, False

    def record_usage(self, usage):
        with self.lock:
            self.usage["requests"] += 1
            for field in ("input_tokens", "output_tokens", "cost"):
                self.usage[field] += usage.get(field) or 0
            if self.usage["cost"] >= self.max_cost:
                self.stop.set()

    def paint(self, on_progress):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.fetch_chunk, i): i for i in range(len(self.chunks))}
            try:
                for done, future in enumerate(as_completed(futures), 1):
                    answers, cached = future.result()
                    self.answers.update(answers)
                    self.cached_chunks += cached
                    on_progress(done, len(self.chunks), time.time() - t0, futures[future], answers)
            except FatalAPIError:
                self.stop.set()
                for future in futures:
                    future.cancel()
                raise
        return time.time() - t0

    def pixels(self, decode):
        """Decoded (rgb, confidence) per pixel in raster order; rgb is None if unpainted."""
        return [self.mode.decode(self.answers, row, col, decode)
                for row in range(self.size) for col in range(self.size)]


def save_images(run_dir, size, pixels, truth, suffix):
    painting = Image.new("RGB", (size, size))
    painting.putdata([rgb or UNPAINTED for rgb, _ in pixels])
    painting.save(os.path.join(run_dir, f"painting{suffix}.png"))
    scale = max(1, 768 // size)
    if scale > 1:
        painting.resize((size * scale, size * scale), Image.NEAREST).save(
            os.path.join(run_dir, f"painting{suffix}_x{scale}.png"))
    confidence = Image.new("L", (size, size))
    confidence.putdata([round(conf * 255) for _, conf in pixels])
    confidence.save(os.path.join(run_dir, "confidence.png"))
    if truth:
        expected = Image.new("RGB", (size, size))
        expected.putdata([rgb for rgb, _ in truth])
        expected.save(os.path.join(run_dir, "truth.png"))
        diff = Image.new("RGB", (size, size))
        diff.putdata([(255, 255, 255) if is_right(p, t) else (255, 0, 0) for (p, _), (t, _) in zip(pixels, truth)])
        diff.save(os.path.join(run_dir, f"diff{suffix}.png"))


def is_right(painted, expected):
    return painted is not None and all(abs(p - e) <= TOLERANCE for p, e in zip(painted, expected))


def score_against(pixels, truth):
    """Pixel accuracy overall, on shape pixels, and on background pixels."""
    tally = {"overall": [0, 0], "shapes": [0, 0], "background": [0, 0]}
    for (painted, _), (expected, in_shape) in zip(pixels, truth):
        right = is_right(painted, expected)
        for group in ("overall", "shapes" if in_shape else "background"):
            tally[group][0] += right
            tally[group][1] += 1
    return {group: (right / total if total else None) for group, (right, total) in tally.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene", choices=sorted(SCENES), help="built-in rectangle scene with ground truth")
    source.add_argument("--brief", help="path to a free-text description of the painting")
    ap.add_argument("--size", type=int, default=256, help="canvas is size x size pixels")
    ap.add_argument("--mode", choices=sorted(MODES), default="named",
                    help="named/abstract = three channel scores per pixel; palette = one hex-code choice per pixel")
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="rects", help="how a --scene is put into words")
    ap.add_argument("--name", help="run name (default: scene or brief stem, plus layout and mode)")
    ap.add_argument("--max-pixels", type=int, help="pixels per request (default: what the mode can fit)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cost", type=float, default=2.0, help="hard cap in USD for this run")
    ap.add_argument("--decode", choices=("mean", "argmax"), default="mean",
                    help="mean = Jev's expected colour (soft where unsure); argmax = its single likeliest answer")
    ap.add_argument("--assert-accuracy", type=float, help="exit 1 if overall accuracy vs ground truth is below this")
    args = ap.parse_args()

    mode = MODES[args.mode]()
    if args.scene:
        scene = SCENES[args.scene]
        if 256 % args.size:
            ap.error("--scene needs a size that divides 256 so rectangle edges stay exact")
        brief = LAYOUTS[args.layout](scene, args.size, mode)
        truth = [(hex_to_rgb(PALETTE[colour]), in_shape) for colour, in_shape in scene_colours(scene, args.size)]
        name = args.name or f"{args.scene}-{args.layout}-{args.mode}"
    else:
        with open(args.brief) as f:
            brief = f.read()
        truth = None
        name = args.name or f"{os.path.splitext(os.path.basename(args.brief))[0]}-{args.mode}"
    if args.assert_accuracy is not None and not truth:
        ap.error("--assert-accuracy needs a --scene (free-text briefs have no ground truth)")

    # The fingerprint keys the cache to exactly what Jev is asked and how it is chunked, so an
    # edited brief or a new chunk size lands in a fresh directory instead of reusing stale answers.
    state = build_state(args.size, brief, mode)
    max_pixels = args.max_pixels or mode.max_pixels
    fingerprint = hashlib.sha256(
        json.dumps([MODEL, state, args.size, args.mode, ANCHORS, max_pixels]).encode()).hexdigest()[:8]
    run_dir = os.path.join(ROOT, "runs", f"{name}-{args.size}-{fingerprint}")
    painter = Painter(run_dir, state, args.size, mode, max_pixels, args.workers, args.max_cost)

    todo, questions, estimate = painter.estimate()
    print(f"run: {run_dir}")
    print(f"canvas {args.size}x{args.size} = {args.size ** 2} pixels; {len(painter.chunks)} chunks "
          f"({len(painter.chunks) - todo} cached), {questions} decisions to make, est. ${estimate:.4f}")
    if estimate > args.max_cost:
        raise SystemExit(f"estimated cost ${estimate:.2f} exceeds --max-cost ${args.max_cost:.2f}; nothing sent")

    suffix = "" if args.decode == "mean" else f"_{args.decode}"
    step = max(1, len(painter.chunks) // 10)

    def on_progress(done, total, elapsed, index, answers):
        if done % step == 0 or done == total:
            print(f"  [{done:4d}/{total}] chunks  ${painter.usage['cost']:.4f}  {elapsed:6.1f}s", flush=True)
        if done % (step * 2) == 0 and done != total:
            save_images(run_dir, args.size, painter.pixels(args.decode), truth, suffix)

    try:
        elapsed = painter.paint(on_progress)
    except FatalAPIError as e:
        raise SystemExit(f"aborted: {e}")

    pixels = painter.pixels(args.decode)
    save_images(run_dir, args.size, pixels, truth, suffix)
    unpainted = sum(rgb is None for rgb, _ in pixels)
    accuracy = score_against(pixels, truth) if truth else None
    summary = {
        "name": name, "model": MODEL, "size": args.size, "mode": args.mode, "decode": args.decode,
        "layout": args.layout if args.scene else None, "brief": brief, "state": state,
        "fingerprint": fingerprint, "chunks": len(painter.chunks), "cached_chunks": painter.cached_chunks,
        "failed_chunks": painter.failed_chunks, "unpainted_pixels": unpainted,
        "usage": painter.usage, "seconds": round(elapsed, 2), "accuracy": accuracy,
    }
    with open(os.path.join(run_dir, f"run{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"done in {elapsed:.1f}s: {painter.usage['requests']} requests, "
          f"{painter.usage['input_tokens']} input tokens, ${painter.usage['cost']:.4f} spent this run")
    if painter.stop.is_set():
        print(f"STOPPED EARLY: spend reached --max-cost ${args.max_cost:.2f}")
    if unpainted:
        print(f"WARNING: {unpainted} pixels unpainted (shown mid-grey); failed chunks: {painter.failed_chunks[:5]}")
    if accuracy:
        print("accuracy vs ground truth: " + ", ".join(
            f"{group} {value:.2%}" for group, value in accuracy.items() if value is not None))
    print(f"painting: {os.path.join(run_dir, f'painting{suffix}.png')}")
    if args.assert_accuracy is not None and accuracy["overall"] < args.assert_accuracy:
        sys.exit(1)


if __name__ == "__main__":
    main()
