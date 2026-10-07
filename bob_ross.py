#!/usr/bin/env python3
"""Jev paints a Bob Ross landscape, live.

Stage 1: Jev makes the composition decisions (time of day, where the mountain stands, ...).
Stage 2: Jev paints, choosing among the hex codes of the paints on the palette. Either every
pixel is its own decision, or Jev works in blocks: big blocks first, split only where colours meet.
The run directory gets an index.html showing the canvas fill in, with the running cost.

  python3 bob_ross.py --preview                          # render the intended scene only, no decisions
  python3 bob_ross.py --size 64                          # quick look at the whole pipeline, a few cents
  python3 bob_ross.py --size 512 --open                  # one decision per pixel; opens the live viewer first
  python3 bob_ross.py --size 512 --method blocks --open  # coarse to fine, about 5x cheaper
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

from jev_paint import (MODEL, PRICE_PER_TOKEN, ROOT, TOKENS_PER_REQUEST, URL, FatalAPIError, Painter,
                       PaletteMode, build_state, hex_to_rgb, load_key)

CANVAS = "#F4EFE4"
MAX_BLOCK_QUESTIONS = 256  # per request; answers carry a distribution, so more overflows Jev's output limit
HORIZON = 0.60
TITLE = "Jev paints a Bob Ross landscape"

# Bob's thirteen oils, then the mixes that come off the palette knife.
PAINTS = {
    "Titanium White": "#FFFFFF",
    "Midnight Black": "#0A0A0F",
    "Prussian Blue": "#12284C",
    "Phthalo Blue": "#0F4C81",
    "Phthalo Green": "#123F2D",
    "Sap Green": "#4F7A28",
    "Van Dyke Brown": "#4A2C17",
    "Dark Sienna": "#3C1414",
    "Alizarin Crimson": "#9E1B32",
    "Bright Red": "#D92121",
    "Cadmium Yellow": "#FFD300",
    "Indian Yellow": "#F0A30A",
    "Yellow Ochre": "#C8962C",
    "Deep Sky": "#2F6DB5",
    "Sky Blue": "#5B9BD5",
    "Pale Sky": "#A9CDEB",
    "Horizon Glow": "#DDEBF5",
    "Dusk Violet": "#4B3F72",
    "Sunset Pink": "#E58C9A",
    "Sunset Orange": "#F28C38",
    "Twilight Blue": "#3E6E9E",
    "Mist": "#B5C7D6",
    "Mountain Grey": "#6F8296",
    "Mountain Shadow": "#2B3A4A",
    "Distant Blue": "#7E9BB5",
    "Lake Blue": "#3A78A8",
    "Deep Water": "#1F4A6E",
    "Spring Green": "#9DBF3B",
}

THEMES = {
    "day": {
        "sky": ["Deep Sky", "Sky Blue", "Pale Sky", "Horizon Glow"], "orb": "Cadmium Yellow", "glow": "Horizon Glow",
        "cloud": "Titanium White", "cloud_under": "Mist", "far": "Distant Blue", "snow": "Titanium White",
        "snow_shadow": "Mist", "mist": "Mist", "shore": "Phthalo Green", "lake": "Lake Blue",
        "reflection": "Deep Water", "reflection_snow": "Pale Sky", "waterline": "Horizon Glow",
    },
    "sunset": {
        "sky": ["Dusk Violet", "Sunset Pink", "Sunset Orange", "Indian Yellow"], "orb": "Cadmium Yellow", "glow": "Indian Yellow",
        "cloud": "Dusk Violet", "cloud_under": "Sunset Pink", "far": "Twilight Blue", "snow": "Sunset Pink",
        "snow_shadow": "Dusk Violet", "mist": "Sunset Pink", "shore": "Midnight Black", "lake": "Dusk Violet",
        "reflection": "Prussian Blue", "reflection_snow": "Sunset Pink", "waterline": "Sunset Orange",
    },
    "dusk": {
        "sky": ["Midnight Black", "Prussian Blue", "Phthalo Blue", "Twilight Blue"], "orb": "Titanium White", "glow": "Mist",
        "cloud": "Mountain Shadow", "cloud_under": "Twilight Blue", "far": "Prussian Blue", "snow": "Mist",
        "snow_shadow": "Twilight Blue", "mist": "Twilight Blue", "shore": "Midnight Black", "lake": "Prussian Blue",
        "reflection": "Midnight Black", "reflection_snow": "Twilight Blue", "waterline": "Mist",
    },
}

COMPOSER_STATE = (
    "You are Bob Ross, about to paint a calm landscape on a small square canvas for The Joy of "
    "Painting: a mountain, a lake, and some happy little trees. Before the first brush stroke you "
    "decide the composition. Make each decision the way Bob would, for a painting viewers will love."
)
CHOICES = {
    "time": ("What time of day is it in the painting?", {
        "day": "A bright clear day with a blue sky",
        "sunset": "A warm sunset with a pink and orange sky",
        "dusk": "Deep twilight, a dark blue sky with a moon"}),
    "mountain": ("Where does the big mountain stand?", {
        "left": "Left of centre", "centre": "In the centre", "right": "Right of centre"}),
    "clouds": ("How many clouds float in the sky?", {
        "none": "A clear sky with no clouds",
        "few": "Two or three happy little clouds",
        "many": "A sky full of fluffy clouds"}),
    "trees": ("Where do the big foreground evergreens stand?", {
        "left": "On the left bank", "right": "On the right bank", "both": "On both banks, framing the lake"}),
    "tree_count": ("How many big evergreens are there?", {
        "few": "Two or three", "many": "A whole family of them"}),
}
NOULS = {
    "companion": "The big mountain has a smaller mountain beside it, because everybody needs a friend.",
    "cabin": "There is a little cabin by the lake.",
    "orb": "The sun, or at dusk the moon, is visible in the sky.",
}

SIGNATURE = {
    "J": ["..#", "..#", "..#", "#.#", ".#."],
    "E": ["###", "#..", "##.", "#..", "###"],
    "V": ["#.#", "#.#", "#.#", "#.#", ".#."],
}


def composer_state(prompt):
    """The composer's brief. A viewer's prompt (from the Studio) steers the choices Jev makes."""
    if not prompt:
        return COMPOSER_STATE
    return (COMPOSER_STATE + "\n\nA viewer has asked for this painting:\n\"" + prompt.strip() + "\"\n"
            "Make every decision so the finished painting matches their request as closely as the choices allow.")


def compose(fresh, prompt=None):
    """Jev's composition decisions, asked once and kept so every size paints the same scene.
    With a prompt, the decisions are cached per prompt instead."""
    if prompt:
        digest = hashlib.sha256(prompt.strip().encode()).hexdigest()[:12]
        path = os.path.join(ROOT, "runs", "compositions", f"{digest}.json")
    else:
        path = os.path.join(ROOT, "runs", "bobross_composition.json")
    if os.path.exists(path) and not fresh:
        with open(path) as f:
            return json.load(f)
    questions = {key: {"type": "choice", "instructions": question, "criteria": options}
                 for key, (question, options) in CHOICES.items()}
    for key, statement in NOULS.items():
        questions[key] = {"type": "noul", "instructions": statement,
                          "criteria": {"true": "Yes, it belongs in this painting", "false": "No, leave it out"}}
    r = requests.post(
        URL,
        headers={"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json",
                 "X-OpenRouter-Title": "Jev Paints"},
        data=json.dumps({"model": MODEL, "state": composer_state(prompt), "questions": questions}),
        timeout=60,
    )
    if r.status_code != 200:
        raise SystemExit(f"composition request failed: HTTP {r.status_code}: {r.text[:500]}")
    body = r.json()
    answers = body["answers"]
    values, decisions = {}, []
    for key, (question, options) in CHOICES.items():
        answer = answers[key]
        choice = answer["choice"] if answer.get("choice") in options else next(iter(options))
        confidence = (answer.get("probabilities") or {}).get(choice, answer.get("confidence", 0))
        values[key] = choice
        decisions.append({"question": question, "answer": options[choice], "confidence": confidence})
    for key, statement in NOULS.items():
        p = float(answers[key]["noul"])
        values[key] = p >= 0.5
        decisions.append({"question": statement, "answer": "yes" if p >= 0.5 else "no",
                          "confidence": p if p >= 0.5 else 1 - p})
    composition = {"values": values, "decisions": decisions, "usage": body.get("usage") or {}, "prompt": prompt}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(composition, f, indent=2)
    return composition


def tri(t):
    """Triangle wave between 0 and 1 with period 1: the jagged edge of snow and treetops."""
    t %= 1.0
    return 2 * t if t < 0.5 else 2 - 2 * t


def rect(x0, y0, x1, y1):
    return (lambda x, y: x0 <= x <= x1 and y0 <= y <= y1), (x0, y0, x1, y1)


def ellipse(cx, cy, rx, ry):
    return (lambda x, y: ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1), (cx - rx, cy - ry, cx + rx, cy + ry)


def polygon(points):
    xs, ys = zip(*points)

    def contains(x, y):
        inside = False
        xj, yj = points[-1]
        for xi, yi in points:
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            xj, yj = xi, yi
        return inside

    return contains, (min(xs), min(ys), max(xs), max(ys))


def mountain(ax, ay, half, lit):
    """A craggy peak: (body, lit side, snowline) with `lit` = +1 when the light comes from the right."""
    height = HORIZON - ay
    at = lambda dx, dy: (ax + dx * half * lit, ay + dy * height)
    shadow_ridge = [at(-1.0, 1.0), at(-0.75, 0.70), at(-0.56, 0.58), at(-0.31, 0.30), at(-0.14, 0.19), at(0, 0)]
    lit_ridge = [at(0.11, 0.12), at(0.33, 0.37), at(0.50, 0.47), at(0.72, 0.72), at(1.0, 1.0)]
    spine = [at(0, 0), at(0.06, 0.23), at(-0.05, 0.40), at(0.11, 0.60), at(0.0, 0.81), at(0.14, 1.0)]
    body = polygon(shadow_ridge + lit_ridge)
    lit_side = polygon(spine + lit_ridge[::-1])
    snowline = lambda x: ay + height * (0.40 + 0.11 * tri(x * 9.0))
    return body, lit_side, snowline


def build_scene(c):
    """Layers back to front as (paint, contains, bbox), in unit coordinates with y pointing down."""
    theme = THEMES[c["time"]]
    rng = random.Random(1942)
    layers = []
    add = lambda paint, shape: layers.append((paint, shape[0], shape[1]))
    lit = -1 if c["mountain"] == "right" else 1  # the light comes from the side the mountain leaves open

    # Sky: four bands with wandering edges, darkest at the top.
    add(theme["sky"][3], rect(0, 0, 1, HORIZON + 0.02))
    for k, edge in ((2, 0.45), (1, 0.31), (0, 0.16)):
        add(theme["sky"][k], ((lambda x, y, e=edge, p=k * 1.7: y < e + 0.014 * math.sin(x * 7 + p)), (0, 0, 1, edge + 0.02)))

    if c["orb"]:
        ox = 0.5 + 0.28 * lit
        oy = 0.40 if c["time"] == "sunset" else 0.15
        add(theme["glow"], ellipse(ox, oy, 0.085, 0.085))
        add(theme["orb"], ellipse(ox, oy, 0.05, 0.05))

    count = {"none": 0, "few": 3, "many": 6}[c["clouds"]]
    for i in range(count):
        cx = (i + 0.5) / count + rng.uniform(-0.06, 0.06)
        cy, width = rng.uniform(0.07, 0.27), rng.uniform(0.09, 0.15)
        puff = width * 0.30
        add(theme["cloud_under"], ellipse(cx, cy + puff * 0.55, width * 1.05, puff * 0.7))
        for dx, scale in ((-0.55, 0.75), (0.0, 1.0), (0.5, 0.8), (0.95, 0.5)):
            add(theme["cloud"], ellipse(cx + dx * width * 0.7, cy - puff * (scale - 0.6), width * 0.5 * scale, puff * scale))

    hills = [(0, HORIZON + 0.01)] + [(x / 8, HORIZON - rng.uniform(0.035, 0.085)) for x in range(9)] + [(1, HORIZON + 0.01)]
    add(theme["far"], polygon(hills))

    ax = {"left": 0.32, "centre": 0.5, "right": 0.68}[c["mountain"]]
    peaks = [(ax, 0.17, 0.36)]
    if c["companion"]:
        peaks.insert(0, (ax + (0.35 if c["mountain"] == "left" else -0.35), 0.32, 0.25))
    snowy = []
    for px, py, half in peaks:
        (body, bbox), (lit_side, lit_bbox), snowline = mountain(px, py, half, lit)
        add("Mountain Shadow", (body, bbox))
        add("Mountain Grey", (lit_side, lit_bbox))
        add(theme["snow_shadow"], ((lambda x, y, b=body, s=snowline: y < s(x) and b(x, y)), bbox))
        add(theme["snow"], ((lambda x, y, b=lit_side, s=snowline: y < s(x) and b(x, y)), lit_bbox))
        add(theme["mist"], ((lambda x, y, b=body: y > HORIZON - 0.055 + 0.015 * math.sin(x * 17) and b(x, y)), bbox))
        snowy.append((body, snowline))

    # The far shore: a line of tiny evergreens, then the lake with everything mirrored in it.
    treetops = lambda x: 0.030 + 0.028 * tri(x * 16 + 0.3 * math.sin(x * 50))
    shore = HORIZON + 0.012
    add(theme["shore"], ((lambda x, y: HORIZON - treetops(x) < y <= shore), (0, HORIZON - 0.06, 1, shore)))
    add(theme["lake"], rect(0, shore, 1, 1))
    mirror = lambda y: HORIZON - (y - shore) * 1.5
    for body, snowline in snowy:
        add(theme["reflection"], ((lambda x, y, b=body: b(x, mirror(y))), (0, shore, 1, 0.9)))
        add(theme["reflection_snow"], ((lambda x, y, b=body, s=snowline: mirror(y) < s(x) and b(x, mirror(y))), (0, shore, 1, 0.9)))
    add(theme["shore"], ((lambda x, y: shore < y < shore + 0.6 * treetops(x)), (0, shore, 1, shore + 0.04)))
    for y in (0.685, 0.73, 0.775, 0.82):
        x0 = rng.uniform(0.05, 0.55)
        add(theme["waterline"], rect(x0, y, x0 + rng.uniform(0.15, 0.4), y + 0.007))

    # The near bank rises toward the side the big trees stand on.
    def land_top(x):
        lift = {"left": 1 - x, "right": x, "both": abs(2 * x - 1)}[c["trees"]]
        return 0.885 - 0.13 * lift + 0.010 * math.sin(x * 19)

    add("Van Dyke Brown", ((lambda x, y: y > land_top(x)), (0, 0.74, 1, 1)))
    add("Dark Sienna", ((lambda x, y: y > land_top(x) + 0.05 and math.sin(x * 31) * math.sin(y * 47) > 0.35), (0, 0.78, 1, 1)))
    add("Sap Green", ((lambda x, y: land_top(x) < y < land_top(x) + 0.03), (0, 0.74, 1, 0.94)))
    add("Spring Green", ((lambda x, y: land_top(x) < y < land_top(x) + 0.012 and tri(x * 7.3) > 0.55), (0, 0.74, 1, 0.92)))

    if c["cabin"]:
        cx = {"left": 0.72, "right": 0.28, "both": 0.5}[c["trees"]]
        base = land_top(cx) + 0.025
        top, x0, x1 = base - 0.085, cx - 0.085, cx + 0.085
        add("Van Dyke Brown", rect(x0, top, x1, base))
        add("Dark Sienna", rect(x0, top, x0 + 0.054, base))
        add("Midnight Black", polygon([(x0 - 0.02, top + 0.004), (x0 + 0.035, top - 0.06), (x1 - 0.035, top - 0.06), (x1 + 0.02, top + 0.004)]))
        add("Dark Sienna", rect(x1 - 0.06, top - 0.085, x1 - 0.035, top - 0.05))
        add("Cadmium Yellow", rect(cx + 0.015, top + 0.022, cx + 0.06, top + 0.058))
        add("Midnight Black", rect(cx - 0.035, top + 0.03, cx - 0.005, base))

    sides = {"left": [1], "right": [-1], "both": [1, -1]}[c["trees"]]
    per_side = {"few": 2, "many": 3 if len(sides) == 2 else 4}[c["tree_count"]]
    for side in sides:
        place = lambda u: u if side == 1 else 1 - u
        spots = [0.08 + i * 0.30 / max(1, per_side - 1) for i in range(per_side)]
        heights = sorted((rng.uniform(0.40, 0.62) for _ in spots), reverse=True)
        for u, h in reversed(list(zip(spots, heights))):  # smallest and farthest in first
            tx = place(u)
            base = land_top(tx) + 0.035
            crown, half = base - h, 0.19 * h
            add("Van Dyke Brown", rect(tx - 0.008, crown + h * 0.8, tx + 0.008, base))
            tiers = 6
            for i in range(tiers):
                t0 = crown + h * 0.86 * (i / tiers) * 0.92
                t1 = crown + h * 0.86 * ((i + 1) / tiers) + h * 0.035
                w, th = half * (0.28 + 0.72 * (i + 1) / tiers), t1 - t0
                add("Phthalo Green", polygon([(tx, t0), (tx - w, t1), (tx + w, t1)]))
                add("Sap Green", polygon([(tx + lit * w * 0.25, t0 + 0.55 * th), (tx + lit * w * 0.95, t1 - 0.12 * th),
                                          (tx + lit * w * 0.30, t1 - 0.12 * th)]))
        for i in range(4):
            bx = place(0.04 + i * 0.105 + rng.uniform(-0.015, 0.015))
            rx, ry = rng.uniform(0.05, 0.075), rng.uniform(0.03, 0.045)
            by = land_top(bx) + 0.012
            add("Sap Green", ellipse(bx, by, rx, ry))
            add("Spring Green", ellipse(bx + lit * rx * 0.25, by - ry * 0.35, rx * 0.55, ry * 0.5))
            add("Cadmium Yellow", ellipse(bx + lit * rx * 0.35, by - ry * 0.55, rx * 0.2, ry * 0.18))

    # Signed in red, bottom left, like the man himself.
    dot = 3 / 256
    for n, letter in enumerate("JEV"):
        for r, line in enumerate(SIGNATURE[letter]):
            for k, mark in enumerate(line):
                if mark == "#":
                    x, y = 0.045 + (n * 4 + k) * dot, 0.925 + r * dot
                    add("Bright Red", rect(x, y, x + dot, y + dot))
    return layers


def rasterise(layers, size):
    """Paint name per pixel, sampled at pixel centres; later layers cover earlier ones."""
    grid = [[None] * size for _ in range(size)]
    for paint, contains, (x0, y0, x1, y1) in layers:
        c0, c1 = max(0, int(x0 * size)), min(size - 1, int(x1 * size))
        r0, r1 = max(0, int(y0 * size)), min(size - 1, int(y1 * size))
        for row in range(r0, r1 + 1):
            y, line = (row + 0.5) / size, grid[row]
            for col in range(c0, c1 + 1):
                if contains((col + 0.5) / size, y):
                    line[col] = paint
    assert all(paint for line in grid for paint in line), "scene leaves bare canvas"
    return grid


def row_spans(line):
    spans = []
    for col, paint in enumerate(line):
        if spans and spans[-1][2] == paint:
            spans[-1][1] = col
        else:
            spans.append([col, col, paint])
    return spans


def rows_brief(grid, rows, mode):
    """The rows a request's pixels fall in, each as a partition into column spans."""
    lines = [
        "The painting is described row by row. Inside a row, each range of columns has one colour. "
        "Ranges are inclusive, and every pixel belongs to exactly one column range."
    ]
    for row in rows:
        parts = "; ".join(f"columns {c0}-{c1} are {mode.describe(paint)}" for c0, c1, paint in row_spans(grid[row]))
        lines.append(f"- Row {row}: {parts}.")
    return "\n".join(lines)


def band_brief(grid, r0, b, mode):
    """Rows r0..r0+b-1 seen as one band: each column range is entirely one paint, or mixed.
    Returns the brief and the paints a block in this band could be."""
    labels = []
    for col in range(len(grid)):
        paints = {grid[row][col] for row in range(r0, r0 + b)}
        labels.append(paints.pop() if len(paints) == 1 else None)
    lines = [
        f"This request is about the band of rows {r0}-{r0 + b - 1}. Taking all of these rows together, "
        "each range of columns is either entirely one colour or mixed (more than one colour). Ranges "
        "are inclusive, and every column belongs to exactly one range."
    ]
    for c0, c1, paint in row_spans(labels):
        lines.append(f"- columns {c0}-{c1} are " + (f"entirely {mode.describe(paint)}." if paint else "mixed."))
    return "\n".join(lines), [paint for paint in dict.fromkeys(labels) if paint]


def block_state(size, brief):
    return (
        f"You are painting a {size}x{size} pixel image in blocks. Rows are numbered 0 to {size - 1} from "
        f"top to bottom. Columns are numbered 0 to {size - 1} from left to right. Each question names one "
        "block of columns inside the band described below. If the whole block lies inside a single range "
        "that is entirely one colour, answer that colour's hex code. If the block touches a mixed range, "
        "or crosses from one range into another, answer MIXED.\n\nTHE PAINTING:\n" + brief
    )


def children(blocks, b):
    half = b // 2
    return [(r + dr, c + dc) for r, c in blocks for dr in (0, half) for dc in (0, half)]


def plan_level(grid, size, mode, blocks, b):
    """Requests for one level of blocks, one band of rows per state. A band with no single-colour
    column cannot hold a uniform block, so its blocks go to the next level unasked."""
    by_band = {}
    for r, c in blocks:
        by_band.setdefault(r, []).append(c)
    jobs, forced = [], []
    for r, cols in sorted(by_band.items()):
        if b == 1:
            state = build_state(size, rows_brief(grid, [r], mode), mode)
            ask_for = lambda c, r=r: next(iter(mode.questions(r, c)))
        else:
            brief, paints = band_brief(grid, r, b, mode)
            if not paints:
                forced += [(r, c) for c in cols]
                continue
            state = block_state(size, brief)
            criteria = {mode.palette[paint]: paint for paint in paints}
            criteria["MIXED"] = "More than one colour in the block"
            ask_for = lambda c, r=r, criteria=criteria: (
                f"k{r}_{c}", {"type": "choice", "instructions": f"Block of columns {c}-{c + b - 1}.", "criteria": criteria})
        cols = sorted(cols)
        for k in range(0, len(cols), MAX_BLOCK_QUESTIONS):
            part = cols[k:k + MAX_BLOCK_QUESTIONS]
            jobs.append((f"b{b:02d}_r{r:04d}_{k // MAX_BLOCK_QUESTIONS:02d}", state, dict(ask_for(c) for c in part), r, part))
    return jobs, forced


def estimate_blocks(grid, size, mode, top):
    """Decisions, requests and cost if Jev judges every block correctly. A floor: each needless
    MIXED asks four more questions, and a wrong flat answer saves money but costs accuracy."""
    blocks, b = [(r, c) for r in range(0, size, top) for c in range(0, size, top)], top
    questions = requests = tokens = 0
    while blocks:
        jobs, mixed = plan_level(grid, size, mode, blocks, b)
        for _, state, asked, r, cols in jobs:
            requests += 1
            questions += len(asked)
            tokens += TOKENS_PER_REQUEST + len(state) // 4 + sum(38 + 18.6 * len(q["criteria"]) for q in asked.values())
            mixed += [(r, c) for c in cols
                      if len({grid[y][x] for y in range(r, r + b) for x in range(c, c + b)}) > 1]
        blocks = children(mixed, b) if b > 1 else []
        b //= 2
    return questions, requests, tokens * PRICE_PER_TOKEN


def paint_blocks(painter, grid, size, mode, top, workers, on_level, on_block, on_request):
    """Coarse to fine: fill a block when Jev names one paint for it, split it in four when Jev says MIXED."""
    t0 = time.time()
    blocks, b, asked = [(r, c) for r in range(0, size, top) for c in range(0, size, top)], top, 0
    while blocks and not painter.stop.is_set():
        jobs, mixed = plan_level(grid, size, mode, blocks, b)
        on_level(b, len(blocks), len(jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(painter.ask, name, state, questions): (questions, r, cols)
                       for name, state, questions, r, cols in jobs}
            try:
                for future in as_completed(futures):
                    answers, _ = future.result()
                    questions, r, cols = futures[future]
                    asked += len(questions)
                    elapsed = time.time() - t0
                    for key, c in zip(questions, cols):
                        answer = answers.get(key)
                        choice = answer.get("choice") if isinstance(answer, dict) else None
                        if choice in mode.codes:
                            on_block(r, c, b, hex_to_rgb(choice), elapsed, asked)
                        elif b > 1:
                            mixed.append((r, c))  # MIXED, or an unusable answer: look closer
                    on_request(elapsed, asked)
            except FatalAPIError:
                painter.stop.set()
                for future in futures:
                    future.cancel()
                raise
        blocks = children(mixed, b) if b > 1 else []
        b //= 2
    return time.time() - t0, asked


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class LiveCanvas:
    """The canvas as it fills in: live.png + live.js for the viewer, and the event log for replay."""

    def __init__(self, run_dir, size, static):
        self.run_dir = run_dir
        self.size = size
        self.static = static
        self.image = Image.new("RGB", (size, size), hex_to_rgb(CANVAS))
        self.events = []
        self.painted = 0
        self.version = 0
        self.last_flush = 0.0

    def event(self, start, colours, elapsed, usage, questions):
        self.events.append({"t": round(elapsed, 3), "cost": round(usage["cost"], 6), "requests": usage["requests"],
                            "questions": questions, "start": start, "colours": colours})

    def land(self, start, colours, elapsed, usage, questions):
        """A raster run of pixels, as the per-pixel method paints them."""
        for offset, rgb in enumerate(colours):
            if rgb is not None:
                self.image.putpixel(((start + offset) % self.size, (start + offset) // self.size), rgb)
                self.painted += 1
        self.event(start, "".join("%02X%02X%02X" % (rgb or hex_to_rgb(CANVAS)) for rgb in colours),
                   elapsed, usage, questions)

    def land_block(self, row, col, b, rgb, elapsed, usage, questions):
        """A b x b block of one paint; the replay log gets it as one run per row."""
        self.image.paste(rgb, (col, row, col + b, row + b))
        self.painted += b * b
        for y in range(row, row + b):
            self.event(y * self.size + col, ("%02X%02X%02X" % rgb) * b, elapsed, usage, questions)

    def flush(self, stats, force=False):
        if not force and time.time() - self.last_flush < 0.15:
            return
        self.last_flush = time.time()
        self.version += 1
        tmp = os.path.join(self.run_dir, "live.png.tmp")
        self.image.save(tmp, format="PNG")
        os.replace(tmp, os.path.join(self.run_dir, "live.png"))
        live = dict(self.static, version=self.version, pixels_painted=self.painted, **stats)
        write_atomic(os.path.join(self.run_dir, "live.js"), f"window.__live = {json.dumps(live)};\n")

    def save_replay(self):
        replay = {"size": self.size, "canvas": CANVAS, "events": sorted(self.events, key=lambda e: e["t"])}
        write_atomic(os.path.join(self.run_dir, "replay.js"), f"window.__replay = {json.dumps(replay)};\n")


def save_png(pixels, size, path, scale=1):
    image = Image.new("RGB", (size, size))
    image.putdata(pixels)
    if scale > 1:
        image = image.resize((size * scale, size * scale), Image.NEAREST)
    image.save(path)


LAYA_QUESTION_VERSION = 1  # bump when the per-pixel question changes, so old cached answers are not reused


def laya_question(grid, row, col, reach, row_paints):
    """What Laya is asked about one pixel: the colour spans within `reach` columns of it, and a menu of at
    least three paints (those in view, then the rest of the row's, then nearby rows'), so no answer is forced."""
    spans = [s for s in row_spans(grid[row]) if s[1] >= col - reach and s[0] <= col + reach]
    state = "; ".join(f"columns {a} to {b}: {p}" for a, b, p in spans) + "."
    menu = list(dict.fromkeys(p for _, _, p in spans))
    for r in [row] + [row + d * s for d in range(1, len(grid)) for s in (-1, 1)]:
        if len(menu) >= 3:
            break
        if 0 <= r < len(grid):
            menu += [p for p in row_paints[r] if p not in menu][:3 - len(menu)]
    return state, {"type": "choice", "instructions": f"Which paint covers column {col}?", "criteria": {p: None for p in menu}}


def compose_with_laya(laya):
    """The same composition questions Jev answered, asked of Laya. Its English checkpoint lets noul answers
    follow their option labels (model card, issue #156), so yes/no questions go as two-option choices."""
    questions = {key: {"type": "choice", "instructions": q, "criteria": opts} for key, (q, opts) in CHOICES.items()}
    for key, statement in NOULS.items():
        questions[key] = {"type": "choice", "instructions": statement,
                          "criteria": {"A": "Yes, it belongs in this painting", "B": "No, leave it out"}}
    (answers,), _ = laya.answer([(COMPOSER_STATE, questions)])
    values, decisions = {}, []
    for key, (question, options) in CHOICES.items():
        choice = answers[key]["choice"]
        values[key] = choice
        decisions.append({"question": question, "answer": options[choice], "confidence": answers[key]["probabilities"][choice]})
    for key, statement in NOULS.items():
        yes = answers[key]["probabilities"]["A"]
        values[key] = yes >= 0.5
        decisions.append({"question": statement, "answer": "yes" if yes >= 0.5 else "no", "confidence": max(yes, 1 - yes)})
    return {"values": values, "decisions": decisions, "usage": {"cost": 0.0}, "composer": "Laya"}


def open_viewer(run_dir, open_it):
    viewer, index = os.path.join(ROOT, "viewer.html"), os.path.join(run_dir, "index.html")
    if not os.path.exists(viewer):
        if open_it:
            print("viewer.html is missing, so there is no live view for this run")
        return
    shutil.copyfile(viewer, index)
    print(f"viewer: {index}")
    if open_it:
        chrome = "/Applications/Google Chrome.app"  # the viewer is verified in Chrome; fall back to the default browser
        subprocess.run(["open", "-a", chrome, index] if os.path.exists(chrome) else ["open", index], check=False)
        time.sleep(4)  # let the page load so the first strokes are seen


def paint_with_laya(args, laya, composition, grid, palette, truth, row_paints):
    """One Laya decision per pixel, on this Mac, for $0. Same live view and replay log as the Jev runs."""
    size, reach = args.size, args.reach
    pixels = [(r, c) for r in range(size) for c in range(size)]
    chunks = [pixels[i:i + args.chunk] for i in range(0, len(pixels), args.chunk)]
    variant = ["laya", laya.name, LAYA_QUESTION_VERSION, reach, args.chunk]
    fingerprint = hashlib.sha256(json.dumps([variant, size, palette, grid]).encode()).hexdigest()[:8]
    run_dir = os.path.join(ROOT, "runs", f"laya-{laya.name}-{size}-{fingerprint}")
    os.makedirs(os.path.join(run_dir, "chunks"), exist_ok=True)
    cached = sum(os.path.exists(os.path.join(run_dir, "chunks", f"chunk_{i:05d}.json")) for i in range(len(chunks)))
    print(f"run: {run_dir}")
    print(f"canvas {size}x{size} = {size * size} pixels, one Laya decision each, in {len(chunks)} batches "
          f"({cached} cached); cost $0")
    composer = composition.get("composer", "Jev")
    live = LiveCanvas(run_dir, size, {
        "title": "Laya paints a Bob Ross landscape", "size": size, "pixels_total": size * size, "composer": composer,
        "byline": f"{size} × {size} pixels, decided one by one by Laya, running on this Mac",
        "est_cost": 0.0, "decisions": composition["decisions"],
        "palette": [{"name": name, "hex": code} for name, code in palette.items()],
    })
    progress = {"done": 0, "asked": 0, "tokens": 0}

    def stats(elapsed, finished=False, accuracy=None):
        return {"finished": finished, "chunks_done": progress["done"], "chunks_total": len(chunks),
                "questions": progress["asked"], "requests": progress["done"], "input_tokens": progress["tokens"],
                "cost": 0.0, "elapsed": round(elapsed, 2), "accuracy": accuracy}

    live.flush(stats(0.0), force=True)
    open_viewer(run_dir, args.open)
    hard, soft = [None] * (size * size), [None] * (size * size)
    step, t0 = max(1, len(chunks) // 20), time.time()
    for index, part in enumerate(chunks):
        path = os.path.join(run_dir, "chunks", f"chunk_{index:05d}.json")
        asked = [laya_question(grid, r, c, reach, row_paints) for r, c in part]
        if os.path.exists(path):
            with open(path) as f:
                answers = json.load(f)["answers"]
        else:
            out, usage = laya.answer([(state, {"p": q}) for state, q in asked])
            answers = [o["p"] for o in out]
            progress["tokens"] += usage["input_tokens"]
            write_atomic(path, json.dumps({"answers": answers, "usage": usage}))
        colours = []
        for (r, c), a in zip(part, answers):
            rgb = hex_to_rgb(palette[a["choice"]])
            probs = [(hex_to_rgb(palette[name]), p) for name, p in a["probabilities"].items()]
            total = sum(p for _, p in probs) or 1.0
            hard[r * size + c] = rgb
            soft[r * size + c] = tuple(round(sum(col[i] * p for col, p in probs) / total) for i in range(3))
            colours.append(rgb)
        progress["done"], progress["asked"] = index + 1, progress["asked"] + len(part)
        elapsed = time.time() - t0
        live.land(part[0][0] * size + part[0][1], colours, elapsed, {"cost": 0.0, "requests": index + 1}, progress["asked"])
        live.flush(stats(elapsed))
        if (index + 1) % step == 0 or index + 1 == len(chunks):
            rate = progress["asked"] / max(elapsed, 1e-9)
            print(f"  [{index + 1:5d}/{len(chunks)}] {progress['asked']:>8,} decisions  {elapsed:7.1f}s  {rate:5.0f}/s", flush=True)
    elapsed = time.time() - t0
    accuracy = sum(p == t for p, t in zip(hard, truth)) / len(truth)
    scale = max(1, 768 // size)
    save_png(hard, size, os.path.join(run_dir, "painting.png"))
    save_png(soft, size, os.path.join(run_dir, "painting_soft.png"))
    if scale > 1:
        save_png(hard, size, os.path.join(run_dir, f"painting_x{scale}.png"), scale)
        save_png(soft, size, os.path.join(run_dir, f"painting_soft_x{scale}.png"), scale)
    save_png(truth, size, os.path.join(run_dir, "truth.png"))
    save_png([(255, 255, 255) if p == t else (255, 0, 0) for p, t in zip(hard, truth)], size, os.path.join(run_dir, "diff.png"))
    live.save_replay()
    live.flush(stats(elapsed, finished=True, accuracy=round(accuracy, 4)), force=True)
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump({"title": "Laya paints a Bob Ross landscape", "engine": "laya", "checkpoint": laya.name,
                   "device": str(laya.device), "size": size, "variant": variant, "fingerprint": fingerprint,
                   "composition": composition, "palette": palette, "decisions": progress["asked"],
                   "cached_batches": cached, "input_tokens": progress["tokens"], "total_cost": 0.0,
                   "seconds": round(elapsed, 2), "accuracy": accuracy}, f, indent=2)
    print(f"done in {elapsed:.1f}s: {progress['asked']:,} decisions ({progress['asked'] / max(elapsed, 1e-9):.0f}/s), "
          f"{progress['tokens']:,} input tokens, cost $0")
    print(f"pixels matching the intended scene: {accuracy:.2%}")
    print(f"painting: {os.path.join(run_dir, f'painting_x{scale}.png' if scale > 1 else 'painting.png')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=256, help="canvas is size x size pixels")
    ap.add_argument("--method", choices=("pixels", "blocks"), default="pixels",
                    help="pixels = one decision per pixel; blocks = big blocks first, split only where colours meet")
    ap.add_argument("--menu", choices=("row", "full"), default="row",
                    help="paints offered per pixel: only those in its row (about 3x cheaper) or the whole palette")
    ap.add_argument("--top", type=int, default=16, help="blocks method: side of the largest block, a power of two")
    ap.add_argument("--workers", type=int, default=6, help="requests in flight; fewer paints slower")
    ap.add_argument("--max-cost", type=float, default=2.0, help="hard cap in USD for this run")
    ap.add_argument("--preview", action="store_true", help="render the intended scene and stop; no pixel decisions")
    ap.add_argument("--recompose", action="store_true", help="ask Jev for the composition again")
    ap.add_argument("--open", action="store_true", help="open the live viewer before the first brush stroke")
    ap.add_argument("--engine", choices=("jev", "laya"), default="jev",
                    help="jev = TypeSafe's cloud API (paid); laya = the open Laya model on this Mac (free)")
    ap.add_argument("--ckpt", choices=("english", "multilingual", "typed-decisions"), default="typed-decisions",
                    help="laya engine: which checkpoint answers")
    ap.add_argument("--composer", choices=("jev", "laya"), default="jev",
                    help="who makes the composition decisions (jev reuses the cached ones, so both engines paint one scene)")
    ap.add_argument("--reach", type=int, default=12, help="laya engine: columns each side of a pixel shown to Laya")
    ap.add_argument("--chunk", type=int, default=128, help="laya engine: pixels per batch (and per live update)")
    ap.add_argument("--prompt", help="what the painting should look like; steers Jev's composition decisions")
    ap.add_argument("--run-dir", help="write the run here instead of runs/bobross-<method>-<size>-<hash> (used by the Studio)")
    args = ap.parse_args()
    size, blocks = args.size, args.method == "blocks"
    if blocks and (args.top & (args.top - 1) or size % args.top):
        ap.error("--top must be a power of two that divides --size")

    laya = None
    if args.engine == "laya" or args.composer == "laya":
        from laya_local import Laya  # torch loads only when Laya is used
        t_load = time.time()
        laya = Laya(args.ckpt, dtype="fp16")
        print(f"Laya ({args.ckpt}) loaded in {time.time() - t_load:.0f}s on {laya.device}")
    composition = compose_with_laya(laya) if args.composer == "laya" else compose(args.recompose, args.prompt)
    print(f"{composition.get('composer', 'Jev')}'s composition:")
    for d in composition["decisions"]:
        print(f"  {d['question']}  ->  {d['answer']}  ({d['confidence']:.0%})")

    grid = rasterise(build_scene(composition["values"]), size)
    used = {paint for line in grid for paint in line}
    palette = {name: code for name, code in PAINTS.items() if name in used}
    truth = [hex_to_rgb(palette[paint]) for line in grid for paint in line]
    spans = [len(row_spans(line)) for line in grid]
    row_paints = [[name for name in palette if name in set(line)] for line in grid]
    print(f"scene: {len(palette)} paints on the palette; per row: mean {sum(map(len, row_paints)) / size:.1f} paints "
          f"(max {max(map(len, row_paints))}), mean {sum(spans) / size:.1f} column spans (max {max(spans)})")

    if args.preview:
        path = os.path.join(ROOT, "runs", f"bobross_preview_{size}.png")
        save_png(truth, size, path, scale=max(1, 768 // size))
        print(f"preview: {path}")
        return
    if args.engine == "laya":
        return paint_with_laya(args, laya, composition, grid, palette, truth, row_paints)

    mode = PaletteMode(palette, row_paints if blocks or args.menu == "row" else None)
    # Answers carry a distribution over the options, so short menus leave room for more pixels per request.
    max_pixels = 256 if mode.row_paints and max(map(len, row_paints)) <= 12 else mode.max_pixels

    def state_for(index):
        first, last = index * max_pixels, min((index + 1) * max_pixels, size * size) - 1
        return build_state(size, rows_brief(grid, range(first // size, last // size + 1), mode), mode)

    variant = [args.method, args.top] if blocks else [args.method, args.menu, max_pixels]
    fingerprint = hashlib.sha256(json.dumps([MODEL, size, variant, palette, grid]).encode()).hexdigest()[:8]
    run_dir = (os.path.abspath(args.run_dir) if args.run_dir
               else os.path.join(ROOT, "runs", f"bobross-{args.method}-{size}-{fingerprint}"))
    painter = Painter(run_dir, state_for, size, mode, max_pixels, args.workers, args.max_cost)
    compose_cost = composition["usage"].get("cost") or 0.0
    print(f"run: {run_dir}")
    if blocks:
        planned, requests, estimate = estimate_blocks(grid, size, mode, args.top)
        print(f"canvas {size}x{size} = {size * size} pixels, painted in blocks from {args.top}px down; if Jev judges "
              f"every block right: {planned} decisions in {requests} requests, est. ${estimate:.4f}")
    else:
        todo, planned, estimate = painter.estimate()
        print(f"canvas {size}x{size} = {size * size} pixels; {len(painter.chunks)} requests "
              f"({len(painter.chunks) - todo} cached), {planned} pixel decisions to make, est. ${estimate:.4f}")
    if estimate > args.max_cost:
        raise SystemExit(f"estimated cost ${estimate:.2f} exceeds --max-cost ${args.max_cost:.2f}; nothing sent")

    live = LiveCanvas(run_dir, size, {
        "title": args.prompt or TITLE, "size": size, "pixels_total": size * size,
        "byline": (f"{size} × {size} pixels, painted by Jev in blocks, big strokes first" if blocks
                   else f"{size} × {size} pixels, decided one by one by Jev"),
        "est_cost": round(estimate + compose_cost, 4), "decisions": composition["decisions"],
        "palette": [{"name": name, "hex": code} for name, code in palette.items()],
    })
    progress = {"done": 0, "total": 0 if blocks else len(painter.chunks), "asked": 0}

    def stats(elapsed, finished=False, accuracy=None):
        return {"finished": finished, "chunks_done": progress["done"], "chunks_total": progress["total"],
                "questions": progress["asked"], "requests": painter.usage["requests"],
                "input_tokens": painter.usage["input_tokens"],
                "cost": round(painter.usage["cost"] + compose_cost, 6), "elapsed": round(elapsed, 2), "accuracy": accuracy}

    live.flush(stats(0.0), force=True)
    viewer = os.path.join(ROOT, "viewer.html")
    index = os.path.join(run_dir, "index.html")
    if os.path.exists(viewer):
        shutil.copyfile(viewer, index)
        print(f"viewer: {index}")
        if args.open:
            chrome = "/Applications/Google Chrome.app"  # the viewer is verified in Chrome; fall back to the default browser
            subprocess.run(["open", "-a", chrome, index] if os.path.exists(chrome) else ["open", index], check=False)
            time.sleep(4)  # let the page load so the first strokes are seen
    elif args.open:
        print("viewer.html is missing, so there is no live view for this run")

    canvas = [None] * (size * size)
    try:
        if blocks:
            def on_level(b, count, requests):
                progress["total"] += requests
                print(f"  {b:>2}px blocks: {count:6d} to decide in {requests:4d} requests   "
                      f"${painter.usage['cost']:.4f} so far", flush=True)

            def on_block(row, col, b, rgb, elapsed, asked):
                for y in range(row, row + b):
                    canvas[y * size + col:y * size + col + b] = [rgb] * b
                live.land_block(row, col, b, rgb, elapsed, painter.usage, asked)

            def on_request(elapsed, asked):
                progress["done"] += 1
                progress["asked"] = asked
                live.flush(stats(elapsed))

            elapsed, _ = paint_blocks(painter, grid, size, mode, args.top, args.workers, on_level, on_block, on_request)
        else:
            step = max(1, len(painter.chunks) // 10)

            def on_progress(done, total, elapsed, chunk, answers):
                pixels = painter.chunks[chunk]
                colours = [mode.decode(answers, row, col, "argmax")[0] for row, col in pixels]
                start = pixels[0][0] * size + pixels[0][1]
                canvas[start:start + len(pixels)] = colours
                progress["done"], progress["asked"] = done, progress["asked"] + len(pixels)
                live.land(start, colours, elapsed, painter.usage, progress["asked"])
                live.flush(stats(elapsed))
                if done % step == 0 or done == total:
                    print(f"  [{done:4d}/{total}] requests  ${painter.usage['cost']:.4f}  {elapsed:6.1f}s", flush=True)

            elapsed = painter.paint(on_progress)
    except FatalAPIError as e:
        raise SystemExit(f"aborted: {e}")

    blank = hex_to_rgb(CANVAS)
    unpainted = sum(rgb is None for rgb in canvas)
    accuracy = sum(p == t for p, t in zip(canvas, truth)) / len(truth)
    scale = max(1, 768 // size)
    shown = os.path.join(run_dir, f"painting_x{scale}.png" if scale > 1 else "painting.png")
    save_png([rgb or blank for rgb in canvas], size, os.path.join(run_dir, "painting.png"))
    if scale > 1:
        save_png([rgb or blank for rgb in canvas], size, shown, scale)
    save_png(truth, size, os.path.join(run_dir, "truth.png"))
    save_png([(255, 255, 255) if p == t else (255, 0, 0) for p, t in zip(canvas, truth)], size, os.path.join(run_dir, "diff.png"))
    live.save_replay()
    live.flush(stats(elapsed, finished=True, accuracy=round(accuracy, 4)), force=True)

    total_cost = painter.usage["cost"] + compose_cost
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump({
            "title": TITLE, "prompt": args.prompt, "model": MODEL, "size": size, "method": args.method, "variant": variant,
            "fingerprint": fingerprint, "composition": composition, "palette": palette,
            "decisions": progress["asked"], "failed_requests": painter.failed_chunks,
            "unpainted_pixels": unpainted, "usage": painter.usage, "composition_cost": compose_cost,
            "total_cost": total_cost, "seconds": round(elapsed, 2), "accuracy": accuracy,
        }, f, indent=2)

    print(f"done in {elapsed:.1f}s: {progress['asked']} decisions in {painter.usage['requests']} requests, "
          f"{painter.usage['input_tokens']} input tokens")
    print(f"cost: ${painter.usage['cost']:.4f} for the painting + ${compose_cost:.6f} for the composition = ${total_cost:.4f}")
    if painter.stop.is_set():
        print(f"STOPPED EARLY: spend reached --max-cost ${args.max_cost:.2f}")
    if unpainted:
        print(f"WARNING: {unpainted} pixels unpainted (left as bare canvas); failed requests: {painter.failed_chunks[:5]}")
    print(f"pixels matching the intended scene: {accuracy:.2%}")
    print(f"painting: {shown}")


if __name__ == "__main__":
    main()
