"""Turn any prompt into a flat, layered scene that Jev can paint.

A general chat model (through the same OpenRouter key) plans the picture as coloured shapes
in unit coordinates: rectangles, ellipses, polygons and thick lines, back to front. Python
rasterises the shapes, and Jev then decides every pixel or block from that description, exactly
as it does for the Bob Ross landscape. Plans are cached per (prompt, model), so repainting a
prompt at another size, or after a failure, costs nothing extra.

  from scene_planner import plan, to_layers
  scene = plan("a human face")        # {"title", "paints", "background", "shapes", "usage"}
  layers = to_layers(scene)           # [(paint, contains, bbox), ...] for bob_ross.rasterise
"""
import hashlib
import json
import os
import re
import time

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
CHAT_URL = os.environ.get("PLANNER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"  # fastest cheap option; thinking can be kept low
FALLBACK_MODEL = "z-ai/glm-5.3-flash"  # cheaper per token, but always thinks first, so slower
MAX_PAINTS = 16
MAX_SHAPES = 120
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")

SYSTEM = """You design flat, bold pixel-art pictures for a painter that can only fill coloured shapes.
Reply with ONE compact JSON object and nothing else:
{"t":"short title",
 "k":["key visual features of the subject, 4 to 8 short phrases, e.g. long curved neck, one tall hump, four thin legs with knobbly knees"],
 "p":{"Sky":"#9EC9E8","Fur":"#C8803E","Fur Shade":"#8E5426","Outline":"#3B2414"},
 "bg":"Sky",
 "s":[
  ["r","Sand",0,0.7,1,1],
  ["e","Outline",0.5,0.5,0.23,0.15],
  ["e","Fur",0.5,0.5,0.21,0.13],
  ["p","Fur Shade",0.3,0.55,0.7,0.55,0.62,0.62,0.38,0.62],
  ["l","Outline",0.36,0.6,0.34,0.85,0.05],
  ["l","Fur",0.36,0.6,0.34,0.85,0.03]]}

"k" = what makes the subject recognisable. Write it first, then draw every item in it.
"p" = paints: 5 to 12 entries, name -> #RRGGBB. Names: 1 to 2 plain words, letters and spaces only.
  Include, for the main subject: a base colour, a darker shade of it, and a dark outline colour.
"bg" = the paint that fills the whole canvas first.
"s" = shapes, painted in order, back to front. Each shape is an array:
  ["r", paint, x0, y0, x1, y1]          rectangle
  ["e", paint, cx, cy, rx, ry]          ellipse
  ["p", paint, x1, y1, x2, y2, ...]     polygon, 3 to 16 points, for silhouettes and angled parts
  ["l", paint, x0, y0, x1, y1, width]   thick line: legs, necks, tails, arms, brows, mouths, stems
Canvas: unit square, x 0 (left) to 1 (right), y 0 (top) to 1 (bottom). At most 2 decimal places.

How to draw well:
- Build the subject from separate parts (head, neck, body, each leg, tail, ears, eyes...), never one blob.
- Outline the subject: draw each main part first in the outline paint, slightly bigger (or a wider line),
  then the same part again in its base colour on top.
- Add a darker shade on the underside or the side away from the light; add 1 or 2 highlights.
- Ground the scene: a horizon or floor, and a simple shadow under the subject.
- The subject fills 50 to 80 percent of the canvas, centred. Nothing thinner than 0.02.
- Use 30 to 70 shapes. Paint exactly what is asked for, whatever it is. No text."""


class PlanError(Exception):
    pass


def load_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        with open(os.path.join(ROOT, ".env")) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    raise PlanError("OPENROUTER_API_KEY not found in the environment or .env")


def num(value, lo=-0.5, hi=1.5):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("not a number")
    if v != v:  # NaN
        raise ValueError("not a number")
    return max(lo, min(hi, v))


def clean_name(name):
    name = re.sub(r"[^A-Za-z ]+", " ", str(name or "")).strip()
    name = re.sub(r"\s+", " ", name)[:24].strip()
    return name.title() if name else ""


KINDS = {"r": "rect", "e": "ellipse", "p": "polygon", "l": "line"}


def expand(raw):
    """The compact reply ({"t","p","bg","s":[[kind, paint, numbers...]]}) as the long form validate() reads."""
    if not isinstance(raw, dict) or "s" not in raw:
        return raw
    paints = raw.get("p") or {}
    if isinstance(paints, dict):
        paints = [{"name": n, "hex": h} for n, h in paints.items()]
    shapes = []
    for item in raw.get("s") or []:
        if not isinstance(item, list) or len(item) < 3:
            continue
        kind, paint, nums = KINDS.get(str(item[0]).lower()[:1]), item[1], item[2:]
        if kind == "polygon":
            shapes.append({"type": kind, "paint": paint, "points": [nums[i:i + 2] for i in range(0, len(nums) - 1, 2)]})
        elif kind in ("rect", "ellipse", "line"):
            keys = {"rect": ("x0", "y0", "x1", "y1"), "ellipse": ("cx", "cy", "rx", "ry"),
                    "line": ("x0", "y0", "x1", "y1", "width")}[kind]
            if len(nums) >= 4:
                shapes.append(dict(zip(keys, nums), type=kind, paint=paint))
    return {"title": raw.get("t"), "paints": paints, "background": raw.get("bg"), "shapes": shapes}


def validate(raw):
    """Keep what is usable from the model's plan, drop the rest, and fail only if nothing paintable is left."""
    raw = expand(raw)
    if not isinstance(raw, dict):
        raise PlanError("the planner did not return a JSON object")
    paints, rename = {}, {}
    for p in raw.get("paints") or []:
        if not isinstance(p, dict):
            continue
        name, code = clean_name(p.get("name")), str(p.get("hex") or "").strip()
        if not name or not HEX.match(code) or len(paints) >= MAX_PAINTS:
            continue
        base, n = name, 2
        while name in paints:
            name, n = f"{base} {chr(64 + n)}", n + 1
        paints[name] = code.upper()
        rename[str(p.get("name"))] = name
        rename[clean_name(p.get("name"))] = name
    if len(paints) < 2:
        raise PlanError("the planner's palette had fewer than two usable paints")

    def paint_of(name):
        return rename.get(str(name)) or rename.get(clean_name(name))

    background = paint_of(raw.get("background")) or next(iter(paints))
    shapes = []
    for s in (raw.get("shapes") or [])[:MAX_SHAPES]:
        if not isinstance(s, dict) or not paint_of(s.get("paint")):
            continue
        kind = s.get("type")
        try:
            if kind == "rect":
                shape = {k: num(s[k]) for k in ("x0", "y0", "x1", "y1")}
                shape["x0"], shape["x1"] = sorted((shape["x0"], shape["x1"]))
                shape["y0"], shape["y1"] = sorted((shape["y0"], shape["y1"]))
            elif kind == "ellipse":
                shape = {"cx": num(s["cx"]), "cy": num(s["cy"]),
                         "rx": num(s["rx"], 0.004, 1.5), "ry": num(s["ry"], 0.004, 1.5)}
            elif kind == "polygon":
                points = [[num(x), num(y)] for x, y in s["points"]][:48]
                if len(points) < 3:
                    continue
                shape = {"points": points}
            elif kind == "line":
                shape = {k: num(s[k]) for k in ("x0", "y0", "x1", "y1")}
                shape["width"] = num(s.get("width", 0.02), 0.006, 0.5)
            else:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        shape.update(type=kind, paint=paint_of(s.get("paint")), label=str(s.get("label") or kind)[:40])
        shapes.append(shape)
    if not shapes:
        raise PlanError("the planner's scene had no usable shapes")
    title = " ".join(str(raw.get("title") or "").split())[:80]
    return {"title": title, "paints": paints, "background": background, "shapes": shapes}


def plan(prompt, model=DEFAULT_MODEL, fresh=False, timeout=40, fallback=FALLBACK_MODEL, effort="low"):
    """Plan with `model`; if it is slow, failing or returns nothing usable, plan once with `fallback`."""
    if effort in ("medium", "high"):
        timeout = max(timeout, 90)  # thinking harder takes longer before the first word
    try:
        return _plan(prompt, model, fresh, timeout, effort)
    except PlanError as e:
        if not fallback or fallback == model:
            raise
        print(f"planner: {model} failed ({str(e)[:160]}); falling back to {fallback}", flush=True)
        return _plan(prompt, fallback, fresh, timeout, "low")


def _plan(prompt, model, fresh, timeout, effort="low"):
    """The scene for a prompt: from the cache if this prompt and model were planned before."""
    prompt = " ".join(prompt.split())
    digest = hashlib.sha256(json.dumps([prompt, model, SYSTEM] + ([effort] if effort != "low" else [])).encode()).hexdigest()[:12]  # new instructions, new plan
    path = os.path.join(ROOT, "runs", "plans", f"{digest}.json")
    if os.path.exists(path) and not fresh:
        with open(path) as f:
            scene = json.load(f)
        scene["usage"] = {"cost": 0.0, "cached": True}
        return scene

    print(f"planner: asking {model} to plan the shapes...", flush=True)
    t0 = time.time()
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Picture to paint: {prompt}"}],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_tokens": 12000,  # room for a model that insists on thinking before it writes the plan
        "usage": {"include": True},
        # a layout needs little thought; some models refuse to switch thinking off, so ask for the least
        "reasoning": {"effort": effort, "exclude": True},
    }
    raw, cost, errors = None, 0.0, []
    for attempt in range(2):
        try:
            r = requests.post(
                CHAT_URL,
                headers={"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json",
                         "X-OpenRouter-Title": "Jev Studio"},
                data=json.dumps(payload),
                timeout=(10, timeout),
            )
        except requests.Timeout:
            raise PlanError(f"no reply within {timeout}s")
        except requests.RequestException as e:
            raise PlanError(f"could not reach the planner: {e}")
        if r.status_code != 200:
            if attempt == 0 and r.status_code == 400:
                errors.append(f"HTTP 400: {r.text[:200]}")
                payload.pop("response_format", None)  # some providers refuse JSON mode; the prompt asks for JSON anyway
                payload.pop("reasoning", None)
                continue
            raise PlanError(f"planner request failed: HTTP {r.status_code}: {r.text[:400]}")
        body = r.json()
        cost += float((body.get("usage") or {}).get("cost") or 0.0)
        try:
            choice = body["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError):
            raise PlanError(f"planner returned no message: {json.dumps(body)[:400]}")
        text = (message.get("content") or message.get("reasoning") or message.get("reasoning_content") or "").strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                raw = json.loads(text[start:end + 1])
                break
            except ValueError:
                errors.append(f"not valid JSON (finish_reason={choice.get('finish_reason')}): {text[:150]}")
        else:
            errors.append(f"empty reply (finish_reason={choice.get('finish_reason')})")
        payload.pop("response_format", None)  # retry once in plain mode
        if choice.get("finish_reason") == "length":
            payload["max_tokens"] = 24000  # thinking ate the budget: give the retry more room
    if raw is None:
        raise PlanError(f"{model} did not return a usable plan: " + " | ".join(errors))

    scene = validate(raw)
    scene.update(prompt=prompt, model=model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)
    scene["usage"] = {"cost": cost}
    print(f"planner: {model} planned {len(scene['shapes'])} shapes in {time.time() - t0:.1f}s", flush=True)
    return scene


# ---- geometry: same (contains, bbox) contract as bob_ross.rect / ellipse / polygon ----

def _rect(x0, y0, x1, y1):
    return (lambda x, y: x0 <= x <= x1 and y0 <= y <= y1), (x0, y0, x1, y1)


def _ellipse(cx, cy, rx, ry):
    return (lambda x, y: ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1), (cx - rx, cy - ry, cx + rx, cy + ry)


def _polygon(points):
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


def _line(x0, y0, x1, y1, width):
    """A thick segment with round ends: every point within width/2 of it."""
    half, dx, dy = width / 2, x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy

    def contains(x, y):
        t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / length2))
        px, py = x0 + t * dx - x, y0 + t * dy - y
        return px * px + py * py <= half * half

    return contains, (min(x0, x1) - half, min(y0, y1) - half, max(x0, x1) + half, max(y0, y1) + half)


def to_layers(scene):
    """Layers back to front as (paint, contains, bbox): the background, then every shape in order."""
    layers = [(scene["background"],) + _rect(-1, -1, 2, 2)]
    for s in scene["shapes"]:
        if s["type"] == "rect":
            shape = _rect(s["x0"], s["y0"], s["x1"], s["y1"])
        elif s["type"] == "ellipse":
            shape = _ellipse(s["cx"], s["cy"], s["rx"], s["ry"])
        elif s["type"] == "polygon":
            shape = _polygon([tuple(p) for p in s["points"]])
        else:
            shape = _line(s["x0"], s["y0"], s["x1"], s["y1"], s["width"])
        layers.append((s["paint"], shape[0], shape[1]))
    return layers
