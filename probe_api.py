"""Exploratory probes against the Jev decisions API. Facts feed the PRD.

Usage: python3 probe_api.py basic|score|choice|scale N
"""
import json
import os
import sys
import time

import requests

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
ROOT = os.path.dirname(os.path.abspath(__file__))


def load_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("OPENROUTER_API_KEY not found in env or .env")


def ask(state, questions, timeout=120):
    t0 = time.time()
    r = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {load_key()}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "Jev Image",
        },
        data=json.dumps({"model": MODEL, "state": state, "questions": questions}),
        timeout=timeout,
    )
    dt = time.time() - t0
    try:
        body = r.json()
    except ValueError:
        body = {"_raw": r.text[:2000]}
    return r.status_code, dt, body


def basic():
    """The user's exact example: verifies the key and shows the full response shape."""
    status, dt, body = ask(
        "Help! My payouts have been failing for 3 days.",
        {
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this message convey urgency?",
                "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
            },
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                },
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            },
        },
    )
    print(f"HTTP {status} in {dt:.2f}s")
    print(json.dumps(body, indent=2))


PAINT_STATE = (
    "You are painting a 16x16 pixel image, one pixel at a time. Rows and columns are "
    "numbered 0-15; row 0 is the top, column 0 is the left. The painting: the top half "
    "(rows 0-7) is pure red (R=255, G=0, B=0). The bottom half (rows 8-15) is pure white "
    "(R=255, G=255, B=255)."
)


def score():
    """What range does `score` return? Try 3 anchors vs 5 anchors on known-answer pixels."""
    three = ["None of this colour (0)", "Half intensity (128)", "Full intensity (255)"]
    five = ["0", "64", "128", "192", "255"]
    qs = {}
    for name, (row, col) in {"top": (2, 5), "bottom": (12, 5)}.items():
        for ch in ("red", "green", "blue"):
            for label, crit in (("a3", three), ("a5", five)):
                qs[f"{name}_{ch}_{label}"] = {
                    "type": "score",
                    "instructions": f"Amount of {ch} in the pixel at row {row}, column {col}.",
                    "criteria": crit,
                }
    status, dt, body = ask(PAINT_STATE, qs)
    print(f"HTTP {status} in {dt:.2f}s  usage={body.get('usage')}")
    for k, v in (body.get("answers") or {}).items():
        print(f"  {k:22s} {json.dumps(v)}")
    if "answers" not in body:
        print(json.dumps(body, indent=2)[:3000])


def choice():
    """A 16-way hex-digit choice: does the full distribution come back?"""
    hexd = "0123456789ABCDEF"
    qs = {
        "top_green_hi": {
            "type": "choice",
            "instructions": "High hex digit of the GREEN channel for the pixel at row 2, column 5.",
            "criteria": {d: f"Green high nibble = {d} (channel value {int(d, 16) * 16}-{int(d, 16) * 16 + 15})" for d in hexd},
        },
        "bottom_green_hi": {
            "type": "choice",
            "instructions": "High hex digit of the GREEN channel for the pixel at row 12, column 5.",
            "criteria": {d: f"Green high nibble = {d} (channel value {int(d, 16) * 16}-{int(d, 16) * 16 + 15})" for d in hexd},
        },
    }
    status, dt, body = ask(PAINT_STATE, qs)
    print(f"HTTP {status} in {dt:.2f}s  usage={body.get('usage')}")
    print(json.dumps(body.get("answers", body), indent=2)[:3000])


def scale(n):
    """How many questions fit in one request? n score questions over distinct pixels."""
    crit = ["0", "64", "128", "192", "255"]
    qs = {}
    i = 0
    for row in range(16):
        for col in range(16):
            for ch in ("red", "green", "blue"):
                if i >= n:
                    break
                qs[f"r{row}c{col}{ch[0]}"] = {
                    "type": "score",
                    "instructions": f"Amount of {ch} in the pixel at row {row}, column {col}.",
                    "criteria": crit,
                }
                i += 1
    status, dt, body = ask(PAINT_STATE, qs, timeout=300)
    answers = body.get("answers") or {}
    print(f"n={len(qs)}  HTTP {status} in {dt:.2f}s  answered={len(answers)}  usage={body.get('usage')}")
    if status != 200 or not answers:
        print(json.dumps(body, indent=2)[:2000])
        return
    # Known ground truth: green/blue are 0 on rows 0-7 and 255 on rows 8-15; red is always 255.
    sample = list(answers.items())
    for k, v in sample[:3] + sample[-3:]:
        print(f"  {k:10s} {json.dumps(v)[:160]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "basic"
    if cmd == "scale":
        scale(int(sys.argv[2]))
    else:
        {"basic": basic, "score": score, "choice": choice}[cmd]()
