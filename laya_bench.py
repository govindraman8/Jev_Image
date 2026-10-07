"""Does local Laya paint? Sanity, equivalence, precision/speed, then accuracy on painting questions.

  .venv/bin/python laya_bench.py sanity   [--ckpt english]
  .venv/bin/python laya_bench.py speed    [--ckpt english]
  .venv/bin/python laya_bench.py paint    [--ckpt english] [--size 64] [--n 400] [--formats jev,compact,window]
"""
import argparse
import json
import random
import sys
import time

import numpy as np

import bob_ross as br
from jev_paint import PaletteMode, build_state
from laya_local import Laya, internal
from rl_common import build_sequence

EMAIL = {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
         "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."}
EMAIL_Q = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}


def scene(size):
    comp = br.compose(False)["values"]
    grid = br.rasterise(br.build_scene(comp), size)
    used = {p for line in grid for p in line}
    palette = {n: c for n, c in br.PAINTS.items() if n in used}
    row_paints = [[n for n in palette if n in set(line)] for line in grid]
    return grid, palette, row_paints


def sanity(laya):
    t0 = time.time()
    (ans,), usage = laya.answer([(EMAIL, EMAIL_Q)])
    print(f"README example in {time.time() - t0:.2f}s: department={ans['department']['choice']} "
          f"{ans['department']['probabilities']}, urgency={ans['urgency']['score']}, churn_risk={ans['churn_risk']['noul']}")
    # my cached sequence builder must match the repo's builder token for token
    grid, palette, row_paints = scene(64)
    mode = PaletteMode(palette, row_paints)
    mismatches = 0
    for row in (5, 30, 50):
        state = build_state(64, br.rows_brief(grid, [row], mode), mode)
        sids = laya.state_ids(state)
        for col in (0, 31, 63):
            _, qdef = next(iter(mode.questions(row, col)))
            q = internal(qdef)
            ref = build_sequence(laya.tok, state, q, laya.max_len, laya.head_max_len)
            mine = laya.sequence(sids, q)[:2]
            mismatches += (list(ref[0]), list(ref[1])) != (list(mine[0]), list(mine[1]))
    print(f"sequence builder vs repo's build_sequence: {9 - mismatches}/9 identical")


def speed(ckpt):
    grid, palette, row_paints = scene(256)
    mode = PaletteMode(palette, row_paints)
    reqs = []
    for row in range(0, 256, 32):
        state = build_state(256, br.rows_brief(grid, [row], mode), mode)
        reqs.append((state, {k: q for col in range(0, 256, 4) for k, q in mode.questions(row, col)}))
    ref = None
    for dtype in ("fp32", "bf16", "fp16"):
        try:
            laya = Laya(ckpt, dtype=dtype)
            laya.answer(reqs[:1])  # warm-up
            t0 = time.time()
            out, usage = laya.answer(reqs)
            dt = time.time() - t0
        except Exception as e:  # a dtype the GPU cannot run
            print(f"{dtype}: failed: {e!r}"[:300])
            continue
        choices = [a[k]["choice"] for a in out for k in sorted(a)]
        agree = "" if ref is None else f", agrees with fp32 on {np.mean([x == y for x, y in zip(choices, ref)]):.1%}"
        ref = ref or choices
        n = usage["questions"]
        print(f"{dtype}: {n} questions in {dt:.2f}s = {n / dt:.0f} q/s, {usage['input_tokens'] / n:.0f} tokens/question, "
              f"{usage['truncated_states']} states truncated{agree}")
        del laya


def formats(grid, palette, row_paints, size, mode):
    """Question builders: (row, col) -> (state, question id, question, answer key)."""
    def jev(row, col):  # exactly what Jev was asked
        state = build_state(size, br.rows_brief(grid, [row], mode), mode)
        key, q = next(iter(mode.questions(row, col)))
        return state, key, q, palette[grid[row][col]]

    def compact(row, col):
        spans = br.row_spans(grid[row])
        state = f"Row {row} of a {size}-column painting. " + "; ".join(f"columns {a} to {b}: {p}" for a, b, p in spans) + "."
        q = {"type": "choice", "instructions": f"Which paint covers column {col} of row {row}?",
             "criteria": {p: None for p in row_paints[row]}}
        return state, f"p{row}_{col}", q, grid[row][col]

    def window(row, col):  # only the spans near the column
        spans = [s for s in br.row_spans(grid[row]) if s[1] >= col - 12 and s[0] <= col + 12]
        state = "; ".join(f"columns {a} to {b}: {p}" for a, b, p in spans) + "."
        opts = list(dict.fromkeys(p for _, _, p in spans))
        q = {"type": "choice", "instructions": f"Which paint covers column {col}?", "criteria": {p: None for p in opts}}
        return state, f"p{row}_{col}", q, grid[row][col]

    return {"jev": jev, "compact": compact, "window": window}


def paint(laya, size, n, names):
    grid, palette, row_paints = scene(size)
    mode = PaletteMode(palette, row_paints)
    rng = random.Random(7)
    # half the sample sits on a colour edge (within 2 columns of a span boundary), where lookups are hard
    edges = [(r, c) for r in range(size) for c in range(size) if any(abs(c - b) <= 2 for _, b, _ in br.row_spans(grid[r])[:-1])]
    flat = [(r, c) for r in range(size) for c in range(size)]
    sample = rng.sample(edges, min(n // 2, len(edges))) + rng.sample(flat, n // 2)
    builders = formats(grid, palette, row_paints, size, mode)
    for name in names:
        reqs, keys, truth, n_opts = [], [], [], []
        for row, col in sample:
            state, key, q, answer = builders[name](row, col)
            reqs.append((state, {key: q}))
            keys.append(key)
            truth.append(answer)
            n_opts.append(len(q["criteria"]))
        t0 = time.time()
        out, usage = laya.answer(reqs)
        dt = time.time() - t0
        got = [a[k]["choice"] for a, k in zip(out, keys)]
        right = np.array([g == t for g, t in zip(got, truth)])
        half = len(sample) // 2
        # the honest baseline: always answer the paint that covers most of the row
        majority = np.array([max(set(grid[r]), key=grid[r].count) == grid[r][c] for r, c in sample])
        chance = np.mean([1 / k for k in n_opts])
        print(f"[{name:8s}] accuracy {right.mean():6.1%} (edges {right[:half].mean():.1%}, anywhere {right[half:].mean():.1%}) | "
              f"row-majority baseline {majority.mean():.1%} | chance {chance:.1%} | "
              f"{usage['input_tokens'] / len(sample):.0f} tok/q, {len(sample) / dt:.0f} q/s, {usage['truncated_states']} truncated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("sanity", "speed", "paint"))
    ap.add_argument("--ckpt", default="english")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--formats", default="jev,compact,window")
    args = ap.parse_args()
    if args.what == "speed":
        return speed(args.ckpt)
    laya = Laya(args.ckpt, dtype=args.dtype)
    if args.what == "sanity":
        return sanity(laya)
    paint(laya, args.size, args.n, args.formats.split(","))


if __name__ == "__main__":
    main()
