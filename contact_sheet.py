#!/usr/bin/env python3
"""Tile several runs' paintings into one labelled image for side-by-side comparison.

  python3 contact_sheet.py OUT.png [--cols 3] [--truth] RUN_DIR[:argmax] [RUN_DIR[:argmax] ...]
"""
import argparse
import json
import os

from PIL import Image, ImageDraw

TILE = 256
LABEL = 44
PAD = 12


def parse_run(arg):
    """RUN_DIR, or RUN_DIR:argmax for that run's hard-decision render."""
    run_dir, _, decode = arg.partition(":")
    return run_dir, f"_{decode}" if decode else ""


def load_tile(run_dir, filename):
    image = Image.open(os.path.join(run_dir, filename)).convert("RGB")
    return image.resize((TILE, TILE), Image.NEAREST)


def label_for(run_dir, suffix):
    with open(os.path.join(run_dir, f"run{suffix}.json")) as f:
        run = json.load(f)
    top = f"{run.get('layout') or 'free text'} / {run['mode']}  {run['size']}px{suffix.replace('_', '  ')}"
    accuracy = run.get("accuracy")
    if accuracy:
        bottom = "  ".join(f"{group[:5]} {value:.0%}" for group, value in accuracy.items() if value is not None)
    else:
        bottom = f"${run['usage']['cost']:.3f}"
    return top, bottom


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--truth", action="store_true", help="lead with the first run's ground truth")
    args = ap.parse_args()

    runs = [parse_run(arg) for arg in args.runs]
    tiles = []
    if args.truth:
        truth_dir = next(run_dir for run_dir, _ in runs if os.path.exists(os.path.join(run_dir, "truth.png")))
        tiles.append((load_tile(truth_dir, "truth.png"), ("ground truth", "")))
    tiles += [(load_tile(run_dir, f"painting{suffix}.png"), label_for(run_dir, suffix)) for run_dir, suffix in runs]

    rows = -(-len(tiles) // args.cols)
    sheet = Image.new("RGB", (PAD + args.cols * (TILE + PAD), PAD + rows * (TILE + LABEL + PAD)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (tile, (top, bottom)) in enumerate(tiles):
        x = PAD + (i % args.cols) * (TILE + PAD)
        y = PAD + (i // args.cols) * (TILE + LABEL + PAD)
        sheet.paste(tile, (x, y))
        draw.text((x, y + TILE + 6), top, fill=(235, 235, 235))
        draw.text((x, y + TILE + 24), bottom, fill=(160, 160, 160))
    sheet.save(args.out)
    print(f"wrote {args.out} ({len(tiles)} tiles)")


if __name__ == "__main__":
    main()
