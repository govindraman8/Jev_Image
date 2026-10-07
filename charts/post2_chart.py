"""Post #2 chart: latency vs throughput vs accuracy, Jev vs Laya, from the measured benchmark files.

  .venv/bin/python charts/post2_chart.py   -> charts/post2_latency_vs_throughput.html (screenshot it to PNG)
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
one = [json.loads(line) for line in open(os.path.join(ROOT, "ports", "results.jsonl"))]
six = [json.loads(line) for line in open(os.path.join(ROOT, "ports", "results_six.jsonl"))]
find = lambda rows, **kw: next(r for r in rows if all(r.get(k) == v for k, v in kw.items()))

jev1 = find(one, runtime="jev-cloud (OpenRouter)")
ne_fast1 = find(one, runtime="coreml", checkpoint="multilingual", batch_mode="loop")
ne_acc1 = find(one, runtime="coreml", checkpoint="typed-decisions")
jev6 = find(six, runtime="jev", checkpoint_or_inflight="128")
ne_fast6 = find(six, runtime="coreml", checkpoint_or_inflight="multilingual")
ne_acc6 = find(six, runtime="coreml", checkpoint_or_inflight="typed-decisions")

# categorical slots 1-3, dark steps (validated: all-pairs CVD dE >= 9.4, normal-vision >= 20.9)
ROWS = [("Jev", "cloud API, via OpenRouter", "#3987e5"),
        ("Laya · fastest", "Neural Engine, multilingual", "#d95926"),
        ("Laya · most accurate", "Neural Engine, typed-decisions", "#199e70")]
PANELS = [
    ("One decision", "milliseconds · lower is better",
     [jev1["single_ms_p50"], ne_fast1["single_ms_p50"], ne_acc1["single_ms_p50"]], lambda v: f"{v:.0f} ms"),
    ("Bulk job", "decisions per second · higher is better",
     [jev6["decisions_per_s"], ne_fast6["decisions_per_s"], ne_acc6["decisions_per_s"]], lambda v: f"{v:,.0f}/s"),
    ("Sorted correctly", "emails sent to the right team",
     [jev6["dept_accuracy"] * 100, ne_fast6["dept_accuracy"] * 100, ne_acc6["dept_accuracy"] * 100], lambda v: f"{v:.0f}%"),
]
MAX = [None, None, 100.0]

W, H, PAD = 1600, 900, 72
LABEL_W, GAP = 300, 56
PANEL_X0 = PAD + LABEL_W
PANEL_W = (W - PAD - PANEL_X0 - 2 * GAP) / 3
BAR_MAX = PANEL_W - 96
ROW_Y = [420, 500, 580]
BAR_H = 24


def bar(x0, y, length, color):
    r = min(4, length / 2)
    x1 = x0 + max(length, 1)
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} V{y + BAR_H - r:.1f} '
            f'A{r},{r} 0 0 1 {x1 - r:.1f},{y + BAR_H:.1f} H{x0:.1f} Z" fill="{color}"/>')


svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
       f'aria-label="Laya wins one decision; Jev wins the bulk job and sorts every email correctly">',
       f'<rect width="{W}" height="{H}" fill="#1a1a19"/>',
       f'<text x="{PAD}" y="118" class="title">Laya wins one decision. Jev wins the bulk job.</text>',
       f'<text x="{PAD}" y="164" class="sub">256 support emails × 6 questions = 1,536 decisions · Laya on an M3 Pro laptop, '
       f'Jev through OpenRouter · measured 24 Sep 2026</text>']
for i, (title, unit, values, fmt) in enumerate(PANELS):
    px = PANEL_X0 + i * (PANEL_W + GAP)
    top = max(values) if MAX[i] is None else MAX[i]
    svg.append(f'<text x="{px:.1f}" y="292" class="ph">{title}</text>')
    svg.append(f'<text x="{px:.1f}" y="324" class="pu">{unit}</text>')
    svg.append(f'<line x1="{px:.1f}" y1="{ROW_Y[0] - 22}" x2="{px:.1f}" y2="{ROW_Y[-1] + BAR_H + 22}" stroke="#383835" stroke-width="1"/>')
    for (name, detail, color), v, y in zip(ROWS, values, ROW_Y):
        length = BAR_MAX * v / top
        svg.append(bar(px + 1, y, length, color))
        svg.append(f'<text x="{px + 1 + length + 12:.1f}" y="{y + 19}" class="val">{fmt(v)}</text>')
for (name, detail, color), y in zip(ROWS, ROW_Y):
    svg.append(f'<circle cx="{PAD + 7}" cy="{y + 9}" r="7" fill="{color}"/>')
    svg.append(f'<text x="{PAD + 26}" y="{y + 16}" class="rl">{name}</text>')
    svg.append(f'<text x="{PAD + 26}" y="{y + 40}" class="rd">{detail}</text>')
foot = [
    "One decision: median of 30 calls for Jev (including the internet round trip) and 50 for Laya. Bulk job: Jev with 128 requests in flight,",
    "a 2-second burst; Laya via Core ML on the Neural Engine, fp16, $0. All of today's Jev benchmarks cost $0.04. Emails are synthetic;",
    "\"right team\" is the team each email was written for.",
]
for k, line in enumerate(foot):
    svg.append(f'<text x="{PAD}" y="{756 + 30 * k}" class="ft">{line}</text>')
svg.append("</svg>")

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Jev vs Laya chart</title><style>
html,body{{margin:0;background:#1a1a19}}
svg text{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
.title{{font-size:46px;font-weight:650;fill:#ffffff;letter-spacing:-0.01em}}
.sub{{font-size:21px;fill:#c3c2b7}}
.ph{{font-size:26px;font-weight:600;fill:#ffffff}}
.pu{{font-size:18px;fill:#898781;letter-spacing:0.01em}}
.val{{font-size:22px;font-weight:600;fill:#ffffff}}
.rl{{font-size:22px;font-weight:600;fill:#ffffff}}
.rd{{font-size:17px;fill:#898781}}
.ft{{font-size:17px;fill:#898781}}
</style></head><body>{"".join(svg)}</body></html>"""
out = os.path.join(ROOT, "charts", "post2_latency_vs_throughput.html")
open(out, "w").write(html)
print(out)
print({p[0]: [round(v, 1) for v in p[2]] for p in PANELS})
