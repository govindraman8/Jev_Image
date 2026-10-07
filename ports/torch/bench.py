"""Reference runtime benchmark: the repo's own PyTorch code on the Apple GPU (via laya_local), same output
format as ports/mlx/bench.py and ports/coreml/bench.py.

  .venv/bin/python ports/torch/bench.py [english|multilingual|typed-decisions] [fp16|fp32]
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PORTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PORTS))
from laya_local import Laya  # noqa: E402

ckpt = sys.argv[1] if len(sys.argv) > 1 else "multilingual"
precision = sys.argv[2] if len(sys.argv) > 2 else "fp16"
parity_q = json.load(open(os.path.join(PORTS, "parity_questions.json")))
reference = json.load(open(os.path.join(PORTS, "parity_reference.json")))[ckpt]
work = json.load(open(os.path.join(PORTS, "bench_workload.json")))

t0 = time.time()
laya = Laya(ckpt, dtype=precision)
load_s = time.time() - t0

req0 = (parity_q[0]["state"], {"department": parity_q[0]["questions"]["department"]})
for _ in range(5):
    laya.answer([req0])
lat = []
for _ in range(50):
    t = time.perf_counter()
    laya.answer([req0])
    lat.append((time.perf_counter() - t) * 1000)
lat.sort()

reqs = [(w["state"], w["questions"]) for w in work]
laya.answer(reqs[:32])
t = time.perf_counter()
out, _ = laya.answer(reqs)
dt = time.perf_counter() - t
acc = sum(o["dept"]["choice"] == w["label"] for o, w in zip(out, work)) / len(work)

mine, _ = laya.answer([(p["state"], p["questions"]) for p in parity_q])
same = total = 0
maxdiff = 0.0
for got, ref in zip(mine, reference):
    for qid, a in ref.items():
        total += 1
        b = got[qid]
        if a["type"] == "noul":
            same += (a["noul"] >= 0.5) == (b["noul"] >= 0.5)
            maxdiff = max(maxdiff, abs(a["noul"] - b["noul"]))
        else:
            same += a.get("choice", None) == b.get("choice", None) if a["type"] == "choice" else \
                max(a["probabilities"], key=a["probabilities"].get) == max(b["probabilities"], key=b["probabilities"].get)
            maxdiff = max(maxdiff, max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]))
print(json.dumps({"runtime": "pytorch-mps", "checkpoint": ckpt, "precision": precision, "load_s": round(load_s, 1),
                  "single_ms_p50": round(statistics.median(lat), 1), "single_ms_p95": round(lat[int(0.95 * len(lat)) - 1], 1),
                  "batch_qps": round(len(work) / dt, 1), "workload_accuracy": round(acc, 3),
                  "parity": f"{same}/{total} same, max prob diff {maxdiff:.4f}"}))
