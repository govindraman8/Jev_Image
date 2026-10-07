"""Jev (TypeSafe, via OpenRouter) on the same benchmark as the Laya ports: one-question latency, then the
256-email bulk job with 1, 8 and 32 requests in flight. Costs well under a cent per run.

  .venv/bin/python ports/jev/bench.py
"""
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PORTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PORTS))
from jev_paint import MODEL, URL, load_key  # noqa: E402

KEY = load_key()
parity_q = json.load(open(os.path.join(PORTS, "parity_questions.json")))
work = json.load(open(os.path.join(PORTS, "bench_workload.json")))
local = __import__("threading").local()
spent = {"cost": 0.0, "requests": 0}


def ask(state, questions):
    if not hasattr(local, "s"):
        local.s = requests.Session()
    body = json.dumps({"model": MODEL, "state": state, "questions": questions})
    for attempt in range(6):
        r = local.s.post(URL, data=body, timeout=60, headers={"Authorization": f"Bearer {KEY}",
                         "Content-Type": "application/json", "X-OpenRouter-Title": "Jev Paints bench"})
        if r.status_code == 200:
            out = r.json()
            spent["cost"] += (out.get("usage") or {}).get("cost") or 0
            spent["requests"] += 1
            return out["answers"]
        if r.status_code in (429,) or r.status_code >= 500:
            time.sleep(0.5 * 2 ** attempt)
            continue
        raise SystemExit(f"Jev HTTP {r.status_code}: {r.text[:300]}")
    raise SystemExit("Jev: gave up after retries")


req0 = (parity_q[0]["state"], {"department": parity_q[0]["questions"]["department"]})
for _ in range(3):
    ask(*req0)
lat = []
for _ in range(30):
    t = time.perf_counter()
    ask(*req0)
    lat.append((time.perf_counter() - t) * 1000)
lat.sort()

result = {"runtime": "jev-cloud (OpenRouter)", "checkpoint": "typesafe/jev-1.13", "precision": "n/a",
          "single_ms_p50": round(statistics.median(lat), 1), "single_ms_p95": round(lat[int(0.95 * len(lat)) - 1], 1)}
for inflight in (1, 8, 32):
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=inflight) as pool:
        answers = list(pool.map(lambda w: ask(w["state"], w["questions"]), work))
    dt = time.perf_counter() - t
    result[f"batch_qps_{inflight}_in_flight"] = round(len(work) / dt, 1)
    result["workload_accuracy"] = round(sum(a["dept"]["choice"] == w["label"] for a, w in zip(answers, work)) / len(work), 3)
result["cost_usd"] = round(spent["cost"], 5)
result["requests"] = spent["requests"]
print(json.dumps(result))
