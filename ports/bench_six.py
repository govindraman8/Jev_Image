"""The realistic bulk job: 256 emails x 6 questions each (the Laya repo's own email fan-out) = 1,536 decisions.
Run each runtime with its own Python (one JSON line out):

  .venv/bin/python               ports/bench_six.py pytorch typed-decisions
  ports/mlx/.venv/bin/python     ports/bench_six.py mlx typed-decisions
  ports/coreml/.venv/bin/python  ports/bench_six.py coreml typed-decisions [batch]
  .venv/bin/python               ports/bench_six.py jev 32 [six|one]     # requests in flight; six or one question per email
"""
import json
import os
import sys
import time

PORTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PORTS)
work = json.load(open(os.path.join(PORTS, "bench_workload.json")))

# models/laya/email_utils.email_questions(), with the workload's own team question as "dept"
SIX = {
    "dept": work[0]["questions"]["dept"],
    "is_spam": {"type": "noul", "instructions": "Is this email unsolicited spam or bulk marketing?"},
    "is_phishing": {"type": "noul", "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
                    "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email"}},
    "urgency": {"type": "score", "instructions": "How urgent is the issue described in `body`?",
                "criteria": ["no time pressure", "needs attention soon", "blocking issue or hard deadline"]},
    "needs_reply": {"type": "noul", "instructions": "Does the sender expect a reply?"},
    "sentiment": {"type": "score", "instructions": "What is the sender's tone in `body`?",
                  "criteria": ["angry or very negative", "negative", "neutral", "positive"]},
}

runtime, arg = sys.argv[1], sys.argv[2]
shape = sys.argv[3] if len(sys.argv) > 3 else ("six" if runtime == "jev" else "")
questions = SIX if shape != "one" else {"dept": SIX["dept"]}
reqs = [(w["state"], questions) for w in work]
extra = {}

if runtime == "pytorch":
    sys.path.insert(0, ROOT)
    from laya_local import Laya
    laya = Laya(arg, dtype="fp16")
    run = lambda rs: laya.answer(rs)[0]
elif runtime == "mlx":
    sys.path.insert(0, os.path.join(PORTS, "mlx"))
    from laya_port import answer_many, load_agent
    agent, _ = load_agent(arg, "fp16")
    run = lambda rs: answer_many(agent, rs, batch_size=64)
elif runtime == "coreml":
    sys.path.insert(0, os.path.join(PORTS, "coreml"))
    from laya_coreml import LayaCoreML
    laya = LayaCoreML(arg, "CPU_AND_NE")
    batch = len(sys.argv) > 3 and sys.argv[3] == "batch"
    extra["coreml_mode"] = "batch" if batch else "loop"
    run = lambda rs: laya.answer(rs, batch=batch)
elif runtime == "jev":
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import requests
    sys.path.insert(0, ROOT)
    from jev_paint import MODEL, URL, load_key
    key, local, spent = load_key(), threading.local(), {"cost": 0.0, "retries": 0}

    def ask(req):
        if not hasattr(local, "s"):
            local.s = requests.Session()
        body = json.dumps({"model": MODEL, "state": req[0], "questions": req[1]})
        for attempt in range(8):
            r = local.s.post(URL, data=body, timeout=60, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            if r.status_code == 200:
                out = r.json()
                spent["cost"] += (out.get("usage") or {}).get("cost") or 0
                return out["answers"]
            if r.status_code == 429 or r.status_code >= 500:
                spent["retries"] += 1
                time.sleep(0.5 * 2 ** attempt)
                continue
            raise SystemExit(f"Jev HTTP {r.status_code}: {r.text[:300]}")
        raise SystemExit("Jev: gave up")

    def run(rs):
        with ThreadPoolExecutor(max_workers=int(arg)) as pool:
            return list(pool.map(ask, rs))
    extra.update(in_flight=int(arg), cost=spent, questions_per_email=len(questions))
else:
    raise SystemExit(f"unknown runtime {runtime}")

run(reqs[:16])  # warm-up
t0 = time.perf_counter()
answers = run(reqs)
dt = time.perf_counter() - t0
n_dec = sum(len(q) for _, q in reqs)
acc = sum(a["dept"]["choice"] == w["label"] for a, w in zip(answers, work)) / len(work)
if runtime == "jev":
    extra["cost_usd"] = round(extra.pop("cost")["cost"], 5)
    extra["retries"] = spent["retries"]
print(json.dumps({"runtime": runtime, "checkpoint_or_inflight": arg, "emails": len(work), "decisions": n_dec,
                  "seconds": round(dt, 2), "decisions_per_s": round(n_dec / dt, 1), "emails_per_s": round(len(work) / dt, 1),
                  "dept_accuracy": round(acc, 3), **extra}), flush=True)
os._exit(0)  # the CoreML runtime's workaround for a coremltools teardown crash; harmless elsewhere
