"""Smoke benchmark of Laya on Core ML. Prints ONE JSON line.

  .venv/bin/python bench.py                                        # multilingual, CPU_AND_NE, looped workload
  .venv/bin/python bench.py --ckpt english --units ALL --mode batch

Loads the checkpoint's Core ML buckets once (load_s: tokenizer + compiled models; the coremltools import is
reported separately as import_s), warms up, then measures
  (a) single-question latency, p50/p95 over --calls calls (default 50) of parity request 0's "department"
      question, end to end: tokenize (no state cache) + build inputs + predict + calibrated decode;
  (b) throughput over ports/bench_workload.json (256 email-sorting requests, one 4-option choice each):
      batch_qps = 256 / wall seconds for the whole workload (load and warm-up excluded), and
      workload_accuracy = share of "dept" choices equal to "label". --mode loop runs one predict per question;
      --mode batch hands each bucket's questions to Core ML's batch API (model.predict(list of dicts));
  and parity on ports/parity_questions.json against ports/parity_reference.json.
The models take one question per prediction (fixed [1, L] shapes), so "batch" means Core ML batch
prediction, not a batched tensor.
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

_t = time.perf_counter()
import coremltools  # noqa: E402,F401  (imports torch too when it is installed: seconds, so timed separately)

IMPORT_S = time.perf_counter() - _t

from laya_coreml import HERE, LayaCoreML, parity  # noqa: E402

PORTS = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="multilingual", choices=["english", "multilingual", "typed-decisions"])
    ap.add_argument("--units", default="CPU_AND_NE", choices=["CPU_AND_NE", "ALL", "CPU_AND_GPU", "CPU_ONLY"])
    ap.add_argument("--mode", default="loop", choices=["loop", "batch"])
    ap.add_argument("--lengths", default="", help="bucket lengths to load, e.g. 128,256 (default: all found)")
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")] if args.lengths else None

    t0 = time.perf_counter()
    laya = LayaCoreML(args.ckpt, args.units, lengths=lengths)
    load_s = time.perf_counter() - t0

    with open(os.path.join(PORTS, "parity_questions.json")) as f:
        req0 = json.load(f)[0]
    single = [(req0["state"], {"department": req0["questions"]["department"]})]
    with open(os.path.join(PORTS, "bench_workload.json")) as f:
        workload = json.load(f)
    requests = [(r["state"], r["questions"]) for r in workload]

    # warm-up: every loaded bucket, the single question, and a slice of the workload in the timed mode
    for L in laya.lengths:
        laya.models[L].predict(laya.arrays([laya.tok.cls_token_id], [0], 0, L))
    for _ in range(args.warmup):
        laya.answer(single)
    laya.answer(requests[:16], batch=args.mode == "batch")

    # (a) single-question latency, end to end, no tokenizer cache
    laya.cache_states = False
    times = []
    for _ in range(args.calls):
        t = time.perf_counter()
        laya.answer(single)
        times.append((time.perf_counter() - t) * 1000.0)
    laya.cache_states = True

    # (b) the 256-request workload
    laya._state_cache.clear()
    t = time.perf_counter()
    answers = laya.answer(requests, batch=args.mode == "batch")
    wall = time.perf_counter() - t
    correct = sum(a["dept"]["choice"] == r["label"] for a, r in zip(answers, workload))

    par = parity(laya)
    print(json.dumps({
        "runtime": "coreml",
        "checkpoint": args.ckpt,
        "precision": laya.precision,
        "compute_units": args.units,
        "load_s": round(load_s, 3),
        "single_ms_p50": round(float(np.percentile(times, 50)), 3),
        "single_ms_p95": round(float(np.percentile(times, 95)), 3),
        "batch_qps": round(len(requests) / wall, 1),
        "workload_accuracy": round(correct / len(workload), 4),
        "parity": {"same_option": "%d/%d" % (par["same"], par["total"]), "max_abs_prob_diff": par["max_abs_prob_diff"]},
        "batch_mode": args.mode,
        "buckets": laya.lengths,
        "import_s": round(IMPORT_S, 3),
    }))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip interpreter teardown; see laya_coreml._KEEPALIVE
