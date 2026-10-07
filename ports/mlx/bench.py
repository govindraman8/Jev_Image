"""Smoke benchmark of the laya-mlx port on one checkpoint and precision; prints ONE JSON line.

  .venv/bin/python bench.py --checkpoint english --precision fp16
  .venv/bin/python bench.py --checkpoint multilingual --precision fp32 --batch-size 32

Loads the local checkpoint once, checks parity against ports/parity_reference.json, warms up,
then measures:
  (a) single-question latency: Agent.system_one on the README email "department" question
      (parity_questions.json request 0), --single-n calls, median and p95 in ms;
  (b) throughput on ports/bench_workload.json (256 one-question email-sorting requests):
      questions from different requests share forward passes of --batch-size (the port's
      prepare/collate_items/forward); batch_qps = questions / wall seconds for the whole
      workload (tokenize + forward + decode; load and warm-up excluded), plus the share of
      dept choices equal to the label. --batch-mode per-request calls Agent.system_one once
      per request instead (the port's stock API, which batches only within a request).
Diagnostics go to stderr; stdout carries only the JSON line.
"""
import argparse
import json
import sys
import time

import numpy as np

import laya_port as lp


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", choices=list(lp.CHECKPOINTS), default="english")
    ap.add_argument("--precision", choices=list(lp.PRECISIONS), default="fp16")
    ap.add_argument("--batch-size", type=int, default=64, help="questions per forward pass for (b)")
    ap.add_argument("--batch-mode", choices=("cross", "per-request"), default="cross")
    ap.add_argument("--single-n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--compile", action="store_true", help="laya-mlx opt-in mx.compile of the model")
    ap.add_argument("--pad-to-multiple", type=int, default=None, help="laya-mlx opt-in padding bucket")
    args = ap.parse_args()

    import mlx.core as mx

    import laya_mlx

    agent, load_s = lp.load_agent(args.checkpoint, args.precision, device=args.device,
                                  batch_size=args.batch_size, compile=args.compile,
                                  pad_to_multiple=args.pad_to_multiple)
    log(f"loaded {args.checkpoint}/{args.precision} in {load_s:.2f}s on {agent.device}")

    # parity (untimed): the port's own system_one, and the cross-request batched path used in (b)
    par, _ = lp.parity(agent, args.checkpoint)
    parity_requests = lp.load_json(lp.PARITY_QUESTIONS)
    batched = lp.answer_many(agent, [(r["state"], r["questions"]) for r in parity_requests], args.batch_size)
    par_b = lp.compare(batched, lp.load_json(lp.PARITY_REFERENCE)[args.checkpoint])
    parity = {"same_option": par["same_option"], "questions": par["questions"],
              "max_abs_prob_diff": par["max_abs_prob_diff"],
              "batched_same_option": par_b["same_option"], "batched_max_abs_prob_diff": par_b["max_abs_prob_diff"]}
    log(f"parity: {parity}")

    # (a) single question: README email, department only
    state0 = parity_requests[0]["state"]
    q0 = {"department": parity_requests[0]["questions"]["department"]}
    for _ in range(args.warmup):
        agent.system_one(state0, q0)
    samples = []
    for _ in range(args.single_n):
        t0 = time.perf_counter()
        agent.system_one(state0, q0)  # returns after mx.eval + host copy, so this is completed work
        samples.append((time.perf_counter() - t0) * 1000)
    log(f"single: p50 {np.median(samples):.2f} ms, p95 {np.percentile(samples, 95):.2f} ms, "
        f"min {min(samples):.2f}, max {max(samples):.2f}")

    # (b) bulk workload
    workload = lp.load_json(lp.WORKLOAD)
    requests = [(w["state"], w["questions"]) for w in workload]
    n_questions = sum(len(q) for _, q in requests)

    def run(reqs):
        if args.batch_mode == "cross":
            return lp.answer_many(agent, reqs, args.batch_size)
        return [agent.system_one(s, q)["answers"] for s, q in reqs]

    run(requests[:args.batch_size])  # warm-up at the measured batch shape
    mx.synchronize()
    t0 = time.perf_counter()
    answers = run(requests)
    mx.synchronize()
    wall = time.perf_counter() - t0
    correct = [a["dept"]["choice"] == w["label"] for a, w in zip(answers, workload)]
    log(f"workload: {n_questions} questions in {wall:.2f}s ({args.batch_mode}, batch "
        f"{args.batch_size if args.batch_mode == 'cross' else 1}), "
        f"accuracy {np.mean(correct):.3f}")

    options = [o for o, on in (("cpu", args.device == "cpu"), ("compile", args.compile),
                                ("pad%s" % args.pad_to_multiple, args.pad_to_multiple),
                                ("per-request", args.batch_mode == "per-request")) if on]
    runtime = f"laya-mlx {laya_mlx.__version__} / mlx {mx.__version__}"
    print(json.dumps({
        "runtime": runtime + (" [%s]" % ", ".join(options) if options else ""),
        "checkpoint": args.checkpoint,
        "precision": args.precision,
        "load_s": round(load_s, 3),
        "single_ms_p50": round(float(np.median(samples)), 2),
        "single_ms_p95": round(float(np.percentile(samples, 95)), 2),
        "batch_qps": round(n_questions / wall, 1),
        "workload_accuracy": round(float(np.mean(correct)), 4),
        "parity": parity,
        "batch_size": args.batch_size if args.batch_mode == "cross" else 1,
    }))


if __name__ == "__main__":
    main()
