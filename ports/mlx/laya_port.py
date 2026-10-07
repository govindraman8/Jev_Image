"""Shared helpers for checking the laya-mlx port against the local Laya checkpoints.

Loads the SHA-verified local weights in models/laya/ (no conversion, no download), answers
Jev-shaped requests, batches questions across requests, and scores parity against
ports/parity_reference.json (reference PyTorch fp32, calibrated probabilities).
"""
import os

# Local weights only: forbid Hub network access and telemetry before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # .../Jev_Image
LAYA = ROOT / "models" / "laya"
CHECKPOINTS = {
    "english": LAYA,
    "multilingual": LAYA / "multilingual",
    "typed-decisions": LAYA / "typed-decisions",
}
PRECISIONS = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
PARITY_QUESTIONS = ROOT / "ports" / "parity_questions.json"
PARITY_REFERENCE = ROOT / "ports" / "parity_reference.json"
WORKLOAD = ROOT / "ports" / "bench_workload.json"


def load_agent(checkpoint, precision, **kwargs):
    """Build a laya_mlx.Agent on the local checkpoint; returns (agent, load seconds)."""
    import mlx.core as mx
    from laya_mlx import Agent

    t0 = time.perf_counter()
    agent = Agent(str(CHECKPOINTS[checkpoint]), dtype=PRECISIONS[precision], **kwargs)
    mx.synchronize()
    return agent, time.perf_counter() - t0


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- parity
def picked(answer):
    """The option an answer selects: choice label, argmax score level, or noul true/false."""
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "score":
        probs = answer["probabilities"]
        return max(probs, key=lambda k: probs[k])
    return answer["noul"] >= 0.5


def prob_vector(answer):
    if answer["type"] == "noul":
        return {"false": 1.0 - answer["noul"], "true": answer["noul"]}
    return answer["probabilities"]


def compare(answers, reference):
    """answers/reference: [{qid: answer}] per request -> parity summary."""
    same, total, max_diff, rows = 0, 0, 0.0, []
    for r, (mine, ref) in enumerate(zip(answers, reference)):
        for qid, ref_answer in ref.items():
            ans = mine[qid]
            p, q = prob_vector(ans), prob_vector(ref_answer)
            diff = max(abs(float(p[k]) - float(q[k])) for k in q)
            ok = picked(ans) == picked(ref_answer)
            same += ok
            total += 1
            max_diff = max(max_diff, diff)
            rows.append({"request": r, "qid": qid, "same_option": ok, "max_abs_prob_diff": round(diff, 6),
                         "port": picked(ans), "reference": picked(ref_answer)})
    return {"same_option": same, "questions": total, "max_abs_prob_diff": round(max_diff, 6), "rows": rows}


def parity(agent, checkpoint):
    """Answer every parity request with Agent.system_one (the port's own API) and compare."""
    requests = load_json(PARITY_QUESTIONS)
    reference = load_json(PARITY_REFERENCE)[checkpoint]
    answers = [agent.system_one(req["state"], req["questions"])["answers"] for req in requests]
    return compare(answers, reference), answers


# ----------------------------------------------------------------------------- cross-request batching
def decode(agent, q, logits):
    """Calibrated Jev-shaped answer from one row of logits, exactly as Agent.system_one does it."""
    from laya_mlx.common import QTYPES, confidence_from_probs, temp_bucket

    k, qt = len(logits), QTYPES[q["t"]]
    z = logits / agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
    p = np.exp(z - z.max())
    p /= p.sum()
    if q["t"] == "choice":
        labels = list(q["crit"])
        return {"type": "choice", "choice": labels[int(p.argmax())],
                "probabilities": {label: round(float(v), 4) for label, v in zip(labels, p)},
                "confidence": round(confidence_from_probs(p, k), 4)}
    if q["t"] == "score":
        return {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": round(confidence_from_probs(p, k), 4)}
    return {"type": "noul", "noul": round(float(p[1]), 4)}


def answer_many(agent, requests, batch_size=64, sort_by_length=True):
    """Answer [(state, questions)] with questions from different requests sharing forward passes.

    Uses the port's own Agent.prepare / collate_items / Agent.forward; system_one only batches
    questions within one request, so bulk single-question requests need this to fill the GPU.
    """
    from laya_mlx.agent import collate_items

    items = []  # (request index, qid, internal question, prepared item)
    for r, (state, questions) in enumerate(requests):
        prepared, internal = agent.prepare(state, questions)
        for qid, item, q in zip(questions, prepared, internal):
            items.append((r, qid, q, item))
    order = list(range(len(items)))
    if sort_by_length:
        order.sort(key=lambda i: len(items[i][3]["ids"]))
    out = [dict() for _ in requests]
    max_len = agent.cfg.get("max_len", 512)
    for start in range(0, len(order), batch_size):
        chunk = [items[i] for i in order[start:start + batch_size]]
        batch = collate_items([it[3] for it in chunk], agent.tok.pad_token_id,
                              pad_to_multiple=agent.pad_to_multiple, max_length=max_len)
        logits, _act = agent.forward(batch)
        logits = np.asarray(logits)
        if not np.isfinite(logits).all():
            raise FloatingPointError("Non-finite model outputs; retry with fp32")
        for row, (r, qid, q, item) in enumerate(chunk):
            out[r][qid] = decode(agent, q, logits[row, :len(item["markers"])])
    return out
