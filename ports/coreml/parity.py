"""Core ML vs the PyTorch fp32 reference on ports/parity_questions.json.

  .venv/bin/python parity.py                                  # every checkpoint with buckets, CPU_AND_NE
  .venv/bin/python parity.py --ckpt multilingual --units CPU_AND_NE,ALL

First checks the runtime's sequence builder (tokenizers backend) token-for-token against the repo's
rl_common.build_sequence (transformers tokenizer) on every parity and bench_workload question, then answers
all 14 parity questions and compares with ports/parity_reference.json: same-option count and max absolute
probability difference (the reference is rounded to 4 decimals, so differences under 5e-5 are rounding).
Writes reports/parity-<ckpt>-<units>.json.
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True  # never drop __pycache__ into models/laya/

from laya_coreml import CHECKPOINTS, HERE, LAYA_DIR, LayaCoreML, Tok, build_sequence, find_buckets, internal, parity  # noqa: E402

PORTS = os.path.dirname(HERE)


def check_sequences(ckpt):
    """-> (identical, total) comparing our builder with rl_common.build_sequence + AutoTokenizer."""
    sys.path.insert(0, LAYA_DIR)
    import rl_common
    from transformers import AutoTokenizer
    src = os.path.join(LAYA_DIR, CHECKPOINTS[ckpt])
    with open(os.path.join(src, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    hf = AutoTokenizer.from_pretrained(os.path.join(src, "tokenizer"))
    tok = Tok(os.path.join(src, "tokenizer"))
    assert (tok.cls_token_id, tok.sep_token_id, tok.mask_token_id, tok.pad_token_id) == \
        (hf.cls_token_id, hf.sep_token_id, hf.mask_token_id, hf.pad_token_id), "special token ids differ"
    reqs = []
    for name in ("parity_questions.json", "bench_workload.json"):
        with open(os.path.join(PORTS, name)) as f:
            reqs += json.load(f)
    same = total = 0
    for r in reqs:
        for qdef in r["questions"].values():
            q = internal(qdef)
            for max_len in (cfg["max_len"], 64):  # also a truncating length
                ref = rl_common.build_sequence(hf, r["state"], q, max_len, cfg["head_max_len"])
                mine = build_sequence(tok, r["state"], q, max_len, cfg["head_max_len"])
                same += (list(ref[0]), list(ref[1])) == (list(mine[0]), list(mine[1]))
                total += 1
    return same, total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--units", default="CPU_AND_NE")
    ap.add_argument("--no-seq-check", action="store_true")
    args = ap.parse_args()
    ckpts = args.ckpt.split(",") if args.ckpt else [c for c in CHECKPOINTS if _has_buckets(c)]
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    for ckpt in ckpts:
        if not args.no_seq_check:
            same, total = check_sequences(ckpt)
            print("[%s] sequence builder vs rl_common.build_sequence: %d/%d identical" % (ckpt, same, total), flush=True)
            if same != total:
                raise SystemExit("sequence mismatch: fix the builder before trusting parity")
        for units in args.units.split(","):
            t0 = time.time()
            laya = LayaCoreML(ckpt, units)
            load_s = time.time() - t0
            res = parity(laya)
            used = {}
            for r in json.load(open(os.path.join(PORTS, "parity_questions.json"))):
                for qdef in r["questions"].values():
                    L = laya.encode(r["state"], qdef)[3]
                    used[L] = used.get(L, 0) + 1
            res.update({"checkpoint": ckpt, "compute_units": units, "buckets": {str(k): v for k, v in laya.paths.items()},
                        "questions_per_bucket": {str(k): v for k, v in sorted(used.items())}, "load_s": round(load_s, 2)})
            print("[%s %s] same option %d/%d, max |dprob| %.4f, questions per bucket %s, load %.1fs" % (
                ckpt, units, res["same"], res["total"], res["max_abs_prob_diff"], res["questions_per_bucket"], load_s),
                flush=True)
            for row in res["rows"]:
                if not row["same_option"] or row["max_abs_prob_diff"] > 0.02:
                    print("   request %d %s: coreml %s vs reference %s" % (
                        row["request"], row["question"], json.dumps(row["coreml"]), json.dumps(row["reference"])))
            with open(os.path.join(HERE, "reports", "parity-%s-%s.json" % (ckpt, units)), "w") as f:
                json.dump(res, f, indent=1)


def _has_buckets(ckpt):
    try:
        return bool(find_buckets(ckpt))
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip interpreter teardown; see laya_coreml._KEEPALIVE
