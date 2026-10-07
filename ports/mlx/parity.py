"""Parity of the laya-mlx port against the reference PyTorch fp32 answers.

  .venv/bin/python parity.py                                   # all checkpoints, fp32 + fp16
  .venv/bin/python parity.py --checkpoints english --precisions fp32,fp16,bf16 --verbose

Writes parity_results.json next to this file.
"""
import argparse
import gc
import json

import laya_port as lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default=",".join(lp.CHECKPOINTS))
    ap.add_argument("--precisions", default="fp32,fp16")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=str(lp.HERE / "parity_results.json"))
    args = ap.parse_args()
    results = {}
    for ckpt in args.checkpoints.split(","):
        for prec in args.precisions.split(","):
            agent, load_s = lp.load_agent(ckpt, prec)
            summary, answers = lp.parity(agent, ckpt)
            print(f"{ckpt:16s} {prec}: same option {summary['same_option']}/{summary['questions']}, "
                  f"max |dp| {summary['max_abs_prob_diff']:.4f}  (load {load_s:.1f}s)", flush=True)
            if args.verbose:
                for row in summary["rows"]:
                    flag = "" if row["same_option"] else "   <-- differs"
                    print(f"   req {row['request']:2d} {row['qid']:11s} port={row['port']!s:16s} "
                          f"ref={row['reference']!s:16s} |dp|={row['max_abs_prob_diff']:.4f}{flag}")
            results.setdefault(ckpt, {})[prec] = {"load_s": round(load_s, 2), **summary, "answers": answers}
            del agent
            gc.collect()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
