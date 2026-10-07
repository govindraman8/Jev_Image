#!/bin/bash
# The fair comparison: every runtime, one at a time on a quiet machine, same 256-email job. Appends to ports/results.jsonl.
cd "$(dirname "$0")/.."
OUT=ports/results.jsonl
: > "$OUT"
stamp() { echo "[$(date +%H:%M:%S)] $*" >&2; }
for ck in multilingual english typed-decisions; do
  stamp "pytorch $ck"; .venv/bin/python ports/torch/bench.py "$ck" fp16 2>/dev/null | tail -1 >> "$OUT"
  stamp "mlx $ck";     (cd ports/mlx && .venv/bin/python bench.py --checkpoint "$ck" --precision fp16 2>/dev/null | tail -1) >> "$OUT"
  stamp "coreml $ck";  (cd ports/coreml && .venv/bin/python bench.py --ckpt "$ck" --units CPU_AND_NE --mode loop 2>/dev/null | tail -1) >> "$OUT"
done
stamp "coreml multilingual batch-api"; (cd ports/coreml && .venv/bin/python bench.py --ckpt multilingual --units CPU_AND_NE --mode batch 2>/dev/null | tail -1) >> "$OUT"
stamp "jev cloud"; .venv/bin/python ports/jev/bench.py 2>/dev/null | tail -1 >> "$OUT"
stamp "done: $(wc -l < "$OUT") results"
