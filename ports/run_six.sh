#!/bin/bash
# Six questions per email on every runtime, then Jev scaling; one at a time. Appends to ports/results_six.jsonl.
cd "$(dirname "$0")/.."
OUT=ports/results_six.jsonl
: > "$OUT"
stamp() { echo "[$(date +%H:%M:%S)] $*" >&2; }
for ck in typed-decisions multilingual; do
  stamp "pytorch $ck"; .venv/bin/python ports/bench_six.py pytorch "$ck" 2>/dev/null | tail -1 >> "$OUT"
  stamp "mlx $ck";     ports/mlx/.venv/bin/python ports/bench_six.py mlx "$ck" 2>/dev/null | tail -1 >> "$OUT"
  stamp "coreml $ck";  ports/coreml/.venv/bin/python ports/bench_six.py coreml "$ck" batch 2>/dev/null | tail -1 >> "$OUT"
done
for n in 32 128; do stamp "jev six, $n in flight"; .venv/bin/python ports/bench_six.py jev "$n" six 2>/dev/null | tail -1 >> "$OUT"; done
for n in 64 128 256; do stamp "jev one, $n in flight"; .venv/bin/python ports/bench_six.py jev "$n" one 2>/dev/null | tail -1 >> "$OUT"; done
stamp "done: $(wc -l < "$OUT") results"
