"""Where Core ML runs each op of a Laya bucket: MLComputePlan preferred device and estimated cost per op.

  .venv/bin/python placement.py                              # every bucket found, CPU_AND_NE and ALL
  .venv/bin/python placement.py --ckpt english --units CPU_AND_NE

For each (bucket, compute units) prints non-const op counts per preferred device, the estimated-cost share per
device, and every op that is not on the preferred accelerator with its type and supported devices.
Writes reports/placement.json. This is Core ML's plan for the loaded model on this Mac (what the
coreml-cli / Xcode performance report shows), not a hardware trace.
"""
import argparse
import json
import os
from collections import Counter

from coremltools.models.compute_plan import MLComputePlan

from laya_coreml import CHECKPOINTS, HERE, compute_units, find_buckets

DEVICE = {"MLNeuralEngineComputeDevice": "ANE", "MLGPUComputeDevice": "GPU", "MLCPUComputeDevice": "CPU"}


def placement(path, units):
    plan = MLComputePlan.load_from_path(path, compute_units(units))
    ops = plan.model_structure.program.functions["main"].block.operations
    counts, cost, off = Counter(), Counter(), []
    for op in ops:
        if op.operator_name == "const":
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = DEVICE.get(type(usage.preferred_compute_device).__name__, "?") if usage else "unknown"
        est = plan.get_estimated_cost_for_mlprogram_operation(op)
        counts[dev] += 1
        cost[dev] += est.weight if est else 0.0
        if dev == "CPU" or dev == "unknown":
            off.append({"op": op.operator_name, "output": op.outputs[0].name if op.outputs else "",
                        "device": dev, "cost": round(est.weight, 5) if est else None,
                        "supported": sorted(DEVICE.get(type(d).__name__, "?") for d in usage.supported_compute_devices)
                        if usage else []})
    total = sum(counts.values())
    total_cost = sum(cost.values()) or 1.0
    return {"ops": total, "ops_by_device": dict(counts),
            "ops_share": {d: round(100.0 * n / total, 1) for d, n in counts.items()},
            "estimated_cost_share": {d: round(100.0 * c / total_cost, 1) for d, c in cost.items()},
            "cpu_ops": off}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=",".join(CHECKPOINTS))
    ap.add_argument("--units", default="CPU_AND_NE,ALL")
    args = ap.parse_args()
    report = {}
    for ckpt in args.ckpt.split(","):
        try:
            buckets = find_buckets(ckpt)
        except FileNotFoundError:
            continue
        for length, path in buckets.items():
            for units in args.units.split(","):
                r = placement(path, units)
                report["%s L%d %s" % (ckpt, length, units)] = r
                print("%-16s L%-4d %-10s ops %d by device %s | est. cost share %% %s | CPU ops: %s" % (
                    ckpt, length, units, r["ops"], r["ops_by_device"], r["estimated_cost_share"],
                    ", ".join("%s(%s)" % (o["op"], o["output"]) for o in r["cpu_ops"])), flush=True)
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    with open(os.path.join(HERE, "reports", "placement.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
