"""Convert a local, SHA-verified Laya checkpoint to a fixed-shape FP16 Core ML bucket.

  .venv/bin/python convert.py --ckpt typed-decisions --length 128

Same recipe as FluidInference/mobius convert-coreml.py, which built the published multilingual and English
buckets: the vendored export_model.LayaExport adapter, an ML program for iOS17/macOS14, FLOAT16 compute
precision, 32 option slots, inputs input_ids/attention_mask int32 [1, L], marker_map float32 [1, 32, L],
question_type float32 [1, 3]. Differences: weights come from models/laya/<ckpt>/ through the repo's own
rl_common.build_model (no laya PyPI package), and before converting, the FP32 adapter is checked against
the reference DecisionModel on every parity question that fits the bucket.

Writes models/<ckpt>/laya_<ckpt>_fp16_L<L>_options32.{mlpackage,mlmodelc} and a .conversion.json manifest.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

sys.dont_write_bytecode = True  # never drop __pycache__ into models/laya/

import numpy as np  # noqa: E402
import torch  # noqa: E402

from laya_coreml import (CHECKPOINTS, LAYA_DIR, MAX_OPTIONS, MODELS_DIR, QTYPES, Tok, build_sequence,  # noqa: E402
                         internal, render_options)

sys.path.insert(0, LAYA_DIR)
from rl_common import build_model, collate_items  # noqa: E402

from export_model import LayaExport  # noqa: E402

# models/laya/download_manifest.json; identical to the LFS hashes of convaiinnovations/laya at
# 1c5edc17a7acd8701df6fc341c0d179f1c62c982, the revision the FluidInference buckets were converted from.
WEIGHTS_SHA256 = {
    "english": "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
    "multilingual": "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
    "typed-decisions": "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="typed-decisions", choices=sorted(CHECKPOINTS))
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    import coremltools as ct
    from safetensors.torch import load_file

    torch.set_num_threads(args.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    src = os.path.join(LAYA_DIR, CHECKPOINTS[args.ckpt])
    weights = os.path.join(src, "model.safetensors")
    t0 = time.time()
    digest = sha256(weights)
    if digest != WEIGHTS_SHA256[args.ckpt]:
        raise SystemExit("%s sha256 %s != expected %s" % (weights, digest, WEIGHTS_SHA256[args.ckpt]))
    print("weights sha256 ok (%s) in %.1fs" % (digest[:12], time.time() - t0), flush=True)

    with open(os.path.join(src, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    model = build_model(cfg, encoder_dir=os.path.join(src, "encoder"))
    model.load_state_dict(load_file(weights), strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    tok = Tok(os.path.join(src, "tokenizer"))
    L = args.length
    export = LayaExport(model, L, MAX_OPTIONS).eval()

    # FP32 adapter vs reference DecisionModel on every parity question that fits this bucket
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parity_questions.json")) as f:
        reqs = json.load(f)
    worst, checked, example = 0.0, 0, None
    for r in reqs:
        for qid, qdef in r["questions"].items():
            q = internal(qdef)
            ids, markers = build_sequence(tok, r["state"], q, cfg["max_len"], cfg["head_max_len"])
            if len(ids) > L or len(markers) != len(render_options(q)):
                continue
            qtype = QTYPES[q["t"]]
            b = collate_items([[{"ids": ids, "markers": markers, "qtype": qtype, "target": [0.0] * len(markers),
                                 "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1, "src": "api"}]], tok.pad_token_id)
            arrays = {
                "input_ids": np.full((1, L), tok.pad_token_id, dtype=np.int32),
                "attention_mask": np.zeros((1, L), dtype=np.int32),
                "marker_map": np.zeros((1, MAX_OPTIONS, L), dtype=np.float32),
                "question_type": np.zeros((1, 3), dtype=np.float32),
            }
            arrays["input_ids"][0, :len(ids)] = ids
            arrays["attention_mask"][0, :len(ids)] = 1
            arrays["marker_map"][0, np.arange(len(markers)), markers] = 1.0
            arrays["question_type"][0, qtype] = 1.0
            inputs = tuple(torch.from_numpy(arrays[k]) for k in ("input_ids", "attention_mask", "marker_map", "question_type"))
            with torch.no_grad():
                ref_logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
                logits, _, _ = export(*inputs)
            k = len(markers)
            worst = max(worst, float((ref_logits[0, :k] - logits[0, :k]).abs().max()))
            checked += 1
            example = example or inputs
    print("fp32 adapter vs reference DecisionModel: %d questions, max |logit diff| %.2e" % (checked, worst), flush=True)
    if worst > 1e-3:
        raise SystemExit("export adapter does not match the reference model; not converting")

    with torch.no_grad():
        traced = torch.jit.trace(export, example)
    started = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, L), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, L), dtype=np.int32),
            ct.TensorType(name="marker_map", shape=(1, MAX_OPTIONS, L), dtype=np.float32),
            ct.TensorType(name="question_type", shape=(1, 3), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32), ct.TensorType(name="probabilities", dtype=np.float32),
                 ct.TensorType(name="action_probabilities", dtype=np.float32)],
        skip_model_load=True,
    )
    mlmodel.short_description = "laya %s: typed decision scoring (choice/score/noul), %d tokens, %d option slots" % (
        args.ckpt, L, MAX_OPTIONS)
    mlmodel.author = "Convai Innovations (weights); conversion recipe FluidInference/mobius; converted locally"
    mlmodel.license = "Apache-2.0"
    mlmodel.user_defined_metadata.update({
        "source_repo": "convaiinnovations/laya", "source_revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
        "source_subfolder": CHECKPOINTS[args.ckpt], "weights_sha256": digest, "length": str(L),
        "max_options": str(MAX_OPTIONS), "head_max_len": str(cfg["head_max_len"]),
        "temperature": json.dumps(cfg.get("temperature", [1.0, 1.0, 1.0])),
        "temperature_by_options": json.dumps(cfg.get("temperature_by_options", {})),
        "sequence_format": "[CLS] <type> question: instructions [SEP] ([MASK] option)* [SEP] state [SEP]",
    })
    out_dir = os.path.join(MODELS_DIR, args.ckpt)
    os.makedirs(out_dir, exist_ok=True)
    name = "laya_%s_fp16_L%d_options%d" % (args.ckpt, L, MAX_OPTIONS)
    package = os.path.join(out_dir, name + ".mlpackage")
    mlmodel.save(package)
    convert_s = time.time() - started
    compiled = os.path.join(out_dir, name + ".mlmodelc")
    ct.utils.compile_model(package, destination_path=compiled)
    files = {}
    for root in (package, compiled):
        for dirpath, _, names in os.walk(root):
            for n in sorted(names):
                p = os.path.join(dirpath, n)
                files[os.path.relpath(p, out_dir)] = sha256(p)
    manifest = {
        "model": name, "checkpoint": args.ckpt, "precision": "float16", "length": L, "max_options": MAX_OPTIONS,
        "head_max_len": cfg["head_max_len"], "parameters": n_params, "weights_sha256": digest,
        "fp32_adapter_max_logit_diff": worst, "fp32_adapter_questions": checked,
        "convert_seconds": round(convert_s, 1), "python": platform.python_version(), "torch": torch.__version__,
        "coremltools": ct.__version__, "recipe": "FluidInference/mobius@2696c9e8 models/computer-use/laya/coreml",
        "files_sha256": files,
    }
    with open(os.path.join(out_dir, name + ".conversion.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("saved %s (%s parameters, converted in %.0fs)" % (package, format(n_params, ","), convert_s), flush=True)


if __name__ == "__main__":
    main()
