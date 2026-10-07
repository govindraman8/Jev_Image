"""Laya typed decisions on Core ML (Apple Neural Engine / GPU / CPU), driven from Python.

Needs only coremltools, tokenizers and numpy at run time (no torch, no transformers).

  from laya_coreml import LayaCoreML
  laya = LayaCoreML("multilingual", compute_units="CPU_AND_NE")
  answers = laya.answer([(state, questions), ...])        # Jev request shape in, Jev answer shape out

Models are fixed-shape, one question per prediction: input_ids/attention_mask int32 [1, L],
marker_map float32 [1, 32, L] (one-hot [MASK] position per option), question_type float32 [1, 3];
outputs raw option logits [1, 32]. Buckets of several lengths L can be loaded; each question runs on the
smallest bucket that holds its sequence, and only a sequence longer than the largest bucket has its state
cut on the right (as laya does at max_len). Calibration (temperature_by_options) is applied here on the
host, exactly as models/laya/rl_agent_api.py does.

The sequence builder below is a line-for-line copy of models/laya/rl_common.build_sequence; parity.py
checks it token-for-token against the original (transformers tokenizer) before any parity numbers count.
"""
import glob
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
LAYA_DIR = os.path.join(PROJECT, "models", "laya")
MODELS_DIR = os.path.join(HERE, "models")
CHECKPOINTS = {"english": "", "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
MAX_OPTIONS = 32

# Every input buffer handed to Core ML, and every loaded model, stays referenced here for the life of the process.
# coremltools 9.0 wraps each numpy input in a PybindCompatibleArray that owns a py::array; on macOS 26 Core ML's
# E5 runtime keeps the last inputs "lingering" and releases them later on its own dispatch queue
# (MLE5ExecutionStream resetAfterLingering), which drops the py::array without the GIL. If that was the last
# reference, numpy memory is freed off-thread while Python runs: we hit exactly this segfault
# (_PyObject_Free <- MLFeatureValue dealloc <- resetAfterLingering). Reusing persistent buffers that we always
# hold (two references each) means Core ML's late release never frees anything. Scripts end with os._exit()
# for the same reason, so interpreter teardown cannot race a late release either.
_KEEPALIVE = []


# ----------------------------------------------------------------------------- rendering (copied from rl_common)
def serialize_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(q):
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def internal(qdef):
    """Jev question -> internal form (same conversion as rl_agent_api._to_internal)."""
    crit = qdef.get("criteria")
    if qdef["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"] if isinstance(qdef["instructions"], str) else json.dumps(qdef["instructions"])
    return {"t": qdef["type"], "ins": ins, "crit": crit}


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def confidence_from_probs(p, k: int) -> float:
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return float(1 - ent / math.log(k))


class Tok:
    """The subset of the HF tokenizer interface build_sequence uses, on the raw `tokenizers` backend."""

    def __init__(self, tok_dir):
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
        with open(os.path.join(tok_dir, "tokenizer_config.json")) as f:
            cfg = json.load(f)
        self.mask_token = cfg["mask_token"]
        ids = {}
        for name in ("cls", "sep", "mask", "pad"):
            tid = self.backend.token_to_id(cfg[name + "_token"])
            if tid is None:
                raise ValueError("%s_token %r not in vocabulary" % (name, cfg[name + "_token"]))
            ids[name] = tid
        self.cls_token_id, self.sep_token_id = ids["cls"], ids["sep"]
        self.mask_token_id, self.pad_token_id = ids["mask"], ids["pad"]

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids


def build_sequence(tok, state, q, max_len, head_max_len, state_ids=None, return_cut=False):
    """[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP] (rl_common.build_sequence).

    state_ids: pre-tokenized state (same tokens the function would compute). return_cut: also report whether
    the state was cut to fit max_len.
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok.encode("%s question: %s" % (q["t"], ins))
    opt_ids = [[tok.mask_token_id] + tok.encode(" " + o.replace(mask_tok, " "))[:48] for o in opts]
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:  # too many / too long options: shrink every option text evenly
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[:max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = state_ids if state_ids is not None else tok.encode(serialize_state(state).replace(mask_tok, " "))
    cut = len(st) > room
    ids = ids + st[:room] + [tok.sep_token_id]
    if return_cut:
        return ids[:max_len], [m for m in markers if m < max_len], cut
    return ids[:max_len], [m for m in markers if m < max_len]


# ----------------------------------------------------------------------------- Core ML runtime
def compute_units(name):
    import coremltools as ct
    return {"CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE, "ALL": ct.ComputeUnit.ALL, "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU}[name]


def find_buckets(checkpoint, precision="fp16", models_dir=MODELS_DIR):
    """{length: compiled .mlmodelc path}. A .mlpackage without a compiled twin is compiled once, next to it."""
    import coremltools as ct
    tag = "laya_%s_%s_L" % (checkpoint, precision)
    found = {}
    for path in sorted(glob.glob(os.path.join(models_dir, checkpoint, tag + "*_options%d.ml*" % MAX_OPTIONS))):
        m = re.search(r"_L(\d+)_options\d+\.(mlmodelc|mlpackage)$", path)
        if not m:
            continue
        length = int(m.group(1))
        if m.group(2) == "mlpackage":
            compiled = path[:-len(".mlpackage")] + ".mlmodelc"
            if not os.path.isdir(compiled):
                ct.utils.compile_model(path, destination_path=compiled)
            path = compiled
        found[length] = path
    if not found:
        raise FileNotFoundError("no %s*.mlmodelc/.mlpackage buckets under %s" % (tag, os.path.join(models_dir, checkpoint)))
    return dict(sorted(found.items()))


class LayaCoreML:
    """Core ML buckets of one checkpoint. Not thread-safe: input buffers are reused across calls."""

    def __init__(self, checkpoint="multilingual", compute_units_name="CPU_AND_NE", lengths=None, precision="fp16",
                 per_bucket_units=None):
        import coremltools as ct
        self.name, self.precision = checkpoint, precision
        src = os.path.join(LAYA_DIR, CHECKPOINTS[checkpoint])
        with open(os.path.join(src, "rl_agent_config.json")) as f:
            self.cfg = json.load(f)
        self.tok = Tok(os.path.join(src, "tokenizer"))
        self.max_len, self.head_max_len = self.cfg["max_len"], self.cfg["head_max_len"]
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        available = find_buckets(checkpoint, precision)
        lengths = sorted(lengths or available)
        missing = [L for L in lengths if L not in available]
        if missing:
            raise FileNotFoundError("buckets %s not found for %s (have %s)" % (missing, checkpoint, sorted(available)))
        self.units = {L: (per_bucket_units or {}).get(L, compute_units_name) for L in lengths}
        self.models = {L: ct.models.CompiledMLModel(available[L], compute_units(self.units[L])) for L in lengths}
        self.paths = {L: available[L] for L in lengths}
        self.lengths = lengths
        self._pool = {L: [] for L in lengths}
        _KEEPALIVE.append(self)
        self._state_cache = {}
        self.cache_states = True  # tokenize each distinct state once; bench.py turns this off for latency
        self.truncated_states = 0  # questions whose state was cut to fit the largest loaded bucket / max_len

    # -- sequences
    def state_ids(self, state):
        text = serialize_state(state)
        ids = self._state_cache.get(text) if self.cache_states else None
        if ids is None:
            if len(self._state_cache) > 10000:
                self._state_cache.clear()
            ids = self._state_cache[text] = self.tok.encode(text.replace(self.tok.mask_token, " "))
        return ids

    def encode(self, state, qdef, sids=None):
        """-> (ids, markers, qtype, bucket length, state truncated?) for one Jev question."""
        q = internal(qdef)
        k = len(render_options(q))
        if k > MAX_OPTIONS:
            raise ValueError("%d options exceed the %d option slots of the Core ML models" % (k, MAX_OPTIONS))
        sids = self.state_ids(state) if sids is None else sids
        ids, markers, cut = build_sequence(self.tok, state, q, self.max_len, self.head_max_len, sids, True)
        fits = [L for L in self.lengths if L >= len(ids)]
        if fits:
            length = fits[0]
        else:  # longer than the largest loaded bucket: truncate the state on the right at that length
            length = self.lengths[-1]
            ids, markers, cut = build_sequence(self.tok, state, q, length, self.head_max_len, sids, True)
        if len(markers) != k:
            raise ValueError("options do not fit (head_max_len=%d, largest loaded bucket %d tokens)"
                             % (self.head_max_len, self.lengths[-1]))
        return ids, markers, QTYPES[q["t"]], length, cut

    def arrays(self, ids, markers, qtype, length, slot=0):
        """Fill persistent input buffer set `slot` of this bucket (see _KEEPALIVE) and return it."""
        pool = self._pool[length]
        while len(pool) <= slot:
            bufs = {"input_ids": np.empty((1, length), dtype=np.int32),
                    "attention_mask": np.empty((1, length), dtype=np.int32),
                    "marker_map": np.empty((1, MAX_OPTIONS, length), dtype=np.float32),
                    "question_type": np.empty((1, 3), dtype=np.float32)}
            pool.append(bufs)
            _KEEPALIVE.extend(bufs.values())
        b = pool[slot]
        b["input_ids"].fill(self.tok.pad_token_id)
        b["input_ids"][0, :len(ids)] = ids
        b["attention_mask"].fill(0)
        b["attention_mask"][0, :len(ids)] = 1
        b["marker_map"].fill(0.0)
        b["marker_map"][0, np.arange(len(markers)), markers] = 1.0
        b["question_type"].fill(0.0)
        b["question_type"][0, qtype] = 1.0
        return b

    # -- inference
    def raw(self, state, qdef):
        """One question -> (option logits [k], act probability, bucket length, truncated)."""
        ids, markers, qtype, length, cut = self.encode(state, qdef)
        out = self.models[length].predict(self.arrays(ids, markers, qtype, length))
        return np.asarray(out["logits"], dtype=np.float32)[0, :len(markers)], float(out["action_probabilities"][0, 0]), length, cut

    def answer(self, requests, batch=False):
        """requests: [(state, {qid: jev_question})] -> [{qid: jev_answer}].

        batch=True hands all questions of one bucket to Core ML's batch API in one call
        (model.predict(list of dicts)); False loops one predict per question.
        """
        items = []
        out = [{qid: None for qid in questions} for _, questions in requests]  # keep question order
        for r, (state, questions) in enumerate(requests):
            sids = self.state_ids(state)
            for qid, qdef in questions.items():
                ids, markers, qtype, length, cut = self.encode(state, qdef, sids)
                self.truncated_states += cut
                items.append((r, qid, qdef, ids, markers, qtype, length))
        if batch:
            for length in self.lengths:
                sel = [it for it in items if it[6] == length]
                if not sel:
                    continue
                preds = self.models[length].predict([self.arrays(it[3], it[4], it[5], length, slot)
                                                     for slot, it in enumerate(sel)])
                for it, p in zip(sel, preds):
                    out[it[0]][it[1]] = self.decode(it[2], np.asarray(p["logits"], dtype=np.float32)[0, :len(it[4])])
        else:
            for r, qid, qdef, ids, markers, qtype, length in items:
                p = self.models[length].predict(self.arrays(ids, markers, qtype, length))
                out[r][qid] = self.decode(qdef, np.asarray(p["logits"], dtype=np.float32)[0, :len(markers)])
        return out

    def decode(self, qdef, z):
        """Calibrated Jev-shaped answer, built as rl_agent_api.system_one does (without the rl_agent extension)."""
        q = internal(qdef)
        k, qt = len(z), QTYPES[q["t"]]
        z = z / self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
        p = np.exp(z - z.max())
        p = p / p.sum()
        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            return {"type": "choice", "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": round(confidence_from_probs(p, k), 4)}
        if q["t"] == "score":
            return {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": round(confidence_from_probs(p, k), 4)}
        return {"type": "noul", "noul": round(float(p[1]), 4)}


# ----------------------------------------------------------------------------- parity against the PyTorch reference
def same_option(a, b):
    """Do two Jev answers pick the same option? choice: key; score: argmax level; noul: side of 0.5."""
    if a["type"] == "choice":
        return a["choice"] == b["choice"]
    if a["type"] == "score":
        pa, pb = a["probabilities"], b["probabilities"]
        return max(pa, key=pa.get) == max(pb, key=pb.get)
    return (a["noul"] >= 0.5) == (b["noul"] >= 0.5)


def max_prob_diff(a, b):
    if a["type"] == "noul":
        return abs(a["noul"] - b["noul"])
    return max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])


def parity(laya, questions_path=None, reference_path=None):
    """Answer every parity request; -> {same, total, max_abs_prob_diff, rows}."""
    ports = os.path.dirname(HERE)
    with open(questions_path or os.path.join(ports, "parity_questions.json")) as f:
        reqs = json.load(f)
    with open(reference_path or os.path.join(ports, "parity_reference.json")) as f:
        ref = json.load(f)[laya.name]
    got = laya.answer([(r["state"], r["questions"]) for r in reqs])
    rows, same, worst = [], 0, 0.0
    for i, (r, a_ref, a_got) in enumerate(zip(reqs, ref, got)):
        for qid in r["questions"]:
            s, d = same_option(a_got[qid], a_ref[qid]), max_prob_diff(a_got[qid], a_ref[qid])
            same += s
            worst = max(worst, d)
            rows.append({"request": i, "question": qid, "same_option": s, "max_abs_prob_diff": round(d, 4),
                         "coreml": a_got[qid], "reference": a_ref[qid]})
    return {"same": same, "total": len(rows), "max_abs_prob_diff": round(worst, 4), "rows": rows}
