"""Laya on this Mac: Jev-shaped decisions from a local checkpoint, batched for the Apple GPU.

Wraps the model repo's own rl_common.py (read in full before use). Differences from its
rl_agent_api.py are about speed only: each state is tokenized once per request instead of once
per question, and questions from many requests share length-sorted GPU batches.

  from laya_local import Laya
  laya = Laya("english")                      # or "multilingual", "typed-decisions"
  answers, usage = laya.answer([(state, questions), ...])
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
LAYA_DIR = os.path.join(ROOT, "models", "laya")
sys.path.insert(0, LAYA_DIR)
from rl_common import QTYPES, build_model, confidence_from_probs, render_options, serialize_state, temp_bucket  # noqa: E402

CHECKPOINTS = {"english": "", "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def internal(qdef):
    """Jev question -> the repo's internal form (same conversion as rl_agent_api._to_internal)."""
    crit = qdef.get("criteria")
    if qdef["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"] if isinstance(qdef["instructions"], str) else json.dumps(qdef["instructions"])
    return {"t": qdef["type"], "ins": ins, "crit": crit}


class Laya:
    def __init__(self, checkpoint="english", device=None, dtype="bf16", max_tokens=16384, max_seqs=256):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        self.name = checkpoint
        path = os.path.join(LAYA_DIR, CHECKPOINTS[checkpoint])
        with open(os.path.join(path, "rl_agent_config.json")) as f:
            self.cfg = json.load(f)
        # multilingual/ and typed-decisions/ ship their own tokenizer; the English root keeps it at tokenizer/
        self.tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        self.model = build_model(self.cfg, encoder_dir=os.path.join(path, "encoder"))
        self.model.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=True)
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.model.encoder.config.reference_compile = False
        self.dtype = DTYPES[dtype]
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.max_tokens, self.max_seqs = max_tokens, max_seqs
        self.max_len, self.head_max_len = self.cfg["max_len"], self.cfg["head_max_len"]
        self._opt_cache, self._head_cache, self._state_cache = {}, {}, {}

    # -- sequences: the same token layout as rl_common.build_sequence, with caching
    def _head(self, q):
        key = (q["t"], q["ins"], json.dumps(q["crit"], sort_keys=False))
        if key not in self._head_cache:
            tok, mask = self.tok, self.tok.mask_token
            opts = render_options(q)
            head_ids = tok("%s question: %s" % (q["t"], str(q["ins"]).replace(mask, " ")), add_special_tokens=False)["input_ids"]
            opt_ids = []
            for text in opts:
                if text not in self._opt_cache:
                    self._opt_cache[text] = tok(" " + text.replace(mask, " "), add_special_tokens=False)["input_ids"][:48]
                opt_ids.append([tok.mask_token_id] + self._opt_cache[text])
            opt_budget = self.head_max_len - sum(len(o) for o in opt_ids)
            if opt_budget < 16:
                per = max(4, (self.head_max_len - 16) // max(1, len(opt_ids)))
                opt_ids = [o[:per] for o in opt_ids]
                opt_budget = self.head_max_len - sum(len(o) for o in opt_ids)
            head_ids = head_ids[:max(8, opt_budget)]
            ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
            markers = []
            for o in opt_ids:
                markers.append(len(ids))
                ids.extend(o)
            ids.append(tok.sep_token_id)
            self._head_cache[key] = (ids, markers, len(opts))
        return self._head_cache[key]

    def state_ids(self, state):
        text = serialize_state(state)
        ids = self._state_cache.get(text)
        if ids is None:  # neighbouring pixels often see the same description; tokenize it once
            if len(self._state_cache) > 50000:
                self._state_cache.clear()
            ids = self._state_cache[text] = self.tok(text.replace(self.tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        return ids

    def sequence(self, state_ids, q):
        head, markers, n_opts = self._head(q)
        room = max(0, self.max_len - len(head) - 1)
        ids = (head + state_ids[:room] + [self.tok.sep_token_id])[:self.max_len]
        markers = [m for m in markers if m < self.max_len]
        if len(markers) != n_opts:
            raise ValueError("options do not fit in head_max_len=%d tokens" % self.head_max_len)
        return ids, markers, len(state_ids) > room

    # -- inference
    @torch.no_grad()
    def _forward(self, batch):
        n, L = len(batch), max(len(it[0]) for it in batch)
        # at least two option slots: the model's act head takes a top-2 and crashes on all-single-option
        # batches (fixed the same way in the laya package); the spare slot is masked out
        kmax = max(2, max(len(it[1]) for it in batch))
        ids = torch.full((n, L), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((n, L), dtype=torch.long)
        mpos = torch.zeros((n, kmax), dtype=torch.long)
        mmask = torch.zeros((n, kmax), dtype=torch.bool)
        qtype = torch.tensor([it[2] for it in batch])
        for i, (seq, markers, _) in enumerate(batch):
            ids[i, :len(seq)] = torch.tensor(seq)
            att[i, :len(seq)] = 1
            mpos[i, :len(markers)] = torch.tensor(markers)
            mmask[i, :len(markers)] = True
        dev = self.device
        with torch.autocast(device_type=dev.type, dtype=self.dtype or torch.float32, enabled=self.dtype is not None):
            logits, _ = self.model(ids.to(dev), att.to(dev), mpos.to(dev), mmask.to(dev), qtype.to(dev))
        return logits.float().cpu().numpy(), int(att.sum())

    def answer(self, requests, on_batch=None):
        """requests: [(state, {qid: jev_question})] -> ([{qid: jev_answer}], usage)."""
        items = []  # (request index, qid, internal q, ids, markers)
        truncated = 0
        for r, (state, questions) in enumerate(requests):
            sids = self.state_ids(state)
            for qid, qdef in questions.items():
                q = internal(qdef)
                ids, markers, cut = self.sequence(sids, q)
                truncated += cut
                items.append((r, qid, q, ids, markers))
        order = sorted(range(len(items)), key=lambda i: len(items[i][3]))
        out = [dict() for _ in requests]
        tokens, i = 0, 0
        while i < len(order):
            j, L = i, 0
            while j < len(order) and j - i < self.max_seqs and max(L, len(items[order[j]][3])) * (j - i + 1) <= self.max_tokens:
                L = max(L, len(items[order[j]][3]))
                j += 1
            j = max(j, i + 1)
            chunk = [items[order[t]] for t in range(i, j)]
            logits, n_tok = self._forward([(it[3], it[4], QTYPES[it[2]["t"]]) for it in chunk])
            tokens += n_tok
            for row, (r, qid, q, _, markers) in enumerate(chunk):
                out[r][qid] = self._decode(q, logits[row, :len(markers)])
            if on_batch:
                on_batch(j, len(order))
            i = j
        return out, {"input_tokens": tokens, "questions": len(items), "truncated_states": truncated}

    def _decode(self, q, z):
        """Calibrated Jev-shaped answer, as rl_agent_api.system_one builds it."""
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
