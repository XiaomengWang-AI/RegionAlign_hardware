"""Task-panel runtime for language-only matrix experiments.

This adapter keeps the ``BaseRuntime`` contract of ``harness.core`` and feeds
it explicit MMLU or WikiText rows.  It does
not change the quantizer or the legal layout transform.  Rows contain frozen
token ids, so a search cannot silently resample text or change the prompt
after gradients have been measured.
"""
from __future__ import annotations

import torch

from regionalign.harness import core as P


class MatrixTextRuntime(P.BaseRuntime):
    """A causal-LM runtime whose rows expose ``prompt_ids`` and ``answer_ids``."""

    task = "text_matrix"
    max_new_tokens = 1

    def __init__(self, lm, tokenizer, rows, device):
        self.lm = lm
        self.tok = tokenizer
        self.rows = rows
        self.device = torch.device(device)
        self.eos = tokenizer.eos_token_id
        self.residual = None
        self.format = P.weight_format(lm)
        self.quantization = P.quantization_audit(lm)
        self._prompts = {row["sid"]: torch.as_tensor(row["prompt_ids"], dtype=torch.long)
                         for split in rows.values() for row in split}
        self._answers = {row["sid"]: torch.as_tensor(row["answer_ids"], dtype=torch.long)
                         for split in rows.values() for row in split}

    def seq_embeds(self, row, answer_ids=None):
        prompt = self._prompts[row["sid"]].to(self.device)
        embed = self.lm.get_input_embeddings()
        parts = [embed(prompt)]
        if answer_ids is not None:
            parts.append(embed(answer_ids.to(self.device)))
        return torch.cat(parts, 0), prompt.numel()

    def answer_ids(self, row):
        return self._answers[row["sid"]]

    def decode(self, tokens):
        return self.tok.decode(tokens, skip_special_tokens=True).strip()

    def score(self, prediction, answers):
        return float(prediction == answers)

    def mc_accuracy(self, rows):
        """Exact MMLU-style label accuracy without generation."""
        correct = 0
        for row in rows:
            with torch.inference_mode():
                logits = self.prefill_logits([row])[0]
            choice = torch.as_tensor(row["choice_token_ids"], device=logits.device)
            prediction = int(logits[choice].argmax().item())
            correct += prediction == int(row["answer"])
        return {"acc": correct / len(rows) if rows else 0.0, "count": len(rows)}


class FormalTextRuntime(MatrixTextRuntime):
    """Matrix runtime whose objective is token-weighted within each frozen split."""

    def seq_embeds(self, row, answer_ids=None):
        # WikiText scores every token after the first, while the clean layout gate
        # should compare logits after a representative full prefix rather than
        # after the first token alone.
        if answer_ids is None and row.get("gate_prompt_ids"):
            prompt = torch.as_tensor(row["gate_prompt_ids"], dtype=torch.long,
                                     device=self.device)
            return self.lm.get_input_embeddings()(prompt), prompt.numel()
        return super().seq_embeds(row, answer_ids)

    def _nll_batch(self, rows):
        values = super()._nll_batch(rows)
        weights = torch.as_tensor([row.get("objective_weight", 1.0) for row in rows],
                                  dtype=values.dtype, device=values.device)
        return values * weights
