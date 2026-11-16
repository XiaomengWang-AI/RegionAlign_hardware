"""Tiny CPU fixtures shared by the RegionAlign tests: no checkpoint, no GPU.

Every fixture is built from a random two-layer Llama-shaped model, so a test
exercises the real kernel and runtimes of this package on something that runs
in milliseconds.
"""
from types import SimpleNamespace

import torch
torch.set_num_threads(1)
from transformers import LlamaConfig, LlamaForCausalLM

from regionalign.harness import core as P
from regionalign.runtimes.text import MatrixTextRuntime


def tiny_lm(dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                         tie_word_embeddings=False)
    lm = LlamaForCausalLM(config).to(dtype).eval().requires_grad_(False)
    for p in lm.parameters():             # random norm gains so their permutation matters
        if p.dim() == 1:
            p.data.uniform_(0.5, 1.5)
    return lm


def tiny_gqa_lm(dtype=torch.float32, seed=0, kv_heads=2, attention_bias=False):
    """Tiny Llama-shaped grouped-query model with configurable KV groups."""
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=kv_heads, max_position_embeddings=128,
                         tie_word_embeddings=False, attention_bias=attention_bias)
    lm = LlamaForCausalLM(config).to(dtype).eval().requires_grad_(False)
    for p in lm.parameters():
        if p.dim() == 1:
            p.data.uniform_(0.5, 1.5)
    return lm


def tiny_layernorm_lm(dtype=torch.float32, seed=0, kv_heads=4):
    """Tiny StableLM-shaped decoder: LayerNorm gains and biases, q/k/v biases, partial RoPE."""
    from transformers import StableLmConfig, StableLmForCausalLM
    torch.manual_seed(seed)
    config = StableLmConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=kv_heads,
                            max_position_embeddings=128, partial_rotary_factor=0.25,
                            use_qkv_bias=True, tie_word_embeddings=False)
    lm = StableLmForCausalLM(config).to(dtype).eval().requires_grad_(False)
    for p in lm.parameters():             # random gains and biases so their permutation matters
        if p.dim() == 1:
            p.data.uniform_(0.5, 1.5)
    return lm


TINY_GROUP = 16      # int4 group in tests: the tiny model's head_dim, so heads move whole groups


def grouped_perm(n, group, rng):
    """Random order that moves whole groups and shuffles columns inside each group."""
    blocks = list(range(n // group))
    rng.shuffle(blocks)
    return [b * group + i for b in blocks for i in rng.sample(range(group), group)]


def random_spec(d, rng):
    def perm(n):
        order = list(range(n))
        rng.shuffle(order)
        return order
    hd = d["head_dim"]
    vo = [h * hd + i for h in range(d["heads"]) for i in rng.sample(range(hd), hd)]
    return {"residual": perm(d["hidden"]), "mlp": {"0": perm(d["mlp"]), "1": perm(d["mlp"])},
            "heads": {"0": perm(d["heads"])}, "vo": {"0": vo, "1": vo}}


class TinyRuntime(P.BaseRuntime):
    def __init__(self, lm, rows):
        self.lm, self.device, self.eos, self.residual = lm, torch.device("cpu"), 1, None
        self.rows = {"search": rows, "dev": rows}

    def seq_embeds(self, row, answer_ids=None):
        embed = self.lm.get_input_embeddings()
        prompt = torch.tensor(row["prompt"])
        tokens = embed(prompt)
        features = row["features"]
        if self.residual is not None:
            features = features.index_select(-1, self.residual)
        parts = [tokens[:1], features.to(tokens.dtype), tokens[1:]]
        if answer_ids is not None:
            parts.append(embed(answer_ids))
        return torch.cat(parts), 1 + features.shape[0] + len(prompt) - 1

    def answer_ids(self, row):
        return torch.tensor(row["answer"] + [self.eos])

    def decode(self, tokens):
        return " ".join(map(str, tokens))

    def score(self, prediction, answers):
        return float(prediction in answers)


def tiny_runtime(dtype=torch.bfloat16, n=6, fmt="bf16"):
    lm = tiny_lm(dtype)
    P.quantize(lm, fmt, group=TINY_GROUP)
    g = torch.Generator().manual_seed(1)
    rows = [{"sid": f"s{i}", "prompt": torch.randint(2, 64, (5,), generator=g).tolist(),
             "features": torch.randn(3, 64, generator=g), "answer": [5, 6], "answers": []}
            for i in range(n)]
    rt = TinyRuntime(lm, rows)
    for row, record in zip(rows, rt.generate(rows)):
        row["answers"] = [record["prediction"]]        # clean accuracy is 1 by construction
    return rt


class StubTokenizer:
    """Deterministic stand-in for a HF tokenizer, inside the tiny model's vocab.

    Only the four behaviours ``LlamaTextRuntime`` relies on are implemented:
    calling it with/without special tokens, ``decode`` and ``eos_token_id``.
    """
    bos_token_id, eos_token_id, vocab_size = 1, 2, 64

    def _ids(self, text):
        return [3 + (ord(ch) % (self.vocab_size - 3)) for ch in text]

    def __call__(self, text, truncation=False, add_special_tokens=True,
                 return_tensors=None, max_length=24):
        ids = self._ids(text)[:max_length]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    def decode(self, tokens, skip_special_tokens=True):
        special = {self.bos_token_id, self.eos_token_id}
        return " ".join(str(int(t)) for t in tokens
                        if not (skip_special_tokens and int(t) in special))


def tiny_text_runtime(dtype=torch.bfloat16, n=6, fmt="bf16"):
    """A ``MatrixTextRuntime`` on the tiny model whose clean accuracy is 1 by construction.

    Rows carry frozen prompt and answer token ids, as the MMLU and WikiText
    panels of the experiments do.  The answer of each row is the clean model's
    own greedy continuation, so every row scores 1 before any fault.
    """
    lm = tiny_lm(dtype)
    P.quantize(lm, fmt, group=TINY_GROUP)
    tok = StubTokenizer()
    rows = [{"sid": f"s{i}", "answers": "", "answer_ids": [tok.eos_token_id],
             "prompt_ids": tok(f"what is object {i} made of").input_ids[0].tolist()}
            for i in range(n)]
    rt = MatrixTextRuntime(lm, tok, {"search": rows, "dev": rows}, "cpu")
    for row, record in zip(rows, rt.generate(rows)):
        row["answers"] = record["prediction"]
        row["answer_ids"] = record["token_ids"] + ([] if record["finish"] == "eos" else [tok.eos_token_id])
        rt._answers[row["sid"]] = torch.as_tensor(row["answer_ids"], dtype=torch.long)
    return rt
