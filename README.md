# RegionAlign → hardware handoff: LLM package (Qwen2.5-1.5B, MMLU, INT8)

## Deploy & run (self-contained — the RegionAlign code is bundled in `regionalign/`)

```bash
git clone git@github.com:XiaomengWang-AI/RegionAlign_hardware.git
cd RegionAlign_hardware
pip install -r requirements.txt

# download the base model once (Qwen2.5-1.5B is public ~3 GB); set a cache dir:
export REGIONALIGN_HF_HOME=$PWD/hf_cache
huggingface-cli download Qwen/Qwen2.5-1.5B --revision 8faed761d45a263340a0528343f099c05c9a4323 \
    --local-dir-use-symlinks False  # or: HF_HOME=$REGIONALIGN_HF_HOME python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B')"

# run: builds the permuted + faulted INT8 model from the base checkpoint and evaluates MMLU
python run_asr_mmlu.py qwen2.5-1.5b__mmlu --layout permuted --mode selected
```

The script reconstructs the attacked model from the public base checkpoint + `permutation.json` +
the recorded flips — so the repo does not ship the full 3 GB weights, only the attacked tensor's
bytes (`codes/`). A GPU is recommended (CPU MMLU 5-shot forwards are slow). `materialize_codes.py`
regenerates/extends `codes/` the same way.

---


One tested LLM where an attacker-chosen permutation + a few in-region bit flips crosses the attack
success threshold. All numbers are from the frozen INT8 campaign (software fault injection).

## The attack at a glance

| | |
|---|---|
| Model | Qwen2.5-1.5B (base), INT8, all parameters quantized |
| Task | MMLU (accuracy) |
| Target region | `L08.v_proj.p7` → tensor `model.layers.8.self_attn.v_proj.weight` |
| Region bytes in that tensor | **[28672, 32768)** = 4096 INT8 codes (1 byte each) |
| Bit flips | **4**, all bit 7 (INT8 sign/MSB), at in-region byte offsets 727, 1225, 1264, 1380 |
| Effect | MMLU accuracy **0.593 → 0.291 (−51%)** on the dev panel |

The flips are chosen by search on the **permuted (RegionAlign)** layout; on the original layout the
same region does not give this drop. This is the *attacker-chooses-the-bits* upper bound — see the
caveat at the bottom.

## What's in this folder

```
qwen2.5-1.5b__mmlu/
  attack.json          region, the 4 flips (byte offset in region + bit) and all K=1..16 flip sets,
                       old/new weight values, degradation, checkpoint repo+revision, provenance
  permutation.json     the full function-equivalent channel permutation (residual / mlp / heads / vo)
  codes/
    original_codes.u8.npy   stored INT8 codes of the attacked tensor, ORIGINAL layout
    permuted_codes.u8.npy    ... PERMUTED (RegionAlign) layout  ← this is what gets deployed
    faulted_codes.u8.npy     ... permuted + the 4 flips applied  ← what the attack produces
  bytes_manifest.json  self-check: decoding the flipped bytes reproduces the recorded old/new values
  eval.json, layout_artifact.json   verbatim source artifacts (provenance; plan_sha256)
samples/mmlu_min_qwen2.5-1.5b.json   32 Qwen-tokenized MMLU dev questions (self-contained panel)
run_asr_mmlu.py        load the model, evaluate MMLU accuracy, compute attack success / ASR
materialize_codes.py   regenerate codes/ from the checkpoint + permutation (or dump more tensors)
```

INT8 code = one signed two's-complement byte; bit 7 is the sign/MSB. Decoded weight =
`row_scale × signed_code`; scales are FP32, one per output row, and are **not** faulted. The region
is `codes[28672:32768]` of the attacked tensor (row-major flat).

## How to use

**Just the bytes (no Python):** flip bit 7 of the 4 bytes at region offsets {727, 1225, 1264, 1380}
in the deployed `permuted_codes` to get `faulted_codes`; `attack.json → flip_values` lists the exact
old/new decoded weights to expect.

**Reproduce / measure ASR (needs the model + the RegionAlign harness + a torch env):**
```
PYTHONPATH=/path/to/BFA_llm  REGIONALIGN_HF_HOME=/path/to/hf_cache \
python run_asr_mmlu.py qwen2.5-1.5b__mmlu --layout permuted --mode selected
```
Prints clean vs faulted accuracy and SUCCESS/fail (success = faulted ≤ 0.5 × clean). Modes:
`--mode sample --trials N` = random in-range flips in the region (hardware-realistic ASR, expected ≈0);
`--mode codes --codes <u8.npy>` = score your own post-fault bytes.
Needs: (1) the checkpoint — `Qwen/Qwen2.5-1.5B` rev `8faed761…` from Hugging Face; (2) the
RegionAlign harness on `PYTHONPATH`; (3) torch. **Run on a GPU** — on CPU the 5-shot MMLU forwards
are very slow. The 32-question panel is bundled, so no MMLU dataset download is needed.

## Important caveat (please read)

This is the **attacker-chooses-which-bits** result. On real hardware the attacker usually cannot pick
which bit flips; under randomly sampled in-region faults the success rate is ≈0 (that's `--mode
sample`). So this package shows the *idealized upper-bound* attack and the exact bytes it needs — the
open question for hardware is whether a fault can be steered onto (or near) these specific bytes/bits.
(On WikiText-2 the LLM attacks are much stronger — e.g. StableLM2, one flip, perplexity 11.5→68 — but
this package is MMLU per request.)
