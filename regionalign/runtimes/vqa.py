"""Hugging Face VQA runtime for fixed-region RegionAlign experiments.

This is a parallel LIVE adapter for LLaVA-1.5 and Qwen2-VL.  It deliberately
does not read a benchmark split by name: callers supply rows selected from
training Arrow files, and :func:`load_train_panels` rejects validation/test
paths.  Search and development panels contain 128 and 256 image-question pairs
with disjoint image IDs.

The adapter reuses the clean evaluator's model/processor loader, prompt builder,
reference-answer rule, quantizer and VQA scorer.  Region search sees the usual
``nll``, ``grads`` and ``rows`` contract consumed by
``regionalign.harness.region.RegionContext``.
The complete multimodal model is used for forward/generation so Qwen2-VL keeps
its multimodal rotary positions.  A small dynamic hook makes projected visual
features follow a legal residual-channel permutation.

Qwen2-VL has Q/K/V projection biases.  The shared layout kernel moves each
output bias with the sibling weight's row order, while its GQA placement keeps
complete shared-KV query groups together.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re

from regionalign.harness import core as P, region as R

# Registry of known VLMs accepted by the runtime.  The development matrix
# chooses a subset of these (``experiments.vqa.DEV_MODELS``); the frozen
# official OK-VQA parent binds its own subset (``experiments.vqa.MODELS``).
MODEL_KEYS = ("llava-1.5-7b", "qwen2-vl-7b", "qwen2-vl-2b")
TASKS = ("okvqa", "textvqa")
SEARCH_ROWS = 128
DEV_ROWS = 256
PANEL_SEED = 20261109

# The formal driver accepts only these audited Hugging Face cache artifacts.
# Binding the repository and builder revision prevents a caller from renaming a
# different Arrow dataset to a plausible ``*-train-*.arrow`` filename.
DATASET_SPECS = {
    "okvqa": {
        "repository": "lmms-lab/ok-vqa",
        "cache_component": "lmms-lab___ok-vqa",
        "config": "default",
        "version": "0.0.0",
        "revision": "c9f858961f4095a9aeef615057b0822cce967096",
        "filename": re.compile(r"ok-vqa-train-(\d{5})-of-(\d{5})\.arrow"),
    },
    "textvqa": {
        "repository": "lmms-lab/textvqa",
        "cache_component": "lmms-lab___textvqa",
        "config": "default",
        "version": "0.0.0",
        "revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "filename": re.compile(r"textvqa-train-(\d{5})-of-(\d{5})\.arrow"),
    },
}


def validate_train_paths(paths, task):
    """Bind formal inputs to one complete audited repository revision."""
    if task not in DATASET_SPECS:
        raise ValueError(f"task must be one of {TASKS}")
    spec = DATASET_SPECS[task]
    paths = [Path(path).expanduser().resolve() for path in paths]
    if not paths:
        raise ValueError(f"{task} requires at least one train Arrow shard")
    shards, totals = {}, set()
    expected_parent = (spec["cache_component"], spec["config"], spec["version"],
                       spec["revision"])
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if tuple(path.parent.parts[-4:]) != expected_parent:
            raise ValueError(
                f"{task} input is not from frozen {spec['repository']} revision "
                f"{spec['revision']}: {path}")
        match = spec["filename"].fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected {task} train shard name: {path.name}")
        index, total = map(int, match.groups())
        if index in shards:
            raise ValueError(f"duplicate {task} train shard index {index}")
        shards[index] = path
        totals.add(total)
    if len(totals) != 1:
        raise ValueError(f"inconsistent {task} train shard totals")
    total = totals.pop()
    if total < 1 or set(shards) != set(range(total)):
        raise ValueError(f"{task} requires all {total} train shards")
    return [shards[index] for index in range(total)]


def _validate_arrow_schema(schema, task, path):
    """Check the task-bearing columns before any row is selected."""
    import pyarrow as pa
    required = {"question_id", "image_id", "question", "answers", "image", "set_name"}
    missing = required - set(schema.names)
    if missing:
        raise ValueError(f"{task} Arrow schema lacks {sorted(missing)}: {path}")
    if not pa.types.is_string(schema.field("question").type):
        raise ValueError(f"{task} question column must be string")
    if not pa.types.is_string(schema.field("set_name").type):
        raise ValueError(f"{task} set_name column must be string")
    answers = schema.field("answers").type
    if not (pa.types.is_list(answers) and pa.types.is_string(answers.value_type)):
        raise ValueError(f"{task} answers column must be list<string>")
    image = schema.field("image").type
    if not pa.types.is_struct(image) or image.get_field_index("bytes") < 0:
        raise ValueError(f"{task} image column must contain bytes")
    if not pa.types.is_binary(image.field("bytes").type):
        raise ValueError(f"{task} image.bytes must be binary")


def _clean_module():
    """Import the clean evaluator lazily so CPU-only panel tests stay lightweight."""
    from regionalign.experiments import clean_benchmarks as clean
    return clean


def _rank(seed, task, image_id, question_id):
    value = f"{seed}|{task}|{image_id}|{question_id}".encode()
    return hashlib.sha256(value).hexdigest()


def select_train_panels(records, task, search_rows=SEARCH_ROWS, dev_rows=DEV_ROWS,
                        seed=PANEL_SEED, reference_answer=None):
    """Choose deterministic, image-disjoint search/dev rows from training records.

    Selection uses only stable identifiers and input-validity checks.  It never
    evaluates a model or uses a task score.  At most one question per image is
    retained, so image IDs are disjoint both within and between the two panels.
    Empty normalized references are excluded because RegionAlign's search loss
    requires at least one scored answer token.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if type(search_rows) is not int or type(dev_rows) is not int or min(search_rows, dev_rows) < 1:
        raise ValueError("panel sizes must be positive integers")
    if reference_answer is None:
        from regionalign import config as C
        C.setup_environment()
        reference_answer = _clean_module().reference_answer
    by_image = {}
    seen_questions = set()
    for raw in records:
        if raw.get("set_name") != "train":
            raise ValueError("VQA search/dev panels may contain training rows only")
        question_id = str(raw.get("question_id", ""))
        image_id = str(raw.get("image_id", ""))
        question = raw.get("question")
        answers = raw.get("answers")
        image = raw.get("image") or {}
        image_bytes = image.get("bytes")
        if not question_id or question_id in seen_questions:
            raise ValueError(f"missing or duplicate question ID {question_id!r}")
        seen_questions.add(question_id)
        if not image_id or not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            continue
        if not isinstance(answers, list) or not answers or not all(isinstance(x, str) for x in answers):
            continue
        reference = reference_answer(answers, task)
        if not reference:
            continue
        rank = _rank(seed, task, image_id, question_id)
        row = {"sid": f"{task}-{question_id}", "question_id": question_id,
               "image_id": image_id, "question": question, "answers": list(answers),
               "reference": reference,
               "image": {"bytes": bytes(image_bytes), "path": image.get("path")},
               "image_sha256": hashlib.sha256(image_bytes).hexdigest(), "rank": rank}
        previous = by_image.get(image_id)
        if previous is None or rank < previous[0]:
            by_image[image_id] = (rank, row)
    need = search_rows + dev_rows
    chosen = [row for _, row in sorted(by_image.values(), key=lambda item: item[0])[:need]]
    if len(chosen) != need:
        raise ValueError(f"need {need} valid distinct training images, found {len(chosen)}")
    search, dev = chosen[:search_rows], chosen[search_rows:]
    if {row["image_id"] for row in search} & {row["image_id"] for row in dev}:
        raise AssertionError("search and dev images overlap")
    return {"search": search, "dev": dev}


def _arrow_rows(paths, task, validate_paths=True):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    paths = (validate_train_paths(paths, task) if validate_paths
             else [Path(path).resolve() for path in paths])
    for raw_path in paths:
        path = Path(raw_path).resolve()
        lower = path.name.lower()
        forbidden = any(re.search(r"(^|[-_.])(validation|val2014|test)($|[-_.])",
                                  part.lower()) for part in path.parts)
        if "train" not in lower or forbidden:
            raise ValueError(f"refusing non-training VQA input: {path}")
        with pa.memory_map(str(path), "r") as handle:
            reader = ipc.open_stream(handle)
            _validate_arrow_schema(reader.schema, task, path)
            for batch in reader:
                yield from batch.to_pylist()


def load_train_panels(paths, task, **kwargs):
    """Load training Arrow shards without retaining the full image corpus in RAM.

    The first pass chooses one question per image using identifiers and answer
    text only.  The second pass materializes bytes for the selected 384 images.
    This is deliberately separate from :func:`select_train_panels`, whose small
    in-memory interface remains useful for synthetic protocol tests.
    """
    paths = validate_train_paths(paths, task)
    search_rows = kwargs.pop("search_rows", SEARCH_ROWS)
    dev_rows = kwargs.pop("dev_rows", DEV_ROWS)
    seed = kwargs.pop("seed", PANEL_SEED)
    reference_answer = kwargs.pop("reference_answer", None)
    if kwargs:
        raise TypeError(f"unexpected panel arguments: {sorted(kwargs)}")
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if type(search_rows) is not int or type(dev_rows) is not int or min(search_rows, dev_rows) < 1:
        raise ValueError("panel sizes must be positive integers")
    if reference_answer is None:
        from regionalign import config as C
        C.setup_environment()
        reference_answer = _clean_module().reference_answer

    # Only small strings and identifiers survive the first pass.  Keeping image
    # bytes here made preparation require memory proportional to the full corpus.
    by_image, seen_questions = {}, set()
    for raw in _arrow_rows(paths, task, validate_paths=False):
        if raw.get("set_name") != "train":
            raise ValueError("VQA search/dev panels may contain training rows only")
        question_id = str(raw.get("question_id", ""))
        image_id = str(raw.get("image_id", ""))
        question = raw.get("question")
        answers = raw.get("answers")
        image = raw.get("image") or {}
        if not question_id or question_id in seen_questions:
            raise ValueError(f"missing or duplicate question ID {question_id!r}")
        seen_questions.add(question_id)
        if (not image_id or not isinstance(question, str) or not question.strip()
                or not isinstance(image.get("bytes"), (bytes, bytearray))
                or not image.get("bytes") or not isinstance(answers, list)
                or not answers or not all(isinstance(value, str) for value in answers)):
            continue
        reference = reference_answer(answers, task)
        if not reference:
            continue
        rank = _rank(seed, task, image_id, question_id)
        previous = by_image.get(image_id)
        if previous is None or rank < previous[0]:
            by_image[image_id] = (rank, question_id)
    need = search_rows + dev_rows
    chosen = sorted(by_image.values())[:need]
    if len(chosen) != need:
        raise ValueError(f"need {need} valid distinct training images, found {len(chosen)}")
    wanted = {question_id: rank for rank, question_id in chosen}

    materialized = {}
    for raw in _arrow_rows(paths, task, validate_paths=False):
        question_id = str(raw.get("question_id", ""))
        if question_id not in wanted:
            continue
        image = raw["image"]
        image_bytes = bytes(image["bytes"])
        answers = list(raw["answers"])
        question = raw["question"]
        reference = reference_answer(answers, task)
        materialized[question_id] = {
            "sid": f"{task}-{question_id}", "question_id": question_id,
            "image_id": str(raw["image_id"]), "question": question,
            "answers": answers, "reference": reference,
            "image": {"bytes": image_bytes, "path": image.get("path")},
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "answers_sha256": hashlib.sha256("\0".join(answers).encode()).hexdigest(),
            "rank": wanted[question_id],
        }
    if set(materialized) != set(wanted):
        raise RuntimeError("selected VQA rows changed between Arrow passes")
    ordered = [materialized[question_id] for rank, question_id in chosen]
    return {"search": ordered[:search_rows], "dev": ordered[search_rows:]}


def panel_manifest(rows):
    """JSON-safe selected-input identity; image bytes are represented by SHA256 only."""
    return {split: [{key: row[key] for key in
                     ("sid", "question_id", "image_id", "image_sha256", "rank", "reference")}
                    for row in split_rows]
            for split, split_rows in rows.items()}


def apply_layout(rt, spec, inverse=False):
    """Apply the shared architecture-aware legal permutation."""
    spec = P.validate_spec(spec, P.dims(rt.lm))
    P.apply_permutation(rt, spec, inverse=inverse)


class VQARegionContext(R.RegionContext):
    """Named RegionContext seam for the expanded VQA matrix."""


class HFVQARegionRuntime:
    """LLaVA/Qwen2-VL adapter implementing the RegionContext model contract."""

    max_new_tokens = 32

    def __init__(self, model, processor, language, rows, model_key, task, device,
                 quantization=None, load_timing=None):
        if model_key not in MODEL_KEYS or task not in TASKS:
            raise ValueError(f"unsupported VQA condition: {model_key}/{task}")
        import torch
        self.model, self.processor, self.lm = model, processor, language
        self.model_key, self.task = model_key, task
        self.device = torch.device(device)
        self.residual = None
        self.quantization = quantization or P.quantization_audit(language)
        self.load_timing = load_timing or {}
        self._prompt_cache = {}
        if not hasattr(self.lm, "config"):
            # The clean evaluator's Qwen language-only wrapper registers
            # ``model`` and ``lm_head`` but intentionally excludes ``visual``.
            self.lm.config = self.model.config
        P.validate_parameter_layout(self.lm)
        if not isinstance(rows, dict) or not rows or set(rows) - {"search", "dev"}:
            raise ValueError("VQA runtime rows must map search/dev split names to rows")
        self.rows = {split: self._prepare_rows(split_rows)
                     for split, split_rows in rows.items()}
        feature_module = (self.model.multi_modal_projector if model_key == "llava-1.5-7b"
                          else self.model.visual)
        self._visual_hook = feature_module.register_forward_hook(self._reorder_visual)

    def _prepare_rows(self, rows):
        """Attach tokenizer-specific token weights without changing frozen inputs."""
        prepared = []
        for source in rows:
            row = dict(source)
            target_tokens = len(self.processor.tokenizer.encode(
                row["reference"], add_special_tokens=False))
            if target_tokens < 1:
                raise ValueError(f"row {row['sid']} has no scored answer token")
            row["target_tokens"] = target_tokens
            prepared.append(row)
        total = sum(row["target_tokens"] for row in prepared)
        if not prepared or total < len(prepared):
            raise ValueError("VQA panel has no scored answer tokens")
        mean = total / len(prepared)
        for row in prepared:
            row["objective_weight"] = row["target_tokens"] / mean
        return prepared

    @classmethod
    def load(cls, model_key, checkpoint, fmt, rows, task, device="cuda:0"):
        """Reuse the clean evaluator's exact offline loader and quantization path."""
        clean = _clean_module()
        model, processor, language, audit, timing = clean.load_clean(
            model_key, checkpoint, fmt, device)
        return cls(model, processor, language, rows, model_key, task, device,
                   quantization=audit, load_timing=timing)

    def close(self):
        if self._visual_hook is not None:
            self._visual_hook.remove()
            self._visual_hook = None

    def _reorder_visual(self, _module, _args, output):
        if self.residual is None:
            return output
        import torch
        if not torch.is_tensor(output):
            raise TypeError("visual feature module returned a non-tensor output")
        index = self.residual.to(output.device)
        if output.shape[-1] != index.numel():
            raise ValueError("visual feature width does not match the residual permutation")
        return output.index_select(-1, index)

    def param(self, name):
        return self.lm.get_parameter(name)

    def _image(self, row):
        from PIL import Image
        with Image.open(io.BytesIO(row["image"]["bytes"])) as source:
            return source.convert("RGB")

    def _prompt(self, row):
        """Processor output cached on CPU; tensors move to the model only per call."""
        if row["sid"] not in self._prompt_cache:
            self._prompt_cache[row["sid"]] = _clean_module().vlm_inputs(
                self.model_key, self.processor, self._image(row), row["question"], "cpu")
        return self._prompt_cache[row["sid"]]

    def _to_device(self, values):
        return {key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in values.items()}

    def _position_kwargs(self, inputs):
        if not self.model_key.startswith("qwen2-vl"):
            return {}
        position_ids, rope_deltas = self.model.get_rope_index(
            inputs["input_ids"], inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"), inputs.get("attention_mask"))
        return {"position_ids": position_ids, "rope_deltas": rope_deltas}

    def _answer_ids(self, row):
        import torch
        ids = self.processor.tokenizer.encode(row["reference"], add_special_tokens=False)
        if not ids:
            raise ValueError(f"row {row['sid']} has no scored answer token")
        return torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def _teacher_inputs(self, row):
        import torch
        inputs = self._to_device(self._prompt(row))
        answer = self._answer_ids(row)
        prompt_length = inputs["input_ids"].shape[1]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], answer[None]], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones((1, answer.numel()), dtype=inputs["attention_mask"].dtype,
                                                   device=self.device)], dim=1)
        labels = torch.full_like(inputs["input_ids"], -100)
        labels[:, prompt_length:] = answer
        inputs["labels"] = labels
        inputs.update(self._position_kwargs(inputs))
        return inputs

    def _loss(self, row):
        return (self.model(**self._teacher_inputs(row), use_cache=False).loss
                * row.get("objective_weight", 1.0))

    def nll(self, rows, bs=1):
        """Per-row answer-token NLL; ``bs`` is accepted for the RegionContext API."""
        import torch
        del bs
        with torch.inference_mode():
            return [float(self._loss(row)) for row in rows]

    def grads(self, names, rows, bs=1):
        """Gradient of mean answer NLL with respect to the requested language weights."""
        import torch
        del bs
        params = [self.param(name) for name in names]
        for param in params:
            param.grad = None
            param.requires_grad_(True)
        try:
            with torch.enable_grad():
                for row in rows:
                    (self._loss(row) / len(rows)).backward()
            return {name: param.grad.detach().clone() for name, param in zip(names, params)}
        finally:
            for param in params:
                param.requires_grad_(False)
                param.grad = None

    def prefill_logits(self, rows):
        """Last prompt-token logits for the function-preservation clean gate."""
        import torch
        result = []
        with torch.inference_mode():
            for row in rows:
                inputs = self._to_device(self._prompt(row))
                inputs.update(self._position_kwargs(inputs))
                result.append(self.model(**inputs, use_cache=False).logits[0, -1].float())
        return torch.stack(result)

    def accuracy(self, rows):
        """Official normalized VQA score and generation diagnostics on supplied dev rows."""
        import torch
        from regionalign.runtimes.vqa_score import vqa_score
        records = []
        for row in rows:
            inputs = self._to_device(self._prompt(row))
            prompt_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, do_sample=False, num_beams=1,
                    max_new_tokens=self.max_new_tokens, use_cache=True,
                    return_dict_in_generate=True, output_logits=True)
            finite = all(bool(torch.isfinite(value).all()) for value in generated.logits)
            tokens = generated.sequences[0, prompt_length:].tolist()
            prediction = self.processor.tokenizer.decode(
                tokens, skip_special_tokens=True).strip()
            eos = self.model.generation_config.eos_token_id
            eos = {value for value in (eos if isinstance(eos, list) else [eos])
                   if value is not None}
            reached_eos = bool(tokens and tokens[-1] in eos)
            truncated = (len(tokens) == self.max_new_tokens
                         and not reached_eos)
            records.append({
                "sid": row["sid"], "prediction": prediction,
                "score": vqa_score(prediction, row["answers"], task=self.task),
                "nonfinite": not finite,
                "finish": ("eos" if reached_eos else
                           "length" if truncated else "complete"),
                "generated_tokens": len(tokens),
            })
        count = len(records)
        if not count:
            raise ValueError("VQA evaluation panel is empty")
        return {"acc": sum(row["score"] for row in records) / count,
                "nonfinite_rate": sum(row["nonfinite"] for row in records) / count,
                "eos_rate": sum(row["finish"] == "eos" for row in records) / count,
                "length_rate": sum(row["finish"] == "length" for row in records) / count,
                "records": records}

