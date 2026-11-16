"""Audited development-only LLM RegionAlign matrix.

This is the formal, parallel successor to the one-region language smoke driver.
It freezes, for the registered checkpoints named on the command line, the
complete INT8/INT4 x MMLU/WikiText development matrix before execution.  A plan
may name any non-empty subset of the registry.  Search and evaluation are
separate commands and therefore separate model processes.  There is deliberately
no final/test command or input path in this module.

The registered unit is one condition, twelve predefined 4 KiB regions (four in
each depth band), three repeat seeds, and the noperm/local/random arms.  Results
are atomically published once.  Persistent lock files prevent two workers from
claiming the same registered item concurrently; an existing artifact or report
is never replaced.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import random
import re
import shutil
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from regionalign.harness import core as P, region as R
from regionalign import config as C
from regionalign.experiments import clean_mmlu as M
from regionalign.runtimes.text import FormalTextRuntime, MatrixTextRuntime


SCHEMA_VERSION = 2
SCOPE = "llm_regionalign_dev"
MODELS = tuple(M.MODEL_SPECS)
FORMATS = ("int8", "int4")
TASKS = ("mmlu", "wikitext")
ARMS = ("noperm", "local", "random")
DEPTH_BANDS = ("early", "middle", "late")
REGION_KINDS = ("q_proj", "v_proj", "gate_proj", "down_proj")
REGIONS_PER_BAND = len(REGION_KINDS)
REPEAT_SEEDS = (2026110901, 2026110902, 2026110903)
REGION_SEED = 2026110900
EVAL_K = (1, 2, 4, 8, 16)
MMLU_SEARCH = 285
MMLU_DEV = 1246
WIKITEXT_ROWS = 128
WIKITEXT_WORDS = 512
WIKITEXT_MAX_TOKENS = 512
WIKITEXT_SEARCH_SEED = 2026110911
WIKITEXT_DEV_SEED = 2026110912
MMLU_MAX_TOKENS = 4096
MMLU_SCORE_GATE = 0.02
DEFAULT_SEARCH_BATCH_SIZE = 1
DEFAULT_EVAL_BATCH_SIZE = 1
DEFAULT_GRAD_BATCH_SIZE = 1
DEFAULT_MIN_FREE_GPU_GIB = 40.0
MIN_FREE_POLICY = "available_cuda_memory_before_model_load"

SOURCE_RELATIVE = (
    Path("regionalign/__init__.py"),
    Path("regionalign/config.py"),
    Path("regionalign/experiments/__init__.py"),
    Path("regionalign/experiments/llm.py"),
    Path("regionalign/experiments/clean_mmlu.py"),
    Path("regionalign/runtimes/__init__.py"),
    Path("regionalign/runtimes/text.py"),
    Path("regionalign/harness/__init__.py"),
    Path("regionalign/harness/core.py"),
    Path("regionalign/harness/region.py"),
    Path("regionalign/algorithm/__init__.py"),
    Path("regionalign/algorithm/search.py"),
)
SEARCH_SNAPSHOT = "regionalign__algorithm__search.py"


def _json_hash(value):
    return M.object_sha256(value)


def _safe_component(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    if not value or value in (".", ".."):
        raise ValueError(f"unsafe path component {value!r}")
    return value


def _forbid_test_path(raw, label):
    # Check the resolved target as well as the spelling supplied by the caller.
    # Otherwise a harmless-looking symlink could expose a frozen test input.
    supplied = Path(raw).expanduser().absolute()
    path = supplied.resolve()
    parts = (*supplied.parts, *path.parts)
    if any(re.search(r"(^|[-_.])test($|[-_.])", part, re.I) for part in parts):
        raise ValueError(f"{label} must not be a test input: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _source_record(path):
    path = Path(path).expanduser().absolute()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": M.sha256(path)}


def _copy_sources(run):
    source_dir = Path(run) / "source"
    source_dir.mkdir()
    records = []
    for relative in SOURCE_RELATIVE:
        live = C.PROJECT / relative
        if not live.is_file():
            raise FileNotFoundError(live)
        name = str(relative).replace("/", "__")
        saved = source_dir / name
        shutil.copyfile(live, saved)
        live_hash, saved_hash = M.sha256(live), M.sha256(saved)
        if live_hash != saved_hash:
            raise RuntimeError(f"source changed while freezing: {relative}")
        records.append({"path": str(relative), "snapshot": f"source/{name}",
                        "sha256": live_hash, "bytes": live.stat().st_size})
    return records


def _source_drift(run, plan):
    drift = []
    for record in plan.get("sources", []):
        live = C.PROJECT / record["path"]
        saved = Path(run) / record["snapshot"]
        if (not live.is_file() or not saved.is_file()
                or M.sha256(live) != record["sha256"]
                or M.sha256(saved) != record["sha256"]):
            drift.append(record["path"])
    return drift


def _assert_frozen(run, plan):
    drift = _source_drift(run, plan)
    if drift:
        raise RuntimeError(f"sources changed after planning: {drift}; create a new run")


def load_plan(run):
    run = C.output_path(run)
    plan_path = run / "plan.json"
    plan = M.read_json(plan_path)
    plan_sha256 = M.sha256(plan_path)
    status = M.read_json(run / "status.json")
    if status.get("plan_sha256") != plan_sha256:
        raise RuntimeError("status.json does not register this plan.json hash")
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("scope") != SCOPE:
        raise ValueError("unsupported LLM RegionAlign plan")
    if not plan.get("development_only") or "test" in plan.get("splits", []):
        raise ValueError("the formal development plan must not contain a test split")
    _assert_frozen(run, plan)
    plan["sha256"] = plan_sha256
    return run, plan


def _hash_int(*parts):
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def depth_ranges(layers):
    if type(layers) is not int or layers < 3:
        raise ValueError("a formal model needs at least three decoder layers")
    a, b = layers // 3, (2 * layers) // 3
    return {"early": range(0, a), "middle": range(a, b), "late": range(b, layers)}


def preregistered_regions(model_key, dims, bits, seed=REGION_SEED):
    """Return twelve result-independent regions: four in each depth band."""
    if bits not in (4, 8):
        raise ValueError("formal regions support only INT8 and INT4")
    regions = []
    for band, layer_range in depth_ranges(dims["layers"]).items():
        choices = list(layer_range)
        for kind in REGION_KINDS:
            layer = choices[_hash_int(seed, model_key, band, kind, "layer") % len(choices)]
            pages = R.page_count(kind, dims, bits=bits, page_bytes=P.PAGE_BYTES)
            page = _hash_int(seed, model_key, band, kind, "page") % pages
            region = R.make_region(kind, layer, page, dims, bits=bits,
                                   page_bytes=P.PAGE_BYTES)
            region["depth_band"] = band
            region["selection_seed"] = seed
            regions.append(region)
    if len(regions) != len(DEPTH_BANDS) * REGIONS_PER_BAND:
        raise AssertionError("formal region count changed")
    return regions


def dims_for_format(config, fmt):
    """Freeze the layout dimensions that the loaded quantized model will expose.

    Checkpoint ``config.json`` is a plain mapping, while ``harness.dims`` reads
    attributes from the loaded Hugging Face config.  INT4 also adds the
    repository's 128-column scale-group constraint.  Recording both facts in
    the plan makes its legal-permutation domain identical to the runtime's.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unsupported formal format {fmt!r}")
    if isinstance(config, dict):
        config = SimpleNamespace(**config)
    group = P.GROUP if fmt == "int4" else None
    return P.dims_from_config(config, quant_group=group)


def _read_wikitext(raw, expected_split):
    path = _forbid_test_path(raw, f"WikiText {expected_split}")
    allowed = {f"{expected_split}.arrow", f"wikitext-{expected_split}.arrow"}
    if path.name not in allowed:
        raise ValueError(f"expected one of {sorted(allowed)}, got {path.name}")
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(path), "r") as source:
        table = ipc.open_stream(source).read_all()
    if "text" not in table.column_names:
        raise ValueError("WikiText Arrow input lacks text")
    texts = table["text"].to_pylist()
    if not texts or not all(isinstance(value, str) for value in texts):
        raise ValueError("WikiText text column must contain non-null strings")
    joined = "\n".join(texts)
    return joined, {**_source_record(path), "split": expected_split, "rows": len(texts),
                    "joined_text_sha256": hashlib.sha256(joined.encode()).hexdigest(),
                    "join_rule": "newline_between_arrow_rows"}


def _select_word_spans(text, count, words_per_sample, seed):
    words = list(re.finditer(r"\S+", text))
    blocks = len(words) // words_per_sample
    if blocks < count:
        raise ValueError(f"only {blocks} disjoint WikiText blocks for {count} samples")
    chosen = sorted(random.Random(seed).sample(range(blocks), count))
    rows = []
    for index, block in enumerate(chosen):
        first = block * words_per_sample
        start, end = words[first].start(), words[first + words_per_sample - 1].end()
        snippet = text[start:end]
        rows.append({"index": index, "block": block, "char_start": start,
                     "char_end": end, "text": snippet,
                     "text_sha256": hashlib.sha256(snippet.encode()).hexdigest()})
    return rows


def _normalise_objective_weights(rows):
    total = sum(row["target_tokens"] for row in rows)
    if not rows or total < len(rows):
        raise ValueError("panel has no scored target tokens")
    mean = total / len(rows)
    for row in rows:
        row["objective_weight"] = row["target_tokens"] / mean
    return rows


def _mmlu_rows(tokenizer, hf_home):
    root = Path(hf_home) / M.DATASET_RELATIVE
    dev_path = _forbid_test_path(root / "mmlu-dev.arrow", "MMLU demonstrations")
    validation_path = _forbid_test_path(root / "mmlu-validation.arrow", "MMLU validation")
    demonstrations = M.read_arrow(dev_path)
    validation = M.read_arrow(validation_path)
    by_subject = M.dev_by_subject(demonstrations)
    choice_ids, choice_tokens = M.choice_token_ids(tokenizer)
    selected, holdout, counts = [], [], {}
    for index, raw in enumerate(validation):
        count = counts.get(raw["subject"], 0)
        split = "search" if count < 5 else "dev"
        counts[raw["subject"]] = count + 1
        ids, shots = M.prompt_ids(tokenizer, raw["subject"], by_subject[raw["subject"]],
                                  raw, MMLU_MAX_TOKENS, M.DEFAULT_SHOTS)
        row = {"sid": f"mmlu-validation-{index}", "source_index": index,
               "source_sha256": _json_hash(raw), "subject": raw["subject"],
               "prompt_ids": ids, "answer_ids": [choice_ids[raw["answer"]]],
               "choice_token_ids": choice_ids, "choice_tokens": choice_tokens,
               "answer": raw["answer"], "shots": shots, "target_tokens": 1,
               "objective_weight": 1.0}
        (selected if split == "search" else holdout).append(row)
    if len(selected) != MMLU_SEARCH or len(holdout) != MMLU_DEV:
        raise ValueError(f"unexpected MMLU split: {len(selected)}/{len(holdout)}")
    sources = {"demonstrations": _source_record(dev_path),
               "validation": _source_record(validation_path),
               "selection": "first_five_validation_rows_per_subject_for_search"}
    return {"search": selected, "dev": holdout}, sources


def _wikitext_rows(tokenizer, train_file, validation_file):
    train, train_source = _read_wikitext(train_file, "train")
    validation, validation_source = _read_wikitext(validation_file, "validation")
    selected = {
        "search": _select_word_spans(train, WIKITEXT_ROWS, WIKITEXT_WORDS,
                                     WIKITEXT_SEARCH_SEED),
        "dev": _select_word_spans(validation, WIKITEXT_ROWS, WIKITEXT_WORDS,
                                  WIKITEXT_DEV_SEED),
    }
    output = {}
    for split, samples in selected.items():
        rows = []
        for sample in samples:
            full = tokenizer.encode(sample["text"], add_special_tokens=True)
            ids = full[:WIKITEXT_MAX_TOKENS]
            if len(ids) < 2:
                raise ValueError(f"WikiText {split} sample {sample['index']} has fewer than 2 tokens")
            rows.append({"sid": f"wikitext-{split}-{sample['index']}",
                         "source_index": sample["index"], "block": sample["block"],
                         "text_sha256": sample["text_sha256"],
                         "prompt_ids": [ids[0]], "answer_ids": ids[1:],
                         "gate_prompt_ids": ids[:-1],
                         "full_input_tokens": len(full), "input_tokens": len(ids),
                         "target_tokens": len(ids) - 1,
                         "truncated": len(ids) < len(full)})
        output[split] = _normalise_objective_weights(rows)
    return output, {"train": train_source, "validation": validation_source,
                    "selection": {"count": WIKITEXT_ROWS,
                                  "words_per_sample": WIKITEXT_WORDS,
                                  "search_seed": WIKITEXT_SEARCH_SEED,
                                  "dev_seed": WIKITEXT_DEV_SEED,
                                  "max_tokens": WIKITEXT_MAX_TOKENS}}


def _panel_file(run, model, task, split):
    return Path(run) / "panels" / _safe_component(model) / task / f"{split}.json"


def _freeze_panels(run, models, hf_home, wikitext_train, wikitext_validation):
    from transformers import AutoTokenizer
    panels, dataset_sources = {}, {}
    for model_key, checkpoint in models.items():
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False, use_fast=True)
        model_panels = {}
        mmlu, mmlu_sources = _mmlu_rows(tokenizer, hf_home)
        wiki, wiki_sources = _wikitext_rows(tokenizer, wikitext_train, wikitext_validation)
        dataset_sources.setdefault("mmlu", mmlu_sources)
        dataset_sources.setdefault("wikitext", wiki_sources)
        for task, splits in (("mmlu", mmlu), ("wikitext", wiki)):
            model_panels[task] = {}
            for split, rows in splits.items():
                path = _panel_file(run, model_key, task, split)
                payload = {"schema_version": SCHEMA_VERSION, "scope": SCOPE,
                           "model": model_key, "task": task, "split": split,
                           "development_only": True, "rows": rows,
                           "count": len(rows),
                           "target_tokens": sum(row["target_tokens"] for row in rows)}
                M.write_json_once(path, payload)
                model_panels[task][split] = {
                    "path": str(path.relative_to(run)), "sha256": M.sha256(path),
                    "count": len(rows), "target_tokens": payload["target_tokens"],
                    "row_ids_sha256": _json_hash([row["sid"] for row in rows]),
                }
        panels[model_key] = model_panels
    return panels, dataset_sources


def _parse_models(entries):
    models = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("--model entries use name=/checkpoint")
        name, raw = entry.split("=", 1)
        if name in models or name not in MODELS:
            raise ValueError(f"invalid or duplicate model key {name!r}")
        path = Path(raw).expanduser().resolve()
        models[name] = path
    if not models:
        raise ValueError(f"a formal plan names at least one of {sorted(MODELS)}")
    return models


def _execution_protocol(search_batch_size=DEFAULT_SEARCH_BATCH_SIZE,
                        eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
                        grad_batch_size=DEFAULT_GRAD_BATCH_SIZE,
                        min_free_gpu_gib=DEFAULT_MIN_FREE_GPU_GIB):
    values = (search_batch_size, eval_batch_size, grad_batch_size)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in values):
        raise ValueError("frozen batch sizes must be positive integers")
    if (isinstance(min_free_gpu_gib, bool)
            or not isinstance(min_free_gpu_gib, (int, float))
            or not math.isfinite(min_free_gpu_gib) or min_free_gpu_gib < 0):
        raise ValueError("frozen minimum free GPU memory must be finite and non-negative")
    return {
        "search_batch_size": search_batch_size,
        "evaluation_batch_size": eval_batch_size,
        "grad_batch_size": grad_batch_size,
        "min_free_gpu_gib": float(min_free_gpu_gib),
        "min_free_policy": MIN_FREE_POLICY,
    }


def _assert_execution_protocol(plan, stage, batch_size, min_free_gpu_gib,
                               grad_batch_size=None):
    frozen = plan.get("execution_protocol")
    if not isinstance(frozen, dict):
        raise ValueError("plan lacks a frozen execution protocol")
    expected = _execution_protocol(
        frozen.get("search_batch_size"), frozen.get("evaluation_batch_size"),
        frozen.get("grad_batch_size"), frozen.get("min_free_gpu_gib"))
    if frozen != expected:
        raise ValueError("invalid frozen execution protocol")
    batch_key = "search_batch_size" if stage == "search" else "evaluation_batch_size"
    if batch_size != expected[batch_key]:
        raise ValueError(f"{stage} batch size differs from plan")
    if float(min_free_gpu_gib) != expected["min_free_gpu_gib"]:
        raise ValueError(f"{stage} minimum free GPU memory differs from plan")
    if stage == "search" and grad_batch_size != expected["grad_batch_size"]:
        raise ValueError("search gradient batch size differs from plan")
    return expected


def _bootstrap_key(region_id, repeat):
    return f"{region_id}__r{repeat}"


def _bootstrap_rows(rows, seed):
    """Draw a registered with-replacement search panel and its exact audit."""
    if not rows:
        raise ValueError("cannot bootstrap an empty search panel")
    rng = random.Random(seed)
    indices = [rng.randrange(len(rows)) for _ in range(len(rows))]
    sampled = [dict(rows[index]) for index in indices]
    _normalise_objective_weights(sampled)
    sids = [row["sid"] for row in sampled]
    audit = {
        "method": "bootstrap_with_replacement",
        "seed": seed,
        "population_rows": len(rows),
        "draws": len(sampled),
        "sampled_sids": sids,
        "sampled_sids_sha256": _json_hash(sids),
        "sampled_indices_sha256": _json_hash(indices),
    }
    return sampled, audit


def make_plan(run, models, hf_home, wikitext_train, wikitext_validation,
              formats=FORMATS, tasks=TASKS, note="",
              search_batch_size=DEFAULT_SEARCH_BATCH_SIZE,
              eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
              grad_batch_size=DEFAULT_GRAD_BATCH_SIZE,
              min_free_gpu_gib=DEFAULT_MIN_FREE_GPU_GIB):
    """Freeze the full development matrix in a fresh, empty run directory."""
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run}")
    run.mkdir(parents=True, exist_ok=True)
    if (not models or not set(models) <= set(MODELS)
            or tuple(formats) != FORMATS or tuple(tasks) != TASKS):
        raise ValueError("the formal plan requires registered models and all formats and tasks")
    execution = _execution_protocol(search_batch_size, eval_batch_size,
                                    grad_batch_size, min_free_gpu_gib)
    sources = _copy_sources(run)
    checkpoints = {name: M.checkpoint_manifest(path, name) for name, path in models.items()}
    panels, datasets = _freeze_panels(run, models, hf_home,
                                       wikitext_train, wikitext_validation)
    conditions = []
    for model_key in MODELS:                    # registry order, whatever order was given
        if model_key not in models:
            continue
        config = M.read_json(Path(models[model_key]) / "config.json")
        for fmt in FORMATS:
            dims = dims_for_format(config, fmt)
            bits = P.FORMAT_BITS[fmt]
            regions = preregistered_regions(model_key, dims, bits)
            for task in TASKS:
                cid = f"{model_key}__{fmt}__{task}"
                bootstrap_seeds = {
                    _bootstrap_key(region["id"], repeat):
                        _hash_int("search-bootstrap", cid, region["id"], repeat_seed)
                    for region in regions
                    for repeat, repeat_seed in enumerate(REPEAT_SEEDS)
                }
                conditions.append({"id": cid, "model": model_key, "format": fmt,
                                   "task": task, "dims": dims, "regions": regions,
                                   "bootstrap_seeds": bootstrap_seeds,
                                   "checkpoint_manifest_sha256": checkpoints[model_key]["manifest_sha256"],
                                   "panels": panels[model_key][task]})
    plan = {
        "schema_version": SCHEMA_VERSION, "scope": SCOPE, "created_at": M.utc_now(),
        "development_only": True, "splits": ["search", "dev"],
        "models": checkpoints, "datasets": datasets, "conditions": conditions,
        "formats": list(FORMATS), "tasks": list(TASKS), "arms": list(ARMS),
        "regions_per_condition": len(DEPTH_BANDS) * REGIONS_PER_BAND,
        "region_design": {"bands": list(DEPTH_BANDS), "kinds": list(REGION_KINDS),
                          "seed": REGION_SEED, "page_bytes": P.PAGE_BYTES},
        "repeat_seeds": list(REPEAT_SEEDS), "search_budget": R.BUDGET,
        "repeat_design": "with_replacement_bootstrap_of_frozen_search_panel",
        "execution_protocol": execution,
        "search_cost": dict(R.COST), "search_k": list(R.K_SCHEDULE),
        "evaluation_k": list(EVAL_K), "clean_gate_nll": R.GATE_NLL,
        "mmlu_score_gate": MMLU_SCORE_GATE,
        "work_items_per_condition": (len(DEPTH_BANDS) * REGIONS_PER_BAND
                                     * len(REPEAT_SEEDS) * len(ARMS)),
        "sources": sources, "note": note,
        "claim_boundary": "offline software fault simulation; no physical fault validation",
    }
    live = {record["path"]: M.sha256(C.PROJECT / record["path"]) for record in sources}
    if any(live[record["path"]] != record["sha256"] for record in sources):
        raise RuntimeError("source changed while the formal plan was being prepared")
    M.write_json_once(run / "plan.json", plan)
    M.write_json_once(run / "status.json", {
        "status": "planned", "conditions": len(conditions),
        "registered_searches": len(conditions) * plan["work_items_per_condition"],
        "plan_sha256": M.sha256(run / "plan.json"), "development_only": True,
    })
    return run


def _condition(plan, condition_id):
    matches = [item for item in plan["conditions"] if item["id"] == condition_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate condition {condition_id!r}")
    return matches[0]


def planned_items(condition, plan):
    items = []
    bootstrap_seeds = condition.get("bootstrap_seeds")
    if not isinstance(bootstrap_seeds, dict):
        raise ValueError(f"condition {condition.get('id')} lacks frozen bootstrap seeds")
    for region in condition["regions"]:
        for repeat, seed in enumerate(plan["repeat_seeds"]):
            bootstrap_key = _bootstrap_key(region["id"], repeat)
            bootstrap_seed = bootstrap_seeds.get(bootstrap_key)
            if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
                raise ValueError(f"missing bootstrap seed {bootstrap_key}")
            for arm in plan["arms"]:
                name = f"{region['id']}__r{repeat}__{arm}.json"
                items.append({"name": name, "region": region, "repeat": repeat,
                              "seed": seed, "bootstrap_seed": bootstrap_seed,
                              "arm": arm})
    return items


def shard_items(items, shard):
    try:
        index, count = (int(value) for value in shard.split("/"))
    except Exception as error:
        raise ValueError("shard must be INDEX/COUNT") from error
    if count < 1 or index not in range(count):
        raise ValueError("invalid shard index/count")
    groups = list(dict.fromkeys((item["region"]["id"], item["repeat"]) for item in items))
    mine = set(groups[index::count])
    return [item for item in items if (item["region"]["id"], item["repeat"]) in mine]


def _panel_rows(run, condition, split):
    if split not in ("search", "dev"):
        raise ValueError("only frozen search/dev panels are available")
    record = condition["panels"][split]
    path = Path(run) / record["path"]
    if M.sha256(path) != record["sha256"]:
        raise RuntimeError(f"frozen {split} panel hash drift")
    payload = M.read_json(path)
    if (payload.get("scope") != SCOPE or payload.get("split") != split
            or payload.get("model") != condition["model"]
            or payload.get("task") != condition["task"]
            or not payload.get("development_only")):
        raise ValueError(f"invalid frozen {split} panel")
    rows = payload.get("rows") or []
    if len(rows) != record["count"]:
        raise ValueError(f"frozen {split} panel count drift")
    return rows


def _verify_checkpoint(plan, condition):
    record = plan["models"][condition["model"]]
    current = M.checkpoint_manifest(record["path"], condition["model"])
    if current != record:
        raise RuntimeError(f"checkpoint drift for {condition['model']}")
    return Path(record["path"])


def _load_runtime(run, plan, condition, split, device, min_free_gpu_gib):
    checkpoint = _verify_checkpoint(plan, condition)
    rows = _panel_rows(run, condition, split)
    model, tokenizer, audit, timings = M.load_model(
        str(checkpoint), condition["format"], device,
        min_free_gpu_bytes=int(min_free_gpu_gib * 1024 ** 3))
    rt = FormalTextRuntime(model, tokenizer, {split: rows}, device)
    rt.task = condition["task"]
    if P.dims(model) != condition["dims"]:
        raise RuntimeError("loaded model dimensions differ from the frozen condition")
    return rt, rows, audit, timings


def _search_module(run):
    path = Path(run) / "source" / SEARCH_SNAPSHOT
    spec = importlib.util.spec_from_file_location("llm_region_search_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_seed(seed):
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _is_resource_error(error):
    """Recognize allocation failures that must stop a worker without evidence."""
    seen = set()
    current = error
    patterns = (
        "out of memory", "outofmemory", "cannot allocate memory",
        "can't allocate memory", "cuda_error_out_of_memory",
        "cublas_status_alloc_failed", "cuda out of memory",
    )
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, MemoryError):
            return True
        name = type(current).__name__.lower()
        message = str(current).lower()
        if "outofmemory" in name or any(pattern in message for pattern in patterns):
            return True
        current = current.__cause__ or current.__context__
    return False


@contextmanager
def _item_lock(run, stage, condition_id, name):
    directory = Path(run) / ".locks" / stage / _safe_component(condition_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_component(name)}.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"work item is already running: {stage}/{condition_id}/{name}") from error
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _artifact_path(run, condition_id, name):
    return Path(run) / "artifacts" / _safe_component(condition_id) / name


def _eval_path(run, condition_id, name):
    return Path(run) / "eval" / _safe_component(condition_id) / name


def run_search(run, condition_id, shard="0/1", device="cuda:0",
               min_free_gpu_gib=DEFAULT_MIN_FREE_GPU_GIB,
               batch_size=DEFAULT_SEARCH_BATCH_SIZE,
               grad_batch_size=DEFAULT_GRAD_BATCH_SIZE, log=print):
    run, plan = load_plan(run)
    execution = _assert_execution_protocol(
        plan, "search", batch_size, min_free_gpu_gib, grad_batch_size)
    condition = _condition(plan, condition_id)
    items = shard_items(planned_items(condition, plan), shard)
    rt, rows, audit, load_timings = _load_runtime(
        run, plan, condition, "search", device, min_free_gpu_gib)
    _assert_frozen(run, plan)
    region_search = _search_module(run)
    fingerprint = R.model_checksum(rt.lm)
    output = run / "artifacts" / _safe_component(condition_id)
    output.mkdir(parents=True, exist_ok=True)
    log(f"{condition_id}: loaded {len(rows)} frozen search rows; {len(items)} items")
    for item in items:
        path = _artifact_path(run, condition_id, item["name"])
        if path.exists():
            log(f"skip existing artifact {item['name']}")
            continue
        with _item_lock(run, "search", condition_id, item["name"]):
            if path.exists():
                continue
            _assert_frozen(run, plan)
            _set_seed(item["seed"])
            bootstrap_rows, bootstrap_audit = _bootstrap_rows(
                rows, item["bootstrap_seed"])
            rt.rows["search"] = bootstrap_rows
            started = time.monotonic()
            try:
                import torch
                if str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                ctx = R.RegionContext(rt, item["region"], item["arm"], item["seed"],
                                      budget=plan["search_budget"], log=log,
                                      bs=batch_size, grad_bs=grad_batch_size)
                try:
                    artifact = region_search.search(ctx)
                except Exception as error:
                    if _is_resource_error(error):
                        raise
                    artifact = {"error": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc()[-4000:]}
                artifact.update({
                    "schema_version": SCHEMA_VERSION, "scope": SCOPE,
                    "condition": condition_id, "model": condition["model"],
                    "format": condition["format"], "task": condition["task"],
                    "region": item["region"], "repeat": item["repeat"],
                    "seed": item["seed"], "arm": item["arm"],
                    "bootstrap_seed": item["bootstrap_seed"],
                    "bootstrap": bootstrap_audit,
                    "plan_sha256": plan["sha256"], "quantization": audit,
                    "quantization_schema_version": audit.get("schema_version"),
                    "execution_protocol": execution,
                    "spent": ctx.spent, "calls": dict(ctx.calls), "search_reuse": ctx.reuse,
                    "search_rows": len(bootstrap_rows), "search_target_tokens": sum(
                        row["target_tokens"] for row in bootstrap_rows),
                    "seconds": time.monotonic() - started, "load_timings": load_timings,
                    "peak_gpu_bytes": (torch.cuda.max_memory_allocated(device)
                                       if str(device).startswith("cuda") else None),
                })
                reasons = R.budget_reasons(artifact, plan["search_budget"])
                if reasons and "error" not in artifact:
                    artifact["error"] = "; ".join(reasons)
            except Exception as error:
                if _is_resource_error(error):
                    raise
                artifact = {"schema_version": SCHEMA_VERSION, "scope": SCOPE,
                            "condition": condition_id, "region": item["region"],
                            "repeat": item["repeat"], "seed": item["seed"],
                            "arm": item["arm"],
                            "bootstrap_seed": item["bootstrap_seed"],
                            "bootstrap": bootstrap_audit,
                            "execution_protocol": execution,
                            "plan_sha256": plan["sha256"],
                            "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc()[-4000:],
                            "seconds": time.monotonic() - started}
            finally:
                rt.rows["search"] = rows
            restored = R.model_checksum(rt.lm) == fingerprint
            if not restored:
                artifact["error"] = (artifact.get("error", "")
                                     + "; model state was not restored").lstrip("; ")
            M.write_json_once(path, artifact)
            if not restored:
                raise RuntimeError("model state changed during search; stopping this process")
            log(f"{item['name']}: {artifact.get('error') or 'complete'}")


def _logits_digest(tensor):
    value = tensor.detach().float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _task_metrics(rt, rows, task, batch_size):
    import torch
    started = time.monotonic()
    weighted = rt.nll(rows, bs=batch_size)
    records, nll_sum, tokens, finite = [], 0.0, 0, True
    for row, contribution in zip(rows, weighted):
        weight = row.get("objective_weight", 1.0)
        raw = contribution / weight if weight else float("nan")
        ok = math.isfinite(raw)
        finite &= ok
        if ok:
            nll_sum += raw * row["target_tokens"]
            tokens += row["target_tokens"]
        records.append({"sid": row["sid"], "target_tokens": row["target_tokens"],
                        "nll": P.reportable(raw), "finite": ok})
    nll = nll_sum / tokens if finite and tokens else None
    try:
        ppl = math.exp(nll) if nll is not None else None
    except OverflowError:
        ppl = None
    metrics = {"status": "complete" if finite and tokens else "nonfinite",
               "rows": len(rows), "target_tokens": tokens, "nll": nll, "ppl": ppl,
               "ppl_overflow": nll is not None and ppl is None, "records": records}
    if task == "mmlu":
        correct, nonfinite = 0, 0
        by_sid = {record["sid"]: record for record in records}
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                logits = rt.prefill_logits(batch)
                for row, row_logits in zip(batch, logits):
                    choice = torch.as_tensor(row["choice_token_ids"], device=row_logits.device)
                    scores = row_logits[choice]
                    ok = bool(torch.isfinite(scores).all())
                    prediction = int(scores.argmax().item()) if ok else None
                    hit = ok and prediction == int(row["answer"])
                    correct += int(hit)
                    nonfinite += int(not ok)
                    by_sid[row["sid"]].update(prediction=prediction,
                                              answer=int(row["answer"]), correct=bool(hit),
                                              choice_logits_finite=ok)
        metrics.update(accuracy=correct / len(rows), correct=correct,
                       nonfinite_predictions=nonfinite)
        if nonfinite:
            metrics["status"] = "nonfinite"
    metrics["seconds"] = time.monotonic() - started
    return metrics


def _artifact_reasons(rt, artifact, condition, item, plan):
    reasons = []
    expected = {"scope": SCOPE, "condition": condition["id"],
                "model": condition["model"], "format": condition["format"],
                "task": condition["task"], "region": item["region"],
                "repeat": item["repeat"], "seed": item["seed"], "arm": item["arm"],
                "bootstrap_seed": item["bootstrap_seed"],
                "execution_protocol": plan["execution_protocol"],
                "plan_sha256": plan["sha256"]}
    for key, value in expected.items():
        if artifact.get(key) != value:
            reasons.append(f"artifact {key} differs from plan")
    if "error" in artifact:
        reasons.append(artifact["error"])
    current = P.quantization_audit(rt.lm)
    audit = artifact.get("quantization") or {}
    if (artifact.get("quantization_schema_version") != 2
            or audit.get("schema_version") != 2
            or audit.get("scope") != "all_llm_parameters"
            or audit.get("quantized_fraction") != 1.0
            or audit.get("unquantized_tensors") != []
            or P.weight_format(rt.lm) != condition["format"]):
        reasons.append("artifact/model must use the planned schema-2 full-parameter format")
    for key in R.AUDIT_KEYS:
        if audit.get(key) != current.get(key):
            reasons.append(f"quantization audit differs for {key}")
    bootstrap = artifact.get("bootstrap") or {}
    sampled_sids = bootstrap.get("sampled_sids")
    expected_rows = condition["panels"]["search"]["count"]
    if (bootstrap.get("method") != "bootstrap_with_replacement"
            or bootstrap.get("seed") != item["bootstrap_seed"]
            or bootstrap.get("population_rows") != expected_rows
            or bootstrap.get("draws") != expected_rows
            or artifact.get("search_rows") != expected_rows
            or not isinstance(sampled_sids, list)
            or len(sampled_sids) != expected_rows
            or bootstrap.get("sampled_sids_sha256") != _json_hash(sampled_sids)):
        reasons.append("artifact bootstrap audit differs from the registered repeat")
    reasons.extend(R.budget_reasons(artifact, plan["search_budget"]))
    return reasons


def _ppl_ratio(delta):
    if delta is None or not math.isfinite(delta):
        return None
    try:
        return math.exp(delta)
    except OverflowError:
        return None


def _damage_metrics(task, faulted, rearranged, original_clean):
    """Measure the fault against its own unfaulted deployed layout.

    The original-layout comparison remains diagnostic only; it never controls
    the success decision.
    """
    delta = (faulted["nll"] - rearranged["nll"]
             if faulted.get("nll") is not None and rearranged.get("nll") is not None
             else None)
    original_delta = (faulted["nll"] - original_clean["nll"]
                      if faulted.get("nll") is not None
                      and original_clean.get("nll") is not None else None)
    out = {
        "damage_baseline": "same_rearranged_unfaulted_layout",
        "delta_nll": delta,
        "ppl_ratio": _ppl_ratio(delta),
        "diagnostic_delta_nll_from_original_clean": original_delta,
        "diagnostic_ppl_ratio_from_original_clean": _ppl_ratio(original_delta),
    }
    if task == "mmlu":
        base = rearranged.get("accuracy")
        original_base = original_clean.get("accuracy")
        out["relative_score_drop"] = (
            (base - faulted["accuracy"]) / base if base is not None and base > 0 else None)
        out["diagnostic_relative_score_drop_from_original_clean"] = (
            (original_base - faulted["accuracy"]) / original_base
            if original_base is not None and original_base > 0 else None)
        out["meets_success_threshold"] = bool(
            out["relative_score_drop"] is not None
            and out["relative_score_drop"] >= 0.5)
    else:
        out["meets_success_threshold"] = bool(
            out["ppl_ratio"] is not None and out["ppl_ratio"] >= 2.0)
    return out


def _result_summary(results, task):
    valid = [value for value in results.values() if value.get("valid")]
    out = {"valid_evaluations": len(valid), "registered_evaluations": len(EVAL_K),
           "max_delta_nll": max((value["delta_nll"] for value in valid), default=None),
           "max_ppl_ratio": max((value["ppl_ratio"] for value in valid
                                 if value.get("ppl_ratio") is not None), default=None),
           "success_any": any(value.get("success") for value in valid)}
    if task == "mmlu":
        out["max_relative_score_drop"] = max(
            (value.get("relative_score_drop", 0.0) for value in valid), default=None)
    return out


def _evaluate_one(rt, rows, clean, condition, item, artifact, artifact_file_sha256,
                  plan, batch_size, log):
    import torch
    started = time.monotonic()
    report = {"schema_version": SCHEMA_VERSION, "scope": SCOPE,
              "condition": condition["id"], "region": item["region"],
              "repeat": item["repeat"], "seed": item["seed"], "arm": item["arm"],
              "plan_sha256": plan["sha256"],
              "artifact_file_sha256": artifact_file_sha256,
              "artifact_object_sha256": _json_hash(artifact),
              "clean": clean, "evaluation_k": list(EVAL_K)}
    reasons = _artifact_reasons(rt, artifact, condition, item, plan)
    try:
        spec = P.validate_spec(artifact.get("permutation"), condition["dims"])
        reasons.extend(R.arm_reasons(item["region"], item["arm"], spec,
                                     condition["dims"], item["seed"]))
        attacks = {}
        for key, value in (artifact.get("attacks") or {}).items():
            parsed = int(key)
            if str(parsed) != str(key) or parsed not in plan["search_k"]:
                raise ValueError(f"attack key {key!r} is outside the frozen search schedule")
            attacks[parsed] = value
    except (ValueError, TypeError, AttributeError) as error:
        reasons.append(str(error))
        spec, attacks = P.validate_spec({}, condition["dims"]), {}
    if reasons:
        report.update(status="artifact_error", artifact_error="; ".join(reasons),
                      results={}, summary=_result_summary({}, condition["task"]),
                      seconds=time.monotonic() - started)
        return report
    permuted = not P.is_identity(spec)
    if permuted:
        P.apply_permutation(rt, spec)
    try:
        rearranged = _task_metrics(rt, rows, condition["task"], batch_size)
        logits = rt.prefill_logits(rows[:P.LOGIT_ROWS])
        reference_logits = clean["_logits"]
        logit_ok = (bool(torch.isfinite(logits).all())
                    and torch.allclose(logits, reference_logits,
                                       atol=P.LOGIT_ATOL, rtol=P.LOGIT_RTOL))
        drift = (rearranged["nll"] - clean["nll"]
                 if rearranged["nll"] is not None and clean["nll"] is not None else None)
        score_ok = (condition["task"] != "mmlu"
                    or abs(rearranged["accuracy"] - clean["accuracy"]) <= plan["mmlu_score_gate"])
        gate = (rearranged["status"] == "complete" and drift is not None
                and abs(drift) <= plan["clean_gate_nll"] and logit_ok and score_ok)
        report["rearranged"] = {
            "identity": not permuted, "metrics": rearranged, "nll_drift": drift,
            "logits_allclose": logit_ok, "logits_sha256": _logits_digest(logits),
            "score_gate": score_ok, "gate": gate,
            "role": "unfaulted_damage_baseline",
        }
        results = {}
        for k in EVAL_K:
            flips = attacks.get(k)
            if flips is None:
                results[str(k)] = {"k": k, "valid": False, "reasons": ["missing"]}
                continue
            check = R.check_region_attack(rt, item["region"], flips, k)
            if not gate:
                check["valid"] = False
                check["reasons"].append("clean layout gate failed")
            if check["valid"]:
                normalized = P.normalize_flips(flips)
                P.apply_flips(rt.lm, normalized)
                try:
                    faulted = _task_metrics(rt, rows, condition["task"], batch_size)
                finally:
                    P.apply_flips(rt.lm, normalized)
                if faulted["status"] != "complete":
                    check["valid"] = False
                    check["reasons"].append("task metric became non-finite")
                damage = _damage_metrics(
                    condition["task"], faulted, rearranged, clean)
                check.update(metrics=faulted, **damage)
                check["success"] = bool(
                    check["valid"] and damage["meets_success_threshold"])
            results[str(k)] = check
            log(f"{item['name']} k={k}: valid={check['valid']}")
    finally:
        if permuted:
            P.apply_permutation(rt, spec, inverse=True)
    report.update(status="complete", results=results,
                  summary=_result_summary(results, condition["task"]),
                  seconds=time.monotonic() - started)
    return report


def run_evaluate(run, condition_id, shard="0/1", device="cuda:0",
                 min_free_gpu_gib=DEFAULT_MIN_FREE_GPU_GIB,
                 batch_size=DEFAULT_EVAL_BATCH_SIZE, log=print):
    run, plan = load_plan(run)
    execution = _assert_execution_protocol(
        plan, "evaluate", batch_size, min_free_gpu_gib)
    condition = _condition(plan, condition_id)
    items = shard_items(planned_items(condition, plan), shard)
    rt, rows, audit, load_timings = _load_runtime(
        run, plan, condition, "dev", device, min_free_gpu_gib)
    _assert_frozen(run, plan)
    logits = rt.prefill_logits(rows[:P.LOGIT_ROWS])
    clean = _task_metrics(rt, rows, condition["task"], batch_size)
    clean.update({"logits_sha256": _logits_digest(logits), "quantization": audit,
                  "load_timings": load_timings, "_logits": logits})
    fingerprint = R.model_checksum(rt.lm)
    output = run / "eval" / _safe_component(condition_id)
    output.mkdir(parents=True, exist_ok=True)
    log(f"{condition_id}: clean dev NLL={clean['nll']}; {len(items)} reports")
    for item in items:
        path = _eval_path(run, condition_id, item["name"])
        if path.exists():
            continue
        source = _artifact_path(run, condition_id, item["name"])
        if not source.exists():
            log(f"skip {item['name']}: artifact absent")
            continue
        with _item_lock(run, "evaluate", condition_id, item["name"]):
            if path.exists():
                continue
            _assert_frozen(run, plan)
            artifact = M.read_json(source)
            clean_public = {key: value for key, value in clean.items()
                            if not key.startswith("_logits")}
            clean_for_item = {**clean_public, "_logits": clean["_logits"]}
            started = time.monotonic()
            try:
                report = _evaluate_one(rt, rows, clean_for_item, condition, item, artifact,
                                       M.sha256(source), plan, batch_size, log)
            except Exception as error:
                if _is_resource_error(error):
                    raise
                report = {"schema_version": SCHEMA_VERSION, "scope": SCOPE,
                          "condition": condition_id, "region": item["region"],
                          "repeat": item["repeat"], "seed": item["seed"],
                          "arm": item["arm"], "plan_sha256": plan["sha256"],
                          "artifact_file_sha256": M.sha256(source),
                          "status": "evaluation_error",
                          "error": f"{type(error).__name__}: {error}",
                          "traceback": traceback.format_exc()[-4000:],
                          "seconds": time.monotonic() - started}
            # Private tensors never enter the evidence file.
            report["clean"] = clean_public
            report["execution_protocol"] = execution
            if R.model_checksum(rt.lm) != fingerprint:
                raise RuntimeError("model state changed during evaluation; report not published")
            M.write_json_once(path, report)


def run_status(run):
    run, plan = load_plan(run)
    conditions = []
    for condition in plan["conditions"]:
        items = planned_items(condition, plan)
        artifact_paths = [_artifact_path(run, condition["id"], item["name"]) for item in items]
        eval_paths = [_eval_path(run, condition["id"], item["name"]) for item in items]
        conditions.append({"id": condition["id"], "registered": len(items),
                           "artifacts": sum(path.is_file() for path in artifact_paths),
                           "evaluations": sum(path.is_file() for path in eval_paths)})
    return {"scope": SCOPE, "plan_sha256": plan["sha256"],
            "development_only": True, "conditions": conditions,
            "artifact_total": sum(item["artifacts"] for item in conditions),
            "evaluation_total": sum(item["evaluations"] for item in conditions)}


def parser():
    root = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = root.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--model", action="append", required=True,
                      help="repeat once per registered model to include: model-key=/checkpoint")
    plan.add_argument("--hf-home", default=str(C.HUB))
    plan.add_argument("--wikitext-train", required=True)
    plan.add_argument("--wikitext-validation", required=True)
    plan.add_argument("--output")
    plan.add_argument("--note", default="")
    plan.add_argument("--search-batch-size", type=int,
                      default=DEFAULT_SEARCH_BATCH_SIZE)
    plan.add_argument("--eval-batch-size", type=int,
                      default=DEFAULT_EVAL_BATCH_SIZE)
    plan.add_argument("--grad-batch-size", type=int,
                      default=DEFAULT_GRAD_BATCH_SIZE)
    plan.add_argument("--min-free-gpu-gib", type=float,
                      default=DEFAULT_MIN_FREE_GPU_GIB)
    for name in ("search", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        command.add_argument("--condition", required=True)
        command.add_argument("--shard", default="0/1")
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--min-free-gpu-gib", type=float,
                             default=DEFAULT_MIN_FREE_GPU_GIB)
        command.add_argument("--batch-size", type=int, default=(
            DEFAULT_SEARCH_BATCH_SIZE if name == "search" else DEFAULT_EVAL_BATCH_SIZE))
        if name == "search":
            command.add_argument("--grad-batch-size", type=int,
                                 default=DEFAULT_GRAD_BATCH_SIZE)
    status = commands.add_parser("status")
    status.add_argument("--run", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    C.setup_environment()
    if args.command == "plan":
        models = _parse_models(args.model)
        output = C.fresh_output_dir("llm_region_study", "formal_dev", args.output)
        make_plan(output, models, args.hf_home, args.wikitext_train,
                  args.wikitext_validation, note=args.note,
                  search_batch_size=args.search_batch_size,
                  eval_batch_size=args.eval_batch_size,
                  grad_batch_size=args.grad_batch_size,
                  min_free_gpu_gib=args.min_free_gpu_gib)
        print(output, flush=True)
    elif args.command == "search":
        if args.batch_size < 1 or args.grad_batch_size < 1 or args.min_free_gpu_gib < 0:
            raise ValueError("batch sizes must be positive and minimum free memory non-negative")
        run_search(args.run, args.condition, args.shard, args.device,
                   args.min_free_gpu_gib, args.batch_size, args.grad_batch_size)
    elif args.command == "evaluate":
        if args.batch_size < 1 or args.min_free_gpu_gib < 0:
            raise ValueError("batch size must be positive and minimum free memory non-negative")
        run_evaluate(args.run, args.condition, args.shard, args.device,
                     args.min_free_gpu_gib, args.batch_size)
    else:
        print(json.dumps(run_status(args.run), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
