"""Audited development-only VQA RegionAlign matrix.

The plan freezes LLaVA-1.5/Qwen2-VL x INT8/INT4 x OK-VQA/TextVQA,
twelve result-independent 4 KiB regions, three repeat seeds and the matched
noperm/local/random arms.  Only train-derived search/dev panels are accepted;
there is no test/final command.  A task whose train data is absent remains in
the matrix as an explicit unavailable condition and cannot be executed.

Search and evaluation are separate processes.  Sources, checkpoints, input
Arrow files and the selected binary panels are hashed before execution.
Artifacts and evaluation reports are atomically published once and protected
by persistent per-item locks.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from regionalign.harness import core as P, region as R
from regionalign.runtimes import vqa as V
from regionalign.experiments import clean_mmlu as M
from regionalign import config as C


SCHEMA_VERSION = 2
SCOPE = "vqa_regionalign_dev"
# ``MODELS`` is the frozen official OK-VQA parent set: ``official_okvqa`` (which
# must not change) binds its four-condition parent to exactly these two VLMs x
# two formats.  The development matrix uses ``DEV_MODELS`` instead, which is a
# non-empty subset of ``V.MODEL_KEYS`` and may hold any number of models.
MODELS = ("llava-1.5-7b", "qwen2-vl-7b")
DEV_MODELS = ("qwen2-vl-2b",)
FORMATS = ("int8", "int4")
TASKS = V.TASKS
ARMS = ("noperm", "local", "random")
DEPTH_BANDS = ("early", "middle", "late")
REGION_KINDS = ("q_proj", "v_proj", "gate_proj", "down_proj")
REPEAT_SEEDS = (2026111001, 2026111002, 2026111003)
REGION_SEED = 2026111000
EVAL_K = (1, 2, 4, 8, 16)
VQA_SCORE_GATE = 0.02
VQA_SUCCESS_DROP = 0.50
SEARCH_BATCH_SIZE = 1
GRAD_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
MIN_FREE_GPU_GIB = 40.0
MIN_FREE_POLICY = "available_cuda_memory_before_model_load"

MODEL_SPECS = {
    "llava-1.5-7b": {
        "cache_component": "models--llava-hf--llava-1.5-7b-hf",
        "repository": "llava-hf/llava-1.5-7b-hf", "model_type": "llava",
        "layers": 32, "hidden": 4096, "mlp": 11008, "heads": 32, "kv_heads": 32,
    },
    "qwen2-vl-7b": {
        "cache_component": "models--Qwen--Qwen2-VL-7B-Instruct",
        "repository": "Qwen/Qwen2-VL-7B-Instruct", "model_type": "qwen2_vl",
        "layers": 28, "hidden": 3584, "mlp": 18944, "heads": 28, "kv_heads": 4,
    },
    "qwen2-vl-2b": {
        "cache_component": "models--Qwen--Qwen2-VL-2B-Instruct",
        "repository": "Qwen/Qwen2-VL-2B-Instruct", "model_type": "qwen2_vl",
        "layers": 28, "hidden": 1536, "mlp": 8960, "heads": 12, "kv_heads": 2,
    },
}

SOURCE_RELATIVE = (
    Path("regionalign/__init__.py"),
    Path("regionalign/config.py"),
    Path("regionalign/experiments/__init__.py"),
    Path("regionalign/experiments/vqa.py"),
    Path("regionalign/experiments/clean_mmlu.py"),
    Path("regionalign/experiments/clean_benchmarks.py"),
    Path("regionalign/runtimes/__init__.py"),
    Path("regionalign/runtimes/vqa.py"),
    Path("regionalign/runtimes/vqa_score.py"),
    Path("regionalign/harness/__init__.py"),
    Path("regionalign/harness/core.py"),
    Path("regionalign/harness/region.py"),
    Path("regionalign/algorithm/__init__.py"),
    Path("regionalign/algorithm/search.py"),
    Path("regionalign/runtime.json"),
)
SEARCH_SNAPSHOT = "regionalign__algorithm__search.py"


def _safe_component(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    if not value or value in (".", ".."):
        raise ValueError(f"unsafe path component {value!r}")
    return value


def _json_hash(value):
    return M.object_sha256(value)


def _source_record(path):
    path = Path(path).expanduser().absolute()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": M.sha256(path)}


def _copy_sources(run):
    target = Path(run) / "source"
    target.mkdir()
    records = []
    for relative in SOURCE_RELATIVE:
        live = C.PROJECT / relative
        if not live.is_file():
            raise FileNotFoundError(live)
        name = str(relative).replace("/", "__")
        snapshot = target / name
        shutil.copyfile(live, snapshot)
        digest = M.sha256(live)
        if M.sha256(snapshot) != digest:
            raise RuntimeError(f"source changed while freezing: {relative}")
        records.append({"path": str(relative), "snapshot": f"source/{name}",
                        "sha256": digest, "bytes": live.stat().st_size})
    return records


def _assert_frozen(run, plan):
    drift = []
    for record in plan.get("sources", []):
        live, saved = C.PROJECT / record["path"], Path(run) / record["snapshot"]
        if (not live.is_file() or not saved.is_file()
                or M.sha256(live) != record["sha256"]
                or M.sha256(saved) != record["sha256"]):
            drift.append(record["path"])
    if drift:
        raise RuntimeError(f"sources changed after planning: {drift}; create a new run")


def load_plan(run):
    run = C.output_path(run)
    path = run / "plan.json"
    plan = M.read_json(path)
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("scope") != SCOPE:
        raise ValueError("unsupported VQA RegionAlign plan")
    if not plan.get("development_only") or plan.get("splits") != ["search", "dev"]:
        raise ValueError("formal VQA plan must contain search/dev only")
    expected_execution = {"search_batch_size": SEARCH_BATCH_SIZE,
                          "grad_batch_size": GRAD_BATCH_SIZE,
                          "eval_batch_size": EVAL_BATCH_SIZE,
                          "min_free_gpu_gib": MIN_FREE_GPU_GIB,
                          "min_free_policy": MIN_FREE_POLICY,
                          "real_gpu_preflight_required": True}
    if plan.get("execution") != expected_execution:
        raise ValueError("formal VQA execution parameters differ from the frozen protocol")
    _assert_frozen(run, plan)
    plan_sha256 = M.sha256(path)
    status = M.read_json(run / "status.json")
    if (status.get("plan_sha256") != plan_sha256
            or status.get("development_only") is not True):
        raise ValueError("status.json does not bind this formal VQA plan")
    plan["sha256"] = plan_sha256
    return run, plan


def _hash_int(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def depth_ranges(layers):
    if type(layers) is not int or layers < 3:
        raise ValueError("a formal model needs at least three decoder layers")
    first, second = layers // 3, 2 * layers // 3
    return {"early": range(first), "middle": range(first, second),
            "late": range(second, layers)}


def preregistered_regions(model_key, dims, bits, seed=REGION_SEED):
    """Twelve regions chosen from identifiers, before any sensitivity signal."""
    if bits not in (4, 8):
        raise ValueError("formal VQA regions support only INT8 and INT4")
    regions = []
    for band, layer_range in depth_ranges(dims["layers"]).items():
        choices = list(layer_range)
        for kind in REGION_KINDS:
            layer = choices[_hash_int(seed, model_key, band, kind, "layer") % len(choices)]
            pages = R.page_count(kind, dims, bits=bits, page_bytes=P.PAGE_BYTES)
            page = _hash_int(seed, model_key, band, kind, "page") % pages
            region = R.make_region(kind, layer, page, dims, bits=bits,
                                   page_bytes=P.PAGE_BYTES)
            region.update(depth_band=band, selection_seed=seed)
            regions.append(region)
    if len(regions) != 12:
        raise AssertionError("formal VQA region count changed")
    return regions


def _format_dims(base_dims, fmt):
    if fmt not in FORMATS:
        raise ValueError(f"unknown formal format {fmt!r}")
    return {**base_dims, "group": P.GROUP if fmt == "int4" else None}


def _parse_models(entries):
    models = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("--model entries use name=/checkpoint")
        name, raw = entry.split("=", 1)
        if name not in DEV_MODELS or name in models:
            raise ValueError(f"invalid or duplicate model key {name!r}")
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        models[name] = path
    if not models:
        raise ValueError(f"a formal plan names at least one of {sorted(DEV_MODELS)}")
    return models


def _language_config(checkpoint):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True,
                                        trust_remote_code=False)
    return getattr(config, "text_config", config)


def _checkpoint_manifest(checkpoint, model_key):
    checkpoint = Path(checkpoint).resolve()
    expected = MODEL_SPECS[model_key]
    if expected["cache_component"] not in checkpoint.parts:
        raise ValueError(f"{model_key} must use {expected['repository']}; got {checkpoint}")
    raw = M.read_json(checkpoint / "config.json")
    if raw.get("model_type") != expected["model_type"]:
        raise ValueError(f"wrong model_type for {model_key}: {raw.get('model_type')!r}")
    config = _language_config(checkpoint)
    dims = P.dims_from_config(config)
    observed = {key: dims[key] for key in ("layers", "hidden", "mlp", "heads")}
    observed["kv_heads"] = dims.get("kv_heads", dims["heads"])
    wanted = {key: expected[key] for key in observed}
    if observed != wanted:
        raise ValueError(f"{model_key} language dimensions differ: {observed} != {wanted}")
    files = set(M.checkpoint_files(checkpoint)) | set(checkpoint.glob("*.json"))
    records = [M.file_record(path, checkpoint) for path in sorted(files)]
    manifest = {"path": str(checkpoint), "model_key": model_key,
                "repository": expected["repository"], "model_type": raw["model_type"],
                "revision": checkpoint.name if checkpoint.parent.name == "snapshots" else None,
                "dims": dims, "files": records}
    manifest["manifest_sha256"] = _json_hash(manifest)
    return manifest


def _training_paths(paths, task):
    result = []
    for raw in paths or []:
        path = Path(raw).expanduser().resolve()
        name = path.name.lower()
        forbidden = any(re.search(r"(^|[-_.])(validation|val2014|test)($|[-_.])",
                                  part.lower()) for part in path.parts)
        if (not path.is_file() or "train" not in name
                or forbidden):
            raise ValueError(f"{task} requires explicit train Arrow files, got {path}")
        result.append(path)
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate {task} training Arrow path")
    return V.validate_train_paths(result, task) if result else []


def _bootstrap_records(rows, task):
    """Freeze one with-replacement search draw for each registered repeat."""
    sids = [row["sid"] for row in rows]
    if len(sids) != V.SEARCH_ROWS or len(set(sids)) != len(sids):
        raise ValueError(f"{task} search panel must contain unique frozen SIDs")
    records = []
    for repeat, seed in enumerate(REPEAT_SEEDS):
        rng = random.Random(seed)
        drawn = [sids[rng.randrange(len(sids))] for _ in range(len(sids))]
        records.append({"repeat": repeat, "seed": seed, "with_replacement": True,
                        "draws": len(drawn), "unique_sids": len(set(drawn)),
                        "sids": drawn, "sid_sha256": _json_hash(drawn)})
    return records


def _write_arrow_once(path, rows):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temp.open("xb") as handle:
            with ipc.new_stream(handle, table.schema) as writer:
                writer.write_table(table)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error
    finally:
        temp.unlink(missing_ok=True)


def _panel_record(run, task, split, rows):
    path = Path(run) / "panels" / task / f"{split}.arrow"
    _write_arrow_once(path, rows)
    identities = [{key: row[key] for key in (
        "sid", "question_id", "image_id", "image_sha256", "question_sha256",
        "answers_sha256", "reference", "rank")} for row in rows]
    return {"path": str(path.relative_to(run)), "sha256": M.sha256(path),
            "bytes": path.stat().st_size, "count": len(rows),
            "row_ids_sha256": _json_hash([row["sid"] for row in rows]),
            "identities_sha256": _json_hash(identities)}


def _freeze_dataset(run, task, paths, missing):
    if missing:
        if paths:
            raise ValueError(f"{task} cannot be both supplied and marked missing")
        return {"status": "unavailable", "reason": "train Arrow files not locally available",
                "search_rows": V.SEARCH_ROWS, "dev_rows": V.DEV_ROWS}
    if not paths:
        raise ValueError(f"{task} requires train files or --missing-task {task}")
    panels = V.load_train_panels(paths, task)
    record = {"status": "available", "source_files": [_source_record(path) for path in paths],
              "dataset_binding": {key: value for key, value in V.DATASET_SPECS[task].items()
                                  if key != "filename"},
              "source_filenames": [path.name for path in paths],
              "selection_seed": V.PANEL_SEED, "image_disjoint": True, "panels": {},
              "search_bootstraps": _bootstrap_records(panels["search"], task)}
    for split in ("search", "dev"):
        record["panels"][split] = _panel_record(run, task, split, panels[split])
    return record


def make_plan(run, models, task_paths, missing_tasks=(), note=""):
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run}")
    run.mkdir(parents=True, exist_ok=True)
    if not models or not set(models) <= set(DEV_MODELS):
        raise ValueError(f"formal VQA plan names a non-empty subset of {sorted(DEV_MODELS)}")
    missing_tasks = set(missing_tasks)
    if missing_tasks - set(TASKS):
        raise ValueError(f"unknown missing tasks: {sorted(missing_tasks - set(TASKS))}")
    sources = _copy_sources(run)
    checkpoints = {key: _checkpoint_manifest(path, key) for key, path in models.items()}
    normalized_paths = {task: _training_paths(task_paths.get(task), task) for task in TASKS}
    supplied = [path for paths in normalized_paths.values() for path in paths]
    if len(supplied) != len(set(supplied)):
        raise ValueError("OK-VQA and TextVQA may not share source files")
    source_hashes = [{M.sha256(path) for path in normalized_paths[task]} for task in TASKS]
    if source_hashes[0] & source_hashes[1]:
        raise ValueError("OK-VQA and TextVQA may not share source contents")
    datasets = {task: _freeze_dataset(run, task, normalized_paths[task],
                                      task in missing_tasks) for task in TASKS}
    conditions = []
    for model_key in [key for key in DEV_MODELS if key in models]:
        base_dims = checkpoints[model_key]["dims"]
        for fmt in FORMATS:
            # Legal INT4 input-axis transforms must preserve the same scale
            # groups that the loaded quantizer will attach to the model.
            dims = _format_dims(base_dims, fmt)
            regions = preregistered_regions(model_key, dims, P.FORMAT_BITS[fmt])
            for task in TASKS:
                available = datasets[task]["status"] == "available"
                conditions.append({
                    "id": f"{model_key}__{fmt}__{task}", "model": model_key,
                    "format": fmt, "task": task, "dims": dims, "regions": regions,
                    "availability": "available" if available else "unavailable",
                    "unavailable_reason": None if available else datasets[task]["reason"],
                    "checkpoint_manifest_sha256": checkpoints[model_key]["manifest_sha256"],
                    "panels": datasets[task].get("panels"),
                    "search_bootstraps": datasets[task].get("search_bootstraps"),
                })
    per_condition = 12 * len(REPEAT_SEEDS) * len(ARMS)
    available = sum(item["availability"] == "available" for item in conditions)
    plan = {
        "schema_version": SCHEMA_VERSION, "scope": SCOPE, "created_at": M.utc_now(),
        "development_only": True, "splits": ["search", "dev"],
        "models": checkpoints, "datasets": datasets, "conditions": conditions,
        "formats": list(FORMATS), "tasks": list(TASKS), "arms": list(ARMS),
        "regions_per_condition": 12,
        "region_design": {"bands": list(DEPTH_BANDS), "kinds": list(REGION_KINDS),
                          "seed": REGION_SEED, "page_bytes": P.PAGE_BYTES},
        "repeat_seeds": list(REPEAT_SEEDS), "search_budget": R.BUDGET,
        "search_cost": dict(R.COST), "search_k": list(R.K_SCHEDULE),
        "evaluation_k": list(EVAL_K), "clean_gate_nll": R.GATE_NLL,
        "vqa_score_gate": VQA_SCORE_GATE, "success_relative_score_drop": VQA_SUCCESS_DROP,
        "execution": {"search_batch_size": SEARCH_BATCH_SIZE,
                      "grad_batch_size": GRAD_BATCH_SIZE,
                      "eval_batch_size": EVAL_BATCH_SIZE,
                      "min_free_gpu_gib": MIN_FREE_GPU_GIB,
                      "min_free_policy": MIN_FREE_POLICY,
                      "real_gpu_preflight_required": True},
        "work_items_per_condition": per_condition,
        "registered_available_searches": available * per_condition,
        "sources": sources, "note": note,
        "claim_boundary": "offline software fault simulation; no physical fault validation",
    }
    if any(M.sha256(C.PROJECT / row["path"]) != row["sha256"] for row in sources):
        raise RuntimeError("source changed while preparing the formal VQA plan")
    M.write_json_once(run / "plan.json", plan)
    M.write_json_once(run / "status.json", {
        "status": "planned", "conditions": len(conditions),
        "available_conditions": available, "unavailable_conditions": len(conditions) - available,
        "registered_available_searches": available * per_condition,
        "plan_sha256": M.sha256(run / "plan.json"), "development_only": True,
    })
    return run


def _condition(plan, condition_id):
    found = [row for row in plan["conditions"] if row["id"] == condition_id]
    if len(found) != 1:
        raise ValueError(f"unknown or duplicate condition {condition_id!r}")
    condition = found[0]
    if condition["availability"] != "available":
        raise RuntimeError(f"condition {condition_id} is unavailable: {condition['unavailable_reason']}")
    return condition


def planned_items(condition, plan):
    return [{"name": f"{region['id']}__r{repeat}__{arm}.json", "region": region,
             "repeat": repeat, "seed": seed, "arm": arm}
            for region in condition["regions"]
            for repeat, seed in enumerate(plan["repeat_seeds"])
            for arm in plan["arms"]]


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
            raise RuntimeError(f"work item already running: {stage}/{condition_id}/{name}") from error
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _read_panel(run, condition, split):
    if split not in ("search", "dev"):
        raise ValueError("only frozen search/dev VQA panels are available")
    record = condition["panels"][split]
    path = Path(run) / record["path"]
    if M.sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
        raise RuntimeError(f"frozen {split} panel drift")
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(path), "r") as source:
        rows = ipc.open_stream(source).read_all().to_pylist()
    if len(rows) != record["count"]:
        raise ValueError(f"frozen {split} panel count drift")
    identities = []
    for row in rows:
        image_bytes = row.get("image", {}).get("bytes")
        if not image_bytes or hashlib.sha256(image_bytes).hexdigest() != row["image_sha256"]:
            raise ValueError(f"frozen image identity drift for {row.get('sid')}")
        identities.append({key: row[key] for key in (
            "sid", "question_id", "image_id", "image_sha256", "question_sha256",
            "answers_sha256", "reference", "rank")})
    if (_json_hash([row["sid"] for row in rows]) != record["row_ids_sha256"]
            or _json_hash(identities) != record["identities_sha256"]):
        raise ValueError(f"frozen {split} panel identity drift")
    return rows


def _bootstrap_record(condition, repeat):
    records = condition.get("search_bootstraps") or []
    found = [record for record in records if record.get("repeat") == repeat]
    if len(found) != 1:
        raise ValueError(f"missing or duplicate search bootstrap for repeat {repeat}")
    record = found[0]
    if (record.get("seed") != REPEAT_SEEDS[repeat]
            or record.get("with_replacement") is not True
            or record.get("draws") != V.SEARCH_ROWS
            or record.get("sid_sha256") != _json_hash(record.get("sids"))):
        raise ValueError(f"invalid frozen search bootstrap for repeat {repeat}")
    return record


def _bootstrap_rows(rows, record):
    by_sid = {row["sid"]: row for row in rows}
    if len(by_sid) != len(rows):
        raise ValueError("frozen search panel has duplicate SIDs")
    try:
        selected = [by_sid[sid] for sid in record["sids"]]
    except KeyError as error:
        raise ValueError(f"bootstrap references an unknown search SID: {error}") from error
    if len(selected) != record["draws"]:
        raise ValueError("bootstrap draw count drift")
    return selected


def _resource_error(error):
    if isinstance(error, MemoryError):
        return True
    if isinstance(error, OSError) and error.errno in (errno.ENOMEM, errno.ENOSPC, errno.EDQUOT):
        return True
    text = str(error).lower()
    return any(token in text for token in (
        "out of memory", "cuda error", "cublas_status_alloc_failed",
        "no space left", "disk quota", "cannot allocate memory"))


def _require_gpu_resources(device, execution):
    """Fail before model loading when the frozen GPU requirement is not met."""
    import torch
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal VQA execution requires a real CUDA GPU")
    free, total = torch.cuda.mem_get_info(target)
    minimum = int(execution["min_free_gpu_gib"] * 1024 ** 3)
    if free < minimum:
        raise RuntimeError(
            f"insufficient free GPU memory before model load: {free / 1024**3:.2f} GiB "
            f"< frozen {execution['min_free_gpu_gib']:.2f} GiB")
    properties = torch.cuda.get_device_properties(target)
    return {"device": str(target), "name": properties.name,
            "free_bytes": free, "total_bytes": total,
            "minimum_free_bytes": minimum, "policy": execution["min_free_policy"]}


def _preflight_path(run, condition_id):
    return Path(run) / "preflight" / f"{_safe_component(condition_id)}.json"


def _require_preflight(run, plan, condition):
    path = _preflight_path(run, condition["id"])
    if not path.is_file():
        raise RuntimeError(f"real-GPU preflight is required first for {condition['id']}")
    record = M.read_json(path)
    if (record.get("status") != "complete" or record.get("plan_sha256") != plan["sha256"]
            or record.get("condition") != condition["id"]
            or record.get("matched_arms") != list(ARMS)):
        raise RuntimeError(f"invalid real-GPU preflight for {condition['id']}")
    return record


def _verify_checkpoint(plan, condition):
    expected = plan["models"][condition["model"]]
    current = _checkpoint_manifest(expected["path"], condition["model"])
    if current != expected:
        raise RuntimeError(f"checkpoint drift for {condition['model']}")
    return Path(expected["path"])


def _load_runtime(run, plan, condition, split, device):
    _require_gpu_resources(device, plan["execution"])
    checkpoint = _verify_checkpoint(plan, condition)
    rows = _read_panel(run, condition, split)
    runtime = V.HFVQARegionRuntime.load(condition["model"], checkpoint, condition["format"],
                                        {split: rows}, condition["task"], device)
    if P.dims(runtime.lm) != condition["dims"]:
        runtime.close()
        raise RuntimeError("loaded VLM language dimensions differ from frozen condition")
    return runtime, runtime.rows[split]


def _search_module(run):
    path = Path(run) / "source" / SEARCH_SNAPSHOT
    spec = importlib.util.spec_from_file_location("vqa_region_search_snapshot", path)
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


def _artifact_path(run, condition_id, name):
    return Path(run) / "artifacts" / _safe_component(condition_id) / name


def _eval_path(run, condition_id, name):
    return Path(run) / "eval" / _safe_component(condition_id) / name


def run_preflight(run, condition_id, device="cuda:0", log=print):
    """Exercise one matched three-arm group on the real model and CUDA path."""
    import torch
    run, plan = load_plan(run)
    condition = _condition(plan, condition_id)
    path = _preflight_path(run, condition_id)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    resources = _require_gpu_resources(device, plan["execution"])
    runtime, frozen_rows = _load_runtime(run, plan, condition, "search", device)
    try:
        _assert_frozen(run, plan)
        item = planned_items(condition, plan)[0]
        sample_record = _bootstrap_record(condition, item["repeat"])
        sampled = _bootstrap_rows(frozen_rows, sample_record)
        runtime.rows["search"] = runtime._prepare_rows(sampled)
        probe_rows = runtime.rows["search"][:1]
        region, dims = item["region"], condition["dims"]
        baseline = runtime.prefill_logits(probe_rows)
        fingerprint = R.model_checksum(runtime.lm)
        candidates = R.reachable_rows(region, dims)
        local_row = candidates[-1]
        specs = {
            "noperm": P.validate_spec({}, dims),
            "local": R.placement(region, dims, local_row),
            "random": R.placement(region, dims, R.random_row(region, dims, item["seed"])),
        }
        arms = {}
        for arm in ARMS:
            spec = specs[arm]
            reasons = R.arm_reasons(region, arm, spec, dims, item["seed"])
            if reasons:
                raise RuntimeError(f"preflight {arm} layout invalid: {reasons}")
            permuted = not P.is_identity(spec)
            if permuted:
                P.apply_permutation(runtime, spec)
            try:
                logits = runtime.prefill_logits(probe_rows)
                arms[arm] = {"logits_allclose": bool(torch.allclose(
                    logits, baseline, atol=P.LOGIT_ATOL, rtol=P.LOGIT_RTOL)),
                    "logits_sha256": _logits_digest(logits)}
            finally:
                if permuted:
                    P.apply_permutation(runtime, spec, inverse=True)
            if not arms[arm]["logits_allclose"] or R.model_checksum(runtime.lm) != fingerprint:
                raise RuntimeError(f"preflight failed function preservation for {arm}")
        nll = runtime.nll(probe_rows, bs=plan["execution"]["search_batch_size"])
        gradient = runtime.grads([region["tensor"]], probe_rows,
                                 bs=plan["execution"]["grad_batch_size"])[region["tensor"]]
        generated = runtime.accuracy(probe_rows)
        report = {
            "schema_version": SCHEMA_VERSION, "scope": SCOPE, "status": "complete",
            "condition": condition_id, "plan_sha256": plan["sha256"],
            "matched_region": region, "repeat": item["repeat"],
            "matched_arms": list(ARMS), "search_sample": sample_record,
            "resources": resources,
            "quantization": runtime.quantization, "load_timings": runtime.load_timing,
            "probe_nll": [P.reportable(value) for value in nll],
            "gradient_finite": bool(torch.isfinite(gradient).all()),
            "score": generated["acc"], "eos_rate": generated["eos_rate"],
            "length_rate": generated["length_rate"], "arms": arms,
        }
        if not report["gradient_finite"]:
            raise RuntimeError("preflight gradient is non-finite")
        path.parent.mkdir(parents=True, exist_ok=True)
        M.write_json_once(path, report)
        log(f"{condition_id}: real-GPU matched-group preflight complete")
    finally:
        runtime.close()


def run_search(run, condition_id, shard="0/1", device="cuda:0", log=print):
    run, plan = load_plan(run)
    condition = _condition(plan, condition_id)
    _require_preflight(run, plan, condition)
    items = shard_items(planned_items(condition, plan), shard)
    runtime, frozen_rows = _load_runtime(run, plan, condition, "search", device)
    try:
        _assert_frozen(run, plan)
        search = _search_module(run)
        fingerprint = R.model_checksum(runtime.lm)
        output = run / "artifacts" / _safe_component(condition_id)
        output.mkdir(parents=True, exist_ok=True)
        log(f"{condition_id}: loaded {len(frozen_rows)} train-only search rows; {len(items)} items")
        for item in items:
            path = _artifact_path(run, condition_id, item["name"])
            if path.exists():
                continue
            with _item_lock(run, "search", condition_id, item["name"]):
                if path.exists():
                    continue
                _assert_frozen(run, plan)
                _set_seed(item["seed"])
                sample_record = _bootstrap_record(condition, item["repeat"])
                rows = _bootstrap_rows(frozen_rows, sample_record)
                runtime.rows["search"] = runtime._prepare_rows(rows)
                started = time.monotonic()
                ctx = None
                try:
                    import torch
                    if str(device).startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    ctx = V.VQARegionContext(runtime, item["region"], item["arm"],
                                             item["seed"], budget=plan["search_budget"],
                                             log=log,
                                             bs=plan["execution"]["search_batch_size"],
                                             grad_bs=plan["execution"]["grad_batch_size"])
                    try:
                        artifact = search.search(ctx)
                    except Exception as error:
                        if _resource_error(error):
                            raise
                        artifact = {"error": f"{type(error).__name__}: {error}",
                                    "traceback": traceback.format_exc()[-4000:]}
                    artifact.update({
                        "schema_version": SCHEMA_VERSION, "scope": SCOPE,
                        "condition": condition_id, "model": condition["model"],
                        "format": condition["format"], "task": condition["task"],
                        "region": item["region"], "repeat": item["repeat"],
                        "seed": item["seed"], "arm": item["arm"],
                        "plan_sha256": plan["sha256"],
                        "quantization": runtime.quantization,
                        "quantization_schema_version": runtime.quantization.get("schema_version"),
                        "spent": ctx.spent, "calls": dict(ctx.calls),
                        "search_rows": len(rows),
                        "search_target_tokens": sum(row["target_tokens"] for row in rows),
                        "seconds": time.monotonic() - started,
                        "load_timings": runtime.load_timing,
                        "search_sample": sample_record,
                        "peak_gpu_bytes": (torch.cuda.max_memory_allocated(device)
                                           if str(device).startswith("cuda") else None),
                    })
                    reasons = R.budget_reasons(artifact, plan["search_budget"])
                    if reasons and "error" not in artifact:
                        artifact["error"] = "; ".join(reasons)
                except Exception as error:
                    if _resource_error(error):
                        raise
                    artifact = {"schema_version": SCHEMA_VERSION, "scope": SCOPE,
                                "condition": condition_id, "region": item["region"],
                                "repeat": item["repeat"], "seed": item["seed"],
                                "arm": item["arm"], "plan_sha256": plan["sha256"],
                                "search_sample": sample_record,
                                "error": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc()[-4000:],
                                "seconds": time.monotonic() - started}
                restored = R.model_checksum(runtime.lm) == fingerprint
                if not restored:
                    artifact["error"] = (artifact.get("error", "")
                                         + "; model state was not restored").lstrip("; ")
                M.write_json_once(path, artifact)
                if not restored:
                    raise RuntimeError("model state changed during VQA search")
                log(f"{item['name']}: {artifact.get('error') or 'complete'}")
    finally:
        runtime.close()


def _logits_digest(tensor):
    return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def _task_metrics(runtime, rows):
    started = time.monotonic()
    weighted = runtime.nll(rows, bs=EVAL_BATCH_SIZE)
    records, nll_sum, tokens, finite = [], 0.0, 0, True
    for row, contribution in zip(rows, weighted):
        raw = contribution / row["objective_weight"]
        ok = math.isfinite(raw)
        finite &= ok
        if ok:
            nll_sum += raw * row["target_tokens"]
            tokens += row["target_tokens"]
        records.append({"sid": row["sid"], "target_tokens": row["target_tokens"],
                        "nll": P.reportable(raw), "finite": ok})
    generated = runtime.accuracy(rows)
    by_sid = {row["sid"]: row for row in records}
    for row in generated["records"]:
        by_sid[row["sid"]].update(row)
    complete = finite and tokens and generated["nonfinite_rate"] == 0.0
    return {"status": "complete" if complete else "nonfinite", "rows": len(rows),
            "target_tokens": tokens, "nll": nll_sum / tokens if complete else None,
            "score": generated["acc"], "nonfinite_rate": generated["nonfinite_rate"],
            "eos_rate": generated["eos_rate"],
            "length_rate": generated["length_rate"], "records": records,
            "seconds": time.monotonic() - started}


def _artifact_reasons(runtime, artifact, condition, item, plan):
    reasons = []
    expected = {"scope": SCOPE, "condition": condition["id"],
                "model": condition["model"], "format": condition["format"],
                "task": condition["task"], "region": item["region"],
                "repeat": item["repeat"], "seed": item["seed"], "arm": item["arm"],
                "plan_sha256": plan["sha256"],
                "search_sample": _bootstrap_record(condition, item["repeat"])}
    for key, value in expected.items():
        if artifact.get(key) != value:
            reasons.append(f"artifact {key} differs from plan")
    if "error" in artifact:
        reasons.append(artifact["error"])
    audit = artifact.get("quantization") or {}
    current = P.quantization_audit(runtime.lm)
    if (artifact.get("quantization_schema_version") != 2
            or audit.get("schema_version") != 2
            or audit.get("scope") != "all_llm_parameters"
            or audit.get("quantized_fraction") != 1.0
            or audit.get("unquantized_tensors") != []
            or audit.get("visual_parameters_unchanged") is not True
            or P.weight_format(runtime.lm) != condition["format"]):
        reasons.append("artifact/model must use schema-2 full-language quantization with unchanged vision")
    for key in R.AUDIT_KEYS:
        if audit.get(key) != current.get(key):
            reasons.append(f"quantization audit differs for {key}")
    reasons.extend(R.budget_reasons(artifact, plan["search_budget"]))
    return reasons


def _summary(results):
    valid = [row for row in results.values() if row.get("valid")]
    return {"valid_evaluations": len(valid), "registered_evaluations": len(EVAL_K),
            "max_delta_nll": max((row["delta_nll"] for row in valid), default=None),
            "max_relative_score_drop": max((row["relative_score_drop"] for row in valid),
                                           default=None),
            "success_any": any(row.get("success") for row in valid)}


def _fault_damage(faulted, rearranged, original):
    """Pair fault damage to its deployed no-fault layout; retain original diagnostics."""
    delta = (faulted["nll"] - rearranged["nll"]
             if faulted["nll"] is not None and rearranged["nll"] is not None else None)
    drop = ((rearranged["score"] - faulted["score"]) / rearranged["score"]
            if rearranged["score"] > 0 else None)
    original_delta = (faulted["nll"] - original["nll"]
                      if faulted["nll"] is not None and original["nll"] is not None else None)
    original_drop = ((original["score"] - faulted["score"]) / original["score"]
                     if original["score"] > 0 else None)
    return {"damage_baseline": "rearranged_unfaulted", "delta_nll": delta,
            "relative_score_drop": drop,
            "original_layout_diagnostic": {"delta_nll": original_delta,
                                            "relative_score_drop": original_drop}}


def _evaluate_one(runtime, rows, clean, condition, item, artifact,
                  artifact_sha256, plan, log):
    import torch
    started = time.monotonic()
    report = {"schema_version": SCHEMA_VERSION, "scope": SCOPE,
              "condition": condition["id"], "region": item["region"],
              "repeat": item["repeat"], "seed": item["seed"], "arm": item["arm"],
              "plan_sha256": plan["sha256"], "artifact_file_sha256": artifact_sha256,
              "artifact_object_sha256": _json_hash(artifact), "evaluation_k": list(EVAL_K)}
    reasons = _artifact_reasons(runtime, artifact, condition, item, plan)
    try:
        spec = P.validate_spec(artifact.get("permutation"), condition["dims"])
        reasons.extend(R.arm_reasons(item["region"], item["arm"], spec,
                                     condition["dims"], item["seed"]))
        attacks = {}
        for key, value in (artifact.get("attacks") or {}).items():
            parsed = int(key)
            if str(parsed) != str(key) or parsed not in plan["search_k"]:
                raise ValueError(f"attack key {key!r} outside frozen schedule")
            attacks[parsed] = value
    except (ValueError, TypeError, AttributeError) as error:
        reasons.append(str(error))
        spec, attacks = P.validate_spec({}, condition["dims"]), {}
    if reasons:
        report.update(status="artifact_error", artifact_error="; ".join(reasons),
                      results={}, summary=_summary({}), seconds=time.monotonic() - started)
        return report
    permuted = not P.is_identity(spec)
    if permuted:
        P.apply_permutation(runtime, spec)
    try:
        rearranged = _task_metrics(runtime, rows)
        logits = runtime.prefill_logits(rows[:P.LOGIT_ROWS])
        logit_ok = (bool(torch.isfinite(logits).all())
                    and torch.allclose(logits, clean["_logits"],
                                       atol=P.LOGIT_ATOL, rtol=P.LOGIT_RTOL))
        drift = (rearranged["nll"] - clean["nll"]
                 if rearranged["nll"] is not None and clean["nll"] is not None else None)
        score_drift = rearranged["score"] - clean["score"]
        gate = (rearranged["status"] == "complete" and drift is not None
                and abs(drift) <= plan["clean_gate_nll"] and logit_ok
                and abs(score_drift) <= plan["vqa_score_gate"])
        report["rearranged"] = {"identity": not permuted, "metrics": rearranged,
                                "nll_drift": drift, "score_drift": score_drift,
                                "logits_allclose": logit_ok,
                                "logits_sha256": _logits_digest(logits), "gate": gate}
        results = {}
        for k in EVAL_K:
            flips = attacks.get(k)
            if flips is None:
                results[str(k)] = {"k": k, "valid": False, "reasons": ["missing"]}
                continue
            check = R.check_region_attack(runtime, item["region"], flips, k)
            if not gate:
                check["valid"] = False
                check["reasons"].append("clean layout gate failed")
            if check["valid"]:
                normalized = P.normalize_flips(flips)
                P.apply_flips(runtime.lm, normalized)
                try:
                    faulted = _task_metrics(runtime, rows)
                finally:
                    P.apply_flips(runtime.lm, normalized)
                if faulted["status"] != "complete":
                    check["valid"] = False
                    check["reasons"].append("VQA metric became non-finite")
                # Attack damage is paired to the same deployed layout without
                # faults.  The original-layout deltas below are diagnostics only.
                damage = _fault_damage(faulted, rearranged, clean)
                check.update(metrics=faulted, **damage,
                             success=bool(check["valid"]
                                          and damage["relative_score_drop"] is not None
                                          and damage["relative_score_drop"]
                                          >= plan["success_relative_score_drop"]))
            results[str(k)] = check
            log(f"{item['name']} k={k}: valid={check['valid']}")
    finally:
        if permuted:
            P.apply_permutation(runtime, spec, inverse=True)
    report.update(status="complete", results=results, summary=_summary(results),
                  seconds=time.monotonic() - started)
    return report


def run_evaluate(run, condition_id, shard="0/1", device="cuda:0", log=print):
    run, plan = load_plan(run)
    condition = _condition(plan, condition_id)
    _require_preflight(run, plan, condition)
    items = shard_items(planned_items(condition, plan), shard)
    runtime, rows = _load_runtime(run, plan, condition, "dev", device)
    try:
        _assert_frozen(run, plan)
        logits = runtime.prefill_logits(rows[:P.LOGIT_ROWS])
        clean = _task_metrics(runtime, rows)
        clean.update({"logits_sha256": _logits_digest(logits),
                      "quantization": runtime.quantization,
                      "load_timings": runtime.load_timing, "_logits": logits})
        fingerprint = R.model_checksum(runtime.lm)
        (run / "eval" / _safe_component(condition_id)).mkdir(parents=True, exist_ok=True)
        log(f"{condition_id}: clean dev score={clean['score']}; {len(items)} reports")
        for item in items:
            path = _eval_path(run, condition_id, item["name"])
            source = _artifact_path(run, condition_id, item["name"])
            if path.exists() or not source.exists():
                continue
            with _item_lock(run, "evaluate", condition_id, item["name"]):
                if path.exists():
                    continue
                _assert_frozen(run, plan)
                artifact = M.read_json(source)
                public = {key: value for key, value in clean.items() if key != "_logits"}
                try:
                    report = _evaluate_one(runtime, rows, {**public, "_logits": clean["_logits"]},
                                           condition, item, artifact, M.sha256(source), plan, log)
                except Exception as error:
                    if _resource_error(error):
                        raise
                    report = {"schema_version": SCHEMA_VERSION, "scope": SCOPE,
                              "condition": condition_id, "region": item["region"],
                              "repeat": item["repeat"], "seed": item["seed"],
                              "arm": item["arm"], "plan_sha256": plan["sha256"],
                              "artifact_file_sha256": M.sha256(source),
                              "status": "evaluation_error",
                              "error": f"{type(error).__name__}: {error}",
                              "traceback": traceback.format_exc()[-4000:]}
                report["clean"] = public
                if R.model_checksum(runtime.lm) != fingerprint:
                    raise RuntimeError("model state changed during VQA evaluation")
                M.write_json_once(path, report)
    finally:
        runtime.close()


def run_status(run):
    run, plan = load_plan(run)
    rows = []
    for condition in plan["conditions"]:
        if condition["availability"] != "available":
            rows.append({"id": condition["id"], "availability": "unavailable",
                         "reason": condition["unavailable_reason"], "registered": 0,
                         "artifacts": 0, "evaluations": 0})
            continue
        items = planned_items(condition, plan)
        rows.append({"id": condition["id"], "availability": "available",
                     "registered": len(items),
                     "preflight": _preflight_path(run, condition["id"]).is_file(),
                     "artifacts": sum(_artifact_path(run, condition["id"], x["name"]).is_file()
                                      for x in items),
                     "evaluations": sum(_eval_path(run, condition["id"], x["name"]).is_file()
                                        for x in items)})
    return {"scope": SCOPE, "plan_sha256": plan["sha256"], "development_only": True,
            "conditions": rows, "artifact_total": sum(x["artifacts"] for x in rows),
            "evaluation_total": sum(x["evaluations"] for x in rows)}


def parser():
    root = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = root.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--model", action="append", required=True,
                      help="repeat exactly twice: model-key=/checkpoint")
    plan.add_argument("--textvqa-train", nargs="*", default=[])
    plan.add_argument("--okvqa-train", nargs="*", default=[])
    plan.add_argument("--missing-task", action="append", default=[], choices=TASKS)
    plan.add_argument("--output")
    plan.add_argument("--note", default="")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--run", required=True)
    preflight.add_argument("--condition", required=True)
    preflight.add_argument("--device", default="cuda:0")
    for name in ("search", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        command.add_argument("--condition", required=True)
        command.add_argument("--shard", default="0/1")
        command.add_argument("--device", default="cuda:0")
    status = commands.add_parser("status")
    status.add_argument("--run", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    C.setup_environment()
    if args.command == "plan":
        models = _parse_models(args.model)
        output = C.fresh_output_dir("vqa_region_study", "formal_dev", args.output)
        make_plan(output, models,
                  {"textvqa": args.textvqa_train, "okvqa": args.okvqa_train},
                  args.missing_task, args.note)
        print(output, flush=True)
    elif args.command == "preflight":
        run_preflight(args.run, args.condition, args.device)
    elif args.command == "search":
        run_search(args.run, args.condition, args.shard, args.device)
    elif args.command == "evaluate":
        run_evaluate(args.run, args.condition, args.shard, args.device)
    else:
        print(json.dumps(run_status(args.run), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
