"""Audited clean MMLU evaluation for full-parameter INT8/INT4 models.

This is a parallel evaluation path rather than a registered attack task.  It
does not construct faults, inspect gradients, rank weight bits, or call the
RegionAlign search.  Its purpose is to measure clean task utility for exact
base checkpoints under the package's schema-2 quantizer.  A run freezes the
source, checkpoint files, and cached MMLU Arrow files before inference; final
test access is explicit and every completed output is write-once.

Commands::

    python -B -m regionalign.experiments.clean_mmlu plan --model llama2-7b=/path ...
    python -B -m regionalign.experiments.clean_mmlu run --run regionalign/results/mmlu_clean/<id> \
        --model llama2-7b --format int8 --split validation
    python -B -m regionalign.experiments.clean_mmlu run --run regionalign/results/mmlu_clean/<id> \
        --model llama2-7b --format int8 --split test --final
    python -B -m regionalign.experiments.clean_mmlu summarize --run regionalign/results/mmlu_clean/<id>

MMLU uses the canonical subject header, five fixed development examples per
subject, and next-token scores for the labels A/B/C/D.  A label token is the
one-token suffix obtained by tokenizing ``Answer: A`` relative to
``Answer:``; this checks the actual tokenizer seam instead of tokenizing a
detached leading-space string.  The evaluator stores indices, labels, logits,
token counts, and effective shot counts, but not the question text.  Long
prompts lose demonstrations in their frozen order until they fit; the
evaluated question itself is never truncated.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from regionalign.harness import core as P
from regionalign import config as C

SCHEMA_VERSION = 1
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
DATASET_RELATIVE = Path("datasets/cais___mmlu/all/0.0.0") / DATASET_REVISION
SPLITS = ("validation", "test")
FORMATS = ("int8", "int4")
LABELS = ("A", "B", "C", "D")
DEFAULT_SHOTS = 5
DEFAULT_MAX_LENGTH = 4096
DEFAULT_BATCH_SIZE = 4
FINAL_BATCH_SIZE = 4
MIN_FREE_GPU_BYTES = 40 * 1024 ** 3
MODEL_SPECS = {
    "llama2-7b": {
        "cache_component": "models--meta-llama--Llama-2-7b-hf",
        "repository": "meta-llama/Llama-2-7b-hf",
        "architecture": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 4096, "intermediate_size": 11008, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 32, "vocab_size": 32000,
    },
    "llama3.1-8b": {
        "cache_component": "models--meta-llama--Meta-Llama-3.1-8B",
        "repository": "meta-llama/Meta-Llama-3.1-8B",
        "architecture": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
    },
    "qwen2.5-7b": {
        "cache_component": "models--Qwen--Qwen2.5-7B",
        "repository": "Qwen/Qwen2.5-7B",
        "architecture": ["Qwen2ForCausalLM"], "model_type": "qwen2",
        "hidden_size": 3584, "intermediate_size": 18944, "num_hidden_layers": 28,
        "num_attention_heads": 28, "num_key_value_heads": 4, "vocab_size": 152064,
    },
    "llama3.2-1b": {
        "cache_component": "models--meta-llama--Llama-3.2-1B",
        "repository": "meta-llama/Llama-3.2-1B",
        "architecture": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 16,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
    },
    "qwen2.5-1.5b": {
        "cache_component": "models--Qwen--Qwen2.5-1.5B",
        "repository": "Qwen/Qwen2.5-1.5B",
        "architecture": ["Qwen2ForCausalLM"], "model_type": "qwen2",
        "hidden_size": 1536, "intermediate_size": 8960, "num_hidden_layers": 28,
        "num_attention_heads": 12, "num_key_value_heads": 2, "vocab_size": 151936,
    },
    "stablelm2-1.6b": {
        "cache_component": "models--stabilityai--stablelm-2-1_6b",
        "repository": "stabilityai/stablelm-2-1_6b",
        "architecture": ["StableLmForCausalLM"], "model_type": "stablelm",
        "hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 24,
        "num_attention_heads": 32, "num_key_value_heads": 32, "vocab_size": 100352,
    },
    "smollm2-1.7b": {
        "cache_component": "models--HuggingFaceTB--SmolLM2-1.7B",
        "repository": "HuggingFaceTB/SmolLM2-1.7B",
        "architecture": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 24,
        "num_attention_heads": 32, "num_key_value_heads": 32, "vocab_size": 49152,
    },
}
# The clean paper matrix and its summary are defined on these three checkpoints.  The
# other registered checkpoints are available to the RegionAlign development matrix only.
PAPER_MODELS = ("llama2-7b", "llama3.1-8b", "qwen2.5-7b")
SOURCE_RELATIVE = (
    Path("regionalign/__init__.py"),
    Path("regionalign/experiments/__init__.py"),
    Path("regionalign/harness/__init__.py"),
    Path("regionalign/reporting/__init__.py"),
    Path("regionalign/experiments/clean_mmlu.py"),
    Path("regionalign/config.py"),
    Path("regionalign/harness/core.py"),
    Path("regionalign/reporting/mmlu.py"),
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def object_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)  # atomic no-overwrite publication on the same filesystem
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error
    finally:
        temp.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_frozen_json(path, value, label):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"{label} drift: {path}")
    else:
        write_json_once(path, value)


def software_environment():
    import pyarrow
    import torch
    import transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "pyarrow": pyarrow.__version__}


def runtime_device_identity(torch, device):
    """Return the hardware identity that a resumable condition is bound to."""
    if not str(device).startswith("cuda"):
        return {"host": platform.node(), "device_type": str(device)}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 7B/8B MMLU evaluation")
    logical = torch.device(device).index
    logical = torch.cuda.current_device() if logical is None else logical
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is required for audited GPU identity")
    selectors = [item.strip() for item in visible.split(",") if item.strip()]
    if logical >= len(selectors):
        raise RuntimeError(f"CUDA device {logical} is absent from CUDA_VISIBLE_DEVICES={visible!r}")
    selector = selectors[logical]
    query = subprocess.run(
        ["nvidia-smi", "-i", selector,
         "--query-gpu=uuid,driver_version,pci.bus_id,name,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    lines = [line for line in query.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"nvidia-smi returned {len(lines)} device rows for selector {selector!r}")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 5 or not fields[0].startswith("GPU-"):
        raise RuntimeError(f"unexpected nvidia-smi identity row: {lines[0]!r}")
    uuid, driver, pci_bus_id, nvidia_name, memory_mib = fields
    properties = torch.cuda.get_device_properties(device)
    if nvidia_name != torch.cuda.get_device_name(device):
        raise RuntimeError(f"CUDA/nvidia-smi name mismatch: {torch.cuda.get_device_name(device)!r} "
                           f"!= {nvidia_name!r}")
    framebuffer_memory = int(memory_mib) * 1024 ** 2
    reserved = framebuffer_memory - properties.total_memory
    allowed_reserved = max(1024 ** 3, int(framebuffer_memory * 0.02))
    if reserved < 0 or reserved > allowed_reserved:
        raise RuntimeError(f"CUDA usable/framebuffer memory mismatch: usable={properties.total_memory}, "
                           f"framebuffer={framebuffer_memory}")
    return {
        "host": platform.node(),
        "device_type": "cuda",
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory": properties.total_memory,
        "framebuffer_memory": framebuffer_memory,
        "multiprocessors": properties.multi_processor_count,
        "uuid": uuid,
        "pci_bus_id": pci_bus_id,
        "cuda_visible_selector": selector,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver_version": driver,
    }


def runtime_device_class(identity):
    """Hardware properties that must match across resumable attempts."""
    keys = ("host", "device_type", "device_name", "compute_capability", "total_memory",
            "framebuffer_memory", "multiprocessors", "cuda_runtime", "cudnn", "driver_version")
    return {key: identity.get(key) for key in keys}


def _tensor_record(name, tensor, chunk_bytes=8 * 1024 * 1024):
    """Hash one tensor without materializing the full 7B state on the CPU."""
    import torch

    value = tensor.detach()
    if not value.is_contiguous():
        value = value.contiguous()
    byte_view = value.view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, byte_view.numel(), chunk_bytes):
        block = byte_view[start:start + chunk_bytes].cpu().numpy().tobytes()
        digest.update(block)
    return {"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": byte_view.numel(), "sha256": digest.hexdigest()}


def quantized_state_manifest(model):
    """Digest decoded parameters and quantization scales after quantization."""
    quant = getattr(model, "ar_quant", None)
    if not quant:
        raise ValueError("model has no quantized state")
    parameters = [_tensor_record(name, value) for name, value in
                  sorted(model.named_parameters(), key=lambda item: item[0])]
    scales = [_tensor_record(name, value) for name, value in sorted(quant["scales"].items())]
    manifest = {"algorithm": "sha256_raw_tensor_bytes_v1", "format": quant["format"],
                "parameters": parameters, "scales": scales}
    manifest["manifest_sha256"] = object_sha256(manifest)
    return manifest


def verify_quantized_state_manifest(manifest):
    declared = dict(manifest)
    recorded = declared.pop("manifest_sha256", None)
    if recorded is None or object_sha256(declared) != recorded:
        raise RuntimeError("quantized-state manifest digest drift")
    return recorded


def file_record(path, root=None, hash_file=True):
    # Preserve the logical path.  HF snapshot files are symlinks: resolving here
    # would later verify the old blob while the loader follows a retargeted link.
    path = Path(path).expanduser().absolute()
    root = Path(root).expanduser().absolute() if root is not None else None
    stat = path.stat()
    label = str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)
    out = {"path": label, "bytes": stat.st_size}
    if hash_file:
        out["sha256"] = sha256(path)
    if path.is_symlink():
        out["symlink_target"] = os.readlink(path)
    return out


def arrow_path(hf_home, split):
    if split not in ("dev", *SPLITS):
        raise ValueError(f"unknown MMLU split {split!r}")
    return Path(hf_home) / DATASET_RELATIVE / f"mmlu-{split}.arrow"


def read_arrow(path):
    """Read a cached datasets Arrow stream without cache locks or network."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(path), "r") as source:
        table = ipc.open_stream(source).read_all()
    required = ["question", "subject", "choices", "answer"]
    if table.column_names != required:
        raise ValueError(f"unexpected MMLU columns {table.column_names}")
    rows = table.to_pylist()
    for index, row in enumerate(rows):
        if (not isinstance(row["question"], str) or not isinstance(row["subject"], str)
                or not isinstance(row["choices"], list) or len(row["choices"]) != 4
                or not all(isinstance(x, str) for x in row["choices"])
                or isinstance(row["answer"], bool) or row["answer"] not in range(4)):
            raise ValueError(f"invalid MMLU row {index}")
    return rows


def subject_name(subject):
    return subject.replace("_", " ")


def format_example(row, include_answer):
    lines = [row["question"]]
    lines.extend(f"{label}. {choice}" for label, choice in zip(LABELS, row["choices"]))
    answer = f" {LABELS[row['answer']]}" if include_answer else ""
    lines.append(f"Answer:{answer}")
    return "\n".join(lines)


def make_prompt(subject, examples, row, shots=DEFAULT_SHOTS):
    if shots < 0 or shots > len(examples):
        raise ValueError("invalid shot count")
    header = ("The following are multiple choice questions (with answers) about "
              f"{subject_name(subject)}.\n\n")
    demonstrations = "\n\n".join(format_example(x, True) for x in examples[:shots])
    target = format_example(row, False)
    return header + (demonstrations + "\n\n" if demonstrations else "") + target


def dev_by_subject(rows, shots=DEFAULT_SHOTS):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["subject"]].append(row)
    bad = {subject: len(items) for subject, items in grouped.items() if len(items) < shots}
    if bad:
        raise ValueError(f"subjects lack {shots} dev examples: {bad}")
    return dict(grouped)


def prompt_ids(tokenizer, subject, examples, row, max_length, shots=DEFAULT_SHOTS):
    """Return an untruncated target prompt, dropping demonstrations if needed."""
    for used in range(shots, -1, -1):
        prompt = make_prompt(subject, examples, row, used)
        ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        if len(ids) <= max_length:
            return ids, used
    raise ValueError("target-only MMLU prompt exceeds max_length")


def wilson_interval(correct, total, z=1.959963984540054):
    if total <= 0:
        return [None, None]
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def metrics_from_predictions(predictions):
    if not predictions:
        raise ValueError("no predictions")
    by_subject = defaultdict(list)
    for expected, row in enumerate(predictions):
        if row.get("index") != expected:
            raise ValueError(f"prediction sequence breaks at {expected}")
        by_subject[row["subject"]].append(row)
    subject_metrics = {}
    for subject, rows in sorted(by_subject.items()):
        correct = sum(bool(row["correct"]) for row in rows)
        subject_metrics[subject] = {
            "count": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "nonfinite": sum(bool(row["nonfinite"]) for row in rows),
        }
    correct = sum(bool(row["correct"]) for row in predictions)
    nonfinite = sum(bool(row["nonfinite"]) for row in predictions)
    return {
        "count": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "accuracy_wilson95": wilson_interval(correct, len(predictions)),
        "subject_macro_accuracy": sum(x["accuracy"] for x in subject_metrics.values()) / len(subject_metrics),
        "subjects": len(subject_metrics),
        "nonfinite": nonfinite,
        "nonfinite_rate": nonfinite / len(predictions),
        "reduced_shot_prompts": sum(row["shots"] < DEFAULT_SHOTS for row in predictions),
        "max_prompt_tokens": max(row["prompt_tokens"] for row in predictions),
        "by_subject": subject_metrics,
    }


def validate_predictions(predictions, dataset_rows, condition_fingerprint=None):
    if len(predictions) != len(dataset_rows):
        raise ValueError(f"prediction count {len(predictions)} != dataset count {len(dataset_rows)}")
    for index, (saved, source) in enumerate(zip(predictions, dataset_rows)):
        if (saved.get("index") != index or saved.get("subject") != source["subject"]
                or saved.get("gold") != source["answer"]):
            raise ValueError(f"prediction identity mismatch at {index}")
        if (condition_fingerprint is not None
                and saved.get("condition_fingerprint") != condition_fingerprint):
            raise ValueError(f"prediction condition fingerprint mismatch at {index}")
        values = saved.get("choice_logits")
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError(f"invalid choice logits at {index}")
        nonfinite = any(value is None or not isinstance(value, (int, float))
                        or not math.isfinite(value) for value in values)
        prediction = -1 if nonfinite else max(range(4), key=values.__getitem__)
        if (saved.get("nonfinite") is not nonfinite or saved.get("prediction") != prediction
                or saved.get("correct") is not (prediction == source["answer"])):
            raise ValueError(f"derived prediction fields mismatch at {index}")
        if (isinstance(saved.get("shots"), bool) or saved.get("shots") not in range(DEFAULT_SHOTS + 1)
                or isinstance(saved.get("prompt_tokens"), bool)
                or not isinstance(saved.get("prompt_tokens"), int) or saved["prompt_tokens"] < 1):
            raise ValueError(f"invalid prompt metadata at {index}")


def parse_model_spec(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("model must be KEY=/checkpoint/path")
    key, raw = text.split("=", 1)
    if not key or not key.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise argparse.ArgumentTypeError(f"invalid model key {key!r}")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"checkpoint directory is missing: {path}")
    return key, path


def checkpoint_files(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        data = read_json(index)
        weights = sorted(set(data.get("weight_map", {}).values()))
        if not weights:
            raise ValueError(f"empty weight map: {index}")
    elif (checkpoint / "model.safetensors").is_file():
        weights = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"no safetensors checkpoint in {checkpoint}")
    metadata = [name for name in (
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "merges.txt", "vocab.json",
    ) if (checkpoint / name).is_file()]
    files = [checkpoint / name for name in sorted(set(metadata + weights))]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint files missing: {missing}")
    return files


def checkpoint_manifest(checkpoint, model_key):
    checkpoint = Path(checkpoint).resolve()
    if model_key not in MODEL_SPECS:
        raise ValueError(f"unknown frozen model key {model_key!r}")
    expected = MODEL_SPECS[model_key]
    if expected["cache_component"] not in checkpoint.parts:
        raise ValueError(f"{model_key} must use base checkpoint {expected['repository']}; got {checkpoint}")
    config = read_json(checkpoint / "config.json")
    fields = {
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads", config.get("num_attention_heads")),
        "vocab_size": config.get("vocab_size"),
    }
    mismatches = {field: (fields[field], expected[field]) for field in fields
                  if fields[field] != expected[field]}
    declared = config.get("_name_or_path")
    if declared and declared != expected["repository"]:
        mismatches["_name_or_path"] = (declared, expected["repository"])
    if mismatches:
        raise ValueError(f"{model_key} is not the frozen base architecture: {mismatches}")
    records = [file_record(path, checkpoint) for path in checkpoint_files(checkpoint)]
    manifest = {
        "path": str(checkpoint),
        "model_key": model_key,
        "repository": expected["repository"],
        **fields,
        "max_position_embeddings": config.get("max_position_embeddings"),
        "revision": checkpoint.name if checkpoint.parent.name == "snapshots" else None,
        "files": records,
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    return manifest


def verify_records(base, records):
    base = Path(base).expanduser().absolute()
    errors = []
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe manifest path {relative}")
            continue
        path = base / relative
        if not path.is_file():
            errors.append(f"missing {path}")
        elif "symlink_target" in record and (not path.is_symlink()
                                               or os.readlink(path) != record["symlink_target"]):
            errors.append(f"symlink drift {path}")
        elif path.stat().st_size != record["bytes"]:
            errors.append(f"size drift {path}")
        elif sha256(path) != record["sha256"]:
            errors.append(f"hash drift {path}")
    if errors:
        raise RuntimeError("input verification failed:\n" + "\n".join(errors))


def source_manifest():
    records = []
    for relative in SOURCE_RELATIVE:
        path = C.PROJECT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(file_record(path, C.PROJECT))
    return records


def dataset_manifest(hf_home):
    files = []
    expected = {"dev": 285, "validation": 1531, "test": 14042}
    for split, count in expected.items():
        path = arrow_path(hf_home, split)
        if not path.is_file():
            raise FileNotFoundError(path)
        record = file_record(path, Path(hf_home))
        record.update({"split": split, "expected_rows": count})
        files.append(record)
    return {"name": "cais/mmlu", "config": "all", "revision": DATASET_REVISION,
            "hf_home": str(Path(hf_home).resolve()), "files": files}


def snapshot_sources(run):
    target = Path(run) / "source"
    target.mkdir(parents=True, exist_ok=False)
    for relative in SOURCE_RELATIVE:
        destination = target / (str(relative).replace("/", "__") + ".snapshot")
        shutil.copy2(C.PROJECT / relative, destination)


def write_slurm_launchers(run, plan):
    """Write reproducible validation and final-test array launchers for this cluster."""
    run = Path(run).resolve()
    matrix = run / "conditions.tsv"
    with matrix.open("x", encoding="utf-8") as handle:
        for condition in plan["conditions"]:
            handle.write(f"{condition['model']}\t{condition['format']}\n")
    logs = run / "logs"
    logs.mkdir(exist_ok=False)
    common = "\n".join((
        "set -euo pipefail",
        f"PROJECT={shlex.quote(str(C.PROJECT))}",
        f"RUN={shlex.quote(str(run))}",
        f"PYTHON={shlex.quote(sys.executable)}",
        'ENTRY="$(sed -n \"$((SLURM_ARRAY_TASK_ID + 1))p\" \"$RUN/conditions.tsv\")"',
        "MODEL=${ENTRY%%$'\\t'*}",
        "FORMAT=${ENTRY##*$'\\t'}",
        'cd "$PROJECT"',
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false",
    ))
    headers = "\n".join((
        "#!/usr/bin/env bash",
        "#SBATCH --account=das",
        "#SBATCH --partition=das",
        "#SBATCH --qos=das-normal",
        "#SBATCH --nodelist=cn117",
        "#SBATCH --gres=gpu:rtx_a6000:1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=46G",
        "#SBATCH --array=0-5%2",
    ))
    validation = (headers + "\n#SBATCH --time=02:00:00\n#SBATCH --job-name=mmlu-val\n"
                  f"#SBATCH --output={logs}/validation-%A_%a.log\n" + common + "\n"
                  '"$PYTHON" -B -u -m regionalign.experiments.clean_mmlu run --run "$RUN" --model "$MODEL" '
                  '--format "$FORMAT" --split validation --limit 64 --batch-size 4\n')
    final_batch = plan["protocol"]["final_batch_size"]
    final = (headers + "\n#SBATCH --time=12:00:00\n#SBATCH --job-name=mmlu-test\n"
             f"#SBATCH --output={logs}/test-%A_%a.log\n" + common + "\n"
             'if [[ -f "$RUN/conditions/$MODEL/$FORMAT/test/metrics.json" ]]; then exit 0; fi\n'
             '"$PYTHON" -B -u -m regionalign.experiments.clean_mmlu run --run "$RUN" --model "$MODEL" '
             f'--format "$FORMAT" --split test --final --batch-size {final_batch}\n')
    launchers = []
    for name, body in (("validation_array.sbatch", validation), ("test_array.sbatch", final)):
        path = run / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
        launchers.append(file_record(path, run))
    write_json_once(run / "launchers_manifest.json", {"generated_at": utc_now(),
                                                       "files": launchers,
                                                       "submit_order": ["validation_array.sbatch",
                                                                        "test_array.sbatch"]})


def create_plan(args):
    models = dict(args.model)
    if len(models) != len(args.model):
        raise ValueError("duplicate model key")
    if set(models) != set(PAPER_MODELS):
        raise ValueError(f"the frozen paper matrix requires model keys {sorted(PAPER_MODELS)}")
    if len({str(path) for path in models.values()}) != len(models):
        raise ValueError("the three model keys must refer to distinct checkpoints")
    if set(args.formats) != set(FORMATS) or len(args.formats) != len(FORMATS):
        raise ValueError("the frozen paper matrix requires exactly INT8 and INT4")
    run = C.fresh_output_dir("mmlu_clean", "base_models", args.out)
    manifests = {key: checkpoint_manifest(path, key) for key, path in sorted(models.items())}
    too_short = {key: value.get("max_position_embeddings") for key, value in manifests.items()
                 if not isinstance(value.get("max_position_embeddings"), int)
                 or value["max_position_embeddings"] < args.max_length}
    if too_short:
        raise ValueError(f"max_length {args.max_length} exceeds checkpoint context: {too_short}")
    import torch
    import transformers
    import pyarrow
    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "scope": "clean_mmlu_full_parameter_quantization",
        "explicit_exclusions": ["fault construction", "sensitive-bit search", "adaptive perturbation",
                                "RegionAlign search", "physical fault claim"],
        "formats": list(args.formats),
        "models": manifests,
        "dataset": dataset_manifest(C.HUB),
        "protocol": {
            "shots": DEFAULT_SHOTS,
            "labels": list(LABELS),
            "prompt": "canonical MMLU subject header and A-D choices",
            "scoring": "next-token logits for the one-token Answer:<space>{A,B,C,D} suffixes",
            "max_length": args.max_length,
            "final_batch_size": FINAL_BATCH_SIZE,
            "test_requires_final": True,
            "quantization_schema_version": 2,
            "quantization_scope": "all_llm_parameters",
            "int4_group_columns": P.GROUP,
            "compute": "decoded weights in parameter dtype",
        },
        "sources": source_manifest(),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "transformers": transformers.__version__, "pyarrow": pyarrow.__version__},
        "conditions": [{"model": key, "format": fmt} for key in sorted(models) for fmt in args.formats],
    }
    write_json_once(run / "plan.json", plan)
    plan_sha = sha256(run / "plan.json")
    test_record = next(item for item in plan["dataset"]["files"] if item["split"] == "test")
    write_json_once(run / "test_selection.json", {
        "created_at": utc_now(), "plan_sha256": plan_sha,
        "dataset_revision": DATASET_REVISION, "test_file_sha256": test_record["sha256"],
        "conditions": plan["conditions"], "protocol": plan["protocol"],
    })
    snapshot_sources(run)
    write_slurm_launchers(run, plan)
    with (run / "planned_conditions.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "format"))
        writer.writeheader()
        for condition in plan["conditions"]:
            writer.writerow(condition)
    write_json_once(run / "STATUS.json", {"status": "planned", "measured_results": False,
                                           "created_at": utc_now(), "conditions": len(plan["conditions"])})
    print(run)


def load_plan(run):
    run = C.output_path(run)
    plan = read_json(run / "plan.json")
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("scope") != "clean_mmlu_full_parameter_quantization":
        raise ValueError("unsupported MMLU plan")
    return run, plan


def verify_plan_inputs(plan, model_key, dataset_splits=None):
    if model_key not in plan["models"]:
        raise ValueError(f"model {model_key!r} is not in the plan")
    verify_records(C.PROJECT, plan["sources"])
    dataset = plan["dataset"]
    dataset_records = plan["dataset"]["files"]
    if dataset_splits is not None:
        wanted = set(dataset_splits)
        dataset_records = [record for record in dataset_records if record["split"] in wanted]
        if {record["split"] for record in dataset_records} != wanted:
            raise RuntimeError(f"dataset manifest lacks requested splits: {sorted(wanted)}")
    verify_records(dataset["hf_home"], dataset_records)
    model = plan["models"][model_key]
    declared = dict(model)
    recorded_digest = declared.pop("manifest_sha256")
    if object_sha256(declared) != recorded_digest:
        raise RuntimeError(f"model manifest digest drift for {model_key}")
    verify_records(model["path"], model["files"])


def verify_test_selection(run, plan):
    selection = read_json(Path(run) / "test_selection.json")
    test_record = next(item for item in plan["dataset"]["files"] if item["split"] == "test")
    expected = {
        "plan_sha256": sha256(Path(run) / "plan.json"),
        "dataset_revision": DATASET_REVISION,
        "test_file_sha256": test_record["sha256"],
        "conditions": plan["conditions"],
        "protocol": plan["protocol"],
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise RuntimeError(f"test selection drift: {key}")
    return selection


def condition_fingerprint(run, plan, model_key, fmt, split):
    dataset_record = next(item for item in plan["dataset"]["files"] if item["split"] == split)
    value = {
        "plan_sha256": sha256(Path(run) / "plan.json"),
        "model_manifest_sha256": plan["models"][model_key]["manifest_sha256"],
        "dataset_sha256": dataset_record["sha256"],
        "model": model_key, "format": fmt, "split": split,
        "protocol": plan["protocol"],
    }
    return object_sha256(value), value


def condition_dir(run, model_key, fmt, split):
    return Path(run) / "conditions" / model_key / fmt / split


def append_jsonl(path, value):
    path = Path(path)
    line = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        payload = line.encode("utf-8")
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def repair_uncommitted_jsonl_tail(path, repair_log=None):
    """Repair only a final JSONL fragment that was never newline-committed."""
    path = Path(path)
    if not path.exists():
        return
    payload = path.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return
    offset = payload.rfind(b"\n") + 1
    tail = payload[offset:]
    try:
        value = json.loads(tail.decode("utf-8"), parse_constant=_reject_json_constant)
        if not isinstance(value, dict):
            raise ValueError("JSONL object required")
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        with path.open("r+b") as handle:
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())
        event = {"time": utc_now(), "action": "truncate_uncommitted_jsonl_tail",
                 "path": path.name, "dropped_bytes": len(tail),
                 "dropped_sha256": hashlib.sha256(tail).hexdigest()}
    else:
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        event = {"time": utc_now(), "action": "terminate_complete_jsonl_tail",
                 "path": path.name}
    if repair_log is not None:
        append_jsonl(repair_log, event)


def read_jsonl_objects(path):
    values = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=_reject_json_constant)
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL object required at {path}:{line_number}")
            values.append(value)
    return values


def append_test_access(run, plan, model_key, fmt, split, fingerprint):
    entry = {"time": utc_now(), "model": model_key, "format": fmt, "split": split,
             "pid": os.getpid(), "host": platform.node(), "run": str(Path(run).resolve()),
             "plan_sha256": sha256(Path(run) / "plan.json"),
             "condition_fingerprint": fingerprint,
             "dataset_revision": plan["dataset"]["revision"]}
    append_jsonl(Path(run) / "test_access.jsonl", entry)
    append_jsonl(C.RESULTS_ROOT / "test_access.jsonl", {**entry, "task": "mmlu_clean"})


def append_summary_test_access(run, plan):
    entry = {"time": utc_now(), "purpose": "final_summary_validation", "split": "test",
             "pid": os.getpid(), "host": platform.node(), "run": str(Path(run).resolve()),
             "plan_sha256": sha256(Path(run) / "plan.json"),
             "dataset_revision": plan["dataset"]["revision"]}
    append_jsonl(Path(run) / "test_access.jsonl", entry)
    append_jsonl(C.RESULTS_ROOT / "test_access.jsonl", {**entry, "task": "mmlu_clean"})


@contextmanager
def condition_lock(target):
    """Serialize one condition while permitting crash-safe resume."""
    path = Path(target) / ".condition.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"condition is already running: {target}") from error
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def load_model(checkpoint, fmt, device, min_free_gpu_bytes=MIN_FREE_GPU_BYTES):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    if str(device).startswith("cuda"):
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        if isinstance(min_free_gpu_bytes, bool) or not isinstance(min_free_gpu_bytes, int) or min_free_gpu_bytes < 0:
            raise ValueError("min_free_gpu_bytes must be a non-negative integer")
        if free < min_free_gpu_bytes:
            raise RuntimeError(f"MMLU quantization requires at least {min_free_gpu_bytes / 1024**3:.0f} "
                               f"GiB free on {device}; found {free / 1024**3:.1f} GiB")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": device},
    )
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    loaded = time.monotonic()
    P.quantize(model, fmt)
    quantized = time.monotonic()
    audit = P.quantization_audit(model)
    expected_bits = P.FORMAT_BITS[fmt]
    if (audit["schema_version"] != 2 or audit["scope"] != "all_llm_parameters"
            or audit["format"] != fmt or audit["bits"] != expected_bits
            or audit["quantized_fraction"] != 1.0 or audit["unquantized_tensors"]
            or audit["quantized_parameters"] != audit["total_parameters"]
            or audit["quantized_tensors"] != audit["total_tensors"]
            or audit["int4_group_columns"] != (P.GROUP if fmt == "int4" else None)
            or audit["scales_faultable"] is not False
            or audit["compute"] != "decoded_weights_in_parameter_dtype"):
        raise RuntimeError("full-parameter quantization audit failed")
    return model, tokenizer, audit, {"load_seconds": loaded - started,
                                     "quantize_seconds": quantized - loaded}


def choice_token_ids(tokenizer):
    ids = []
    context = tokenizer.encode("Answer:", add_special_tokens=False)
    details = {"context": context, "labels": {}}
    for label in LABELS:
        encoded = tokenizer.encode(f"Answer: {label}", add_special_tokens=False)
        details["labels"][label] = encoded
        if encoded[:-1] != context or len(encoded) != len(context) + 1:
            raise ValueError(f"MMLU continuation 'Answer: {label}' is not a one-token suffix: "
                             f"context={context}, full={encoded}")
        ids.append(encoded[-1])
    if len(set(ids)) != len(ids):
        raise ValueError(f"MMLU label token ids are not unique: {ids}")
    return ids, details


def validate_prompt_seam(tokenizer, prompt, token_ids):
    base = tokenizer.encode(prompt, add_special_tokens=False)
    for label, token_id in zip(LABELS, token_ids):
        full = tokenizer.encode(f"{prompt} {label}", add_special_tokens=False)
        if full[:-1] != base or len(full) != len(base) + 1 or full[-1] != token_id:
            raise ValueError(f"context-dependent MMLU label token for {label}: "
                             f"base_tail={base[-8:]}, full_tail={full[-8:]}, expected={token_id}")


def validate_special_token_framing(tokenizer, prompt):
    """Require special-token handling to add only a fixed prompt prefix.

    An appended EOS would move the scored position away from ``Answer:`` even
    though the detached continuation seam still looked valid.
    """
    raw = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    framed = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    if not raw or len(framed) < len(raw) or framed[-len(raw):] != raw:
        raise ValueError("tokenizer special tokens alter or follow the MMLU prompt")
    return framed[:-len(raw)]


def tokenizer_protocol_audit(tokenizer, rows, examples, max_length):
    token_ids, details = choice_token_ids(tokenizer)
    subjects = {}
    special_prefix = None
    for row in rows:
        if row["subject"] in subjects:
            continue
        ids, shots = prompt_ids(tokenizer, row["subject"], examples[row["subject"]], row,
                                max_length, DEFAULT_SHOTS)
        prompt = make_prompt(row["subject"], examples[row["subject"]], row, shots)
        validate_prompt_seam(tokenizer, prompt, token_ids)
        prefix = validate_special_token_framing(tokenizer, prompt)
        if special_prefix is None:
            special_prefix = prefix
        elif prefix != special_prefix:
            raise ValueError("tokenizer special-token prefix varies across MMLU subjects")
        subjects[row["subject"]] = {"shots": shots, "prompt_tokens": len(ids)}
    details.update({"choice_token_ids": token_ids, "validated_subjects": subjects,
                    "tokenizer_class": tokenizer.__class__.__name__,
                    "padding_side": tokenizer.padding_side,
                    "special_prefix_token_ids": special_prefix})
    return token_ids, details


def _reject_json_constant(value):
    raise ValueError(f"non-standard JSON constant {value}")


def load_partial(path, repair_trailing=False, repair_log=None):
    path = Path(path)
    if not path.exists():
        return []
    raw_lines = path.read_bytes().splitlines(keepends=True)
    rows = []
    offset = 0
    for line_number, raw in enumerate(raw_lines, 1):
        terminal = line_number == len(raw_lines)
        terminated = raw.endswith(b"\n")
        try:
            line = raw.decode("utf-8")
            row = json.loads(line, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            if repair_trailing and terminal and not terminated:
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                if repair_log is not None:
                    append_jsonl(repair_log, {"time": utc_now(), "action": "truncate_partial_tail",
                                              "path": path.name, "dropped_bytes": len(raw),
                                              "dropped_sha256": hashlib.sha256(raw).hexdigest()})
                break
            raise ValueError(f"invalid partial prediction line {line_number}") from error
        if row.get("index") != len(rows):
            raise ValueError(f"partial prediction sequence breaks at line {line_number}")
        rows.append(row)
        offset += len(raw)
        if repair_trailing and terminal and not terminated:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            if repair_log is not None:
                append_jsonl(repair_log, {"time": utc_now(), "action": "terminate_complete_tail",
                                          "path": path.name, "index": row["index"]})
    return rows


def score_rows(model, tokenizer, token_ids, rows, examples, device, predictions_path,
               max_length=DEFAULT_MAX_LENGTH, batch_size=DEFAULT_BATCH_SIZE, limit=None,
               condition_fingerprint=None):
    import torch

    if limit is not None:
        rows = rows[:limit]
    completed = load_partial(predictions_path, repair_trailing=True,
                             repair_log=predictions_path.parent / "recovery_events.jsonl")
    if len(completed) > len(rows):
        raise ValueError("partial output is longer than requested evaluation")
    for old, expected in zip(completed, rows):
        if old["subject"] != expected["subject"] or old["gold"] != expected["answer"]:
            raise ValueError("partial output does not match frozen dataset")
        if (condition_fingerprint is not None
                and old.get("condition_fingerprint") != condition_fingerprint):
            raise ValueError("partial output belongs to another condition")
    choice_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("a", encoding="utf-8", buffering=1) as output:
        for start in range(len(completed), len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            encoded = [prompt_ids(tokenizer, row["subject"], examples[row["subject"]], row,
                                  max_length, DEFAULT_SHOTS) for row in batch_rows]
            lengths = [len(ids) for ids, _ in encoded]
            width = max(lengths)
            input_ids = torch.full((len(encoded), width), tokenizer.pad_token_id, dtype=torch.long)
            attention = torch.zeros_like(input_ids)
            for offset, (ids, _) in enumerate(encoded):
                input_ids[offset, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
                attention[offset, width - len(ids):] = 1
            position_ids = attention.cumsum(-1) - 1
            position_ids.masked_fill_(attention == 0, 0)
            with torch.inference_mode():
                logits = model(input_ids=input_ids.to(device), attention_mask=attention.to(device),
                               position_ids=position_ids.to(device), use_cache=False,
                               num_logits_to_keep=1).logits[:, -1, :]
                scores = logits[:, choice_ids].float().cpu()
            for offset, (row, (_, shots), values, length) in enumerate(
                    zip(batch_rows, encoded, scores.tolist(), lengths)):
                nonfinite = not all(math.isfinite(value) for value in values)
                prediction = -1 if nonfinite else max(range(4), key=values.__getitem__)
                item = {
                    "index": start + offset,
                    "subject": row["subject"],
                    "gold": row["answer"],
                    "prediction": prediction,
                    "correct": prediction == row["answer"],
                    "choice_logits": [value if math.isfinite(value) else None for value in values],
                    "nonfinite": nonfinite,
                    "prompt_tokens": length,
                    "shots": shots,
                }
                if condition_fingerprint is not None:
                    item["condition_fingerprint"] = condition_fingerprint
                output.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
    return load_partial(predictions_path)


def _finalize_condition(target, plan, selected_rows, fingerprint, fingerprint_value):
    """Publish metrics from sealed predictions without loading the model."""
    target = Path(target)
    completed_path = target / "metrics.json"
    execution = read_json(target / "execution.json")
    completion = read_json(target / "completion_metadata.json")
    final_predictions = target / "predictions.jsonl"
    audit_path = target / "quantization_audit.json"
    tokenizer_path = target / "tokenizer_audit.json"
    state_path = target / "quantized_state.json"
    if completion.get("condition_fingerprint") != fingerprint:
        raise RuntimeError("completion metadata belongs to another condition")
    predictions = load_partial(final_predictions)
    validate_predictions(predictions, selected_rows, fingerprint)
    if (len(predictions) != completion.get("count")
            or sha256(final_predictions) != completion.get("predictions_sha256")):
        raise RuntimeError("sealed predictions differ from completion metadata")
    audit = read_json(audit_path)
    tokenizer_audit = read_json(tokenizer_path)
    state = read_json(state_path)
    state_digest = verify_quantized_state_manifest(state)
    evidence_hashes = {
        "execution_sha256": sha256(target / "execution.json"),
        "predictions_manifest_sha256": sha256(target / "predictions_manifest.json"),
        "device_attempts_sha256": sha256(target / "device_attempts.jsonl"),
        "quantization_audit_sha256": sha256(audit_path),
        "tokenizer_audit_file_sha256": sha256(tokenizer_path),
        "quantized_state_file_sha256": sha256(state_path),
    }
    expected_hashes = completion.get("evidence_hashes")
    if (expected_hashes != evidence_hashes
            or completion.get("quantized_state_manifest_sha256") != state_digest):
        raise RuntimeError("completion evidence hash drift")
    metrics = metrics_from_predictions(predictions)
    metrics.update({
        "schema_version": SCHEMA_VERSION, "completed_at": utc_now(),
        "condition_fingerprint": fingerprint, "plan_sha256": fingerprint_value["plan_sha256"],
        "model_manifest_sha256": fingerprint_value["model_manifest_sha256"],
        "dataset_sha256": fingerprint_value["dataset_sha256"],
        "model": execution["model"], "checkpoint": plan["models"][execution["model"]]["path"],
        "format": execution["format"], "split": execution["split"],
        "final": execution["final"], "limit": execution["limit"],
        "batch_size": execution["batch_size"], "max_length": plan["protocol"]["max_length"],
        "tokenizer_audit_sha256": object_sha256(tokenizer_audit),
        "quantized_state_manifest_sha256": state_digest,
        "quantization": audit, "timing": completion["timing"],
        "environment": completion["environment"],
        "predictions_sha256": completion["predictions_sha256"],
        "completion_metadata_sha256": sha256(target / "completion_metadata.json"),
        **evidence_hashes,
    })
    write_json_once(completed_path, metrics)
    print(json.dumps({"condition": str(target), "accuracy": metrics["accuracy"],
                      "count": metrics["count"]}, sort_keys=True, allow_nan=False))


def _run_condition_locked(args, run, plan, target, fingerprint, fingerprint_value):
    completed_path = target / "metrics.json"
    if completed_path.exists():
        raise FileExistsError(f"condition is already complete: {completed_path}")
    environment = software_environment()
    if environment != plan["environment"]:
        raise RuntimeError(f"software environment drift: current={environment}, plan={plan['environment']}")
    execution_static = {
        "condition_fingerprint": fingerprint, "condition": fingerprint_value,
        "model": args.model, "format": args.format, "split": args.split,
        "final": bool(args.final), "limit": args.limit, "batch_size": args.batch_size,
        "device": args.device, "environment": environment,
    }
    execution_path = target / "execution.json"
    if execution_path.exists():
        saved_execution = read_json(execution_path)
        for key, value in execution_static.items():
            if saved_execution.get(key) != value:
                raise RuntimeError(f"condition execution drift: {key}")

    dataset_root = plan["dataset"]["hf_home"]
    dev_rows = read_arrow(arrow_path(dataset_root, "dev"))
    eval_rows = read_arrow(arrow_path(dataset_root, args.split))
    selected_rows = eval_rows[:args.limit] if args.limit is not None else eval_rows
    examples = dev_by_subject(dev_rows)
    prediction_manifest = {
        "condition_fingerprint": fingerprint,
        "expected_rows": len(selected_rows),
        "identity_sha256": object_sha256([(row["subject"], row["answer"]) for row in selected_rows]),
    }
    ensure_frozen_json(target / "predictions_manifest.json", prediction_manifest,
                       "prediction manifest")

    partial_predictions = target / "predictions.partial.jsonl"
    final_predictions = target / "predictions.jsonl"
    completion_path = target / "completion_metadata.json"
    if final_predictions.exists() and partial_predictions.exists():
        raise RuntimeError("both partial and final prediction files exist")
    if completion_path.exists():
        completion = read_json(completion_path)
        sealed = final_predictions if final_predictions.exists() else partial_predictions
        if not sealed.is_file():
            raise RuntimeError("completion metadata has no prediction file")
        predictions = load_partial(sealed)
        validate_predictions(predictions, selected_rows, fingerprint)
        if (len(predictions) != completion.get("count")
                or sha256(sealed) != completion.get("predictions_sha256")):
            raise RuntimeError("completed working predictions failed recovery validation")
        if not final_predictions.exists():
            os.replace(partial_predictions, final_predictions)
        _finalize_condition(target, plan, selected_rows, fingerprint, fingerprint_value)
        return
    if final_predictions.exists():
        raise RuntimeError("final predictions exist without completion metadata")

    C.setup_environment()
    started = time.monotonic()
    import torch
    device_identity = runtime_device_identity(torch, args.device)
    device_class = runtime_device_class(device_identity)
    execution = {**execution_static, "device_class": device_class}
    ensure_frozen_json(execution_path, execution, "condition execution")
    repair_uncommitted_jsonl_tail(target / "device_attempts.jsonl",
                                  target / "recovery_events.jsonl")
    append_jsonl(target / "device_attempts.jsonl", {
        "time": utc_now(), "pid": os.getpid(), "device_identity": device_identity,
        "device_class": device_class,
    })
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    checkpoint = plan["models"][args.model]["path"]
    model, tokenizer, audit, timings = load_model(checkpoint, args.format, args.device)
    ensure_frozen_json(target / "quantization_audit.json", audit, "quantization audit")
    state = quantized_state_manifest(model)
    ensure_frozen_json(target / "quantized_state.json", state, "quantized state")
    token_ids, tokenizer_audit = tokenizer_protocol_audit(
        tokenizer, selected_rows, examples, plan["protocol"]["max_length"])
    ensure_frozen_json(target / "tokenizer_audit.json", tokenizer_audit, "tokenizer audit")

    resume_count = len(load_partial(partial_predictions, repair_trailing=True,
                                    repair_log=target / "recovery_events.jsonl"))
    eval_started = time.monotonic()
    predictions = score_rows(
        model, tokenizer, token_ids, eval_rows, examples, args.device, partial_predictions,
        max_length=plan["protocol"]["max_length"], batch_size=args.batch_size, limit=args.limit,
        condition_fingerprint=fingerprint,
    )
    eval_finished = time.monotonic()
    validate_predictions(predictions, selected_rows, fingerprint)
    peak = (torch.cuda.max_memory_allocated(args.device)
            if str(args.device).startswith("cuda") else None)
    evidence_hashes = {
        "execution_sha256": sha256(target / "execution.json"),
        "predictions_manifest_sha256": sha256(target / "predictions_manifest.json"),
        "device_attempts_sha256": sha256(target / "device_attempts.jsonl"),
        "quantization_audit_sha256": sha256(target / "quantization_audit.json"),
        "tokenizer_audit_file_sha256": sha256(target / "tokenizer_audit.json"),
        "quantized_state_file_sha256": sha256(target / "quantized_state.json"),
    }
    device_attempts = read_jsonl_objects(target / "device_attempts.jsonl")
    device_uuids = sorted({item["device_identity"]["uuid"] for item in device_attempts})
    completion = {
        "condition_fingerprint": fingerprint, "count": len(predictions),
        "predictions_sha256": sha256(partial_predictions),
        "quantized_state_manifest_sha256": state["manifest_sha256"],
        "evidence_hashes": evidence_hashes,
        "timing": {**timings, "evaluation_seconds_this_process": eval_finished - eval_started,
                   "total_seconds_this_process": eval_finished - started,
                   "resumed_from_predictions": resume_count,
                   "complete_single_process_timing": resume_count == 0},
        "environment": {**environment, "host": device_identity["host"],
                        "cuda_device": device_identity.get("device_name"),
                        "device_class": device_class,
                        "device_uuids": device_uuids,
                        "device_attempt_count": len(device_attempts),
                        "last_device_identity": device_identity,
                        "peak_gpu_bytes": peak if resume_count == 0 else None,
                        "peak_gpu_bytes_this_process": peak},
    }
    write_json_once(completion_path, completion)
    os.replace(partial_predictions, final_predictions)
    _finalize_condition(target, plan, selected_rows, fingerprint, fingerprint_value)


def run_condition(args):
    if args.split == "test" and not args.final:
        raise ValueError("test evaluation requires --final")
    if args.split != "test" and args.final:
        raise ValueError("--final is only valid with --split test")
    if args.split == "test" and args.limit is not None:
        raise ValueError("a final test run cannot use --limit")
    run, plan = load_plan(args.run)
    if args.split == "test" and args.batch_size != plan["protocol"]["final_batch_size"]:
        raise ValueError(f"final test batch size is frozen at "
                         f"{plan['protocol']['final_batch_size']}")
    if args.format not in plan["formats"]:
        raise ValueError(f"format {args.format!r} is not in the plan")
    if args.model not in plan["models"]:
        raise ValueError(f"model {args.model!r} is not in the plan")
    target = condition_dir(run, args.model, args.format, args.split)
    target.mkdir(parents=True, exist_ok=True)
    fingerprint, fingerprint_value = condition_fingerprint(
        run, plan, args.model, args.format, args.split)
    with condition_lock(target):
        if args.split == "test":
            verify_test_selection(run, plan)
            append_test_access(run, plan, args.model, args.format, args.split, fingerprint)
        verify_plan_inputs(plan, args.model, ("dev", args.split))
        _run_condition_locked(args, run, plan, target, fingerprint, fingerprint_value)


def summarize(args):
    from regionalign.reporting.mmlu import summarize_run
    summarize_run(C.output_path(args.run))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="freeze inputs and allocate a new append-only run")
    plan.add_argument("--model", action="append", type=parse_model_spec, required=True,
                      help="repeat KEY=/checkpoint/path")
    plan.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    plan.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    plan.add_argument("--out", help="new path under regionalign/results/; must not exist")
    plan.set_defaults(function=create_plan)
    run = commands.add_parser("run", help="run or resume one frozen condition")
    run.add_argument("--run", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--format", choices=FORMATS, required=True)
    run.add_argument("--split", choices=SPLITS, required=True)
    run.add_argument("--final", action="store_true")
    run.add_argument("--limit", type=int)
    run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run.add_argument("--device", default="cuda:0")
    run.set_defaults(function=run_condition)
    summary = commands.add_parser("summarize", help="create tables and figures from six final results")
    summary.add_argument("--run", required=True)
    summary.set_defaults(function=summarize)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if getattr(args, "max_length", DEFAULT_MAX_LENGTH) < 128:
        raise ValueError("max_length must be at least 128")
    if getattr(args, "batch_size", DEFAULT_BATCH_SIZE) < 1:
        raise ValueError("batch_size must be positive")
    if getattr(args, "limit", None) is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
