"""Audit and execute the reproducible model/data matrix for the new protocol.

The package already contains separate, audited clean MMLU and WikiText
drivers.  This module is the small orchestration layer that freezes the matrix
inputs before a run.  It deliberately refuses to label a condition as an
attack result until a task-specific runtime and region evaluator are present;
a runtime written for one VQA model cannot be silently reused for another.

The ``audit`` command is offline and write-once.  ``download`` is an explicit
opt-in helper for missing public inputs and is never called by ``audit``.
Besides the audit table it accepts every checkpoint of the formal LLM registry
(``clean_mmlu.MODEL_SPECS``), so a newly registered model needs no second entry
here, and it checks such a download against that registry before reporting it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from regionalign import config as C
from regionalign.experiments import clean_mmlu as M

SCHEMA = 1


def cache_root(args):
    """The cache to audit or fill: an explicit ``--hf-home``, else the package's configured one."""
    return Path(args.hf_home or C.HUB).expanduser().absolute()


MODELS = {
    "llama2-7b": {
        "repository": "meta-llama/Llama-2-7b-hf",
        "cache_component": "models--meta-llama--Llama-2-7b-hf",
        "architecture": "MHA",
        "family": "llm",
        "expected_model_type": "llama",
        "paths": [
            "hub/models--meta-llama--Llama-2-7b-hf/snapshots",
            "models--meta-llama--Llama-2-7b-hf/snapshots",
        ],
    },
    "llama3.1-8b": {
        "repository": "meta-llama/Meta-Llama-3.1-8B",
        "cache_component": "models--meta-llama--Meta-Llama-3.1-8B",
        "architecture": "GQA",
        "family": "llm",
        "expected_model_type": "llama",
        "paths": ["models--meta-llama--Meta-Llama-3.1-8B/snapshots"],
    },
    "qwen2.5-7b": {
        "repository": "Qwen/Qwen2.5-7B",
        "cache_component": "models--Qwen--Qwen2.5-7B",
        "architecture": "GQA",
        "family": "llm",
        "expected_model_type": "qwen2",
        "paths": ["models--Qwen--Qwen2.5-7B/snapshots"],
    },
    "llava-1.5-7b": {
        "repository": "llava-hf/llava-1.5-7b-hf",
        "cache_component": "models--llava-hf--llava-1.5-7b-hf",
        "architecture": "MHA",
        "family": "vlm",
        "expected_model_type": None,
        "paths": ["hub/models--llava-hf--llava-1.5-7b-hf/snapshots"],
    },
    "qwen2-vl-7b": {
        "repository": "Qwen/Qwen2-VL-7B-Instruct",
        "cache_component": "models--Qwen--Qwen2-VL-7B-Instruct",
        "architecture": "GQA",
        "family": "vlm",
        "expected_model_type": None,
        "paths": ["hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots"],
    },
}

DATASETS = {
    "mmlu": {
        "repository": "cais/mmlu",
        "split_files": ["mmlu-validation.arrow", "mmlu-test.arrow", "mmlu-dev.arrow"],
        "globs": ["datasets/cais___mmlu/all/0.0.0/*"],
        "protocol": "search=285 validation labels; dev=1246 validation labels; final=all test",
    },
    "wikitext-2-raw": {
        "repository": "wikitext",
        "split_files": ["wikitext-train.arrow", "wikitext-validation.arrow", "wikitext-test.arrow"],
        "globs": ["datasets/wikitext/wikitext-2-raw-v1/0.0.0/*"],
        "protocol": "search=128 train spans; dev=128 validation spans; final=full test",
    },
    "ok-vqa": {
        "repository": "lmms-lab/OK-VQA",
        "split_files": [],
        "globs": ["datasets/lmms-lab___ok-vqa/*"],
        "protocol": "search=128 train image-question pairs; dev=256 train pairs; final=all public test",
    },
    "text-vqa": {
        "repository": "lmms-lab/textvqa",
        "split_files": [],
        "globs": ["datasets/lmms-lab___textvqa/*"],
        "protocol": "search=128 train pairs; dev=256 train pairs; final=5000 public validation pairs",
    },
}

CODE_INVENTORY = {
    "quantization_and_permutation": ["regionalign/harness/core.py"],
    "region_search": ["regionalign/harness/region.py", "regionalign/algorithm/search.py"],
    "clean_mmlu": ["regionalign/experiments/clean_mmlu.py"],
    "clean_wikitext": ["regionalign/experiments/clean_wikitext.py"],
    "vlm_runtime": ["regionalign/runtimes/vqa.py"],
    "formal_studies": ["regionalign/experiments/llm.py", "regionalign/experiments/vqa.py"],
    "input_audit": ["regionalign/experiments/input_audit.py"],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def newest_snapshot(root: Path):
    snapshots = [p for p in root.glob("*/config.json") if p.is_file()]
    if not snapshots:
        return None
    return max((p.parent for p in snapshots), key=lambda p: p.stat().st_mtime)


def model_status(hf_home: Path, spec):
    candidates = [hf_home / rel for rel in spec["paths"]]
    snapshots = []
    for candidate in candidates:
        snap = newest_snapshot(candidate)
        if snap is not None:
            snapshots.append(snap)
    if not snapshots:
        return {"status": "missing", "repository": spec["repository"], "paths": [str(p) for p in candidates]}
    path = max(snapshots, key=lambda p: p.stat().st_mtime)
    try:
        config = json.loads((path / "config.json").read_text())
    except Exception as error:
        return {"status": "invalid", "path": str(path), "error": repr(error)}
    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (path / name).exists()]
    model_type = config.get("model_type")
    expected = spec.get("expected_model_type")
    mismatch = expected is not None and model_type != expected
    return {"status": "mismatch" if mismatch else ("incomplete" if missing else "available"),
            "repository": spec["repository"], "path": str(path),
            "model_type": model_type, "architectures": config.get("architectures"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "missing_files": missing, "expected_model_type": expected,
            "architecture_label": spec["architecture"]}


def dataset_status(hf_home: Path, spec):
    roots = []
    for pattern in spec["globs"]:
        roots.extend(sorted(hf_home.glob(pattern)))
    files = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(p for p in root.rglob("*.arrow") if p.is_file()))
    if not files:
        return {"status": "missing", "repository": spec["repository"], "roots": [str(p) for p in roots]}
    names = sorted({p.name for p in files})
    expected = spec["split_files"]
    missing = [name for name in expected if name not in names]
    return {"status": "incomplete" if missing else "available", "repository": spec["repository"],
            "files": [str(p) for p in files], "file_names": names, "missing_files": missing,
            "protocol": spec["protocol"]}


def audit(args):
    hf_home = cache_root(args)
    code = {group: [{"path": path, "exists": (C.PROJECT / path).is_file()}
                    for path in paths] for group, paths in CODE_INVENTORY.items()}
    result = {
        "schema_version": SCHEMA, "created_at": now(), "host": platform.node(),
        "python": sys.version.split()[0], "project": str(C.PROJECT), "hf_home": str(hf_home),
        "models": {key: model_status(hf_home, spec) for key, spec in MODELS.items()},
        "datasets": {key: dataset_status(hf_home, spec) for key, spec in DATASETS.items()},
        "formats": {"int8": {"bits": 8, "weights_per_4kib": 4096, "group": None},
                    "int4": {"bits": 4, "weights_per_4kib": 8192, "group": 128}},
        "repeats": 3, "k_schedule": [0, 1, 2, 4, 8, 16], "region_bytes": 4096,
        "code_inventory": code,
        "attack_support": {
            "llm_clean": "available via existing audited drivers",
            "llm_region_attack": "regionalign.experiments.llm; execution results are separate from this input audit",
            "llava_region_attack": "regionalign.experiments.vqa; requires its registered train inputs and preflight",
            "qwen2_vl_region_attack": "regionalign.experiments.vqa; requires its registered train inputs and preflight",
        },
    }
    result["audit_sha256"] = digest(result)
    output = C.fresh_output_dir("experimental_matrix", "input_audit", args.output)
    (output / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plan = {"schema_version": 1, "audit": str(output / "audit.json"),
            "conditions": [{"model": model, "format": fmt, "repeats": 3}
                           for model in MODELS for fmt in ("int8", "int4")],
            "datasets": {key: spec["protocol"] for key, spec in DATASETS.items()},
            "result_policy": "unsupported or missing inputs are recorded, never imputed"}
    plan["plan_sha256"] = digest(plan)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(output)
    print(json.dumps({"models": {k: v["status"] for k, v in result["models"].items()},
                      "datasets": {k: v["status"] for k, v in result["datasets"].items()}},
                     sort_keys=True))


def downloadable_models():
    """Model keys ``download`` accepts, each with its Hub repository and cache directory name.

    A key in both tables must name the same checkpoint in both.
    """
    models = {}
    for table in (MODELS, M.MODEL_SPECS):
        for key, spec in table.items():
            entry = {"repository": spec["repository"], "cache_component": spec["cache_component"]}
            if models.setdefault(key, entry) != entry:
                raise ValueError(f"{key} names different checkpoints in the audit and LLM registries")
    return models


def download(args):
    """Download only an explicitly named missing public input."""
    hf_home = cache_root(args)
    if args.model:
        from huggingface_hub import snapshot_download
        spec = downloadable_models()[args.model]
        path = snapshot_download(spec["repository"], revision=args.revision,
                                 cache_dir=str(hf_home / "hub"), resume_download=True)
        print(path)
        if args.model in M.MODEL_SPECS:
            # Fail now, not at plan time, if the snapshot is not the registered architecture.
            manifest = M.checkpoint_manifest(path, args.model)
            print(json.dumps({"model_key": args.model, "revision": manifest["revision"],
                              "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))
        return
    from datasets import load_dataset
    spec = DATASETS[args.dataset]
    dataset = load_dataset(spec["repository"], revision=args.revision,
                           cache_dir=str(hf_home / "datasets"))
    print(dataset)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--hf-home")
    a.add_argument("--output")
    a.set_defaults(function=audit)
    d = sub.add_parser("download")
    group = d.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=sorted(downloadable_models()))
    group.add_argument("--dataset", choices=sorted(DATASETS))
    d.add_argument("--revision", help="Hub commit, branch or tag to fetch (default: main)")
    d.add_argument("--hf-home")
    d.set_defaults(function=download)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
