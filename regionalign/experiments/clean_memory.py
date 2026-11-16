"""Independent clean-only VLM entry point with measured synthetic GPU admission.

The default 20 GiB free-memory threshold is a calibration candidate, not a
claim that all final examples fit.  Configuration replaces dictionaries only
inside this process; existing source files, dtype, quantization, image limits
and final scoring stay unchanged.  Every new plan binds this module and its
threshold.  Calibration never opens benchmark Arrow rows.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import resource
import time

from regionalign.experiments import clean_benchmarks as B


SOURCE = "regionalign/experiments/clean_memory.py"
DEFAULT_MIN_FREE_GPU_GIB = 20.0
_BASE_SETTINGS = dict(B.SETTINGS)
_BASE_SOURCES = tuple(B.SOURCES)
LONG_QUESTION = "Describe the colors and geometric shapes in this synthetic image. " * 48
LONG_ANSWER = "white rectangle " * 64


def configure(min_free_gpu_gib=DEFAULT_MIN_FREE_GPU_GIB):
    """Explicitly configure the shared clean evaluator in this process only."""
    if (isinstance(min_free_gpu_gib, bool)
            or not math.isfinite(float(min_free_gpu_gib)) or float(min_free_gpu_gib) <= 0):
        raise ValueError("minimum free GPU GiB must be finite and positive")
    B.SETTINGS = {**_BASE_SETTINGS, "min_free_gpu_gib": float(min_free_gpu_gib)}
    B.SOURCES = tuple(dict.fromkeys((*_BASE_SOURCES, SOURCE)))
    return dict(B.SETTINGS)


def load_plan(run):
    """Reject legacy, differently configured, or incomplete-source plans."""
    plan = B.load_plan(run)
    if ({r["path"] for r in plan["sources"]} != set(B.SOURCES)
            or SOURCE not in B.SOURCES or plan.get("no_faults_or_search") is not True):
        raise ValueError("clean memory plan must bind this entry point and all clean sources")
    if any(c["model"] not in B.VLM_TYPES or c["format"] not in ("int8", "int4")
           or c["task"] not in ("okvqa", "textvqa") for c in plan["conditions"]):
        raise ValueError("clean memory entry point accepts VLM clean conditions only")
    return plan


def prepare(args):
    """Inherit frozen input identities for unstarted clean VLM conditions only."""
    parent = B.C.output_path(args.parent_run)
    old = B.M.read_json(parent / "plan.json")
    parent_sha = B.M.sha256(parent / "plan.json")
    selection = B.M.read_json(parent / "test_selection.json")
    if (old.get("scope") != "clean_benchmarks_v1"
            or old.get("no_faults_or_search") is not True
            or selection != {"plan_sha256": parent_sha, "conditions": old["conditions"]}):
        raise ValueError("parent must be a frozen clean-only plan")
    strip = lambda settings: {k: v for k, v in settings.items() if k != "min_free_gpu_gib"}
    if strip(old["settings"]) != strip(B.SETTINGS):
        raise ValueError("only the memory admission threshold may change")
    B.M.verify_records(B.C.PROJECT, old["sources"])
    for row in old["sources"]:
        if B.M.sha256(parent / "source" / row["path"].replace("/", "__")) != row["sha256"]:
            raise ValueError("parent source snapshot changed")
    requested = set(args.condition or [])
    conditions = []
    for condition in old["conditions"]:
        name = "/".join(condition[key] for key in ("model", "format", "task"))
        if condition["model"] not in B.VLM_TYPES or (requested and name not in requested):
            continue
        target = parent / "conditions" / condition["model"] / condition["format"] / condition["task"]
        if target.exists():
            if requested:
                raise ValueError(f"selected parent condition already has output: {name}")
            continue
        conditions.append(condition)
    selected = {"/".join(c[k] for k in ("model", "format", "task")) for c in conditions}
    if not conditions or requested - selected:
        raise ValueError("no matching unstarted clean VLM conditions")
    models = {c["model"] for c in conditions}
    tasks = {c["task"] for c in conditions}
    B.freeze_plan(args.output, {k: v for k, v in old["models"].items() if k in models},
                  {k: v for k, v in old["datasets"].items() if k in tasks}, conditions,
                  memory_calibration={"entry_point": SOURCE,
                      "candidate_min_free_gpu_gib": B.SETTINGS["min_free_gpu_gib"],
                      "synthetic_only": True, "fit_not_guaranteed": True},
                  parent={"run": str(parent), "plan_sha256": parent_sha,
                          "prior_final_exposure": True})


def _memory(torch, device):
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {"free_bytes": free, "total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def _shapes(inputs):
    result = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
              for key, value in inputs.items() if hasattr(value, "shape")}
    if "image_grid_thw" in inputs:
        result["image_grid_thw"]["values"] = inputs["image_grid_thw"].detach().cpu().tolist()
    return result


def _stress_generation(torch, model, inputs):
    """Force 32 synthetic tokens independently of the unchanged final scorer."""
    length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, num_beams=1,
            min_new_tokens=32, max_new_tokens=32, use_cache=True,
            return_dict_in_generate=True, output_logits=True)
    generated = output.sequences.shape[1] - length
    if generated != 32 or not all(bool(torch.isfinite(x).all()) for x in output.logits):
        raise RuntimeError("synthetic 32-token generation stress failed")
    return {"stress_only": True, "generated_tokens": generated,
            "min_new_tokens": 32, "max_new_tokens": 32}


def preflight(args):
    """Measure load, original GPU quantization, state audit and synthetic pressure."""
    B.C.setup_environment()
    run = B.C.output_path(args.run)
    plan = load_plan(run)
    if not any(c["model"] == args.model and c["format"] == args.format for c in plan["conditions"]):
        raise ValueError("model/format is not in the frozen plan")
    key = f"{args.model}_{args.format}.json"
    profile_path, preflight_path = run / "memory_profiles" / key, run / "preflight" / key
    if profile_path.exists() or preflight_path.exists():
        raise FileExistsError("memory profile or preflight already exists; use a new plan")
    import torch
    from PIL import Image
    profile = {"status": "failed", "scope": "synthetic_memory_calibration_only",
               "plan_sha256": B.M.sha256(run / "plan.json"), "model": args.model,
               "format": args.format, "min_free_gpu_gib": B.SETTINGS["min_free_gpu_gib"],
               "no_final_data": True, "started_at": B.M.utc_now(), "phases": []}
    started, phase = time.monotonic(), "checkpoint_verification"
    try:
        checkpoint = plan["models"][args.model]
        B.M.verify_records(checkpoint["path"], checkpoint["files"])
        phase = "cuda_setup"
        profile["device"] = B.M.runtime_device_identity(torch, args.device)
        profile["environment"] = B.M.software_environment()
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        profile["initial_memory"] = _memory(torch, args.device)
        if profile["initial_memory"]["free_bytes"] < B.SETTINGS["min_free_gpu_gib"] * 1024 ** 3:
            raise RuntimeError("available GPU memory is below the plan's admission threshold")
        phase = "load_and_original_gpu_quantization"
        model, processor, language, audit, timing = B.load_clean(
            args.model, checkpoint["path"], args.format, args.device)
        profile.update(audit=audit, timing=timing)
        profile["phases"].append({"phase": phase, "memory": _memory(torch, args.device)})
        phase = "quantized_state_manifest"
        state = B.M.quantized_state_manifest(language)
        profile["quantized_state_manifest_sha256"] = state["manifest_sha256"]
        del state
        profile["phases"].append({"phase": phase, "memory": _memory(torch, args.device)})
        cases = [("normal", (336, 336), "What color is this image?", ["white"] * 10),
                 ("mixed_reference", (336, 336), "What color is this image?", ["white"] * 9 + ["a"]),
                 ("raw_reference", (336, 336), "What color is this image?", ["a"] * 10),
                 ("blank_reference", (336, 336), "What color is this image?", [""] * 10),
                 ("landscape_pressure", (1120, 896), LONG_QUESTION, [LONG_ANSWER] * 10),
                 ("portrait_pressure", (896, 1120), LONG_QUESTION, [LONG_ANSWER] * 10)]
        for name, size, question, answers in cases:
            image = Image.new("RGB", size, "white")
            phase = name
            inputs = B.vlm_inputs(args.model, processor, image, question, args.device)
            record = {"phase": name, "image_size": list(size), "processor": _shapes(inputs),
                      "synthetic_question_characters": len(question), "results": []}
            del inputs
            for task in ("okvqa", "textvqa"):
                value = B.score_vqa(model, processor, args.model, image, question, answers, task, args.device)
                if value["status"] != "complete":
                    raise RuntimeError("nonfinite synthetic clean score")
                record["results"].append({"task": task, **value})
            if name in ("normal", "landscape_pressure", "portrait_pressure"):
                inputs = B.vlm_inputs(args.model, processor, image, question, args.device)
                record["forced_generation"] = _stress_generation(torch, model, inputs)
                del inputs
            record["memory"] = _memory(torch, args.device)
            profile["phases"].append(record)
        load_plan(run)  # Refuse successful publication after source drift.
        profile.update(status="complete", memory=_memory(torch, args.device))
    except Exception as error:
        profile.update(error={"type": type(error).__name__, "message": str(error)},
                       failed_phase=phase, oom=isinstance(error, torch.cuda.OutOfMemoryError)
                       or "out of memory" in str(error).lower())
        try:
            profile["memory"] = _memory(torch, args.device)
        except Exception as measurement_error:
            profile["memory_measurement_error"] = str(measurement_error)
        raise
    finally:
        profile.update(completed_at=B.M.utc_now(), seconds=time.monotonic() - started,
                       host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        B.M.write_json_once(profile_path, profile)
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    B.M.write_json_once(preflight_path, {"status": "complete", "scope": profile["scope"],
        "plan_sha256": profile["plan_sha256"], "min_free_gpu_gib": profile["min_free_gpu_gib"],
        "memory_profile": str(profile_path.relative_to(run)),
        "memory_profile_sha256": B.M.sha256(profile_path), "memory": profile["memory"],
        "completed_at": profile["completed_at"]})
    print(preflight_path, flush=True)


def run_condition(args):
    B.require_final(args.split, args.final)
    run = B.C.output_path(args.run)
    load_plan(run)
    profile = run / "memory_profiles" / f"{args.model}_{args.format}.json"
    preflight = B.M.read_json(run / "preflight" / f"{args.model}_{args.format}.json")
    if preflight.get("memory_profile_sha256") != B.M.sha256(profile):
        raise ValueError("missing or changed synthetic memory profile")
    B.run_condition(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--parent-run", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--condition", action="append", help="model/format/task; defaults to all unstarted VLM conditions")
    prep.set_defaults(function=prepare)
    for name, function in (("preflight", preflight), ("run", run_condition)):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        command.add_argument("--model", required=True, choices=sorted(B.VLM_TYPES))
        command.add_argument("--format", required=True, choices=("int8", "int4"))
        command.add_argument("--device", default="cuda:0")
        if name == "run":
            command.add_argument("--task", required=True, choices=("okvqa", "textvqa"))
            command.add_argument("--split", required=True, choices=("test",))
            command.add_argument("--final", action="store_true")
        command.set_defaults(function=function)
    for command in (prep, *[commands.choices[name] for name in ("preflight", "run")]):
        command.add_argument("--min-free-gpu-gib", type=float, default=DEFAULT_MIN_FREE_GPU_GIB)
    args = parser.parse_args(argv)
    configure(args.min_free_gpu_gib)
    args.function(args)


if __name__ == "__main__":
    main()
