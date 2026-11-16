"""Atomically build paper tables and figures for a completed clean MMLU run."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from uuid import uuid4

from regionalign.experiments import clean_mmlu as M

BOOTSTRAP_SEED = 20261108
BOOTSTRAP_SAMPLES = 2000
REGISTERED_METRICS = (
    "count", "correct", "accuracy", "accuracy_wilson95", "subject_macro_accuracy",
    "subjects", "nonfinite", "nonfinite_rate", "reduced_shot_prompts",
    "max_prompt_tokens", "by_subject",
)


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value}")


def _load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line, parse_constant=_reject_constant))
                except (ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def _write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paired_bootstrap(left, right, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    import numpy as np

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired bootstrap inputs must be equal nonempty vectors")
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, samples, 100):
        count = min(100, samples - start)
        indices = rng.integers(0, len(left), size=(count, len(left)))
        estimates.extend((right[indices] - left[indices]).mean(axis=1).tolist())
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _strict_quantization(audit, fmt):
    return (audit.get("schema_version") == 2
            and audit.get("scope") == "all_llm_parameters"
            and audit.get("format") == fmt
            and audit.get("bits") == M.P.FORMAT_BITS[fmt]
            and audit.get("quantized_fraction") == 1.0
            and audit.get("unquantized_tensors") == []
            and audit.get("quantized_parameters") == audit.get("total_parameters")
            and audit.get("quantized_tensors") == audit.get("total_tensors")
            and audit.get("int4_group_columns") == (M.P.GROUP if fmt == "int4" else None)
            and audit.get("scales_faultable") is False
            and audit.get("compute") == "decoded_weights_in_parameter_dtype")


def _condition(run, plan, model, fmt, test_rows):
    root = Path(run) / "conditions" / model / fmt / "test"
    paths = {name: root / name for name in (
        "metrics.json", "predictions.jsonl", "quantization_audit.json",
        "tokenizer_audit.json", "execution.json", "predictions_manifest.json",
        "quantized_state.json", "completion_metadata.json", "device_attempts.jsonl",
    )}
    if not all(path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise RuntimeError(f"missing completed test files for {model}/{fmt}: {missing}")
    metrics = M.read_json(paths["metrics.json"])
    predictions = _load_jsonl(paths["predictions.jsonl"])
    audit = M.read_json(paths["quantization_audit.json"])
    tokenizer_audit = M.read_json(paths["tokenizer_audit.json"])
    execution = M.read_json(paths["execution.json"])
    prediction_manifest = M.read_json(paths["predictions_manifest.json"])
    state = M.read_json(paths["quantized_state.json"])
    completion = M.read_json(paths["completion_metadata.json"])
    device_attempts = M.read_jsonl_objects(paths["device_attempts.jsonl"])
    fingerprint, fingerprint_value = M.condition_fingerprint(run, plan, model, fmt, "test")
    if (metrics.get("condition_fingerprint") != fingerprint
            or execution.get("condition_fingerprint") != fingerprint
            or prediction_manifest.get("condition_fingerprint") != fingerprint
            or execution.get("condition") != fingerprint_value
            or execution.get("device_class") != metrics.get("environment", {}).get("device_class")):
        raise RuntimeError(f"condition fingerprint drift for {model}/{fmt}")
    if (metrics.get("model") != model or metrics.get("format") != fmt
            or metrics.get("split") != "test" or metrics.get("final") is not True
            or metrics.get("limit") is not None or metrics.get("count") != 14042
            or execution.get("final") is not True or execution.get("limit") is not None
            or execution.get("batch_size") != plan["protocol"]["final_batch_size"]
            or metrics.get("batch_size") != plan["protocol"]["final_batch_size"]
            or prediction_manifest.get("expected_rows") != 14042):
        raise RuntimeError(f"invalid final test condition: {model}/{fmt}")
    if prediction_manifest.get("identity_sha256") != M.object_sha256(
            [(row["subject"], row["answer"]) for row in test_rows]):
        raise RuntimeError(f"dataset identity drift for {model}/{fmt}")
    M.validate_predictions(predictions, test_rows, fingerprint)
    recalculated = M.metrics_from_predictions(predictions)
    for key in REGISTERED_METRICS:
        if recalculated[key] != metrics.get(key):
            raise RuntimeError(f"metric drift for {model}/{fmt}: {key}")
    if (M.sha256(paths["predictions.jsonl"]) != metrics.get("predictions_sha256")
            or M.sha256(paths["execution.json"]) != metrics.get("execution_sha256")
            or M.sha256(paths["predictions_manifest.json"]) != metrics.get("predictions_manifest_sha256")
            or M.sha256(paths["device_attempts.jsonl"]) != metrics.get("device_attempts_sha256")
            or M.sha256(paths["quantization_audit.json"]) != metrics.get("quantization_audit_sha256")
            or M.sha256(paths["tokenizer_audit.json"]) != metrics.get("tokenizer_audit_file_sha256")
            or M.sha256(paths["quantized_state.json"]) != metrics.get("quantized_state_file_sha256")
            or M.sha256(paths["completion_metadata.json"]) != metrics.get("completion_metadata_sha256")
            or M.object_sha256(tokenizer_audit) != metrics.get("tokenizer_audit_sha256")):
        raise RuntimeError(f"evidence hash drift for {model}/{fmt}")
    state_digest = M.verify_quantized_state_manifest(state)
    completion_evidence = {
        key: metrics.get(key) for key in (
            "execution_sha256", "predictions_manifest_sha256", "device_attempts_sha256",
            "quantization_audit_sha256", "tokenizer_audit_file_sha256",
            "quantized_state_file_sha256",
        )
    }
    if (state_digest != metrics.get("quantized_state_manifest_sha256")
            or completion.get("condition_fingerprint") != fingerprint
            or completion.get("count") != len(predictions)
            or completion.get("predictions_sha256") != metrics.get("predictions_sha256")
            or completion.get("quantized_state_manifest_sha256") != state_digest
            or completion.get("evidence_hashes") != completion_evidence
            or completion.get("environment") != metrics.get("environment")
            or completion.get("timing") != metrics.get("timing")):
        raise RuntimeError(f"completion metadata drift for {model}/{fmt}")
    if not device_attempts or any(
            attempt.get("device_class") != execution.get("device_class")
            or M.runtime_device_class(attempt.get("device_identity", {})) != execution.get("device_class")
            for attempt in device_attempts):
        raise RuntimeError(f"device attempt class drift for {model}/{fmt}")
    if any(not isinstance(attempt["device_identity"].get("uuid"), str)
           or not attempt["device_identity"]["uuid"].startswith("GPU-")
           for attempt in device_attempts):
        raise RuntimeError(f"invalid device UUID evidence for {model}/{fmt}")
    device_uuids = sorted({attempt["device_identity"].get("uuid") for attempt in device_attempts})
    measured_environment = metrics.get("environment", {})
    if (measured_environment.get("device_attempt_count") != len(device_attempts)
            or measured_environment.get("device_uuids") != device_uuids
            or measured_environment.get("last_device_identity")
            != device_attempts[-1]["device_identity"]):
        raise RuntimeError(f"device attempt summary drift for {model}/{fmt}")
    if audit != metrics.get("quantization") or not _strict_quantization(audit, fmt):
        raise RuntimeError(f"quantization audit drift for {model}/{fmt}")
    for key, value in plan["environment"].items():
        if metrics.get("environment", {}).get(key) != value or execution["environment"].get(key) != value:
            raise RuntimeError(f"environment drift for {model}/{fmt}: {key}")
    return metrics, predictions, {name: M.file_record(path, run) for name, path in paths.items()}


def summarize_run(run):
    run, plan = M.load_plan(run)
    if set(plan.get("models", {})) != set(M.PAPER_MODELS):
        raise ValueError("paper summary requires the three frozen base models")
    if sorted(plan.get("formats", [])) != ["int4", "int8"] or len(plan["formats"]) != 2:
        raise ValueError("paper summary requires exactly INT8 and INT4")
    expected_conditions = {(model, fmt) for model in M.PAPER_MODELS for fmt in M.FORMATS}
    planned_conditions = {(row.get("model"), row.get("format")) for row in plan.get("conditions", [])}
    if planned_conditions != expected_conditions or len(plan["conditions"]) != 6:
        raise ValueError("plan does not contain the exact six-condition matrix")
    if (Path(run) / "paper_artifacts").exists():
        raise FileExistsError(f"paper artifacts already exist: {run / 'paper_artifacts'}")
    required = ("metrics.json", "predictions.jsonl", "quantization_audit.json",
                "tokenizer_audit.json", "execution.json", "predictions_manifest.json",
                "quantized_state.json", "completion_metadata.json", "device_attempts.jsonl")
    missing = []
    for model, fmt in sorted(expected_conditions):
        root = Path(run) / "conditions" / model / fmt / "test"
        missing.extend(f"{model}/{fmt}/{name}" for name in required if not (root / name).is_file())
        if (root / "predictions.partial.jsonl").exists():
            missing.append(f"{model}/{fmt}/predictions.partial.jsonl remains")
    if missing:
        raise RuntimeError(f"paper summary requires six sealed test conditions; missing={missing}")
    M.verify_test_selection(run, plan)
    M.append_summary_test_access(run, plan)
    for model in sorted(plan["models"]):
        M.verify_plan_inputs(plan, model)
    test_rows = M.read_arrow(M.arrow_path(plan["dataset"]["hf_home"], "test"))
    if len(test_rows) != 14042:
        raise RuntimeError("frozen MMLU test size drift")

    loaded = {}
    evidence = {}
    summary_rows = []
    subject_rows = []
    for model in sorted(plan["models"]):
        for fmt in M.FORMATS:
            metrics, predictions, records = _condition(run, plan, model, fmt, test_rows)
            loaded[(model, fmt)] = (metrics, predictions)
            evidence[f"{model}/{fmt}"] = records
            low, high = metrics["accuracy_wilson95"]
            summary_rows.append({
                "model": model, "format": fmt, "count": metrics["count"],
                "accuracy": f"{metrics['accuracy']:.9f}",
                "accuracy_ci95_low": f"{low:.9f}", "accuracy_ci95_high": f"{high:.9f}",
                "subject_macro_accuracy": f"{metrics['subject_macro_accuracy']:.9f}",
                "nonfinite_rate": f"{metrics['nonfinite_rate']:.9f}",
                "reduced_shot_prompts": metrics["reduced_shot_prompts"],
                "max_prompt_tokens": metrics["max_prompt_tokens"],
                "complete_single_process_timing": metrics["timing"]["complete_single_process_timing"],
                "evaluation_seconds": (f"{metrics['timing']['evaluation_seconds_this_process']:.3f}"
                                       if metrics["timing"]["complete_single_process_timing"] else ""),
                "peak_gpu_bytes": metrics["environment"]["peak_gpu_bytes"],
            })
            for subject, values in metrics["by_subject"].items():
                subject_rows.append({"model": model, "format": fmt, "subject": subject, **values})
    cuda_devices = {metrics["environment"].get("cuda_device") for metrics, _ in loaded.values()}
    if len(cuda_devices) != 1 or None in cuda_devices:
        raise RuntimeError(f"conditions used inconsistent GPU models: {cuda_devices}")
    hardware_classes = {
        json.dumps(metrics["environment"].get("device_class"), sort_keys=True)
        for metrics, _ in loaded.values()
    }
    if len(hardware_classes) != 1:
        raise RuntimeError(f"conditions used inconsistent hardware classes: {hardware_classes}")

    deltas = []
    for index, model in enumerate(sorted(plan["models"])):
        m8, p8 = loaded[(model, "int8")]
        m4, p4 = loaded[(model, "int4")]
        identity8 = [(row["index"], row["subject"], row["gold"]) for row in p8]
        identity4 = [(row["index"], row["subject"], row["gold"]) for row in p4]
        if identity8 != identity4:
            raise RuntimeError(f"INT8/INT4 predictions are not paired for {model}")
        interval = _paired_bootstrap([row["correct"] for row in p8],
                                     [row["correct"] for row in p4],
                                     seed=BOOTSTRAP_SEED + index)
        deltas.append({"model": model, "int8_accuracy": m8["accuracy"],
                       "int4_accuracy": m4["accuracy"],
                       "int4_minus_int8": m4["accuracy"] - m8["accuracy"],
                       "paired_bootstrap_ci95": interval})

    stage = run / f".paper_artifacts.{uuid4().hex}.tmp"
    stage.mkdir(exist_ok=False)
    summary = {
        "schema_version": M.SCHEMA_VERSION, "created_at": M.utc_now(),
        "scope": plan["scope"], "models": sorted(plan["models"]),
        "formats": list(M.FORMATS), "conditions": 6,
        "test_examples_per_condition": 14042, "measured_results": True,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES,
                      "unit": "paired MMLU test item"},
        "int4_minus_int8": deltas, "evidence": evidence,
        "environment": {**plan["environment"], "cuda_device": next(iter(cuda_devices))},
    }
    M.write_json_once(stage / "summary.json", summary)
    _write_csv(stage / "tables" / "mmlu_overall.csv", list(summary_rows[0]), summary_rows)
    _write_csv(stage / "tables" / "mmlu_by_subject.csv", list(subject_rows[0]), subject_rows)
    completed_rows = [{"model": row["model"], "format": row["format"], "split": "test",
                       "status": "complete", "examples": row["count"]} for row in summary_rows]
    _write_csv(stage / "completed_matrix.csv", list(completed_rows[0]), completed_rows)

    tex_lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Clean MMLU accuracy with up to five in-context examples under full-parameter quantization. Accuracy intervals are 95\% Wilson intervals; $\Delta$ is INT4 minus INT8 with a paired item-bootstrap interval.}",
        r"\label{tab:mmlu-quantization}", r"\begin{tabular}{lccc}", r"\toprule",
        r"Model & INT8 accuracy [95\% CI] & INT4 accuracy [95\% CI] & $\Delta$ pp [95\% CI] \\",
        r"\midrule",
    ]
    for delta in deltas:
        model = delta["model"]
        m8 = loaded[(model, "int8")][0]
        m4 = loaded[(model, "int4")][0]
        l8, h8 = m8["accuracy_wilson95"]
        l4, h4 = m4["accuracy_wilson95"]
        ld, hd = delta["paired_bootstrap_ci95"]
        tex_lines.append(
            f"{_escape(model)} & {100*m8['accuracy']:.2f} [{100*l8:.2f}, {100*h8:.2f}] & "
            f"{100*m4['accuracy']:.2f} [{100*l4:.2f}, {100*h4:.2f}] & "
            f"{100*delta['int4_minus_int8']:+.2f} [{100*ld:+.2f}, {100*hd:+.2f}] \\\\")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    _write_text(stage / "tables" / "mmlu_overall.tex", "\n".join(tex_lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = sorted(plan["models"])
    x = np.arange(len(models), dtype=float)
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for offset, fmt, color in ((-width / 2, "int8", "#4472C4"),
                               (width / 2, "int4", "#ED7D31")):
        values = [loaded[(model, fmt)][0]["accuracy"] * 100 for model in models]
        errors = []
        for model, value in zip(models, values):
            low, high = loaded[(model, fmt)][0]["accuracy_wilson95"]
            errors.append([value - 100 * low, 100 * high - value])
        axis.bar(x + offset, values, width, label=fmt.upper(), color=color,
                 yerr=np.asarray(errors).T, capsize=3)
    axis.set_ylabel("MMLU accuracy (%)")
    axis.set_xticks(x, models)
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    figures = stage / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "mmlu_accuracy.pdf", bbox_inches="tight")
    fig.savefig(figures / "mmlu_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    values = np.asarray([100 * row["int4_minus_int8"] for row in deltas])
    lows = np.asarray([100 * row["paired_bootstrap_ci95"][0] for row in deltas])
    highs = np.asarray([100 * row["paired_bootstrap_ci95"][1] for row in deltas])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.errorbar(x, values, yerr=np.vstack([values - lows, highs - values]),
                  fmt="o", color="#7030A0", capsize=4)
    axis.set_ylabel("INT4 - INT8 accuracy (pp)")
    axis.set_xticks(x, models)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "int4_minus_int8.pdf", bbox_inches="tight")
    fig.savefig(figures / "int4_minus_int8.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    validation = {
        "valid": True, "measured_conditions": len(summary_rows), "expected_conditions": 6,
        "all_full_test": all(row["count"] == 14042 for row in summary_rows),
        "all_accuracy_values_finite": all(math.isfinite(float(row["accuracy"])) for row in summary_rows),
        "total_nonfinite_predictions": sum(loaded[key][0]["nonfinite"] for key in loaded),
        "sources_checkpoints_and_dataset_verified": True, "no_attack_search": True,
    }
    if not all((validation["all_full_test"], validation["all_accuracy_values_finite"],
                validation["sources_checkpoints_and_dataset_verified"])):
        raise RuntimeError(f"summary validation failed: {validation}")
    M.write_json_once(stage / "validation.json", validation)
    M.write_json_once(stage / "FINAL_STATUS.json", {"status": "complete",
                                                     "completed_at": M.utc_now(),
                                                     "measured_results": True})
    os.replace(stage, run / "paper_artifacts")
    print(json.dumps({"run": str(run), **validation}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    summarize_run(parser.parse_args().run)
