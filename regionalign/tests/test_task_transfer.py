"""Independent VQA transfer metadata plus real tiny INT8/INT4 runtime checks."""
from contextlib import ExitStack
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from regionalign.algorithm import search as A
from regionalign.experiments import task_transfer as T
from regionalign.runtimes.vqa import HFVQARegionRuntime
from regionalign.tests.helpers import TINY_GROUP, tiny_lm


class TinyVLM(torch.nn.Module):
    """Real language forward/generation behind the HF VQA runtime API."""
    def __init__(self, language):
        super().__init__()
        self.language_model = language
        self.multi_modal_projector = torch.nn.Identity()
        self.visual = torch.nn.Identity()
        self.config = language.config
        self.generation_config = language.generation_config
        self.generation_config.pad_token_id = 0

    def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        return position_ids, torch.zeros(input_ids.shape[0], 1, dtype=torch.long)

    def forward(self, **kwargs):
        kwargs.pop("rope_deltas", None)
        return self.language_model(**kwargs)

    def generate(self, **kwargs):
        kwargs.pop("rope_deltas", None)
        return self.language_model.generate(**kwargs)


class TaskTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(T.C, "output_path", side_effect=lambda p: Path(p).resolve()))

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def parent(self, name, task):
        run = self.root / name
        run.mkdir()
        sources = T.V._copy_sources(run)
        for relative in (str(T.B.MODULE_SOURCE), T.CM.SELF, T.CM.BOUNDED):
            source = T.C.PROJECT / relative
            target = run / "source" / relative.replace("/", "__")
            T._copy_once(source, target)
            sources.append({"path": relative, "snapshot": str(target.relative_to(run)),
                            "sha256": T.M.sha256(source), "bytes": source.stat().st_size})
        panels = {}
        for split in ("search", "dev"):
            path = run / "panels" / f"{task}_{split}.arrow"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"synthetic train-derived {task} {split}".encode())
            panels[split] = {"path": str(path.relative_to(run)), "sha256": T.M.sha256(path),
                             "bytes": path.stat().st_size, "count": 2,
                             "row_ids_sha256": "rowids", "identities_sha256": "identities"}
        models = {model: {"path": f"/synthetic/{model}", "model_key": model,
                         "repository": T.V.MODEL_SPECS[model]["repository"], "revision": "fixture",
                         "dims": T.P.dims(tiny_lm()),
                         "files": [{"path": "config.json", "bytes": 2, "sha256": "a" * 64}]}
                  for model in T.V.DEV_MODELS}
        for manifest in models.values():
            manifest["manifest_sha256"] = T.M.object_sha256(manifest)
        conditions = []
        for model in T.V.DEV_MODELS:
            for fmt in T.V.FORMATS:
                language = tiny_lm()
                T.P.quantize(language, fmt, group=TINY_GROUP)
                dims = T.P.dims(language)
                regions = [T.R.make_region("gate_proj", 0, page, dims, bits=T.P.FORMAT_BITS[fmt], page_bytes=64)
                           for page in range(12)]
                bootstrap = T.V._bootstrap_records([{"sid": f"{task}-source-{i}"} for i in range(128)], task)
                conditions.append({"id": f"{model}__{fmt}__{task}", "model": model, "format": fmt,
                    "task": task, "availability": "available", "dims": dims, "regions": regions,
                    "checkpoint_manifest_sha256": models[model]["manifest_sha256"],
                    "panels": copy.deepcopy(panels), "search_bootstraps": bootstrap})
        execution = {"search_batch_size": 1, "grad_batch_size": 1, "eval_batch_size": 1,
                     "min_free_gpu_gib": 22.0, "min_free_policy": T.V.MIN_FREE_POLICY,
                     "real_gpu_preflight_required": True}
        policy = {"schema_version": T.B.POLICY_VERSION, "scope": T.B.CAMPAIGN_SCOPE,
                  "formal_loader_scope": T.V.SCOPE, "variant": "baseline",
                  "parameters": {"search_budget": 300}, "verification_sha256": T.M.object_sha256(T.CM.POLICY),
                  "resource_policy": {"min_free_gpu_gib": 22.0},
                  "inherited_panels": [{k: v for k, v in record.items() if k in ("path", "sha256", "bytes")}
                                       for record in panels.values()]}
        plan = {"schema_version": 2, "scope": T.V.SCOPE, "development_only": True,
                "splits": ["search", "dev"], "sources": sources, "models": models,
                "conditions": conditions, "execution": execution, "execution_verification": T.CM.POLICY,
                "campaign_policy": policy, "arms": list(T.V.ARMS), "repeat_seeds": list(T.V.REPEAT_SEEDS),
                "search_budget": 300, "search_cost": dict(T.R.COST), "search_k": list(T.R.K_SCHEDULE),
                "evaluation_k": list(T.V.EVAL_K), "work_items_per_condition": 108,
                "clean_gate_nll": T.R.GATE_NLL, "vqa_score_gate": T.V.VQA_SCORE_GATE,
                "success_relative_score_drop": T.V.VQA_SUCCESS_DROP}
        self.write(run / "plan.json", plan)
        self.write(run / "status.json", {"plan_sha256": T.M.sha256(run / "plan.json"), "development_only": True})
        plan["sha256"] = T.M.sha256(run / "plan.json")
        return run, plan

    def fixture(self, fmt="int8", source_task="textvqa"):
        source_run, source = self.parent("source", source_task)
        target_task = T.OPPOSITE[source_task]
        target_run, target = self.parent("target", target_task)
        run = T.prepare(source_run, source_task, target_run, target_task, self.root / "transfer")
        language = tiny_lm()
        T.P.quantize(language, fmt, group=TINY_GROUP)
        model = TinyVLM(language)
        tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: [5] if text == "cat" else [6, 7],
                                    decode=lambda tokens, skip_special_tokens=True: "cat")
        rows = [{"sid": f"target-{i}", "reference": "cat" if i == 0 else "two tokens",
                 "answers": ["cat"] * 10, "question": "synthetic?"} for i in range(2)]
        audit = T.P.quantization_audit(language)
        audit["visual_parameters_unchanged"] = True
        runtime = HFVQARegionRuntime(model, SimpleNamespace(tokenizer=tokenizer), language,
                    {"dev": rows}, "qwen2-vl-2b", target_task, "cpu", quantization=audit)
        runtime.max_new_tokens = 2
        for index, row in enumerate(runtime.rows["dev"]):
            ids = torch.tensor([[1, 10 + index, 12]])
            runtime._prompt_cache[row["sid"]] = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        condition = next(c for c in source["conditions"] if c["model"] == "qwen2-vl-2b" and c["format"] == fmt)
        fingerprint = T.R.model_checksum(language)
        original_bytes = {}
        for source_condition in source["conditions"]:
            for item in T.V.planned_items(source_condition, source):
                identity = {"schema_version": 2, "scope": T.V.SCOPE, "plan_sha256": source["sha256"],
                    "condition": source_condition["id"], "region": item["region"], "repeat": item["repeat"],
                    "seed": item["seed"], "arm": item["arm"], "campaign_scope": T.B.CAMPAIGN_SCOPE,
                    "campaign_variant": "baseline", "campaign_policy_sha256": T.M.object_sha256(source["campaign_policy"])}
                artifact = {**identity, "error": "registered negative/error retained"}
                chosen = (source_condition["id"] == condition["id"] and item["region"] == condition["regions"][0]
                          and item["repeat"] == 0)
                if chosen:
                    region, dims = item["region"], condition["dims"]
                    row = T.R.random_row(region, dims, item["seed"]) if item["arm"] == "random" else 2 * TINY_GROUP
                    spec = T.P.validate_spec({}, dims) if item["arm"] == "noperm" else T.R.placement(region, dims, row)
                    T.P.apply_permutation(runtime, spec)
                    try:
                        flips = next([[region["tensor"], element, 0]] for element in range(16)
                                     if T.R.check_region_attack(runtime, region, [[region["tensor"], element, 0]], 1)["valid"])
                    finally:
                        T.P.apply_permutation(runtime, spec, inverse=True)
                    artifact = {**identity, "model": condition["model"], "format": fmt, "task": source_task,
                        "search_sample": T.V._bootstrap_record(condition, item["repeat"]),
                        "quantization": audit, "quantization_schema_version": 2,
                        "permutation": T.P.spec_to_json(spec), "attacks": {"1": flips},
                        "calls": {"eval": 1, "grad": 0}, "spent": 1}
                artifact_path = source_run / "artifacts" / source_condition["id"] / item["name"]
                self.write(artifact_path, artifact)
                self.write(source_run / "eval" / source_condition["id"] / item["name"],
                           {**identity, "artifact_file_sha256": T.M.sha256(artifact_path),
                            "status": "complete" if chosen else "artifact_error"})
                if chosen:
                    original_bytes[item["name"]] = artifact_path.read_bytes()
        self.assertEqual(T.R.model_checksum(language), fingerprint)
        selection = T.seal(run)
        self.assertEqual(len(selection["records"]), 216)
        return SimpleNamespace(run=run, source_run=source_run, target_run=target_run, source=source,
            target=target, runtime=runtime, condition=condition, fingerprint=fingerprint,
            original_bytes=original_bytes)

    def execute(self, f, mutate_during_load=False, mutate_parent_during_load=False):
        real_metrics, real_check = T.V._task_metrics, T.V._artifact_reasons
        tasks, checks = [], []
        def metrics(runtime, rows):
            tasks.append(runtime.task)
            return real_metrics(runtime, rows)
        def check(runtime, artifact, condition, item, plan):
            checks.append((artifact.get("task"), condition["task"]))
            return real_check(runtime, artifact, condition, item, plan)
        def load(run, plan, condition, split, device):
            self.assertEqual((run, plan["sha256"], condition["task"], split, device),
                (f.target_run, f.target["sha256"], T.OPPOSITE[f.condition["task"]], "dev", "cpu"))
            if mutate_during_load:
                name = next(name for name in f.original_bytes if name.endswith("__noperm.json"))
                path = f.source_run / "artifacts" / f.condition["id"] / name
                path.write_bytes(path.read_bytes() + b"\n")
            if mutate_parent_during_load:
                path = f.target_run / "plan.json"
                path.write_bytes(path.read_bytes() + b"\n")
            return f.runtime, f.runtime.rows["dev"]
        with ExitStack() as stack:
            stack.enter_context(mock.patch("sys.stdout", new_callable=io.StringIO))
            configuration = stack.enter_context(mock.patch.object(T.B, "configure_from_plan", return_value=f.source))
            stack.enter_context(mock.patch.object(T.V, "load_plan", return_value=(f.target_run, f.target)))
            stack.enter_context(mock.patch.object(T.V, "_require_preflight"))
            stack.enter_context(mock.patch.object(T.V, "_load_runtime", side_effect=load))
            stack.enter_context(mock.patch.object(T.V, "_task_metrics", side_effect=metrics))
            stack.enter_context(mock.patch.object(T.V, "_artifact_reasons", side_effect=check))
            gradients = stack.enter_context(mock.patch.object(f.runtime, "grads", side_effect=AssertionError("no gradients")))
            search = stack.enter_context(mock.patch.object(A, "search", side_effect=AssertionError("no search")))
            stack.enter_context(mock.patch.object(T.V, "_search_module", side_effect=AssertionError("no search loading")))
            T.run_condition(f.run, f.condition["id"], device="cpu", split="dev")
            configuration.assert_called_once_with(str(f.source_run))
            gradients.assert_not_called()
            search.assert_not_called()
        return tasks, checks

    def test_real_vqa_integer_runtime_both_directions_preserves_source_and_restores(self):
        for fmt in ("int8", "int4"):
            for task in ("textvqa", "okvqa"):
                with self.subTest(format=fmt, source_task=task):
                    # Each direction must have an independent on-disk campaign.
                    subroot = self.root / f"{fmt}_{task}"
                    subroot.mkdir()
                    original_root, self.root = self.root, subroot
                    try:
                        f = self.fixture(fmt, task)
                        tasks, checks = self.execute(f)
                        self.assertEqual(set(tasks), {T.OPPOSITE[task]})
                        self.assertEqual(checks[0], (task, task))
                        self.assertEqual(checks[1], (T.OPPOSITE[task], T.OPPOSITE[task]))
                        for name, serialized in f.original_bytes.items():
                            artifact = f.source_run / "artifacts" / f.condition["id"] / name
                            self.assertEqual(artifact.read_bytes(), serialized)
                            report = T.M.read_json(f.run / "eval" / f.condition["id"] / name)
                            self.assertEqual(report["status"], "complete")
                            self.assertTrue(report["rearranged"]["gate"])
                            self.assertTrue(report["results"]["1"]["valid"])
                            self.assertEqual(report["results"]["1"]["damage_baseline"], "rearranged_unfaulted")
                            self.assertEqual(report["artifact_file_sha256"], T.M.sha256(artifact))
                            self.assertEqual(report["artifact_object_sha256"], T.M.object_sha256(json.loads(serialized)))
                            self.assertNotEqual(report["artifact_object_sha256"], report["scoring_view_object_sha256"])
                            self.assertEqual((report["source_plan_sha256"], report["target_plan_sha256"]),
                                             (f.source["sha256"], f.target["sha256"]))
                            self.assertEqual((report["scope"], report["split"], report["development_only"]), (T.SCOPE, "dev", True))
                            self.assertNotIn("_logits", json.dumps(report, allow_nan=False))
                            result = report["results"]["1"]
                            self.assertAlmostEqual(result["delta_nll"], result["metrics"]["nll"] - report["rearranged"]["metrics"]["nll"])
                        self.assertEqual(T.R.model_checksum(f.runtime.lm), f.fingerprint)
                        self.assertIsNone(f.runtime._visual_hook)
                    finally:
                        self.root = original_root

    def test_prepare_rejects_checkpoint_geometry_protocol_and_task_mismatches(self):
        _, left = self.parent("source", "textvqa")
        _, right = self.parent("target", "okvqa")
        for change in ("checkpoint", "geometry", "execution"):
            changed = copy.deepcopy(right)
            if change == "checkpoint":
                changed["models"][T.V.DEV_MODELS[0]]["revision"] = "different"
            elif change == "geometry":
                changed["conditions"][0]["regions"][0]["page_index"] = "different"
            else:
                changed["execution"]["min_free_gpu_gib"] = 40
            with self.subTest(change=change), self.assertRaises(ValueError):
                T._conditions(left, "textvqa", changed, "okvqa")
        with self.assertRaises(ValueError):
            T._conditions(left, "textvqa", right, "textvqa")

    def test_manifest_embedded_digest_uses_pre_digest_body_and_detects_bad_bindings(self):
        _, source = self.parent("source", "textvqa")
        _, target = self.parent("target", "okvqa")
        self.assertEqual(len(T._conditions(source, "textvqa", target, "okvqa")), 2)
        for plan in (source, target):
            for condition in plan["conditions"]:
                manifest = plan["models"][condition["model"]]
                self.assertNotEqual(manifest["manifest_sha256"], T.M.object_sha256(manifest))
                condition["checkpoint_manifest_sha256"] = T.M.object_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            T._conditions(source, "textvqa", target, "okvqa")

    def test_seal_is_all_items_without_filtering_errors_and_selection_drift_fails(self):
        f = self.fixture()
        selection = T.M.read_json(f.run / "selection.json")
        self.assertEqual(len(selection["records"]), 216)
        self.assertEqual(sum(row["development_status"] == "artifact_error" for row in selection["records"]), 213)
        with self.assertRaises(FileExistsError):
            T.seal(f.run)
        selection["records"].pop()
        self.write(f.run / "selection.json", selection)
        with self.assertRaisesRegex(RuntimeError, "selection"):
            T.verify_selection(f.run, T.M.read_json(f.run / "plan.json"), f.source)
        f.runtime.close()

    def test_source_task_rejection_cannot_be_laundered_by_target_scoring_view(self):
        f = self.fixture()
        # Freeze the rejected source into a new selection instead of editing a seal.
        (f.run / "selection.json").unlink()
        for path in (f.source_run / "artifacts" / f.condition["id"]).glob("*__r0__*.json"):
            if path.name not in f.original_bytes:
                continue
            value = T.M.read_json(path)
            value["task"] = "okvqa"
            self.write(path, value)
            development = f.source_run / "eval" / f.condition["id"] / path.name
            report = T.M.read_json(development)
            report["artifact_file_sha256"] = T.M.sha256(path)
            self.write(development, report)
        T.seal(f.run)
        _, checks = self.execute(f)
        self.assertEqual(checks[0], ("okvqa", "textvqa"))
        reports = list((f.run / "eval" / f.condition["id"]).glob("*.json"))
        self.assertEqual(len(reports), 3)
        for path in reports:
            report = T.M.read_json(path)
            self.assertEqual(report["status"], "artifact_error")
            self.assertIn("artifact task differs from plan", report["artifact_error"])
            self.assertEqual(report["results"], {})

    def test_source_drift_during_loading_publishes_no_transfer(self):
        f = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "identity mismatch|sealed source evidence"):
            self.execute(f, mutate_during_load=True)
        self.assertFalse((f.run / "eval").exists())
        self.assertEqual(T.R.model_checksum(f.runtime.lm), f.fingerprint)

    def test_parent_plan_drift_during_loading_is_rejected_before_scoring(self):
        f = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "parent plan changed"):
            self.execute(f, mutate_parent_during_load=True)
        self.assertFalse((f.run / "eval").exists())
        self.assertIsNone(f.runtime._visual_hook)

    def test_missing_source_item_blocks_seal_and_output_directory_is_exclusive(self):
        f = self.fixture()
        before = (f.run / "plan.json").read_bytes()
        with self.assertRaises(FileExistsError):
            T.prepare(f.source_run, "textvqa", f.target_run, "okvqa", f.run)
        self.assertEqual((f.run / "plan.json").read_bytes(), before)
        # A rejected item is still mandatory evidence for the source census.
        path = next((f.source_run / "artifacts").rglob("*__r2__random.json"))
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            T.seal(f.run)
        f.runtime.close()

    def test_only_dev_is_accepted_before_any_plan_read_or_configuration(self):
        with mock.patch.object(T, "load_plan", side_effect=AssertionError("no input read")):
            with self.assertRaisesRegex(ValueError, "only accepts"):
                T.run_condition("unused", "unused", split="test")
        with mock.patch.object(T.C, "setup_environment") as setup, mock.patch("sys.stderr", new_callable=io.StringIO):
            for argv in (["run", "--run", "unused", "--condition", "unused", "--split", "test"],
                         ["run", "--run", "unused", "--condition", "unused", "--split", "dev", "--final"]):
                with self.assertRaises(SystemExit):
                    T.main(argv)
            setup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
