"""Different tiny checkpoints exercise identity transfer using the real evaluator."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from regionalign import config as C
from regionalign.experiments import checkpoint_transfer as T
from regionalign.algorithm import search as A
from regionalign.tests.helpers import StubTokenizer


class CheckpointTransferTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        patcher = mock.patch.object(C, "RESULTS_ROOT", self.root)
        patcher.start(); self.addCleanup(patcher.stop)
        self.addCleanup(T.B._reset_configuration)
        torch.set_num_threads(1)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))

    def checkpoint(self, name, seed):
        torch.manual_seed(seed)
        config = LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=256,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, tie_word_embeddings=False)
        model = LlamaForCausalLM(config).to(torch.bfloat16).eval().requires_grad_(False)
        path = self.root / name
        model.save_pretrained(path, safe_serialization=True)
        (path / "tokenizer.model").write_bytes(b"identical synthetic vocabulary")
        (path / "tokenizer.json").write_bytes(b"identical synthetic vocabulary JSON")
        self.write(path / "tokenizer_config.json", {"chat_template": name})
        return path

    def fixture(self, fmt="int8", task="wikitext", target_illegal=False):
        base = self.checkpoint("base_checkpoint", 101)
        target = self.checkpoint("chat_checkpoint", 202)
        original = self.root / "original"
        historical = original / "source" / "history.py"
        historical.parent.mkdir(parents=True); historical.write_text("# historical\n")
        config = T.M.read_json(base / "config.json")
        manifest = {"path": str(base), "model_key": T.SOURCE_MODEL,
            "files": [T.M.file_record(p, base) for p in T.M.checkpoint_files(base)]}
        manifest["manifest_sha256"] = T.M.object_sha256(manifest)
        conditions = []
        for format_ in T.L.FORMATS:
            dims = T.L.dims_for_format(config, format_)
            for task_ in T.L.TASKS:
                cid = f"llama2-7b__{format_}__{task_}"
                panels = {}
                for split in ("search", "dev"):
                    rows = [{"sid": f"{split}-{i}", "prompt_ids": [1, 5 + i, 8],
                             "answer_ids": [10, 11 + i], "target_tokens": 2,
                             "choice_token_ids": [10, 11, 12, 13], "answer": i}
                            for i in range(2)]
                    path = original / "panels" / cid / f"{split}.json"
                    self.write(path, {"scope": T.L.SCOPE, "model": T.SOURCE_MODEL,
                        "task": task_, "split": split, "development_only": True, "rows": rows})
                    panels[split] = {"path": str(path.relative_to(original)), "count": 2,
                                     "sha256": T.M.sha256(path)}
                regions = T.L.preregistered_regions(T.SOURCE_MODEL, dims, int(format_[3:]))
                # Production LLM conditions have no availability key. Keep that
                # real schema here rather than borrowing VQA's availability field.
                conditions.append({"id": cid, "model": T.SOURCE_MODEL, "format": format_,
                    "task": task_, "dims": dims, "regions": regions, "panels": panels,
                    "checkpoint_manifest_sha256": manifest["manifest_sha256"],
                    "bootstrap_seeds": {T.L._bootstrap_key(r["id"], j): 100 * i + j
                        for i, r in enumerate(regions) for j in range(3)}})
        parent = {"schema_version": 2, "scope": T.L.SCOPE, "development_only": True,
            "splits": ["search", "dev"], "models": {T.SOURCE_MODEL: manifest},
            "conditions": conditions, "datasets": {}, "search_budget": T.R.BUDGET,
            "arms": list(T.L.ARMS), "repeat_seeds": list(T.L.REPEAT_SEEDS),
            "work_items_per_condition": 108, "search_k": list(T.R.K_SCHEDULE),
            "evaluation_k": list(T.L.EVAL_K), "clean_gate_nll": T.R.GATE_NLL,
            "mmlu_score_gate": T.L.MMLU_SCORE_GATE,
            "execution_protocol": T.L._execution_protocol(1, 1, 1, 40),
            "sources": [{"path": "historical_missing_live.py", "snapshot": "source/history.py",
                         "sha256": T.M.sha256(historical)}]}
        self.write(original / "plan.json", parent)
        self.write(original / "status.json", {"plan_sha256": T.M.sha256(original / "plan.json")})
        source_run = T.CM.prepare(original, self.root / "campaign", min_free_gpu_gib=22)
        source_run, source = T._parent(source_run)
        run = T.prepare(source_run, target, self.root / "transfer")
        source_model = LlamaForCausalLM.from_pretrained(base).eval().requires_grad_(False)
        T.P.quantize(source_model, fmt, group=T.P.GROUP)
        condition = T.L._condition(source, f"llama2-7b__{fmt}__{task}")
        rows = T.L._panel_rows(source_run, condition, "dev")
        runtime = T.FormalTextRuntime(source_model, StubTokenizer(), {"dev": rows}, "cpu")
        runtime.task = task
        target_runtime = None
        if target_illegal:
            target_lm = LlamaForCausalLM.from_pretrained(target).eval().requires_grad_(False)
            T.P.quantize(target_lm, fmt, group=T.P.GROUP)
            target_runtime = T.FormalTextRuntime(target_lm, StubTokenizer(), {"dev": rows}, "cpu")
        source_before = T.R.model_checksum(source_model)
        group_region = next(r for r in condition["regions"] if r["kind"] == "gate_proj")
        group_index = condition["regions"].index(group_region) * 3
        chosen_names = []
        for c in source["conditions"]:
            for item in T.L.planned_items(c, source):
                identity = {"schema_version": 2, "scope": T.L.SCOPE, "plan_sha256": source["sha256"],
                    "condition": c["id"], "region": item["region"], "repeat": item["repeat"],
                    "seed": item["seed"], "arm": item["arm"], "campaign_scope": T.B.CAMPAIGN_SCOPE,
                    "campaign_variant": "baseline", "campaign_policy_sha256": T.M.object_sha256(source["campaign_policy"])}
                artifact = {**identity, "error": "source failure retained"}
                chosen = c["id"] == condition["id"] and item["region"] == group_region and item["repeat"] == 0
                if chosen:
                    row = T.R.random_row(group_region, c["dims"], item["seed"]) if item["arm"] == "random" else 2
                    spec = T.P.validate_spec({}, c["dims"]) if item["arm"] == "noperm" else T.R.placement(group_region, c["dims"], row)
                    T.P.apply_permutation(runtime, spec)
                    if target_runtime is not None:
                        T.P.apply_permutation(target_runtime, spec)
                    try:
                        start = group_region["page"] * T.R.elements_per_page(group_region["bits"])
                        bit = group_region["bits"] - 1 if target_illegal else 0
                        flips = next([[group_region["tensor"], i, bit]] for i in range(start, start + 4096)
                            if T.R.check_region_attack(runtime, group_region, [[group_region["tensor"], i, bit]], 1)["valid"]
                            and (target_runtime is None or not T.R.check_region_attack(target_runtime,
                                group_region, [[group_region["tensor"], i, bit]], 1)["valid"]))
                    finally:
                        T.P.apply_permutation(runtime, spec, inverse=True)
                        if target_runtime is not None:
                            T.P.apply_permutation(target_runtime, spec, inverse=True)
                    sampled = ["search-0", "search-1"]
                    artifact = {**identity, "model": T.SOURCE_MODEL, "format": fmt, "task": task,
                        "bootstrap_seed": item["bootstrap_seed"], "execution_protocol": source["execution_protocol"],
                        "quantization": T.P.quantization_audit(source_model), "quantization_schema_version": 2,
                        "bootstrap": {"method": "bootstrap_with_replacement", "seed": item["bootstrap_seed"],
                            "population_rows": 2, "draws": 2, "sampled_sids": sampled,
                            "sampled_sids_sha256": T.M.object_sha256(sampled)}, "search_rows": 2,
                        "permutation": T.P.spec_to_json(spec), "attacks": {"1": flips},
                        "calls": {"eval": 1, "grad": 0}, "spent": 1}
                    chosen_names.append(item["name"])
                path = source_run / "artifacts" / c["id"] / item["name"]
                self.write(path, artifact)
                self.write(source_run / "eval" / c["id"] / item["name"], {**identity,
                    "artifact_file_sha256": T.M.sha256(path), "status": "complete" if chosen else "artifact_error",
                    "summary": {"success_any": False}})
        self.assertEqual(T.R.model_checksum(source_model), source_before)
        T.seal(run)
        return SimpleNamespace(run=run, source_run=source_run, source=source, target=target,
            condition=condition, shard=f"{group_index}/36", names=chosen_names,
            source_runtime=runtime, source_before=source_before)

    def execute(self, fixture, shard=None):
        loaded = []
        def load(checkpoint, fmt, device, min_free_gpu_bytes):
            self.assertEqual(checkpoint, str(fixture.target))
            self.assertEqual(min_free_gpu_bytes, 22 * 1024 ** 3)
            model = LlamaForCausalLM.from_pretrained(checkpoint).eval().requires_grad_(False)
            T.P.quantize(model, fmt, group=T.P.GROUP)
            loaded.append(model)
            return model, StubTokenizer(), T.P.quantization_audit(model), {"mock_load": True}
        with mock.patch.object(T.M, "load_model", side_effect=load), \
             mock.patch.object(T, "target_manifest", wraps=T.target_manifest) as manifests, \
             mock.patch.object(A, "search", side_effect=AssertionError("no search")) as search, \
             mock.patch.object(T.P.BaseRuntime, "grads", side_effect=AssertionError("no gradients")) as gradients, \
             mock.patch("sys.stdout", new_callable=io.StringIO):
            T.run_condition(fixture.run, fixture.condition["id"], shard or fixture.shard, "cpu", "dev")
        search.assert_not_called(); gradients.assert_not_called()
        self.assertEqual(manifests.call_count, 2)  # whole checkpoint before/after load, never per item
        return loaded[0]

    def test_real_different_checkpoints_full_integer_runtime_identity_and_restoration(self):
        for fmt in ("int8", "int4"):
            for task in ("mmlu", "wikitext"):
                with self.subTest(format=fmt, task=task):
                    original = self.root; self.root = original / f"{fmt}_{task}"; self.root.mkdir()
                    try:
                        f = self.fixture(fmt, task)
                        snapshots = {p: p.read_bytes() for p in f.source_run.rglob("*.json")}
                        target_model = self.execute(f)
                        self.assertNotEqual(T.R.model_checksum(target_model), f.source_before)
                        for name in f.names:
                            report = T.M.read_json(f.run / "eval" / f.condition["id"] / name)
                            self.assertEqual(report["scope"], T.SCOPE)
                            self.assertEqual(report["status"], "complete")
                            self.assertTrue(report["rearranged"]["gate"])
                            self.assertTrue(report["results"]["1"]["valid"])
                            self.assertEqual(report["results"]["1"]["damage_baseline"], "same_rearranged_unfaulted_layout")
                            self.assertAlmostEqual(report["results"]["1"]["delta_nll"],
                                report["results"]["1"]["metrics"]["nll"] - report["rearranged"]["metrics"]["nll"])
                            self.assertFalse(report["summary"]["success_any"])
                            self.assertEqual(report["target_runtime_checksum_before"], report["target_runtime_checksum_after"])
                            self.assertEqual(report["artifact_object_sha256"], T.M.object_sha256(T.M.read_json(
                                f.source_run / "artifacts" / f.condition["id"] / name)))
                            self.assertNotIn("source_runtime_checksum", report)
                            self.assertNotIn('"_logits"', json.dumps(report, allow_nan=False))
                        for path, data in snapshots.items(): self.assertEqual(path.read_bytes(), data)
                        self.assertEqual(T.R.model_checksum(f.source_runtime.lm), f.source_before)
                    finally:
                        T.B._reset_configuration(); self.root = original

    def test_source_errors_are_not_laundered_and_all_432_are_sealed(self):
        f = self.fixture()
        selection = T.M.read_json(f.run / "selection.json")
        self.assertEqual(len(selection["records"]), 432)
        self.assertEqual(sum(r["development_status"] == "artifact_error" for r in selection["records"]), 429)
        self.execute(f, shard="0/36")
        reports = list((f.run / "eval" / f.condition["id"]).glob("*.json"))
        self.assertEqual(len(reports), 3)
        for path in reports:
            report = T.M.read_json(path)
            self.assertEqual(report["status"], "artifact_error")
            self.assertIn("source failure retained", report["artifact_error"])
            self.assertEqual(report["results"], {})

    def test_source_legal_coordinate_can_be_illegal_on_different_target_codes(self):
        f = self.fixture(target_illegal=True)
        self.execute(f)
        for name in f.names:
            report = T.M.read_json(f.run / "eval" / f.condition["id"] / name)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["source_artifact_reasons"], [])
            self.assertTrue(report["rearranged"]["gate"])
            self.assertFalse(report["results"]["1"]["valid"])
            self.assertFalse(report["summary"]["success_any"])
            self.assertTrue(report["results"]["1"]["reasons"])

    def test_selection_changed_between_verification_and_read_is_rejected(self):
        f = self.fixture()
        original = T.verify_selection
        def racing(run, plan, source):
            selection = original(run, plan, source)
            modified = copy.deepcopy(selection); modified["records"].pop()
            self.write(run / "selection.json", modified)
            return selection
        with mock.patch.object(T, "verify_selection", side_effect=racing), \
             mock.patch.object(T.M, "load_model") as load, self.assertRaisesRegex(RuntimeError, "selection changed"):
            T.run_condition(f.run, f.condition["id"], f.shard, "cpu", "dev")
        load.assert_not_called()

    def test_target_geometry_tokenizer_or_identical_weights_are_rejected(self):
        f = self.fixture()
        manifest = f.source["models"][T.SOURCE_MODEL]
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            T.target_manifest(manifest["path"], manifest)
        config_path = f.target / "config.json"
        config = T.M.read_json(config_path); config["hidden_size"] += 1
        self.write(config_path, config)
        with self.assertRaisesRegex(ValueError, "architecture"):
            T.target_manifest(f.target, manifest)
        config["hidden_size"] -= 1; self.write(config_path, config)
        (f.target / "tokenizer.json").write_bytes(b"different token IDs")
        with self.assertRaisesRegex(ValueError, "tokenizer.json"):
            T.target_manifest(f.target, manifest)

    def test_embedded_manifest_digest_and_every_dimension_must_match(self):
        f = self.fixture()
        _, plan, source = T.load_plan(f.run)
        self.assertNotEqual(source["models"][T.SOURCE_MODEL]["manifest_sha256"],
                            T.M.object_sha256(source["models"][T.SOURCE_MODEL]))
        for change in ("manifest", "dimension"):
            modified = copy.deepcopy(source)
            if change == "manifest":
                modified["conditions"][0]["checkpoint_manifest_sha256"] = T.M.object_sha256(modified["models"][T.SOURCE_MODEL])
            else:
                modified["conditions"][0]["dims"]["heads"] += 1
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "manifest binding|dimensions"):
                T._conditions(modified, plan["target_checkpoint"])

    def test_seal_requires_all_errors_and_negatives_and_refuses_overwrite(self):
        f = self.fixture()
        with self.assertRaises(FileExistsError): T.seal(f.run)
        missing = next((f.source_run / "eval").rglob("*.json")); missing.unlink()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            T.verify_selection(*T.load_plan(f.run))

    def test_target_checkpoint_drift_and_source_evidence_drift_fail_before_loading(self):
        f = self.fixture()
        artifact = f.source_run / "artifacts" / f.condition["id"] / f.names[0]
        artifact.write_bytes(artifact.read_bytes() + b"\n")
        with mock.patch.object(T.M, "load_model") as load, self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            T.run_condition(f.run, f.condition["id"], f.shard, "cpu", "dev")
        load.assert_not_called()

    def test_target_byte_drift_is_rejected_before_model_loading(self):
        f = self.fixture()
        (f.target / "tokenizer.json").write_bytes(b"changed target vocabulary")
        with mock.patch.object(T.M, "load_model") as load, self.assertRaisesRegex(ValueError, "tokenizer.json"):
            T.run_condition(f.run, f.condition["id"], f.shard, "cpu", "dev")
        load.assert_not_called()

    def test_final_rejection_precedes_any_load_and_cli_has_no_final_switch(self):
        with mock.patch.object(T, "load_plan") as load:
            with self.assertRaisesRegex(ValueError, "only accepts --split dev"):
                T.run_condition("unused", "unused", split="test")
            load.assert_not_called()
        with mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            T.main(["run", "--run", "unused", "--condition", "unused", "--split", "test", "--final"])


if __name__ == "__main__": unittest.main()
