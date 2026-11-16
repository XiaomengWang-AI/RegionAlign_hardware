"""Coverage and final-access tests, using synthetic inputs only."""
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from regionalign.experiments import clean_benchmarks as B


class CleanBenchmarkTests(unittest.TestCase):
    def test_plan_outside_result_roots_is_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(B.M, "read_json") as read:
            with self.assertRaises(ValueError):
                B.load_plan(Path(directory))
            read.assert_not_called()

    def test_absolute_dataset_hashes_are_verified_and_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.arrow"
            path.write_bytes(b"original")
            records = [B.M.file_record(path)]
            B.verify_dataset_files(records)
            path.write_bytes(b"modified")
            with self.assertRaises(RuntimeError):
                B.verify_dataset_files(records)

    def test_sliding_windows_cover_each_target_once(self):
        for length in (2, 255, 256, 511, 512, 513, 767, 768, 769, 1500):
            targets = []
            for start, end, first in B.windows(length):
                self.assertLessEqual(end-start, 512)
                self.assertGreater(first, start)
                targets.extend(range(first, end))
            self.assertEqual(targets, list(range(1, length)))

    def test_final_access_is_checked_before_plan_or_data(self):
        for split, final in (("validation", True), ("test", False)):
            with mock.patch.object(B.M, "read_json") as read:
                with self.assertRaises(ValueError):
                    B.run_condition(SimpleNamespace(split=split, final=final))
                read.assert_not_called()

    def test_uniform_language_model_has_exact_token_weighted_nll(self):
        import torch
        class Tokenizer:
            def encode(self, text, add_special_tokens):
                self_flag = add_special_tokens
                assert self_flag is False
                return [0, 1, 2, 3] * 200 + [1]
        class Model:
            def __call__(self, input_ids, use_cache):
                assert use_cache is False
                return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], 4))
        records = []
        result = B.score_wikitext(Model(), Tokenizer(), "synthetic", "cpu", records.append)
        self.assertEqual(result["target_tokens"], 800)
        self.assertEqual(sum(x["target_tokens"] for x in records), 800)
        self.assertAlmostEqual(result["nll"], math.log(4), places=6)
        self.assertAlmostEqual(result["ppl"], 4, places=5)

    def test_nonfinite_losses_are_not_published(self):
        import torch
        tokenizer = SimpleNamespace(encode=lambda *a, **k: [1, 0, 1])
        model = lambda **k: SimpleNamespace(logits=torch.full((1, 3, 2), float("nan")))
        records = []
        with self.assertRaises(RuntimeError):
            B.score_wikitext(model, tokenizer, "synthetic", "cpu", records.append)
        self.assertEqual(records[0]["status"], "nonfinite")
        self.assertIsNone(records[0]["nll_sum"])

    def test_answer_reference_is_deterministic(self):
        B.C.setup_environment()
        self.assertEqual(B.reference_answer(["A cat", "cat", "dog"], "okvqa"), "cat")
        self.assertEqual(B.reference_answer(["dog", "cat"], "okvqa"), "cat")
        with self.assertRaises(ValueError):
            B.reference_answer([], "okvqa")

    def test_empty_normalization_does_not_reject_valid_answers(self):
        B.C.setup_environment()
        for task in ("okvqa", "textvqa"):
            self.assertEqual(B.reference_answer(["cat", "cat", "a"], task), "cat")
            self.assertEqual(B.reference_answer(["a"] * 10, task), "a")
            self.assertEqual(B.reference_answer(["the", "a"], task), "a")
            self.assertEqual(B.reference_answer(["", " "], task), "")

    def test_blank_references_still_score_generation_without_answer_nll(self):
        import torch
        from transformers import BatchFeature
        B.C.setup_environment()
        inputs = BatchFeature({"input_ids": torch.tensor([[1, 2]]),
                               "attention_mask": torch.ones(1, 2, dtype=torch.long)})
        tokenizer = SimpleNamespace(encode=lambda *a, **k: [], decode=lambda *a, **k: "cat")
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=0),
            generate=mock.Mock(return_value=SimpleNamespace(
                sequences=torch.tensor([[1, 2, 3, 0]]), logits=(torch.zeros(1, 4),))))
        with mock.patch.object(B, "vlm_inputs", return_value=inputs):
            result = B.score_vqa(model, SimpleNamespace(tokenizer=tokenizer), "llava-1.5-7b",
                                 None, "synthetic", [""] * 10, "textvqa", "cpu")
        model.generate.assert_called_once()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["prediction"], "cat")
        self.assertEqual(result["target_tokens"], 0)
        self.assertFalse(result["nll_available"])
        self.assertEqual(result["nll_unavailable_reason"], "empty_reference")

    def test_final_clean_matrix_stays_on_the_paper_models_when_the_registry_grows(self):
        expected = set(B.M.PAPER_MODELS) | set(B.VLM_TYPES)
        self.assertLess(set(B.M.PAPER_MODELS), set(B.M.MODEL_SPECS))
        audit = {"models": {key: {"path": f"/checkpoints/{key}"} for key in expected},
                 "datasets": {"wikitext-2-raw": {"files": ["/data/wikitext-test.arrow"]},
                              "ok-vqa": {"files": ["/data/ok-vqa-val2014-0.arrow"]},
                              "text-vqa": {"files": ["/data/textvqa-validation-0.arrow"]}}}

        def read_json(path):
            path = Path(path)
            if path.name == "audit.json":
                return audit
            return {"model_type": B.VLM_TYPES[path.parent.name]}

        with mock.patch.object(B.M, "read_json", side_effect=read_json), \
                mock.patch.object(B.M, "checkpoint_manifest",
                                  side_effect=lambda path, key: {"model_key": key}), \
                mock.patch.object(B.M, "checkpoint_files", return_value=[]), \
                mock.patch.object(B.M, "file_record",
                                  side_effect=lambda path, *root: {"path": str(path)}), \
                mock.patch.object(B, "freeze_plan") as freeze:
            B.prepare(SimpleNamespace(cache_audit="/audit/audit.json", output="unused"))
        _, models, _, conditions = freeze.call_args.args
        self.assertEqual(set(models), expected)
        self.assertEqual({row["model"] for row in conditions}, expected)

    def test_retry_preserves_parent_and_excludes_completed_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            (parent / "source").mkdir()
            sources = []
            for name in B.SOURCES:
                path = parent / "source" / name.replace("/", "__")
                path.write_text("synthetic source")
                sources.append({**B.M.file_record(path), "path": name})
            complete = {"model": "llama2-7b", "format": "int8", "task": "wikitext"}
            failed = {"model": "llava-1.5-7b", "format": "int8", "task": "okvqa"}
            plan = {"scope": "clean_benchmarks_v1", "no_faults_or_search": True,
                    "conditions": [complete, failed], "settings": B.SETTINGS,
                    "sources": sources, "models": {"llama2-7b": {}, "llava-1.5-7b": {}},
                    "datasets": {"wikitext": {}, "okvqa": {}}}
            B.M.write_json_once(parent / "plan.json", plan)
            original_sha = B.M.sha256(parent / "plan.json")
            B.M.write_json_once(parent / "test_selection.json", {
                "plan_sha256": original_sha, "conditions": plan["conditions"]})
            (parent / "test_access.jsonl").write_text('{}\n')
            for condition, filename, value in (
                (complete, "metrics.json", {"status": "complete"}),
                (failed, "failure.json", {"type": "ValueError", "error": "empty VQA references"}),
            ):
                target = parent / "conditions" / condition["model"] / condition["format"] / condition["task"]
                target.mkdir(parents=True)
                B.M.write_json_once(target / filename, value)
            args = SimpleNamespace(parent_run=str(parent), output=str(parent / "retry"))
            with mock.patch.object(B, "freeze_plan") as freeze:
                B.prepare_retry(args)
                self.assertEqual(freeze.call_args.args[3], [failed])
                metadata = freeze.call_args.kwargs["retry"]
                self.assertTrue(metadata["prior_final_exposure"])
                self.assertEqual(len(metadata["reused_completed_metrics"]), 1)
                self.assertEqual(B.M.sha256(parent / "plan.json"), original_sha)
                (parent / "source" / B.SOURCES[0].replace("/", "__")).write_text("changed")
                with self.assertRaisesRegex(ValueError, "snapshot changed"):
                    B.prepare_retry(args)


if __name__ == "__main__":
    unittest.main()
