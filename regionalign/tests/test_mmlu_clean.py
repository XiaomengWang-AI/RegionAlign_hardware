import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from regionalign.experiments import clean_mmlu as M
from regionalign.reporting import mmlu as S


def row(subject="abstract_algebra", answer=1, question="What is x?"):
    return {"question": question, "subject": subject,
            "choices": ["zero", "one", "two", "three"], "answer": answer}


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False):
        # One deterministic token per whitespace-delimited item plus BOS.
        return {"input_ids": ([99] if add_special_tokens else []) + list(range(len(text.split())))}

    def encode(self, text, add_special_tokens=False):
        mapping = {"Answer:": [7], "Answer: A": [7, 10], "Answer: B": [7, 11],
                   "Answer: C": [7, 12], "Answer: D": [7, 13]}
        return mapping[text]


class SuffixSpecialTokenizer(FakeTokenizer):
    def __call__(self, text, add_special_tokens=True, truncation=False):
        ids = list(range(len(text.split())))
        return {"input_ids": ([99] + ids + [98]) if add_special_tokens else ids}


class MMLUCleanTests(unittest.TestCase):
    def test_module_entrypoint_dispatches_without_loading_inputs(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "regionalign.experiments.clean_mmlu", "--help"],
            cwd=M.C.PROJECT, capture_output=True, text=True, check=True,
        )
        self.assertIn("usage:", result.stdout)
        self.assertIn("{plan,run,summarize}", result.stdout)

    def test_canonical_prompt_has_five_labeled_demonstrations_and_open_target(self):
        examples = [row(answer=i % 4, question=f"demo {i}") for i in range(5)]
        prompt = M.make_prompt("abstract_algebra", examples, row(question="target"))
        self.assertTrue(prompt.startswith(
            "The following are multiple choice questions (with answers) about abstract algebra."))
        self.assertEqual(prompt.count("Answer:"), 6)
        self.assertTrue(prompt.endswith("D. three\nAnswer:"))
        self.assertIn("demo 0\nA. zero", prompt)

    def test_prompt_reduces_shots_without_truncating_target(self):
        tokenizer = FakeTokenizer()
        examples = [row(question="long demonstration words " * 5) for _ in range(5)]
        target = row(question="unique target marker")
        full = M.make_prompt(target["subject"], examples, target, 5)
        zero = M.make_prompt(target["subject"], examples, target, 0)
        max_length = (len(tokenizer(full)["input_ids"]) + len(tokenizer(zero)["input_ids"])) // 2
        ids, shots = M.prompt_ids(tokenizer, target["subject"], examples, target, max_length)
        self.assertLess(shots, 5)
        self.assertGreater(shots, 0)
        self.assertLessEqual(len(ids), max_length)

    def test_choice_labels_are_unique_single_tokens(self):
        ids, details = M.choice_token_ids(FakeTokenizer())
        self.assertEqual(ids, [10, 11, 12, 13])
        self.assertEqual(details["labels"]["C"], [7, 12])

    def test_special_token_framing_rejects_a_trailing_token(self):
        with self.assertRaisesRegex(ValueError, "special tokens"):
            M.validate_special_token_framing(SuffixSpecialTokenizer(), "Question\nAnswer:")

    def test_metrics_keep_nonfinite_as_wrong_and_report_subject_macro(self):
        predictions = [
            {"index": 0, "subject": "a", "gold": 0, "prediction": 0, "correct": True,
             "nonfinite": False, "shots": 5, "prompt_tokens": 20},
            {"index": 1, "subject": "a", "gold": 1, "prediction": -1, "correct": False,
             "nonfinite": True, "shots": 4, "prompt_tokens": 21},
            {"index": 2, "subject": "b", "gold": 2, "prediction": 3, "correct": False,
             "nonfinite": False, "shots": 5, "prompt_tokens": 22},
        ]
        metrics = M.metrics_from_predictions(predictions)
        self.assertEqual(metrics["accuracy"], 1 / 3)
        self.assertEqual(metrics["subject_macro_accuracy"], 0.25)
        self.assertEqual(metrics["nonfinite"], 1)
        self.assertEqual(metrics["reduced_shot_prompts"], 1)
        self.assertEqual(metrics["max_prompt_tokens"], 22)

    def test_arrow_reader_checks_schema_and_values(self):
        # This host has an old system numexpr extension next to NumPy 2.  PyArrow
        # does not need numexpr, so keep that optional pandas probe out of this test.
        with mock.patch.dict(sys.modules, {"numexpr": None}):
            import pyarrow as pa
            import pyarrow.ipc as ipc
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "mmlu.arrow"
                table = pa.Table.from_pylist([row()])
                with pa.OSFile(str(path), "wb") as sink:
                    with ipc.new_stream(sink, table.schema) as writer:
                        writer.write_table(table)
                loaded = M.read_arrow(path)
        self.assertEqual(loaded, [row()])

    def test_checkpoint_files_use_only_indexed_safetensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({"architectures": ["Tiny"]}))
            (root / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}))
            (root / "model-1.safetensors").write_bytes(b"one")
            (root / "model-2.safetensors").write_bytes(b"two")
            files = M.checkpoint_files(root)
        names = {path.name for path in files}
        self.assertEqual(names, {"config.json", "model.safetensors.index.json",
                                 "model-1.safetensors", "model-2.safetensors"})

    def test_manifest_verifies_the_logical_symlink_not_an_old_blob(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "blobs").mkdir()
            (root / "snapshot").mkdir()
            first = root / "blobs" / "first"
            second = root / "blobs" / "second"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            logical = root / "snapshot" / "weight"
            logical.symlink_to(first)
            record = M.file_record(logical, root / "snapshot")
            M.verify_records(root / "snapshot", [record])
            logical.unlink()
            logical.symlink_to(second)
            with self.assertRaisesRegex(RuntimeError, "symlink drift"):
                M.verify_records(root / "snapshot", [record])

    def test_prediction_validation_rejects_derived_field_drift(self):
        source = [row(answer=0)]
        saved = [{"index": 0, "subject": "abstract_algebra", "gold": 0,
                  "prediction": 0, "correct": True, "choice_logits": [1.0, 0.0, -1.0, -2.0],
                  "nonfinite": False, "shots": 5, "prompt_tokens": 20}]
        M.validate_predictions(saved, source)
        saved[0]["correct"] = False
        with self.assertRaisesRegex(ValueError, "derived prediction"):
            M.validate_predictions(saved, source)

    def test_prediction_validation_binds_rows_to_the_condition(self):
        source = [row(answer=0)]
        saved = [{"index": 0, "subject": "abstract_algebra", "gold": 0,
                  "prediction": 0, "correct": True, "choice_logits": [1.0, 0.0, -1.0, -2.0],
                  "nonfinite": False, "shots": 5, "prompt_tokens": 20,
                  "condition_fingerprint": "other"}]
        with self.assertRaisesRegex(ValueError, "condition fingerprint"):
            M.validate_predictions(saved, source, "expected")

    def test_nonfinite_prediction_uses_json_null_and_counts_wrong(self):
        source = [row(answer=0)]
        saved = [{"index": 0, "subject": "abstract_algebra", "gold": 0,
                  "prediction": -1, "correct": False, "choice_logits": [None, 0.0, 1.0, 2.0],
                  "nonfinite": True, "shots": 5, "prompt_tokens": 20}]
        M.validate_predictions(saved, source)
        self.assertNotIn("NaN", json.dumps(saved, allow_nan=False))

    def test_score_rows_uses_left_padding_and_only_last_logits(self):
        import torch

        class Output:
            def __init__(self, logits):
                self.logits = logits

        class Model:
            def __call__(self, input_ids, attention_mask, position_ids, use_cache,
                         num_logits_to_keep):
                self.seen = (input_ids.clone(), attention_mask.clone(), position_ids.clone())
                self.assertions = (use_cache, num_logits_to_keep)
                logits = torch.zeros((input_ids.shape[0], 1, 20))
                logits[0, 0, 10] = 5
                logits[1, 0, 11] = 5
                return Output(logits)

        model = Model()
        rows = [row(answer=0, question="short"), row(answer=1, question="a much longer question")]
        examples = {"abstract_algebra": [row() for _ in range(5)]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            predictions = M.score_rows(model, FakeTokenizer(), [10, 11, 12, 13], rows, examples,
                                       "cpu", path, max_length=1000, batch_size=2)
        input_ids, attention, positions = model.seen
        self.assertEqual(model.assertions, (False, 1))
        self.assertTrue(bool((attention[:, -1] == 1).all()))
        self.assertTrue(bool((positions[:, -1] == attention.sum(-1) - 1).all()))
        self.assertEqual([item["correct"] for item in predictions], [True, True])

    def test_partial_repair_terminates_a_complete_last_line(self):
        saved = {"index": 0, "subject": "a", "gold": 0}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "partial.jsonl"
            path.write_text(json.dumps(saved), encoding="utf-8")
            rows = M.load_partial(path, repair_trailing=True,
                                  repair_log=root / "recovery.jsonl")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"index": 1, "subject": "a", "gold": 1}) + "\n")
            reparsed = M.load_partial(path)
            events = [json.loads(line) for line in (root / "recovery.jsonl").read_text().splitlines()]
        self.assertEqual(rows, [saved])
        self.assertEqual([item["index"] for item in reparsed], [0, 1])
        self.assertEqual(events[0]["action"], "terminate_complete_tail")

    def test_partial_repair_truncates_only_an_uncommitted_bad_tail(self):
        first = json.dumps({"index": 0, "subject": "a", "gold": 0}).encode() + b"\n"
        broken = b'{"index": 1, "subject":'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "partial.jsonl"
            path.write_bytes(first + broken)
            rows = M.load_partial(path, repair_trailing=True,
                                  repair_log=root / "recovery.jsonl")
            repaired = path.read_bytes()
            event = json.loads((root / "recovery.jsonl").read_text().splitlines()[0])
        self.assertEqual([item["index"] for item in rows], [0])
        self.assertEqual(repaired, first)
        self.assertEqual(event["action"], "truncate_partial_tail")
        self.assertEqual(event["dropped_bytes"], len(broken))

    def test_partial_repair_refuses_a_committed_bad_line(self):
        payload = b'{"index": 0}\n{"index": 1,\n'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "partial.jsonl"
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "invalid partial prediction"):
                M.load_partial(path, repair_trailing=True)
            self.assertEqual(path.read_bytes(), payload)

    def test_quantized_state_manifest_detects_state_changes(self):
        import torch

        model = torch.nn.Linear(3, 2)
        M.P.quantize(model, "int8")
        first = M.quantized_state_manifest(model)
        self.assertEqual(M.verify_quantized_state_manifest(first), first["manifest_sha256"])
        model.weight.data.view(-1)[0] += 1
        second = M.quantized_state_manifest(model)
        self.assertNotEqual(first["manifest_sha256"], second["manifest_sha256"])

    def test_gpu_identity_is_bound_to_nvidia_smi_uuid_and_driver(self):
        memory_mib = 49140
        cuda_usable_bytes = 50899386368  # observed cn117/A6000 value; driver reserves ~599 MiB

        class Cuda:
            @staticmethod
            def is_available(): return True
            @staticmethod
            def current_device(): return 0
            @staticmethod
            def get_device_name(device): return "NVIDIA RTX A6000"
            @staticmethod
            def get_device_capability(device): return (8, 6)
            @staticmethod
            def get_device_properties(device):
                return SimpleNamespace(total_memory=cuda_usable_bytes,
                                       multi_processor_count=84)

        fake_torch = SimpleNamespace(
            cuda=Cuda(), device=lambda value: SimpleNamespace(index=0),
            version=SimpleNamespace(cuda="12.1"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100)),
        )
        query = SimpleNamespace(stdout=("GPU-abc, 550.54, 00000000:01:00.0, "
                                        f"NVIDIA RTX A6000, {memory_mib}\n"))
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "6"}), \
                mock.patch.object(M.subprocess, "run", return_value=query) as command:
            identity = M.runtime_device_identity(fake_torch, "cuda:0")
        self.assertEqual(identity["uuid"], "GPU-abc")
        self.assertEqual(identity["driver_version"], "550.54")
        self.assertEqual(identity["cuda_visible_selector"], "6")
        self.assertEqual(identity["total_memory"], cuda_usable_bytes)
        self.assertEqual(identity["framebuffer_memory"], memory_mib * 1024 ** 2)
        self.assertNotIn("uuid", M.runtime_device_class(identity))
        self.assertIn("-i", command.call_args.args[0])

    def test_device_attempt_tail_is_repaired_before_a_new_attempt(self):
        first = json.dumps({"attempt": 0}).encode() + b"\n"
        broken = b'{"attempt":'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "device_attempts.jsonl"
            path.write_bytes(first + broken)
            M.repair_uncommitted_jsonl_tail(path, root / "recovery.jsonl")
            M.append_jsonl(path, {"attempt": 1})
            values = M.read_jsonl_objects(path)
            event = json.loads((root / "recovery.jsonl").read_text().splitlines()[0])
        self.assertEqual(values, [{"attempt": 0}, {"attempt": 1}])
        self.assertEqual(event["action"], "truncate_uncommitted_jsonl_tail")

    def test_completion_metadata_allows_cpu_only_finalization(self):
        source = [row(answer=0)]
        fingerprint = "f" * 64
        fingerprint_value = {"plan_sha256": "p", "model_manifest_sha256": "m",
                             "dataset_sha256": "d", "model": "llama2-7b",
                             "format": "int8", "split": "validation", "protocol": {}}
        environment = {"python": "test"}
        plan = {"environment": environment, "dataset": {"hf_home": "/unused"},
                "protocol": {"max_length": 4096, "final_batch_size": 4},
                "models": {"llama2-7b": {"path": "/checkpoint"}}}
        args = SimpleNamespace(model="llama2-7b", format="int8", split="validation",
                               final=False, limit=None, batch_size=4, device="cuda:0")
        prediction = {"index": 0, "subject": "abstract_algebra", "gold": 0,
                      "prediction": 0, "correct": True,
                      "choice_logits": [2.0, 1.0, 0.0, -1.0], "nonfinite": False,
                      "shots": 5, "prompt_tokens": 20,
                      "condition_fingerprint": fingerprint}
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            execution = {"condition_fingerprint": fingerprint, "condition": fingerprint_value,
                         "model": args.model, "format": args.format, "split": args.split,
                         "final": False, "limit": None, "batch_size": 4,
                         "device": "cuda:0", "environment": environment,
                         "device_class": {"host": "gpu", "device_name": "A6000"}}
            M.write_json_once(target / "execution.json", execution)
            (target / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
            M.write_json_once(target / "predictions_manifest.json", {
                "condition_fingerprint": fingerprint, "expected_rows": 1,
                "identity_sha256": M.object_sha256([("abstract_algebra", 0)]),
            })
            M.write_json_once(target / "quantization_audit.json", {"format": "int8"})
            M.write_json_once(target / "tokenizer_audit.json", {"choice_token_ids": [1, 2, 3, 4]})
            state = {"algorithm": "test", "format": "int8", "parameters": [], "scales": []}
            state["manifest_sha256"] = M.object_sha256(state)
            M.write_json_once(target / "quantized_state.json", state)
            M.append_jsonl(target / "device_attempts.jsonl", {
                "time": "test", "pid": 1,
                "device_identity": {"uuid": "GPU-test", "device_name": "A6000"},
                "device_class": execution["device_class"],
            })
            evidence = {
                "execution_sha256": M.sha256(target / "execution.json"),
                "predictions_manifest_sha256": M.sha256(target / "predictions_manifest.json"),
                "device_attempts_sha256": M.sha256(target / "device_attempts.jsonl"),
                "quantization_audit_sha256": M.sha256(target / "quantization_audit.json"),
                "tokenizer_audit_file_sha256": M.sha256(target / "tokenizer_audit.json"),
                "quantized_state_file_sha256": M.sha256(target / "quantized_state.json"),
            }
            completion = {"condition_fingerprint": fingerprint, "count": 1,
                          "predictions_sha256": M.sha256(target / "predictions.jsonl"),
                          "quantized_state_manifest_sha256": state["manifest_sha256"],
                          "evidence_hashes": evidence,
                          "timing": {"complete_single_process_timing": True},
                          "environment": {**environment, "host": "gpu", "cuda_device": "A6000",
                                          "device_class": execution["device_class"],
                                          "device_uuids": ["GPU-test"],
                                          "device_attempt_count": 1}}
            M.write_json_once(target / "completion_metadata.json", completion)
            with mock.patch.object(M, "software_environment", return_value=environment), \
                    mock.patch.object(M, "read_arrow", side_effect=[[row() for _ in range(5)], source]), \
                    mock.patch.object(M.C, "setup_environment", side_effect=AssertionError("GPU path")), \
                    mock.patch.object(M, "load_model", side_effect=AssertionError("GPU path")):
                M._run_condition_locked(args, target, plan, target, fingerprint, fingerprint_value)
            metrics = M.read_json(target / "metrics.json")
        self.assertEqual(metrics["count"], 1)
        self.assertEqual(metrics["correct"], 1)

    def test_final_batch_must_match_the_frozen_plan(self):
        args = SimpleNamespace(split="test", final=True, limit=None, batch_size=2,
                               run="run", format="int8", model="llama2-7b", device="cuda:0")
        plan = {"protocol": {"final_batch_size": 4}, "formats": ["int8"],
                "models": {"llama2-7b": {}}}
        with mock.patch.object(M, "load_plan", return_value=(Path("/tmp/run"), plan)), \
                mock.patch.object(M, "_run_condition_locked") as execute:
            with self.assertRaisesRegex(ValueError, "batch size is frozen"):
                M.run_condition(args)
        execute.assert_not_called()

    def test_paper_matrix_is_the_three_base_models_not_the_whole_registry(self):
        self.assertEqual(M.PAPER_MODELS, ("llama2-7b", "llama3.1-8b", "qwen2.5-7b"))
        self.assertLess(set(M.PAPER_MODELS), set(M.MODEL_SPECS))
        small = [(key, Path(f"/checkpoints/{key}")) for key in ("llama3.2-1b", "qwen2.5-1.5b")]
        with mock.patch.object(M.C, "fresh_output_dir") as allocate:
            with self.assertRaisesRegex(ValueError, "frozen paper matrix"):
                M.create_plan(SimpleNamespace(model=small, formats=list(M.FORMATS), out=None))
        allocate.assert_not_called()

    def test_summary_preflight_precedes_test_access(self):
        models = {key: {} for key in M.PAPER_MODELS}
        conditions = [{"model": key, "format": fmt} for key in models for fmt in M.FORMATS]
        plan = {"models": models, "formats": list(M.FORMATS), "conditions": conditions,
                "dataset": {"hf_home": "/unused"}}
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(M, "load_plan", return_value=(Path(temporary), plan)), \
                mock.patch.object(M, "append_summary_test_access") as access, \
                mock.patch.object(M, "read_arrow") as read_test:
            with self.assertRaisesRegex(RuntimeError, "six sealed test conditions"):
                S.summarize_run(temporary)
        access.assert_not_called()
        read_test.assert_not_called()

    def test_summary_logs_before_test_hash_or_parse(self):
        models = {key: {} for key in M.PAPER_MODELS}
        conditions = [{"model": key, "format": fmt} for key in models for fmt in M.FORMATS]
        plan = {"models": models, "formats": list(M.FORMATS), "conditions": conditions,
                "dataset": {"hf_home": "/unused"}}
        required = ("metrics.json", "predictions.jsonl", "quantization_audit.json",
                    "tokenizer_audit.json", "execution.json", "predictions_manifest.json",
                    "quantized_state.json", "completion_metadata.json", "device_attempts.jsonl")
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for condition in conditions:
                root = run / "conditions" / condition["model"] / condition["format"] / "test"
                root.mkdir(parents=True)
                for name in required:
                    (root / name).touch()
            with mock.patch.object(M, "load_plan", return_value=(run, plan)), \
                    mock.patch.object(M, "verify_test_selection",
                                      side_effect=lambda *_: events.append("selection")), \
                    mock.patch.object(M, "append_summary_test_access",
                                      side_effect=lambda *_: events.append("access")), \
                    mock.patch.object(M, "verify_plan_inputs",
                                      side_effect=lambda *_: events.append("verify")), \
                    mock.patch.object(M, "read_arrow",
                                      side_effect=lambda *_: (_ for _ in ()).throw(RuntimeError("stop"))):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    S.summarize_run(run)
        self.assertEqual(events[:2], ["selection", "access"])
        self.assertEqual(events[-1], "verify")

    def test_write_once_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.json"
            M.write_json_once(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                M.write_json_once(path, {"a": 2})


if __name__ == "__main__":
    unittest.main()
