"""Synthetic CPU-only checks for independent clean GPU memory admission."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from regionalign.experiments import clean_memory as L


class CleanMemoryTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(L.B, "SETTINGS", dict(L._BASE_SETTINGS)).start()
        mock.patch.object(L.B, "SOURCES", tuple(L._BASE_SOURCES)).start()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        mock.patch.object(L.B.C, "PROJECT", self.root).start()
        mock.patch.object(L.B.C, "setup_environment").start()
        mock.patch("builtins.print").start()
        L.configure(20)
        for source in L.B.SOURCES:
            path = self.root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"synthetic source: {source}\n")
        self.conditions = [{"model": "llava-1.5-7b", "format": "int8", "task": task}
                           for task in ("okvqa", "textvqa")]

    def freeze(self, name="run", conditions=None):
        run = self.root / "regionalign" / "results" / name
        L.B.freeze_plan(run, {"llava-1.5-7b": {"path": str(self.root / "checkpoint"), "files": []}},
                        {task: {"files": [], "expected_count": 10} for task in ("okvqa", "textvqa")},
                        self.conditions if conditions is None else conditions)
        return run

    def args(self, run):
        return SimpleNamespace(run=str(run), model="llava-1.5-7b", format="int8", device="cuda:0",
                               task="okvqa", split="test", final=True)

    def test_configuration_copies_settings_and_changes_only_admission(self):
        previous = L.B.SETTINGS
        L.configure(21)
        self.assertEqual(previous["min_free_gpu_gib"], 20)
        self.assertIsNot(previous, L.B.SETTINGS)
        self.assertEqual({k: v for k, v in L.B.SETTINGS.items() if k != "min_free_gpu_gib"},
                         {k: v for k, v in L._BASE_SETTINGS.items() if k != "min_free_gpu_gib"})
        L.configure(21)
        self.assertEqual(L.B.SOURCES.count(L.SOURCE), 1)
        for value in (True, 0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                L.configure(value)

    def test_frozen_threshold_and_entry_source_are_required(self):
        run = self.freeze()
        self.assertEqual(L.load_plan(run)["settings"]["min_free_gpu_gib"], 20)
        L.configure(21)
        with self.assertRaisesRegex(ValueError, "protocol"):
            L.load_plan(run)
        L.configure(20)
        L.B.SOURCES = tuple(name for name in L.B.SOURCES if name != L.SOURCE)
        legacy = self.freeze("legacy")
        L.configure(20)
        with self.assertRaisesRegex(ValueError, "entry point"):
            L.load_plan(legacy)

    def test_prepare_preserves_parent_and_skips_existing_condition(self):
        parent = self.freeze("parent")
        original = (parent / "plan.json").read_bytes()
        occupied = parent / "conditions" / "llava-1.5-7b" / "int8" / "okvqa"
        occupied.mkdir(parents=True)
        output = self.root / "regionalign" / "results" / "new"
        L.prepare(SimpleNamespace(parent_run=parent, output=output, condition=None))
        plan = L.load_plan(output)
        self.assertEqual(plan["conditions"], [self.conditions[1]])
        self.assertEqual(plan["parent"]["plan_sha256"], L.B.M.sha256(parent / "plan.json"))
        self.assertEqual((parent / "plan.json").read_bytes(), original)
        self.assertTrue(plan["no_faults_or_search"])
        self.assertIn(L.SOURCE, {r["path"] for r in plan["sources"]})
        self.assertEqual(plan["memory_calibration"]["candidate_min_free_gpu_gib"], 20)
        with self.assertRaisesRegex(ValueError, "already has output"):
            L.prepare(SimpleNamespace(parent_run=parent, output=output,
                                     condition=["llava-1.5-7b/int8/okvqa"]))

    def gpu_mocks(self, failure=None):
        import torch
        events = []
        model = SimpleNamespace(generate=mock.Mock(side_effect=lambda **kw: SimpleNamespace(
            sequences=torch.zeros((1, kw["input_ids"].shape[1] + 32), dtype=torch.long),
            logits=(torch.zeros(1, 4),))))
        def load(*args):
            events.append("load")
            if failure is not None:
                raise failure
            return model, SimpleNamespace(), object(), {"quantized_fraction": 1.0}, {"load_seconds": 1}
        def state(_):
            events.append("state")
            return {"manifest_sha256": "synthetic-state"}
        def inputs(*args):
            return {"input_ids": torch.zeros((1, 8), dtype=torch.long),
                    "pixel_values": torch.zeros((1, 3, 4, 4)),
                    "image_grid_thw": torch.tensor([[1, 4, 4]])}
        mock.patch.object(L.B, "load_clean", side_effect=load).start()
        mock.patch.object(L.B.M, "quantized_state_manifest", side_effect=state).start()
        mock.patch.object(L.B.M, "runtime_device_identity", return_value={"device_name": "mock RTX 3090"}).start()
        mock.patch.object(L.B.M, "software_environment", return_value={"python": "synthetic"}).start()
        mock.patch.object(L.B, "vlm_inputs", side_effect=inputs).start()
        scores = mock.patch.object(L.B, "score_vqa", return_value={"status": "complete", "target_tokens": 128}).start()
        rows = mock.patch.object(L.B, "rows_from_arrow", side_effect=AssertionError("final data opened")).start()
        for name, value in (("synchronize", None), ("mem_get_info", (23 * 2**30, 24 * 2**30)),
                            ("memory_allocated", 15 * 2**30), ("memory_reserved", 17 * 2**30),
                            ("max_memory_allocated", 18 * 2**30), ("max_memory_reserved", 19 * 2**30)):
            mock.patch.object(torch.cuda, name, return_value=value).start()
        mock.patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=lambda _: events.append("reset")).start()
        return events, model, scores, rows

    def test_preflight_covers_loading_state_and_synthetic_pressure_only(self):
        run = self.freeze()
        events, model, scores, rows = self.gpu_mocks()
        L.preflight(self.args(run))
        self.assertEqual(events, ["reset", "load", "state"])
        self.assertEqual(scores.call_count, 12)
        rows.assert_not_called()
        profile_path = run / "memory_profiles" / "llava-1.5-7b_int8.json"
        profile = json.loads(profile_path.read_text())
        self.assertEqual(profile["status"], "complete")
        self.assertEqual(profile["memory"]["max_allocated_bytes"], 18 * 2**30)
        self.assertEqual(profile["memory"]["max_reserved_bytes"], 19 * 2**30)
        self.assertGreater(profile["host_maxrss_kib"], 0)
        self.assertEqual(profile["initial_memory"]["free_bytes"], 23 * 2**30)
        images = {tuple(p["image_size"]) for p in profile["phases"] if "image_size" in p}
        self.assertEqual(images, {(336, 336), (1120, 896), (896, 1120)})
        self.assertEqual(model.generate.call_count, 3)
        for call in model.generate.call_args_list:
            self.assertEqual(call.kwargs["min_new_tokens"], 32)
            self.assertEqual(call.kwargs["max_new_tokens"], 32)
            self.assertFalse(call.kwargs["do_sample"])
        preflight = json.loads((run / "preflight" / "llava-1.5-7b_int8.json").read_text())
        self.assertEqual(preflight["plan_sha256"], L.B.M.sha256(run / "plan.json"))
        self.assertEqual(preflight["memory_profile_sha256"], L.B.M.sha256(profile_path))
        with self.assertRaises(FileExistsError):
            L.preflight(self.args(run))

    def test_oom_keeps_failure_profile_without_successful_preflight(self):
        import torch
        run = self.freeze()
        self.gpu_mocks(torch.cuda.OutOfMemoryError("synthetic OOM"))
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            L.preflight(self.args(run))
        profile = json.loads((run / "memory_profiles" / "llava-1.5-7b_int8.json").read_text())
        self.assertTrue(profile["oom"])
        self.assertEqual(profile["status"], "failed")
        self.assertEqual(profile["failed_phase"], "load_and_original_gpu_quantization")
        self.assertFalse((run / "preflight" / "llava-1.5-7b_int8.json").exists())

    def test_real_free_memory_gate_is_honored_before_loading(self):
        import torch
        run = self.freeze()
        events, _, _, _ = self.gpu_mocks()
        with mock.patch.object(torch.cuda, "mem_get_info", return_value=(19 * 2**30, 24 * 2**30)):
            with self.assertRaisesRegex(RuntimeError, "admission threshold"):
                L.preflight(self.args(run))
        self.assertNotIn("load", events)

    def test_final_delegates_unchanged_and_rejects_changed_profile(self):
        run = self.freeze()
        args = self.args(run)
        profile = run / "memory_profiles" / "llava-1.5-7b_int8.json"
        profile.parent.mkdir()
        profile.write_text('{"status":"complete"}\n')
        (run / "preflight").mkdir()
        L.B.M.write_json_once(run / "preflight" / profile.name,
                             {"memory_profile_sha256": L.B.M.sha256(profile)})
        with mock.patch.object(L.B, "run_condition") as run_final:
            L.run_condition(args)
            run_final.assert_called_once_with(args)
            profile.write_text("modified")
            with self.assertRaisesRegex(ValueError, "memory profile"):
                L.run_condition(args)
            self.assertEqual(run_final.call_count, 1)

    def test_missing_final_flag_rejects_before_reading_inputs(self):
        args = self.args("unused")
        args.final = False
        with mock.patch.object(L.B.M, "read_json") as read:
            with self.assertRaises(ValueError):
                L.run_condition(args)
            read.assert_not_called()

    def test_cli_explicitly_configures_threshold_for_each_command(self):
        for command, tail, function in (
            ("prepare", ["--parent-run", "unused", "--output", "unused"], "prepare"),
            ("preflight", ["--run", "unused", "--model", "llava-1.5-7b", "--format", "int8"], "preflight"),
            ("run", ["--run", "unused", "--model", "llava-1.5-7b", "--format", "int8",
                     "--task", "okvqa", "--split", "test", "--final"], "run_condition")):
            with self.subTest(command=command), mock.patch.object(L, function) as operation:
                L.main([command, *tail, "--min-free-gpu-gib", "21"])
                operation.assert_called_once()
                self.assertEqual(L.B.SETTINGS["min_free_gpu_gib"], 21)


if __name__ == "__main__":
    unittest.main()
