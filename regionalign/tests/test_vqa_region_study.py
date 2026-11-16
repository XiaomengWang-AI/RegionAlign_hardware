"""Protocol tests for the train-only formal VQA RegionAlign study."""
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from regionalign.experiments import vqa as S


class RegionRegistrationTests(unittest.TestCase):
    GQA = {"layers": 28, "hidden": 3584, "mlp": 18944, "heads": 28,
           "kv_heads": 4, "query_group_size": 7, "head_dim": 128, "group": 128}

    def test_twelve_regions_cover_bands_and_kinds_for_both_formats(self):
        int8 = S.preregistered_regions("qwen2-vl-7b", self.GQA, 8)
        int4 = S.preregistered_regions("qwen2-vl-7b", self.GQA, 4)
        self.assertEqual(len(int8), 12)
        self.assertEqual({row["depth_band"] for row in int8}, set(S.DEPTH_BANDS))
        for band in S.DEPTH_BANDS:
            rows = [row for row in int8 if row["depth_band"] == band]
            self.assertEqual([row["kind"] for row in rows], list(S.REGION_KINDS))
            self.assertTrue(all(row["layer"] in S.depth_ranges(self.GQA["layers"])[band]
                                for row in rows))
        self.assertTrue(all(row["page_bytes"] == 4096 and row["bits"] == 8 for row in int8))
        self.assertTrue(all(row["page_bytes"] == 4096 and row["bits"] == 4 for row in int4))

    def test_condition_dims_match_loaded_int4_scale_groups(self):
        base = {key: value for key, value in self.GQA.items() if key != "group"}
        self.assertIsNone(S._format_dims(base, "int8")["group"])
        self.assertEqual(S._format_dims(base, "int4")["group"], 128)


class WorkItemTests(unittest.TestCase):
    def setUp(self):
        dims = {"layers": 32, "hidden": 4096, "mlp": 11008, "heads": 32,
                "head_dim": 128, "group": 1}
        self.condition = {"id": "llava-1.5-7b__int8__textvqa",
                          "regions": S.preregistered_regions("llava-1.5-7b", dims, 8)}
        self.plan = {"repeat_seeds": list(S.REPEAT_SEEDS), "arms": list(S.ARMS)}

    def test_cardinality_names_and_matched_sharding(self):
        items = S.planned_items(self.condition, self.plan)
        self.assertEqual(len(items), 108)
        self.assertEqual(len({row["name"] for row in items}), 108)
        seen = set()
        for index in range(7):
            groups = {}
            for row in S.shard_items(items, f"{index}/7"):
                groups.setdefault((row["region"]["id"], row["repeat"]), set()).add(row["arm"])
            self.assertTrue(all(arms == set(S.ARMS) for arms in groups.values()))
            self.assertFalse(seen & set(groups))
            seen |= set(groups)
        self.assertEqual(len(seen), 36)

    def test_concurrent_duplicate_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            with S._item_lock(raw, "search", self.condition["id"], "one.json"):
                with self.assertRaises(RuntimeError):
                    with S._item_lock(raw, "search", self.condition["id"], "one.json"):
                        pass


class DevelopmentIsolationTests(unittest.TestCase):
    def test_source_snapshot_tracks_only_package_sources(self):
        expected = {Path("regionalign/experiments/vqa.py"),
                    Path("regionalign/runtimes/vqa.py"),
                    Path("regionalign/experiments/clean_mmlu.py"),
                    Path("regionalign/experiments/clean_benchmarks.py"),
                    Path("regionalign/config.py"),
                    Path("regionalign/harness/core.py"),
                    Path("regionalign/harness/region.py"),
                    Path("regionalign/algorithm/search.py"),
                    Path("regionalign/runtimes/vqa_score.py"),
                    Path("regionalign/runtime.json")}
        self.assertTrue(expected <= set(S.SOURCE_RELATIVE))
        self.assertTrue(all(path.parts[0] == "regionalign" for path in S.SOURCE_RELATIVE))
        self.assertEqual(S.SEARCH_SNAPSHOT, "regionalign__algorithm__search.py")
        with tempfile.TemporaryDirectory() as raw:
            run = Path(raw)
            sources = S._copy_sources(run)
            plan = {"sources": sources}
            S._assert_frozen(run, plan)
            algorithm = next(record for record in sources
                             if record["path"] == "regionalign/algorithm/search.py")
            snapshot = run / algorithm["snapshot"]
            self.assertEqual(snapshot.name, S.SEARCH_SNAPSHOT)
            snapshot.write_bytes(snapshot.read_bytes() + b"\n# changed snapshot\n")
            with self.assertRaisesRegex(RuntimeError, "algorithm/search.py"):
                S._assert_frozen(run, plan)

    def test_cli_has_no_test_final_or_split_interface(self):
        parser = S.parser()
        commands = next(action for action in parser._actions if getattr(action, "choices", None))
        self.assertEqual(set(commands.choices),
                         {"plan", "preflight", "search", "evaluate", "status"})
        help_text = parser.format_help().lower()
        self.assertNotIn("--final", help_text)
        self.assertNotIn("--split", help_text)

    def test_missing_task_is_registered_but_not_executable(self):
        plan = {"conditions": [{"id": "x", "availability": "unavailable",
                                "unavailable_reason": "train data absent"}]}
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            S._condition(plan, "x")

    def test_nontrain_paths_are_rejected_before_arrow_open(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "textvqa-validation-00000.arrow"
            path.write_bytes(b"not opened")
            with self.assertRaisesRegex(ValueError, "train Arrow"):
                S._training_paths([path], "textvqa")
            parent = Path(raw) / "test"
            parent.mkdir()
            disguised = parent / "textvqa-train-00000.arrow"
            disguised.write_bytes(b"not opened")
            with self.assertRaisesRegex(ValueError, "train Arrow"):
                S._training_paths([disguised], "textvqa")

    def test_only_search_and_dev_panels_are_addressable(self):
        with self.assertRaises(ValueError):
            S._read_panel("unused", {"panels": {}}, "test")

    def test_load_plan_requires_status_to_bind_plan_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            run = Path(raw)
            plan = {"schema_version": S.SCHEMA_VERSION, "scope": S.SCOPE,
                    "development_only": True, "splits": ["search", "dev"],
                    "sources": [],
                    "execution": {"search_batch_size": S.SEARCH_BATCH_SIZE,
                                  "grad_batch_size": S.GRAD_BATCH_SIZE,
                                  "eval_batch_size": S.EVAL_BATCH_SIZE,
                                  "min_free_gpu_gib": S.MIN_FREE_GPU_GIB,
                                  "min_free_policy": S.MIN_FREE_POLICY,
                                  "real_gpu_preflight_required": True}}
            (run / "plan.json").write_text(json.dumps(plan))
            (run / "status.json").write_text(json.dumps(
                {"plan_sha256": "wrong", "development_only": True}))
            with mock.patch.object(S.C, "output_path", side_effect=lambda value: Path(value)):
                with self.assertRaisesRegex(ValueError, "status.json"):
                    S.load_plan(run)


class PanelEvidenceTests(unittest.TestCase):
    def _row(self, index):
        image = f"image-{index}".encode()
        question = f"question {index}?"
        answers = ["cat"] * 10
        return {"sid": f"textvqa-{index}", "question_id": str(index),
                "image_id": f"image-{index}", "question": question,
                "answers": answers, "reference": "cat", "rank": f"{index:04d}",
                "image": {"bytes": image, "path": None},
                "image_sha256": hashlib.sha256(image).hexdigest(),
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "answers_sha256": hashlib.sha256("\0".join(answers).encode()).hexdigest()}

    def test_binary_panel_round_trip_checks_every_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rows = [self._row(i) for i in range(3)]
            record = S._panel_record(root, "textvqa", "search", rows)
            condition = {"panels": {"search": record}}
            loaded = S._read_panel(root, condition, "search")
            self.assertEqual([row["sid"] for row in loaded], [row["sid"] for row in rows])
            with (root / record["path"]).open("ab") as handle:
                handle.write(b"drift")
            with self.assertRaisesRegex(RuntimeError, "panel drift"):
                S._read_panel(root, condition, "search")


class MetricProtocolTests(unittest.TestCase):
    def test_vqa_nll_is_token_weighted_and_official_score_is_retained(self):
        rows = [{"sid": "a", "target_tokens": 1, "objective_weight": 0.5},
                {"sid": "b", "target_tokens": 3, "objective_weight": 1.5}]
        generated = {"acc": 0.7, "nonfinite_rate": 0.0, "length_rate": 0.5,
                     "eos_rate": 0.5,
                     "records": [{"sid": "a", "prediction": "x"},
                                 {"sid": "b", "prediction": "y"}]}
        runtime = SimpleNamespace(
            nll=lambda rows, bs=1: [1.0, 9.0],
            accuracy=lambda rows: generated)
        metrics = S._task_metrics(runtime, rows)
        # Raw row NLLs are 2 and 6; token-weighted mean is (2 + 18) / 4.
        self.assertEqual(metrics["nll"], 5.0)
        self.assertEqual(metrics["score"], 0.7)
        self.assertEqual(metrics["eos_rate"], 0.5)
        self.assertEqual(metrics["length_rate"], 0.5)

    def test_fault_damage_uses_rearranged_layout_baseline(self):
        damage = S._fault_damage(
            {"nll": 5.0, "score": 0.2}, {"nll": 2.0, "score": 0.6},
            {"nll": 1.0, "score": 0.9})
        self.assertEqual(damage["damage_baseline"], "rearranged_unfaulted")
        self.assertEqual(damage["delta_nll"], 3.0)
        self.assertAlmostEqual(damage["relative_score_drop"], 2 / 3)
        self.assertEqual(damage["original_layout_diagnostic"]["delta_nll"], 4.0)
        self.assertNotEqual(damage["relative_score_drop"],
                            damage["original_layout_diagnostic"]["relative_score_drop"])


class BootstrapProtocolTests(unittest.TestCase):
    def test_repeats_are_with_replacement_and_shared_by_all_arms(self):
        rows = [{"sid": f"s{i}"} for i in range(S.V.SEARCH_ROWS)]
        records = S._bootstrap_records(rows, "textvqa")
        self.assertEqual(len(records), 3)
        condition = {"search_bootstraps": records}
        for repeat, record in enumerate(records):
            self.assertEqual(S._bootstrap_record(condition, repeat), record)
            self.assertEqual(record["draws"], S.V.SEARCH_ROWS)
            self.assertTrue(record["with_replacement"])
            self.assertLess(record["unique_sids"], record["draws"])
            # Lookup is keyed only by repeat, so every region/arm uses this draw.
            for _arm in S.ARMS:
                self.assertEqual(S._bootstrap_record(condition, repeat)["sid_sha256"],
                                 record["sid_sha256"])

    def test_summary_excludes_invalid_evaluations(self):
        results = {"1": {"valid": True, "delta_nll": 1.0,
                          "relative_score_drop": 0.6, "success": True},
                   "2": {"valid": False, "delta_nll": 99.0,
                          "relative_score_drop": 0.99, "success": True}}
        summary = S._summary(results)
        self.assertEqual(summary["valid_evaluations"], 1)
        self.assertEqual(summary["max_delta_nll"], 1.0)
        self.assertEqual(summary["max_relative_score_drop"], 0.6)
        self.assertTrue(summary["success_any"])


if __name__ == "__main__":
    unittest.main()
