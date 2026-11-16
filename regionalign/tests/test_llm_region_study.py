import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from regionalign.experiments import llm as S


class RegionRegistrationTests(unittest.TestCase):
    DIMS = {"layers": 32, "hidden": 4096, "mlp": 11008,
            "heads": 32, "kv_heads": 8, "head_dim": 128,
            "group": 4}

    def test_twelve_regions_cover_three_depth_bands_and_four_kinds(self):
        regions = S.preregistered_regions("llama3.1-8b", self.DIMS, 8)
        self.assertEqual(len(regions), 12)
        self.assertEqual({region["depth_band"] for region in regions}, set(S.DEPTH_BANDS))
        for band in S.DEPTH_BANDS:
            band_regions = [region for region in regions if region["depth_band"] == band]
            self.assertEqual([region["kind"] for region in band_regions], list(S.REGION_KINDS))
            allowed = S.depth_ranges(self.DIMS["layers"])[band]
            self.assertTrue(all(region["layer"] in allowed for region in band_regions))

    def test_a_formal_plan_may_name_any_registered_subset(self):
        models = S._parse_models(["qwen2.5-1.5b=/checkpoints/b", "llama3.2-1b=/checkpoints/a"])
        self.assertEqual(set(models), {"llama3.2-1b", "qwen2.5-1.5b"})
        self.assertEqual(set(S._parse_models(["llama2-7b=/checkpoints/c"])), {"llama2-7b"})
        for entries in ([], ["unknown=/x"], ["llama3.2-1b"],
                        ["llama3.2-1b=/x", "llama3.2-1b=/y"]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                S._parse_models(entries)
        with tempfile.TemporaryDirectory() as temporary:
            for models in ({}, {"unregistered": Path(temporary)}):
                with self.subTest(models=models), self.assertRaisesRegex(ValueError, "registered"):
                    S.make_plan(Path(temporary) / "run", models, temporary, "train", "validation")

    def test_every_registered_model_has_legal_regions_in_every_band_and_kind(self):
        self.assertLessEqual({"llama3.2-1b", "qwen2.5-1.5b", "stablelm2-1.6b", "smollm2-1.7b"},
                             set(S.MODELS))
        for key, spec in S.M.MODEL_SPECS.items():
            for fmt, bits in (("int8", 8), ("int4", 4)):
                with self.subTest(model=key, format=fmt):
                    dims = S.dims_for_format(dict(spec), fmt)
                    self.assertEqual(dims["head_dim"] * dims["heads"], dims["hidden"])
                    regions = S.preregistered_regions(key, dims, bits)
                    self.assertEqual(len(regions), 12)
                    self.assertEqual({(r["depth_band"], r["kind"]) for r in regions},
                                     {(band, kind) for band in S.DEPTH_BANDS
                                      for kind in S.REGION_KINDS})
                    for region in regions:
                        self.assertTrue(S.R.reachable_rows(region, dims))

    def test_format_changes_capacity_without_changing_registered_design(self):
        int8 = S.preregistered_regions("llama3.1-8b", self.DIMS, 8)
        int4 = S.preregistered_regions("llama3.1-8b", self.DIMS, 4)
        self.assertEqual([(x["depth_band"], x["kind"]) for x in int8],
                         [(x["depth_band"], x["kind"]) for x in int4])
        self.assertTrue(all(x["page_bytes"] == 4096 and x["bits"] == 8 for x in int8))
        self.assertTrue(all(x["page_bytes"] == 4096 and x["bits"] == 4 for x in int4))

    def test_registration_is_deterministic(self):
        first = S.preregistered_regions("llama3.1-8b", self.DIMS, 8)
        second = S.preregistered_regions("llama3.1-8b", self.DIMS, 8)
        self.assertEqual(first, second)

    def test_planned_dimensions_match_runtime_quantization_groups(self):
        config = {"num_hidden_layers": 32, "hidden_size": 4096,
                  "intermediate_size": 11008, "num_attention_heads": 32,
                  "num_key_value_heads": 32}
        int8 = S.dims_for_format(config, "int8")
        int4 = S.dims_for_format(SimpleNamespace(**config), "int4")
        self.assertIsNone(int8["group"])
        self.assertEqual(int4["group"], 128)
        self.assertEqual({k: v for k, v in int8.items() if k != "group"},
                         {k: v for k, v in int4.items() if k != "group"})


class WorkItemTests(unittest.TestCase):
    def setUp(self):
        dims = {"layers": 32, "hidden": 4096, "mlp": 11008,
                "heads": 32, "kv_heads": 32, "head_dim": 128,
                "group": 1}
        self.condition = {"id": "llama2-7b__int8__mmlu",
                          "regions": S.preregistered_regions("llama2-7b", dims, 8)}
        self.plan = {"repeat_seeds": list(S.REPEAT_SEEDS), "arms": list(S.ARMS)}
        self.condition["bootstrap_seeds"] = {
            S._bootstrap_key(region["id"], repeat): repeat * 1000 + index
            for index, region in enumerate(self.condition["regions"])
            for repeat, _ in enumerate(S.REPEAT_SEEDS)
        }

    def test_registered_cardinality_and_unique_names(self):
        items = S.planned_items(self.condition, self.plan)
        self.assertEqual(len(items), 12 * 3 * 3)
        self.assertEqual(len({item["name"] for item in items}), len(items))

    def test_sharding_keeps_matched_arms_together(self):
        items = S.planned_items(self.condition, self.plan)
        seen = set()
        for index in range(5):
            mine = S.shard_items(items, f"{index}/5")
            groups = {}
            for item in mine:
                groups.setdefault((item["region"]["id"], item["repeat"]), set()).add(item["arm"])
            self.assertTrue(all(arms == set(S.ARMS) for arms in groups.values()))
            keys = set(groups)
            self.assertFalse(seen & keys)
            seen |= keys
        self.assertEqual(len(seen), 12 * 3)

    def test_matched_arms_share_bootstrap_seed_and_repeats_change_it(self):
        items = S.planned_items(self.condition, self.plan)
        groups = {}
        for item in items:
            groups.setdefault((item["region"]["id"], item["repeat"]), set()).add(
                item["bootstrap_seed"])
        self.assertTrue(all(len(seeds) == 1 for seeds in groups.values()))
        first = self.condition["regions"][0]["id"]
        self.assertEqual(len({next(iter(groups[(first, repeat)]))
                              for repeat in range(3)}), 3)

    def test_invalid_shard_is_rejected(self):
        items = S.planned_items(self.condition, self.plan)
        for value in ("1", "2/2", "-1/4", "0/0"):
            with self.assertRaises(ValueError):
                S.shard_items(items, value)

    def test_item_lock_rejects_a_concurrent_duplicate(self):
        with tempfile.TemporaryDirectory() as raw:
            with S._item_lock(raw, "search", self.condition["id"], "item.json"):
                with self.assertRaises(RuntimeError):
                    with S._item_lock(raw, "search", self.condition["id"], "item.json"):
                        pass


class DevelopmentIsolationTests(unittest.TestCase):
    def test_cli_exposes_no_test_or_final_command(self):
        parser = S.parser()
        command_action = next(action for action in parser._actions
                              if getattr(action, "choices", None))
        self.assertEqual(set(command_action.choices), {"plan", "search", "evaluate", "status"})
        help_text = parser.format_help().lower()
        self.assertNotIn("--final", help_text)
        self.assertNotIn("--split", help_text)

    def test_test_named_input_is_rejected_before_open(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            train = root / "wikitext-train.arrow"
            test = root / "wikitext-test.arrow"
            train.write_bytes(b"x")
            test.write_bytes(b"x")
            self.assertEqual(S._forbid_test_path(train, "train"), train.absolute())
            with self.assertRaises(ValueError):
                S._forbid_test_path(test, "test")

    def test_symlink_to_test_named_input_is_rejected_after_resolution(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "wikitext-test.arrow"
            link = root / "apparently-safe.arrow"
            target.write_bytes(b"x")
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                S._forbid_test_path(link, "train")

    def test_only_search_and_dev_panels_are_addressable(self):
        condition = {"panels": {}}
        with self.assertRaises(ValueError):
            S._panel_rows(Path("unused"), condition, "test")


class MetricProtocolTests(unittest.TestCase):
    def test_objective_weights_make_row_mean_token_weighted(self):
        rows = [{"target_tokens": 1}, {"target_tokens": 3}]
        S._normalise_objective_weights(rows)
        self.assertAlmostEqual(sum(row["objective_weight"] for row in rows), 2.0)
        raw_losses = [2.0, 6.0]
        row_mean = sum(loss * row["objective_weight"]
                       for loss, row in zip(raw_losses, rows)) / len(rows)
        token_mean = (2.0 * 1 + 6.0 * 3) / 4
        self.assertAlmostEqual(row_mean, token_mean)

    def test_success_summary_uses_only_valid_registered_results(self):
        results = {
            "1": {"valid": True, "delta_nll": 1.0, "ppl_ratio": 2.8,
                  "success": True, "relative_score_drop": 0.6},
            "2": {"valid": False, "delta_nll": 99.0, "ppl_ratio": 99.0,
                  "success": True, "relative_score_drop": 0.99},
        }
        summary = S._result_summary(results, "mmlu")
        self.assertEqual(summary["valid_evaluations"], 1)
        self.assertEqual(summary["max_delta_nll"], 1.0)
        self.assertEqual(summary["max_relative_score_drop"], 0.6)
        self.assertTrue(summary["success_any"])

    def test_fault_damage_uses_same_rearranged_layout_as_baseline(self):
        original = {"nll": 1.0, "accuracy": 0.8}
        rearranged = {"nll": 1.1, "accuracy": 0.75}
        faulted = {"nll": 2.0, "accuracy": 0.30}
        result = S._damage_metrics("mmlu", faulted, rearranged, original)
        self.assertAlmostEqual(result["delta_nll"], 0.9)
        self.assertAlmostEqual(result["relative_score_drop"], 0.6)
        self.assertTrue(result["meets_success_threshold"])
        self.assertAlmostEqual(result["diagnostic_delta_nll_from_original_clean"], 1.0)
        self.assertAlmostEqual(
            result["diagnostic_relative_score_drop_from_original_clean"], 0.625)


class FrozenProtocolTests(unittest.TestCase):
    def test_formal_runtime_is_the_package_text_runtime(self):
        from regionalign.runtimes import text as runtime
        self.assertIs(S.FormalTextRuntime, runtime.FormalTextRuntime)

    def test_source_snapshot_tracks_only_package_sources(self):
        expected = {Path("regionalign/experiments/llm.py"),
                    Path("regionalign/runtimes/text.py"),
                    Path("regionalign/experiments/clean_mmlu.py"),
                    Path("regionalign/config.py"),
                    Path("regionalign/harness/core.py"),
                    Path("regionalign/harness/region.py"),
                    Path("regionalign/algorithm/search.py")}
        self.assertTrue(expected <= set(S.SOURCE_RELATIVE))
        self.assertTrue(all(path.parts[0] == "regionalign" for path in S.SOURCE_RELATIVE))
        self.assertEqual(S.SEARCH_SNAPSHOT, "regionalign__algorithm__search.py")
        with tempfile.TemporaryDirectory() as raw:
            run = Path(raw)
            sources = S._copy_sources(run)
            plan = {"sources": sources}
            self.assertFalse(S._source_drift(run, plan))
            algorithm = next(record for record in sources
                             if record["path"] == "regionalign/algorithm/search.py")
            snapshot = run / algorithm["snapshot"]
            self.assertEqual(snapshot.name, S.SEARCH_SNAPSHOT)
            snapshot.write_bytes(snapshot.read_bytes() + b"\n# changed snapshot\n")
            self.assertEqual(S._source_drift(run, plan),
                             ["regionalign/algorithm/search.py"])

    def test_bootstrap_is_deterministic_with_replacement_and_reweights_rows(self):
        rows = [{"sid": str(index), "target_tokens": index + 1,
                 "objective_weight": 1.0} for index in range(3)]
        first, first_audit = S._bootstrap_rows(rows, 1)
        second, second_audit = S._bootstrap_rows(rows, 1)
        self.assertEqual([row["sid"] for row in first], ["0", "2", "0"])
        self.assertEqual(first, second)
        self.assertEqual(first_audit, second_audit)
        self.assertEqual(first_audit["method"], "bootstrap_with_replacement")
        self.assertAlmostEqual(sum(row["objective_weight"] for row in first), 3.0)

    def test_execution_arguments_must_equal_frozen_plan(self):
        protocol = S._execution_protocol()
        plan = {"execution_protocol": protocol}
        self.assertEqual(S._assert_execution_protocol(
            plan, "search", 1, 40.0, 1), protocol)
        for args in (("search", 2, 40.0, 1), ("search", 1, 39.0, 1),
                     ("search", 1, 40.0, 2), ("evaluate", 2, 40.0, None)):
            with self.assertRaises(ValueError):
                S._assert_execution_protocol(plan, *args)

    def test_load_plan_rejects_status_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            run = Path(raw)
            S.M.write_json_once(run / "plan.json", {
                "schema_version": S.SCHEMA_VERSION, "scope": S.SCOPE,
                "development_only": True, "splits": ["search", "dev"],
                "sources": [],
            })
            S.M.write_json_once(run / "status.json", {"plan_sha256": "wrong"})
            with mock.patch.object(S.C, "output_path", side_effect=lambda value: Path(value)):
                with self.assertRaisesRegex(RuntimeError, "does not register"):
                    S.load_plan(run)

    def test_resource_allocation_errors_are_fail_fast_classified(self):
        self.assertTrue(S._is_resource_error(MemoryError("cpu allocation")))
        self.assertTrue(S._is_resource_error(RuntimeError("CUDA out of memory")))
        self.assertTrue(S._is_resource_error(
            RuntimeError("CUBLAS_STATUS_ALLOC_FAILED during matmul")))
        self.assertFalse(S._is_resource_error(RuntimeError("invalid candidate")))

    def test_search_oom_publishes_no_algorithm_failure_artifact(self):
        with tempfile.TemporaryDirectory() as raw:
            run = Path(raw)
            region = {"id": "region-0"}
            item = {"name": "region-0__r0__local.json", "region": region,
                    "repeat": 0, "seed": 7, "bootstrap_seed": 11, "arm": "local"}
            condition = {"id": "condition", "model": "model", "format": "int8",
                         "task": "mmlu", "regions": [region]}
            plan = {"execution_protocol": S._execution_protocol(), "search_budget": 1,
                    "sha256": "plan"}
            rows = [{"sid": "row", "target_tokens": 1, "objective_weight": 1.0}]
            runtime = SimpleNamespace(lm=object(), rows={"search": rows})
            context = SimpleNamespace(spent=0, calls={})
            search = SimpleNamespace(search=mock.Mock(side_effect=MemoryError("allocation")))
            patches = (
                mock.patch.object(S, "load_plan", return_value=(run, plan)),
                mock.patch.object(S, "_condition", return_value=condition),
                mock.patch.object(S, "planned_items", return_value=[item]),
                mock.patch.object(S, "_load_runtime",
                                  return_value=(runtime, rows, {}, {})),
                mock.patch.object(S, "_assert_frozen"),
                mock.patch.object(S, "_search_module", return_value=search),
                mock.patch.object(S.R, "model_checksum", return_value="model"),
                mock.patch.object(S.R, "RegionContext", return_value=context),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                    patches[5], patches[6], patches[7]:
                with self.assertRaises(MemoryError):
                    S.run_search(run, "condition", device="cpu", log=lambda *_: None)
            self.assertFalse(S._artifact_path(run, "condition", item["name"]).exists())


if __name__ == "__main__":
    unittest.main()
