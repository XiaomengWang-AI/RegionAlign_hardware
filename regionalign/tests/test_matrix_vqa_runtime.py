"""Synthetic tests for the train-only LLaVA/Qwen2-VL RegionAlign adapter."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from regionalign.runtimes import vqa as V
from regionalign.harness import core as P


def record(index, image_id=None, set_name="train", answer="cat"):
    return {"question_id": index, "image_id": image_id or f"image-{index}",
            "question": f"question {index}?", "answers": [answer] * 10,
            "image": {"bytes": f"image {index}".encode(), "path": f"{index}.jpg"},
            "set_name": set_name}


class PanelTests(unittest.TestCase):
    def test_selection_is_exact_deterministic_and_image_disjoint(self):
        rows = [record(i) for i in range(12)] + [record(99, image_id="image-0")]
        reference = lambda answers, task: answers[0].strip()
        one = V.select_train_panels(rows, "textvqa", 4, 6, seed=7, reference_answer=reference)
        two = V.select_train_panels(reversed(rows), "textvqa", 4, 6, seed=7,
                                    reference_answer=reference)
        self.assertEqual(V.panel_manifest(one), V.panel_manifest(two))
        self.assertEqual((len(one["search"]), len(one["dev"])), (4, 6))
        search_images = {row["image_id"] for row in one["search"]}
        dev_images = {row["image_id"] for row in one["dev"]}
        self.assertFalse(search_images & dev_images)
        self.assertEqual(len(search_images | dev_images), 10)

    def test_nontraining_rows_and_nontraining_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "training rows only"):
            V.select_train_panels([record(0, set_name="validation")], "okvqa", 1, 1,
                                  reference_answer=lambda a, t: a[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ok-vqa-val2014-00000.arrow"
            path.write_bytes(b"not opened")
            with self.assertRaisesRegex(ValueError, "non-training"):
                list(V._arrow_rows([path], "okvqa", validate_paths=False))

    def test_empty_answer_or_image_never_enters_search(self):
        rows = [record(i) for i in range(5)]
        rows[0]["answers"] = [""] * 10
        rows[1]["image"]["bytes"] = b""
        panels = V.select_train_panels(rows, "okvqa", 1, 2,
                                       reference_answer=lambda answers, task: answers[0].strip())
        self.assertEqual({row["question_id"] for split in panels.values() for row in split},
                         {"2", "3", "4"})

    def test_arrow_loader_uses_train_only_two_pass_selection(self):
        import pyarrow as pa
        import pyarrow.ipc as ipc
        rows = [record(i, image_id=f"image-{i // 2}") for i in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            spec = V.DATASET_SPECS["textvqa"]
            parent = (Path(directory) / spec["cache_component"] / spec["config"]
                      / spec["version"] / spec["revision"])
            parent.mkdir(parents=True)
            path = parent / "textvqa-train-00000-of-00001.arrow"
            table = pa.Table.from_pylist(rows)
            with path.open("wb") as handle, ipc.new_stream(handle, table.schema) as writer:
                writer.write_table(table)
            panels = V.load_train_panels(
                [path], "textvqa", search_rows=2, dev_rows=3, seed=9,
                reference_answer=lambda answers, task: answers[0].strip())
        self.assertEqual((len(panels["search"]), len(panels["dev"])), (2, 3))
        images = [row["image_id"] for split in ("search", "dev") for row in panels[split]]
        self.assertEqual(len(images), len(set(images)))
        self.assertTrue(all(row["question_sha256"] and row["answers_sha256"]
                            for split in panels.values() for row in split))

    def test_wrong_repository_revision_and_incomplete_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = V.DATASET_SPECS["textvqa"]
            wrong = (Path(directory) / spec["cache_component"] / spec["config"]
                     / spec["version"] / "wrong")
            wrong.mkdir(parents=True)
            path = wrong / "textvqa-train-00000-of-00001.arrow"
            path.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "frozen"):
                V.validate_train_paths([path], "textvqa")

            correct = wrong.parent / spec["revision"]
            correct.mkdir()
            shard = correct / "textvqa-train-00000-of-00002.arrow"
            shard.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "all 2"):
                V.validate_train_paths([shard], "textvqa")


class LayoutTests(unittest.TestCase):
    def test_gqa_qkv_rows_are_placed_by_complete_query_groups(self):
        from regionalign.harness import region as R
        dims = {"layers": 1, "hidden": 8, "mlp": 12, "heads": 4, "head_dim": 2,
                "kv_heads": 2, "query_group_size": 2, "group": None}
        expected = {"q_proj": ([0, 2, 4, 6], 4),
                    "k_proj": ([0, 2], 2),
                    "v_proj": ([0, 1, 2, 3], 3)}
        for kind, (reachable, source_row) in expected.items():
            region = R.make_region(kind, 0, 0, dims, bits=8, page_bytes=8)
            self.assertEqual(R.reachable_rows(region, dims), reachable)
            spec = R.placement(region, dims, source_row)
            columns = R.tensor_shape(kind, dims)[1]
            deployed = R.deploy(region, dims, spec,
                                [[region["tensor"], source_row * columns, 0]])
            self.assertEqual(deployed, [[region["tensor"], 0, 0]])

    def test_layout_delegates_to_the_shared_architecture_aware_kernel(self):
        rt, spec = SimpleNamespace(lm=object()), {"normalized": True}
        with mock.patch.object(P, "dims", return_value={}), \
                mock.patch.object(P, "validate_spec", return_value=spec), \
                mock.patch.object(P, "apply_permutation") as apply:
            V.apply_layout(rt, {"raw": True}, inverse=True)
        apply.assert_called_once_with(rt, spec, inverse=True)

    def test_visual_features_follow_the_runtime_residual_order(self):
        import torch
        runtime = object.__new__(V.HFVQARegionRuntime)
        runtime.residual = torch.tensor([2, 0, 1])
        features = torch.tensor([[[1., 2., 3.]]])
        moved = runtime._reorder_visual(None, None, features)
        self.assertTrue(torch.equal(moved, torch.tensor([[[3., 1., 2.]]])))


class RuntimeContractTests(unittest.TestCase):
    def test_qwen_positions_use_the_models_multimodal_index(self):
        import torch
        expected_positions = torch.arange(9).reshape(3, 1, 3)
        expected_delta = torch.tensor([[4]])
        model = SimpleNamespace(get_rope_index=mock.Mock(
            return_value=(expected_positions, expected_delta)))
        runtime = object.__new__(V.HFVQARegionRuntime)
        runtime.model_key = "qwen2-vl-7b"
        runtime.model = model
        inputs = {"input_ids": torch.ones(1, 3, dtype=torch.long),
                  "attention_mask": torch.ones(1, 3, dtype=torch.long),
                  "image_grid_thw": torch.tensor([[1, 2, 2]])}
        result = runtime._position_kwargs(inputs)
        self.assertIs(result["position_ids"], expected_positions)
        self.assertIs(result["rope_deltas"], expected_delta)
        model.get_rope_index.assert_called_once()

    def test_prepare_rows_makes_row_mean_token_weighted(self):
        processor = SimpleNamespace(tokenizer=SimpleNamespace(
            encode=lambda text, add_special_tokens=False: list(range(len(text.split())))))
        runtime = object.__new__(V.HFVQARegionRuntime)
        runtime.processor = processor
        rows = runtime._prepare_rows([
            {"sid": "a", "reference": "one"},
            {"sid": "b", "reference": "one two three"},
        ])
        self.assertEqual([row["target_tokens"] for row in rows], [1, 3])
        self.assertAlmostEqual(sum(row["objective_weight"] for row in rows), 2.0)


if __name__ == "__main__":
    unittest.main()
