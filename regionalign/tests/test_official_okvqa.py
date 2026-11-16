"""CPU-only provenance and no-replacement checks for official training inputs."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from PIL import Image

from regionalign import config as C
from regionalign.experiments import clean_mmlu as M, official_okvqa as O, vqa as D
from regionalign.runtimes import vqa as V


def reference(answers, task):
    return answers[0].strip()


def documents(count=12):
    header = {"data_subtype": "train2014", "data_type": "mscoco"}
    questions, annotations = [], []
    for number in range(count):
        questions.append({"question_id": number, "image_id": number // 2,
                          "question": f"Training question {number}?"})
        annotations.append({"question_id": number, "image_id": number // 2,
                            "answers": [{"answer": "yes", "raw_answer": "yes!"}] * 10})
    return {**header, "questions": questions}, {**header, "annotations": annotations}


def zip_document(member, document):
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as container:
        container.writestr(member, json.dumps(document))
    return result.getvalue()


def jpeg():
    target = io.BytesIO()
    Image.new("RGB", (3, 2), "red").save(target, format="JPEG")
    return target.getvalue()


class SelectionTests(unittest.TestCase):
    def test_pre_image_selection_matches_existing_explicit_reference_rule(self):
        questions, annotations = documents()
        panels, audit = O.select_metadata(questions, annotations, reference_answer=reference,
                                         search_rows=2, dev_rows=3)
        records = [{**q, "set_name": "train", "answers": ["yes"] * 10,
                    "image": {"bytes": jpeg()}} for q in questions["questions"]]
        expected = V.select_train_panels(records, "okvqa", search_rows=2, dev_rows=3,
                                        reference_answer=reference)
        for split in ("search", "dev"):
            self.assertEqual([row["question_id"] for row in panels[split]],
                             [row["question_id"] for row in expected[split]])
        self.assertFalse({r["image_id"] for r in panels["search"]}
                         & {r["image_id"] for r in panels["dev"]})
        self.assertEqual(audit["question_count"], 12)
        self.assertEqual(audit["eligible_distinct_images"], 6)
        self.assertEqual(audit["answer_field"], "annotations[].answers[].answer")

    def test_complete_universe_order_does_not_change_selection(self):
        q, a = documents()
        first = O.select_metadata(q, a, reference_answer=reference, search_rows=2, dev_rows=3)
        q["questions"].reverse()
        a["annotations"].reverse()
        self.assertEqual(first, O.select_metadata(q, a, reference_answer=reference,
                                                  search_rows=2, dev_rows=3))

    def test_wrong_split_duplicate_missing_or_misaligned_labels_are_rejected(self):
        for failure in ("split", "duplicate", "missing", "image", "answer"):
            with self.subTest(failure=failure):
                q, a = documents()
                if failure == "split":
                    q["data_subtype"] = "val2014"
                elif failure == "duplicate":
                    q["questions"].append(deepcopy(q["questions"][0]))
                elif failure == "missing":
                    a["annotations"].pop()
                elif failure == "image":
                    a["annotations"][0]["image_id"] = 123
                else:
                    a["annotations"][0]["answers"] = []
                with self.assertRaises(ValueError):
                    O.select_metadata(q, a, reference_answer=reference, search_rows=2, dev_rows=3)

    def test_missing_reference_does_not_rank_as_eligible(self):
        q, a = documents()
        with self.assertRaisesRegex(ValueError, "distinct eligible"):
            O.select_metadata(q, a, reference_answer=lambda *_: "", search_rows=2, dev_rows=3)

    def test_corrupt_or_non_jpeg_image_is_rejected(self):
        self.assertEqual(O._image_bytes_valid(jpeg()), {"width": 3, "height": 2, "format": "JPEG"})
        with self.assertRaises(Exception):
            O._image_bytes_valid(b"not an image")
        target = io.BytesIO()
        Image.new("RGB", (2, 2)).save(target, format="PNG")
        with self.assertRaisesRegex(ValueError, "JPEG"):
            O._image_bytes_valid(target.getvalue())

    def test_failed_chosen_image_never_fetches_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            q, a = documents()
            panels, selection = O.select_metadata(q, a, reference_answer=reference,
                                                  search_rows=2, dev_rows=3)
            M.write_json_once(root / "selection.json", selection)
            rows = panels["search"] + panels["dev"]
            wanted = {O.IMAGE_URL.format(image_id=int(row["image_id"])) for row in rows}
            failed = next(iter(wanted))
            seen = []
            def fetch(url, **kwargs):
                seen.append(url)
                self.assertTrue((root / "selection.json").is_file())
                if url == failed:
                    raise OSError("chosen training image unavailable")
                return jpeg(), {"url": url}
            with mock.patch.object(O, "_fetch", side_effect=fetch):
                with self.assertRaisesRegex(RuntimeError, "no replacements"):
                    O._download_selected(root, rows, workers=2)
            self.assertEqual(set(seen), wanted)
            self.assertEqual(len(seen), 5)
            self.assertFalse((root / "audit.json").exists())
            self.assertEqual(M.read_json(root / "failure.json")["downloaded_count"], 4)


class PreparationTests(unittest.TestCase):
    def test_full_offline_prepare_preserves_geometry_and_audits_new_source(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(C, "RESULTS_ROOT", Path(temp)):
            root = Path(temp)
            q, a = documents(800)
            archive_by_url = {O.QUESTIONS_URL: zip_document(O.JSON_MEMBERS[0], q),
                              O.ANNOTATIONS_URL: zip_document(O.JSON_MEMBERS[1], a)}
            def fetch(url, **kwargs):
                body = archive_by_url.get(url, jpeg())
                return body, {"url": url, "resolved_url": url, "http_status": 200}
            with mock.patch.object(O, "_fetch", side_effect=fetch), \
                    mock.patch.object(C, "setup_environment"), \
                    mock.patch("regionalign.experiments.clean_benchmarks.reference_answer", reference):
                inputs = O.prepare_inputs(root / "inputs", workers=4)
            _, audit, selection = O._verified_inputs(inputs)
            self.assertEqual(audit["image_count"], 384)
            self.assertEqual([audit["panels"][s]["count"] for s in ("search", "dev")], [128, 256])
            self.assertEqual(len(audit["search_bootstraps"]), 3)
            old_root = root / "old"
            old_root.mkdir()
            dims = {"layers": 3, "hidden": 64, "mlp": 128, "heads": 4, "head_dim": 16, "group": 128}
            conditions = []
            for model in D.MODELS:
                for fmt in D.FORMATS:
                    for task in D.TASKS:
                        conditions.append({"id": f"{model}__{fmt}__{task}", "model": model,
                            "format": fmt, "task": task, "dims": dims,
                            "regions": D.preregistered_regions(model, dims, 8 if fmt == "int8" else 4),
                            "checkpoint_manifest_sha256": "inherited-model-manifest",
                            "availability": "unavailable", "unavailable_reason": "train absent",
                            "panels": None, "search_bootstraps": None})
            old = {"schema_version": 2, "scope": D.SCOPE, "development_only": True,
                   "splits": ["search", "dev"], "conditions": conditions,
                   "models": {model: {"manifest_sha256": "inherited-model-manifest"} for model in D.MODELS},
                   "datasets": {}, "sources": [], "repeat_seeds": list(D.REPEAT_SEEDS),
                   "work_items_per_condition": 108, "arms": list(D.ARMS), "search_budget": 300,
                   "execution": {"search_batch_size": 1, "grad_batch_size": 1, "eval_batch_size": 1,
                                 "min_free_gpu_gib": D.MIN_FREE_GPU_GIB,
                                 "min_free_policy": D.MIN_FREE_POLICY, "real_gpu_preflight_required": True}}
            M.write_json_once(old_root / "plan.json", old)
            M.write_json_once(old_root / "status.json", {"plan_sha256": M.sha256(old_root / "plan.json")})
            parent = O.prepare_parent(old_root, inputs, root / "new")
            _, plan = D.load_plan(parent)
            self.assertEqual(plan["tasks"], ["okvqa"])
            self.assertEqual(plan["registered_available_searches"], 432)
            self.assertEqual(plan["models"], old["models"])
            self.assertEqual(len(plan["conditions"]), 4)
            for row in plan["conditions"]:
                previous = next(x for x in old["conditions"] if x["id"] == row["id"])
                self.assertEqual(row["regions"], previous["regions"])
                self.assertEqual(len(D.planned_items(row, plan)), 108)
                self.assertEqual(len(D._read_panel(parent, row, "search")), 128)
                self.assertEqual(len(D._read_panel(parent, row, "dev")), 256)
                self.assertEqual(len(row["search_bootstraps"]), 3)
            binding = plan["datasets"]["okvqa"]["dataset_binding"]
            self.assertFalse(binding["hf_repository_revision_claimed"])
            self.assertEqual(binding["provider"], "official_okvqa_and_coco")
            self.assertNotIn("revision", binding)
            self.assertEqual(plan["official_okvqa_policy"]["parent_plan_sha256"], M.sha256(old_root / "plan.json"))
            self.assertIn(str(O.MODULE_SOURCE), {x["path"] for x in plan["sources"]})
            for split in ("search", "dev"):
                relative = Path(audit["panels"][split]["path"])
                self.assertEqual((inputs / relative).read_bytes(), (parent / relative).read_bytes())
            with self.assertRaises(FileExistsError):
                O.prepare_parent(old_root, inputs, parent)
            with self.assertRaises(FileExistsError):
                O.prepare_inputs(inputs)
            image = inputs / audit["images"][0]["path"]
            image.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "input drift"):
                O._verified_inputs(inputs)


if __name__ == "__main__":
    unittest.main()
