"""Clean WikiText protocol tests using synthetic strings only."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from regionalign.experiments import clean_wikitext as W


class WikiTextCleanDevTests(unittest.TestCase):
    def test_raw_spans_are_fixed_disjoint_and_hash_bound(self):
        text = "\n".join(" ".join(f"word{line}_{word}" for word in range(20))
                         for line in range(30))
        rows = W.select_spans(text, count=8, words_per_sample=16, seed=41)
        self.assertEqual(rows, W.select_spans(text, count=8, words_per_sample=16, seed=41))
        self.assertNotEqual(rows, W.select_spans(text, count=8, words_per_sample=16, seed=42))
        previous_end = 0
        for index, row in enumerate(rows):
            self.assertEqual(row["index"], index)
            self.assertGreaterEqual(row["char_start"], previous_end)
            previous_end = row["char_end"]
            self.assertEqual(row["text"], text[row["char_start"]:row["char_end"]])
            self.assertEqual(len(row["text"].split()), 16)
            self.assertEqual(row["text_sha256"], W.text_sha256(row["text"]))

    def test_token_weighted_nll_is_not_mean_of_sample_means(self):
        result = W.aggregate([
            {"status": "complete", "target_tokens": 2, "nll_sum": 2.0},
            {"status": "complete", "target_tokens": 8, "nll_sum": 24.0},
        ], expected_count=2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["finite_target_tokens"], 10)
        self.assertAlmostEqual(result["nll"], 2.6)
        self.assertNotAlmostEqual(result["nll"], 2.0)

    def test_nonfinite_or_missing_samples_prevent_success(self):
        rows = [{"status": "complete", "target_tokens": 2, "nll_sum": 2.0},
                {"status": "nonfinite", "target_tokens": 8, "nll_sum": None}]
        result = W.aggregate(rows, 2)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["nll"])
        self.assertIsNone(result["ppl"])
        self.assertEqual(W.aggregate(rows[:1], 2)["status"], "failed")

    def test_output_directory_and_json_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(W.C, "PROJECT", root):
                output = W.C.fresh_output_dir("clean", "validation", "regionalign/results/example")
                with self.assertRaises(FileExistsError):
                    W.C.fresh_output_dir("clean", "validation", "regionalign/results/example")
                W.M.write_json_once(output / "panel.json", {"x": 1})
                with self.assertRaises(FileExistsError):
                    W.M.write_json_once(output / "panel.json", {"x": 2})
                with self.assertRaises(ValueError):
                    W.C.fresh_output_dir("clean", "validation", str(root / "outside"))

    def test_nonvalidation_files_and_disguised_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("wikitext-test.arrow", "wikitext-train.arrow", "mmlu-validation.arrow"):
                with self.assertRaises(ValueError):
                    W.validation_path(root / name)
            test = root / "wikitext-test.arrow"
            test.write_bytes(b"synthetic bytes; never read")
            link = root / "wikitext-validation.arrow"
            link.symlink_to(test)
            with self.assertRaises(ValueError):
                W.validation_path(link)


if __name__ == "__main__":
    unittest.main()
