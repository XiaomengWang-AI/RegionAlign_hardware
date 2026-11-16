"""Collect completed clean-matrix metrics without changing any evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from regionalign import config as C


def read(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--audit", required=True)
    p.add_argument("--mmlu-run", required=True)
    p.add_argument("--wikitext-root", required=True)
    p.add_argument("--output")
    args = p.parse_args(argv)
    summary = {"schema_version": 1, "audit": str(Path(args.audit).resolve()),
               "mmlu_clean": [], "wikitext_clean": [],
               "region_attack": {"status": "not_measured_by_clean_summary",
                                  "reason": "inspect each formal RegionAlign run's registered artifacts and evaluations"}}
    for path in sorted(Path(args.mmlu_run).glob("conditions/*/*/*/metrics.json")):
        metrics = read(path)
        summary["mmlu_clean"].append({"model": metrics.get("model"), "format": metrics.get("format"),
                                      "split": metrics.get("split"), "accuracy": metrics.get("accuracy"),
                                      "count": metrics.get("count"), "path": str(path.resolve())})
    for path in sorted(Path(args.wikitext_root).glob("*/metrics.json")):
        metrics = read(path)
        summary["wikitext_clean"].append({"model": metrics.get("model_name"),
                                          "format": metrics.get("format"),
                                          "nll": metrics.get("nll"), "ppl": metrics.get("ppl"),
                                          "count": metrics.get("observed_samples"),
                                          "path": str(path.resolve())})
    summary["mmlu_clean"].sort(key=lambda x: (x["model"], x["format"], x["split"]))
    summary["wikitext_clean"].sort(key=lambda x: (x["model"], x["format"]))
    target = C.fresh_output_dir("experimental_matrix", "results_summary", args.output)
    (target / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(target)


if __name__ == "__main__":
    main()
