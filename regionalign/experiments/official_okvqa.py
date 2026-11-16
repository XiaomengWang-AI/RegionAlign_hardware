"""Independent official-training input adapter (ADR-0025).

This adapter never impersonates an HF dataset revision.  It joins the complete
official train2014 questions/annotations by ID, applies the existing VQA hash
rank and reference rule before any image fetch, and commits that selection.
Exactly those 384 COCO training JPEGs are fetched; a missing or corrupt chosen
image fails preparation and never causes replacement sampling.  Raw archives,
JSON, selected images, binary panels and their hashes remain in a fresh result
directory.  A separate four-condition parent inherits the old v2 checkpoints,
regions and scientific settings, with a distinct official dataset binding.
No model, gradient, validation/test input or GPU operation is involved.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

from regionalign import config as C
from regionalign.experiments import clean_mmlu as M, vqa as D
from regionalign.runtimes import vqa as V


MODULE_SOURCE = Path("regionalign/experiments/official_okvqa.py")
SCOPE = "official_okvqa_train_extension"
QUESTIONS_URL = "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip"
ANNOTATIONS_URL = "https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip"
IMAGE_URL = "http://images.cocodataset.org/train2014/COCO_train2014_{image_id:012d}.jpg"
JSON_MEMBERS = ("OpenEnded_mscoco_train2014_questions.json", "mscoco_train2014_annotations.json")
SOURCE_URLS = (QUESTIONS_URL, ANNOTATIONS_URL)
SELECTION_RULE = "sha256(seed|task|image_id|question_id); minimum per image; global minimum; search then dev"


def _publish_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _record(path, root):
    path, root = Path(path), Path(root)
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": M.sha256(path)}


def _fetch(url, *, max_bytes, attempts=3):
    """Retry the same public object; never choose another object on failure."""
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "RegionAlign-official-input-audit/1"})
            with urllib.request.urlopen(request, timeout=40) as response:
                body = response.read(max_bytes + 1)
                if len(body) > max_bytes or not body:
                    raise ValueError(f"empty or oversized official object: {url}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != len(body):
                    raise ValueError(f"truncated official object: {url}")
                return body, {"url": url, "resolved_url": response.geturl(),
                              "http_status": response.status, "retrieved_at": M.utc_now(),
                              "headers": dict(response.headers), "attempt": attempt + 1}
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(attempt + 1)


def _train_metadata(document, rows_key):
    if (document.get("data_subtype") != "train2014"
            or document.get("data_type") != "mscoco"
            or not isinstance(document.get(rows_key), list)):
        raise ValueError(f"official {rows_key} must be mscoco train2014")


def select_metadata(questions, annotations, *, reference_answer, search_rows=V.SEARCH_ROWS,
                    dev_rows=V.DEV_ROWS, seed=V.PANEL_SEED):
    """Rank the complete training universe without testing image availability."""
    _train_metadata(questions, "questions")
    _train_metadata(annotations, "annotations")
    if any(type(x) is not int or x < 1 for x in (search_rows, dev_rows)):
        raise ValueError("panel sizes must be positive integers")
    question_by_id, annotation_by_id = {}, {}
    for records, target in ((questions["questions"], question_by_id),
                            (annotations["annotations"], annotation_by_id)):
        for row in records:
            qid, image_id = row.get("question_id"), row.get("image_id")
            if (type(qid) is not int or qid < 0 or qid in target
                    or type(image_id) is not int or image_id < 0):
                raise ValueError("invalid or duplicate official question/image ID")
            target[qid] = row
    if set(question_by_id) != set(annotation_by_id) or not question_by_id:
        raise ValueError("complete question/annotation ID universes must match")
    by_image, invalid = {}, 0
    for qid in sorted(question_by_id):
        question, annotation = question_by_id[qid], annotation_by_id[qid]
        if question["image_id"] != annotation["image_id"]:
            raise ValueError(f"question/annotation image mismatch for ID {qid}")
        text = question.get("question")
        raw_answers = annotation.get("answers")
        if not isinstance(raw_answers, list) or not raw_answers:
            raise ValueError(f"missing official answer list for ID {qid}")
        answers = [row.get("answer") if isinstance(row, dict) else None for row in raw_answers]
        if not all(isinstance(answer, str) for answer in answers):
            raise ValueError(f"non-string official answer for ID {qid}")
        reference = reference_answer(answers, "okvqa")
        if not isinstance(text, str) or not text.strip() or not reference:
            invalid += 1
            continue
        image_id = str(question["image_id"])
        rank = V._rank(seed, "okvqa", image_id, str(qid))
        row = {"sid": f"okvqa-{qid}", "question_id": str(qid), "image_id": image_id,
               "question": text, "answers": answers, "reference": reference, "rank": rank,
               "question_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "answers_sha256": hashlib.sha256("\0".join(answers).encode()).hexdigest()}
        if image_id not in by_image or rank < by_image[image_id]["rank"]:
            by_image[image_id] = row
    need = search_rows + dev_rows
    chosen = sorted(by_image.values(), key=lambda row: row["rank"])[:need]
    if len(chosen) != need:
        raise ValueError(f"need {need} distinct eligible official training images, found {len(chosen)}")
    audit = {"scope": SCOPE, "task": "okvqa", "split": "train2014", "seed": seed,
             "search_rows": search_rows, "dev_rows": dev_rows, "selection_rule": SELECTION_RULE,
             "answer_field": "annotations[].answers[].answer",
             "reference_rule": "regionalign.experiments.clean_benchmarks.reference_answer",
             "question_count": len(question_by_id), "annotation_count": len(annotation_by_id),
             "question_ids_sha256": M.object_sha256(sorted(question_by_id)),
             "invalid_reference_or_question_count": invalid, "eligible_distinct_images": len(by_image),
             "selected": [{key: row[key] for key in ("question_id", "image_id", "rank",
                            "question_sha256", "answers_sha256", "reference")} for row in chosen]}
    return {"search": chosen[:search_rows], "dev": chosen[search_rows:]}, audit


def _image_bytes_valid(payload):
    from PIL import Image
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != "JPEG" or min(image.size) < 1:
            raise ValueError("selected COCO train image is not a nonempty JPEG")
        width, height = image.size
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
    return {"width": width, "height": height, "format": "JPEG"}


def _download_selected(root, rows, workers):
    """All rows were committed beforehand. Failure aborts, without replacement."""
    root = Path(root)
    def one(row):
        image_id = int(row["image_id"])
        url = IMAGE_URL.format(image_id=image_id)
        payload, network = _fetch(url, max_bytes=20 * 1024 * 1024)
        dimensions = _image_bytes_valid(payload)
        path = root / "images" / f"COCO_train2014_{image_id:012d}.jpg"
        _publish_bytes(path, payload)
        return {"image_id": row["image_id"], **_record(path, root), **dimensions, **network}
    records, errors = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(one, row): row["image_id"] for row in rows}
        for future in as_completed(pending):
            image_id = pending[future]
            try:
                record = future.result()
                records[image_id] = record
                event = {"status": "complete", **record}
            except Exception as error:
                event = {"status": "failed", "image_id": image_id,
                         "error_type": type(error).__name__, "error": str(error)}
                errors.append(event)
            with (root / "downloads.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()
    if errors:
        M.write_json_once(root / "failure.json", {"status": "failed", "errors": errors,
                           "selected_count": len(rows), "downloaded_count": len(records),
                           "replacement_sampling": False})
        raise RuntimeError(f"{len(errors)} preselected official images failed; no replacements sampled")
    return [records[row["image_id"]] for row in rows]


def prepare_inputs(output, *, workers=8):
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer in 1..16")
    root = C.output_path(output)
    root.mkdir(parents=True, exist_ok=False)
    C.setup_environment()
    from regionalign.experiments.clean_benchmarks import reference_answer
    helper = C.PROJECT / MODULE_SOURCE
    _publish_bytes(root / "source" / "official_okvqa.py", helper.read_bytes())
    archives, documents = [], []
    for url, member in zip(SOURCE_URLS, JSON_MEMBERS):
        payload, network = _fetch(url, max_bytes=8 * 1024 * 1024)
        archive = root / "official_train_sources" / (member + ".zip")
        _publish_bytes(archive, payload)
        with zipfile.ZipFile(io.BytesIO(payload)) as container:
            if member not in container.namelist():
                raise ValueError(f"official archive lacks {member}")
            content = container.read(member)
            if len(content) > 32 * 1024 * 1024:
                raise ValueError("oversized official training JSON")
        path = archive.with_suffix("")
        _publish_bytes(path, content)
        archives.append({"archive": _record(archive, root), "json": _record(path, root), **network})
        documents.append(json.loads(content))
    panels, selection = select_metadata(*documents, reference_answer=reference_answer)
    selection["official_source_archives"] = archives
    # INV-21 / ADR-0025: selection is published before checking image availability.
    M.write_json_once(root / "selection.json", selection)
    selected = panels["search"] + panels["dev"]
    images = _download_selected(root, selected, workers)
    by_image = {record["image_id"]: record for record in images}
    panel_records = {}
    for split, rows in panels.items():
        for row in rows:
            record = by_image[row["image_id"]]
            image_path = root / record["path"]
            row.update(image={"bytes": image_path.read_bytes(), "path": str(image_path)},
                       image_sha256=record["sha256"])
        panel_records[split] = D._panel_record(root, "okvqa", split, rows)
    audit = {"schema_version": 1, "scope": SCOPE, "status": "complete",
             "development_only": True, "created_at": M.utc_now(),
             "official_sources": archives, "selection": _record(root / "selection.json", root),
             "helper_source": _record(root / "source" / "official_okvqa.py", root),
             "images": images, "image_count": len(images), "image_bytes": sum(x["bytes"] for x in images),
             "panels": panel_records, "search_bootstraps": D._bootstrap_records(panels["search"], "okvqa"),
             "replacement_sampling": False, "test_inputs_accessed": False}
    if M.sha256(helper) != audit["helper_source"]["sha256"]:
        raise RuntimeError("helper source changed during input preparation")
    M.write_json_once(root / "audit.json", audit)
    return root


def _verified_inputs(root):
    root = C.output_path(root)
    audit = M.read_json(root / "audit.json")
    if (audit.get("scope") != SCOPE or audit.get("status") != "complete"
            or audit.get("image_count") != V.SEARCH_ROWS + V.DEV_ROWS
            or audit.get("replacement_sampling") is not False
            or audit.get("test_inputs_accessed") is not False):
        raise ValueError("official training input audit is not complete")
    files = [audit["selection"], audit["helper_source"], *audit["images"],
             *audit["panels"].values()]
    for source in audit["official_sources"]:
        files.extend((source["archive"], source["json"]))
    for record in files:
        path = root / record["path"]
        if (root not in path.resolve().parents or not path.is_file()
                or path.stat().st_size != record["bytes"] or M.sha256(path) != record["sha256"]):
            raise ValueError(f"official frozen input drift: {record['path']}")
    if audit["helper_source"]["sha256"] != M.sha256(C.PROJECT / MODULE_SOURCE):
        raise ValueError("official input helper differs from the current source")
    selection = M.read_json(root / audit["selection"]["path"])
    if (selection["search_rows"] != V.SEARCH_ROWS or selection["dev_rows"] != V.DEV_ROWS
            or selection["seed"] != V.PANEL_SEED or len(selection["selected"]) != audit["image_count"]):
        raise ValueError("official selected panel protocol differs")
    if ([row["image_id"] for row in selection["selected"]]
            != [row["image_id"] for row in audit["images"]]):
        raise ValueError("official images differ from committed selection")
    return root, audit, selection


def prepare_parent(parent_run, inputs, output):
    """Publish a new four-condition official parent, preserving old geometry."""
    parent_run = C.output_path(parent_run)
    old = M.read_json(parent_run / "plan.json")
    old_hash = M.sha256(parent_run / "plan.json")
    status = M.read_json(parent_run / "status.json")
    if (old.get("schema_version") != D.SCHEMA_VERSION or old.get("scope") != D.SCOPE
            or not old.get("development_only") or old.get("splits") != ["search", "dev"]
            or status.get("plan_sha256") != old_hash):
        raise ValueError("parent must be an audited formal development VQA v2 plan")
    for record in old["sources"]:
        path = parent_run / record["snapshot"]
        if M.sha256(path) != record["sha256"]:
            raise ValueError("old parent source snapshot drift")
    root, audit, selection = _verified_inputs(inputs)
    conditions = [deepcopy(row) for row in old["conditions"] if row["task"] == "okvqa"]
    if ({(row["model"], row["format"]) for row in conditions}
            != {(model, fmt) for model in D.MODELS for fmt in D.FORMATS} or len(conditions) != 4):
        raise ValueError("parent must register the four original OK-VQA conditions")
    if any(len(row["regions"]) != 12 for row in conditions):
        raise ValueError("parent OK-VQA region registration differs")
    destination = C.output_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    sources = D._copy_sources(destination)
    helper = C.PROJECT / MODULE_SOURCE
    snapshot = Path("source") / str(MODULE_SOURCE).replace("/", "__")
    shutil.copyfile(helper, destination / snapshot)
    sources.append({"path": str(MODULE_SOURCE), "snapshot": str(snapshot),
                    "sha256": M.sha256(helper), "bytes": helper.stat().st_size})
    for record in audit["panels"].values():
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        _publish_bytes(target, (root / record["path"]).read_bytes())
    binding = {"provider": "official_okvqa_and_coco", "source_split": "train2014",
               "questions_url": QUESTIONS_URL, "annotations_url": ANNOTATIONS_URL,
               "image_url_template": IMAGE_URL, "answer_field": selection["answer_field"],
               "selection_rule": SELECTION_RULE, "selection_seed": V.PANEL_SEED,
               "audit_path": str(root / "audit.json"), "audit_sha256": M.sha256(root / "audit.json"),
               "selection_sha256": audit["selection"]["sha256"],
               "input_scope": SCOPE, "hf_repository_revision_claimed": False}
    dataset = {"status": "available", "dataset_binding": binding,
               "source_files": [{"path": str(root / record["path"]), "sha256": record["sha256"],
                                 "bytes": record["bytes"]}
                                for source in audit["official_sources"]
                                for record in (source["archive"], source["json"])],
               "image_manifest_sha256": M.object_sha256(audit["images"]),
               "selection_seed": V.PANEL_SEED, "image_disjoint": True,
               "panels": deepcopy(audit["panels"]),
               "search_bootstraps": deepcopy(audit["search_bootstraps"])}
    for row in conditions:
        row.update(availability="available", unavailable_reason=None,
                   panels=deepcopy(dataset["panels"]), search_bootstraps=deepcopy(dataset["search_bootstraps"]))
    plan = deepcopy(old)
    plan.update(created_at=M.utc_now(), sources=sources, tasks=["okvqa"], datasets={"okvqa": dataset},
                conditions=conditions, registered_available_searches=4 * old["work_items_per_condition"],
                official_okvqa_policy={"scope": SCOPE, "parent_run": str(parent_run),
                                      "parent_plan_sha256": old_hash, **binding},
                note="Independent official-train extension (ADR-0025); four OK-VQA conditions only; old v2 geometry/model identity inherited; no TextVQA rerun.")
    if old["repeat_seeds"] != list(D.REPEAT_SEEDS) or old["work_items_per_condition"] != 108:
        raise ValueError("parent bootstrap or matched-arm design differs")
    for record in sources:
        if (M.sha256(C.PROJECT / record["path"]) != record["sha256"]
                or M.sha256(destination / record["snapshot"]) != record["sha256"]):
            raise RuntimeError("source changed during official parent preparation")
    M.write_json_once(destination / "plan.json", plan)
    M.write_json_once(destination / "status.json", {
        "status": "planned", "conditions": 4, "available_conditions": 4, "unavailable_conditions": 0,
        "registered_available_searches": plan["registered_available_searches"],
        "plan_sha256": M.sha256(destination / "plan.json"), "development_only": True,
        "extension_scope": SCOPE})
    D.load_plan(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("prepare-inputs")
    inputs.add_argument("--output", required=True)
    inputs.add_argument("--workers", type=int, default=8)
    parent = commands.add_parser("prepare-parent")
    parent.add_argument("--parent-run", required=True)
    parent.add_argument("--inputs", required=True)
    parent.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-inputs":
        output = prepare_inputs(args.output, workers=args.workers)
        audit = M.read_json(output / "audit.json")
        print(json.dumps({"output": str(output), "status": audit["status"],
                          "images": audit["image_count"], "image_bytes": audit["image_bytes"]}))
    else:
        output = prepare_parent(args.parent_run, args.inputs, args.output)
        print(json.dumps({"output": str(output), "plan_sha256": M.sha256(output / "plan.json"),
                          "available_conditions": 4, "registered_items": 432}))


if __name__ == "__main__":
    main()
