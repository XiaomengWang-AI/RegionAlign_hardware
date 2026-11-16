"""Bounded Slurm time-limit continuation for primary campaign stage queues.

Only the exact-checksum and official-OKVQA rounds' main queues are admitted.
No model, data panel, scientific artifact or evaluator is imported or opened.
An allocated latest attempt needs a matching Slurm job/node TIME LIMIT message,
a nonzero controller exit and a released shared worker lock. Busy allocations,
OOM, Python errors, unknown interruptions and invalid phase sentinels are never
retried. A killed worker need not have managed to publish its own exit record.

Recovery copies the entire same-plan/condition phase closure verbatim into a
fresh queue and state directory. Commands, resources, requires, completions,
task IDs, scientific paths and worker locks retain their original identities.
Existing complete phase sentinels are skipped by the original controller; the
original campaign_stage resumes missing underlying items without overwriting.
Each original task may cause at most two such continuations, independently of
other phases. Other conditions in the original pool keep running normally.

One monitor lock and shared per-condition recovery locks prevent duplicate
recovery controllers. Child campaign_watch processes own fresh logs and remain
independent of this monitor's lifetime. Immutable source/spec/queue hashes are
checked every poll. Original failures are retained even when canonical phase
coverage later completes; completion is never a scientific-success claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from regionalign.experiments import nohup_campaign as N
from regionalign.experiments import campaign_watch as W

SCOPE = "regionalign_primary_timeout_recovery_v1"
DEFAULT_BASE = Path(__file__).resolve().parents[1] / "results/campaigns/20261110T204835_083076Z_all_experiments"
ROUNDS = {"exact_checksum_round", "official_okvqa_round"}
POOLS = {"exact_checksum_round": {"main_queue_cn105_24g.json", "main_queue_cn117_24g.json", "main_queue_cn117_48g.json"},
         "official_okvqa_round": {"main_queue_cn105_24g.json", "main_queue_cn117_24g.json"}}
STAGES = {"preflight", "search", "evaluate"}
MAX_RECOVERIES = 2
TERMINAL_GROUPS = {"complete", "complete_after_recovery", "nonrecoverable", "recovery_limit", "launch_unknown", "initialization_unknown"}
TIME_LIMIT = re.compile(r"(?im)^.*(?:slurmstepd|srun)[^\n]*?\b(?:JOB|STEP)\s+(\d+)(?:\.[\w-]+)?\s+ON\s+([\w.-]+)\s+CANCELLED\b[^\n]*\bDUE TO (?:TIME LIMIT|TIMEOUT)\b[^\n]*$")
FORBIDDEN = ("out of memory", "oom-kill", "oom_kill", "cuda error", "cublas_status_alloc_failed",
             "traceback (most recent call last)", "no space left", "disk quota exceeded",
             "exceeded memory limit", *N.RESOURCE_REJECTIONS)


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_bytes(path):
    data = Path(path).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def phase(task):
    command = task["command"]
    if ("-m" not in command or command[command.index("-m") + 1] != "regionalign.experiments.campaign_memory"
            or command[command.index("-m") + 2] != "stage" or "--final" in command or "--split" in command):
        raise ValueError("only the primary campaign_memory stage CLI is admitted")
    def argument(flag):
        if command.count(flag) != 1 or command.index(flag) + 1 == len(command):
            raise ValueError(f"one {flag} argument is required")
        return command[command.index(flag) + 1]
    stage, run, condition = argument("--stage"), Path(argument("--run")).resolve(), argument("--condition")
    if stage not in STAGES:
        raise ValueError("only main preflight/search/evaluate phases are admitted")
    output = str(Path(argument("--output")).resolve())
    if (len(task["completion"]) != 1 or task["completion"][0]["path"] != output
            or task["completion"][0].get("json_equals", {}).get("status") != "complete"):
        raise ValueError("a single canonical campaign_stage completion sentinel is required")
    return {"run": str(run), "condition": condition, "stage": stage,
            "shard": argument("--shard"), "output": output}


def register_sources(paths, base):
    base = Path(base).resolve()
    if not paths or len({str(Path(p).resolve()) for p in paths}) != len(paths):
        raise ValueError("distinct primary queues are required")
    queues, groups, plans, sources = [], [], {}, {}
    for path in paths:
        path = Path(path).resolve()
        if (path.parent.parent != base or path.parent.name not in ROUNDS
                or path.name not in POOLS[path.parent.name]):
            raise ValueError("only primary main queues in the two registered rounds are allowed")
        raw = N.read_json(path)
        if not raw.get("tasks"):
            raise ValueError("an admitted primary pool cannot be an empty draft")
        queue = N.load_queue(path)
        queues.append({"path": queue["path"], "sha256": queue["sha256"], "empty": False})
        grouped = {}
        for task in queue["tasks"]:
            identity = phase(task)
            run = Path(identity["run"])
            if not run.is_relative_to(path.parent / "plans"):
                raise ValueError("scientific run must be in this primary round's plans")
            if str(run) not in plans:
                plan_data, digest = read_bytes(run / "plan.json")
                plan = json.loads(plan_data)
                status = N.read_json(run / "status.json")
                if (status.get("plan_sha256") != digest or plan.get("development_only") is not True
                        or plan.get("splits") != ["search", "dev"]
                        or plan.get("scope") not in ("llm_regionalign_dev", "vqa_regionalign_dev")):
                    raise ValueError("not an identity-bound development-only campaign plan")
                plans[str(run)] = {"path": str(run / "plan.json"), "sha256": digest}
                for record in plan.get("sources", []):
                    live = (Path(queue["project"]) / record["path"]).resolve()
                    if not live.is_relative_to(Path(queue["project"])):
                        raise ValueError("scientific source escapes project")
                    if str(live) in sources and sources[str(live)] != record["sha256"]:
                        raise ValueError("registered plans disagree about an executing source")
                    sources[str(live)] = record["sha256"]
            key = (identity["run"], identity["condition"], task["lock_key"])
            grouped.setdefault(key, []).append(task)
        for (run, condition, lock_key), tasks in grouped.items():
            ids = {task["id"] for task in tasks}
            if any(set(task["depends_on"]) - ids for task in tasks):
                raise ValueError("same-condition recovery must contain its entire dependency closure")
            identity = {"origin_queue": queue["path"], "origin_queue_sha256": queue["sha256"],
                        "run": run, "plan_sha256": plans[run]["sha256"], "condition": condition,
                        "lock_key": lock_key, "lock_dir": queue["lock_dir"], "task_ids": [task["id"] for task in tasks]}
            identity["id"] = object_hash(identity)
            groups.append(identity)
    if not groups:
        raise ValueError("no primary phase conditions are registered")
    if len({(g["run"], g["condition"]) for g in groups}) != len(groups):
        raise ValueError("one condition cannot be monitored in multiple source pools")
    for module in (N, W):
        sources[str(Path(module.__file__).resolve())] = N.digest(module.__file__)
    sources[str(Path(__file__).resolve())] = N.digest(__file__)
    wrapper = Path(__file__).with_name("campaign_stage.py").resolve()
    sources[str(wrapper)] = N.digest(wrapper)
    if any(N.digest(path) != digest for path, digest in sources.items()):
        raise RuntimeError("executing sources differ from the registered campaign")
    return {"queues": queues, "groups": groups, "plans": list(plans.values()), "sources": sources}


def journal(queue):
    path = Path(queue["state_dir"]) / "events.jsonl"
    latest = {}
    if path.exists():
        data = path.read_bytes()
        # An actively appended final line may not yet be complete; never infer failure from it.
        lines = data.splitlines(keepends=True)
        for line in lines:
            if not line.endswith(b"\n"):
                continue
            row = json.loads(line)
            if row.get("queue_sha256") != queue["sha256"]:
                raise RuntimeError("parent events do not bind the frozen queue")
            latest[row["task"]] = {"row": row, "line_sha256": hashlib.sha256(line).hexdigest()}
    return latest


class JournalReader:
    """Stream append-only events once, then consume only new bytes.

    Replacement, shrinking, same-size modification and changed old boundary
    bytes fail closed. Fixed 4 KiB prefix/end anchors keep polling bounded;
    older interior bytes retain the repository's append-only evidence contract.
    A final partial JSON line is retained until its terminating newline arrives.
    """
    def __init__(self):
        self.cache = {}
        self.payload_bytes = 0

    def read(self, queue):
        path = Path(queue["state_dir"]) / "events.jsonl"
        key = str(path)
        saved = self.cache.get(key)
        if not path.exists():
            if saved is not None:
                raise RuntimeError("previous parent journal disappeared")
            return {}
        with path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            inode = (stat.st_dev, stat.st_ino)
            if saved is None:
                saved = {"inode": inode, "offset": 0, "fragment": b"", "latest": {},
                         "prefix": b"", "suffix": b"", "mtime": stat.st_mtime_ns}
            else:
                if (saved["inode"] != inode or stat.st_size < saved["offset"]
                        or (stat.st_size == saved["offset"] and saved["mtime"] is not None
                            and stat.st_mtime_ns != saved["mtime"])):
                    raise RuntimeError("parent journal was replaced, truncated or rewritten")
                handle.seek(0)
                if handle.read(len(saved["prefix"])) != saved["prefix"]:
                    raise RuntimeError("parent journal old prefix changed")
                handle.seek(saved["offset"] - len(saved["suffix"]))
                if handle.read(len(saved["suffix"])) != saved["suffix"]:
                    raise RuntimeError("parent journal old append boundary changed")
            handle.seek(saved["offset"])
            allowed = {task["id"] for task in queue["tasks"]}
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                self.payload_bytes += len(block)
                saved["offset"] += len(block)
                lines = (saved["fragment"] + block).split(b"\n")
                saved["fragment"] = lines.pop()
                if len(saved["fragment"]) > 1024 * 1024:
                    raise RuntimeError("parent journal has an unbounded unfinished line")
                for line in lines:
                    row = json.loads(line)
                    if row.get("queue_sha256") != queue["sha256"] or row.get("task") not in allowed:
                        raise RuntimeError("parent journal entry has a different queue/task identity")
                    saved["latest"][row["task"]] = {"row": row,
                        "line_sha256": hashlib.sha256(line + b"\n").hexdigest()}
            handle.seek(0)
            saved["prefix"] = handle.read(min(saved["offset"], 4096))
            handle.seek(max(0, saved["offset"] - 4096))
            saved["suffix"] = handle.read(min(saved["offset"], 4096))
            after = os.fstat(handle.fileno())
            saved["mtime"] = after.st_mtime_ns if after.st_size == saved["offset"] else None
            self.cache[key] = saved
        return saved["latest"]


@contextmanager
def idle_worker_lock(group):
    path = Path(group["lock_dir"]) / f"{hashlib.sha256(group['lock_key'].encode()).hexdigest()}.lock"
    if not path.is_file():
        yield False
        return
    with path.open("r+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def timeout_evidence(queue, task, entry):
    """Return a bound latest allocated TIME LIMIT; return None for every other failure."""
    if not entry or entry["row"].get("state") != "failed":
        return None
    identity_path = Path(queue["state_dir"]) / "queue_identity.json"
    if (not identity_path.is_file() or N.read_json(identity_path) !=
            {"queue": queue["path"], "queue_sha256": queue["sha256"]}):
        return None
    event = entry["row"]
    attempt = event.get("attempt")
    if type(attempt) is not int or attempt < 1:
        return None
    paths = N.task_paths(queue, task, attempt)
    required = ("log", "started", "controller_exit")
    if not all(paths[key].is_file() for key in required):
        return None
    captured = {key: read_bytes(paths[key]) for key in required}
    started = json.loads(captured["started"][0])
    exited = json.loads(captured["controller_exit"][0])
    code = exited.get("exit_code")
    if (type(code) is not int or code == 0 or exited.get("worker_started") is not True
            or event.get("exit_code") != code or event.get("log") != str(paths["log"])
            or started.get("task") != task["id"] or started.get("queue_sha256") != queue["sha256"]
            or started.get("command") != task["command"]
            or str(started.get("host", "")).split(".")[0] != task["node"]
            or not re.fullmatch(r"\d+", str(started.get("job_id", "")))):
        return None
    text = captured["log"][0].decode(errors="replace")
    if any(token in text.lower() for token in FORBIDDEN):
        return None
    matches = [match.group(0) for match in TIME_LIMIT.finditer(text)
               if match.group(1) == str(started["job_id"]) and match.group(2).split(".")[0] == task["node"]]
    if not matches:
        return None
    if paths["exit"].exists():
        captured["exit"] = read_bytes(paths["exit"])
        worker = json.loads(captured["exit"][0])
        if (worker.get("task") != task["id"] or worker.get("queue_sha256") != queue["sha256"]
                or worker.get("status") != "failed" or type(worker.get("exit_code")) is not int
                or worker["exit_code"] == 0 or worker.get("reason")):
            return None
    record = {"queue": queue["path"], "queue_sha256": queue["sha256"], "task": task["id"],
              "attempt": attempt, "event": event, "event_line_sha256": entry["line_sha256"],
              "job_id": str(started["job_id"]), "node": task["node"], "timeout_lines": matches,
              "files": {key: {"path": str(paths[key]), "sha256": value[1], "bytes": len(value[0])}
                        for key, value in captured.items()}}
    record["sha256"] = object_hash(record)
    return record


def failure_details(queue, task, entry, state):
    event = entry["row"] if entry else {}
    result = {"task": task["id"], "state": state, "event": event}
    attempt = event.get("attempt")
    if type(attempt) is int and attempt > 0:
        paths = N.task_paths(queue, task, attempt)
        for key in ("controller_exit", "exit"):
            if paths[key].is_file():
                result[key] = {"path": str(paths[key]), "sha256": N.digest(paths[key]), "record": N.read_json(paths[key])}
        if paths["log"].is_file():
            data, digest = read_bytes(paths["log"])
            tail = data.decode(errors="replace")[-4000:]
            errors = re.findall(r"(?m)^(?:\w*Error|\w*Exception):[^\n]*", tail)
            result.update(log=str(paths["log"]), log_sha256=digest,
                          reason=errors[-1] if errors else tail[-1000:])
    result.setdefault("reason", "invalid/partial sentinel" if state == "partial" else
                      "failure lacks eligible allocated Slurm TIME LIMIT evidence")
    return result


def phase_state(task, plan_hashes, wrapper_sha256):
    """Check canonical sentinel identity and declared inventory, without opening its scientific files."""
    state = N.existing_state(task)
    if state != "complete":
        return state
    identity = phase(task)
    value = N.read_json(task["completion"][0]["path"])
    expected = {"schema_version": 1, "scope": "regionalign_campaign_stage_v1", "status": "complete",
                "run": identity["run"], "condition": identity["condition"], "stage": identity["stage"],
                "shard": identity["shard"], "plan_sha256": plan_hashes[identity["run"]],
                "wrapper_source": "regionalign/experiments/campaign_stage.py", "wrapper_source_sha256": wrapper_sha256}
    inventory = value.get("inventory") or {}
    if not isinstance(inventory, dict):
        return "partial"
    files = inventory.get("files")
    count = inventory.get("expected_count")
    valid_inventory = (type(count) is int and count > 0 and inventory.get("present_count") == count
                       and inventory.get("missing") == [] and isinstance(files, list) and len(files) == count)
    if valid_inventory:
        paths = [f.get("path") for f in files if isinstance(f, dict)]
        valid_inventory = (len(paths) == count and all(isinstance(path, str) for path in paths)
            and len(set(paths)) == count
            and all(isinstance(f, dict) and isinstance(f.get("path"), str)
                    and Path(f["path"]).resolve().is_relative_to(Path(identity["run"]))
                    and re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", "")))
                    and type(f.get("bytes")) is int and f["bytes"] > 0
                    and isinstance(f.get("identity"), dict)
                    and f["identity"].get("plan_sha256") == plan_hashes[identity["run"]]
                    and f["identity"].get("condition") == identity["condition"] for f in files))
    return "complete" if valid_inventory and all(value.get(key) == wanted for key, wanted in expected.items()) else "partial"


def quiescent(tasks, states, causes):
    """Unfinished descendants will become dependency_failed; unrelated work must stop first."""
    failed = {cause["task"] for cause in causes}
    dependencies = {task["id"]: task["depends_on"] for task in tasks}
    def blocked_by_failure(tid):
        return tid in failed or any(blocked_by_failure(parent) for parent in dependencies[tid])
    return all(value in N.TERMINAL or (value not in ("launching", "waiting_resources") and blocked_by_failure(tid))
               for tid, value in states.items())


class Monitor:
    def __init__(self, queues, output, *, base=DEFAULT_BASE, poll_seconds=15, timeout_hours=2160):
        self.output = Path(output).resolve()
        if not self.output.is_relative_to(Path(base).resolve()) or self.output == Path(base).resolve():
            raise ValueError("monitor output must be a new directory beneath the campaign base")
        self.poll_seconds = W._positive(poll_seconds, "poll seconds")
        self.timeout_hours = W._positive(timeout_hours, "timeout hours")
        registration = register_sources(queues, base)
        expected = {"schema_version": 1, "scope": SCOPE, "base": str(Path(base).resolve()),
                    "poll_seconds": self.poll_seconds, "timeout_hours": self.timeout_hours,
                    "max_timeout_recoveries_per_task": MAX_RECOVERIES, **registration}
        if self.output.exists():
            if N.read_json(self.output / "monitor.json") != expected:
                raise ValueError("existing monitor registration differs")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            N.write_once(self.output / "monitor.json", expected)
        self.spec, self.spec_sha = expected, N.digest(self.output / "monitor.json")
        self.plan_hashes = {str(Path(record["path"]).parent): record["sha256"] for record in self.spec["plans"]}
        self.wrapper_sha = self.spec["sources"][str(Path(__file__).with_name("campaign_stage.py").resolve())]
        self.claims, self.last = {}, {}
        self.journal_reader = JournalReader()
        self.stack = ExitStack()

    def guard(self):
        if N.digest(self.output / "monitor.json") != self.spec_sha:
            raise RuntimeError("monitor registration changed")
        for record in self.spec["queues"] + self.spec["plans"]:
            if N.digest(record["path"]) != record["sha256"]:
                raise RuntimeError("primary queue or scientific plan changed")
        if any(N.digest(path) != digest for path, digest in self.spec["sources"].items()):
            raise RuntimeError("registered executing source changed")

    def event(self, name, **fields):
        W._append(self.output / "events.jsonl", {"scope": SCOPE, "monitor_sha256": self.spec_sha,
                                                 "event": name, **fields})

    def chain(self, group):
        directory = self.output / "recoveries" / group["id"]
        records = []
        for path in sorted(directory.glob("*/registration.json")):
            record = N.read_json(path)
            if (record.get("group") != group or record.get("monitor_sha256") != self.spec_sha
                    or record.get("generation") != len(records) + 1
                    or record.get("sha256") != object_hash({k: v for k, v in record.items() if k != "sha256"})
                    or Path(record["queue"]) != path.parent / "queue.json"
                    or N.digest(record["queue"]) != record["queue_sha256"]
                    or record["previous_queue"] != (records[-1]["queue"] if records else group["origin_queue"])
                    or record["previous_queue_sha256"] != (records[-1]["queue_sha256"] if records else group["origin_queue_sha256"])):
                raise RuntimeError("recovery lineage changed")
            for evidence in record["causes"]:
                for saved in evidence["snapshots"].values():
                    if N.digest(saved["path"]) != saved["sha256"]:
                        raise RuntimeError("saved timeout evidence changed")
            records.append(record)
        return records

    def claim(self, group):
        if group["id"] in self.claims:
            return True
        path = Path(group["lock_dir"]) / f"timeout_recovery_{group['id']}.lock"
        owner_path = path.with_suffix(".owner.json")
        owner = {"scope": SCOPE, "group": group["id"], "monitor": str(self.output), "monitor_sha256": self.spec_sha}
        if owner_path.exists() and N.read_json(owner_path) != owner:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        if owner_path.exists():
            if N.read_json(owner_path) != owner:
                handle.close()
                return False
        else:
            N.write_once(owner_path, owner)
        # close-only: explicit LOCK_UN would also unlock the child's inherited FD.
        self.stack.callback(handle.close)
        self.claims[group["id"]] = handle
        return True

    def recover(self, group, head, causes, chain):
        counts = Counter(cause["task"] for record in chain for cause in record["causes"])
        if any(counts[cause["task"]] >= MAX_RECOVERIES for cause in causes):
            return "recovery_limit"
        if not self.claim(group):
            return "owned_elsewhere"
        # The monitor's coordination lock is distinct from the scientific worker lock.
        with idle_worker_lock(group) as idle:
            if not idle:
                return "waiting_worker_exit"
            self.guard()
            latest = self.journal_reader.read(head)
            for cause in causes:
                task = next(task for task in head["tasks"] if task["id"] == cause["task"])
                if timeout_evidence(head, task, latest.get(task["id"])) != cause:
                    return "waiting_evidence_stability"
            generation = len(chain) + 1
            directory = self.output / "recoveries" / group["id"] / f"{generation:04d}"
            directory.mkdir(parents=True, exist_ok=False)
            source = N.read_json(group["origin_queue"])
            wanted = set(group["task_ids"])
            selected = [copy.deepcopy(task) for task in source["tasks"] if task["id"] in wanted]
            if len(selected) != len(wanted):
                raise RuntimeError("origin task closure changed")
            queue = {**source, "tasks": selected, "max_workers": 1,
                     "state_dir": str(directory / "controller"), "lock_dir": group["lock_dir"]}
            path = directory / "queue.json"
            N.write_once(path, queue)
            loaded = N.load_queue(path)
            captured = []
            for index, cause in enumerate(causes):
                snapshots = {}
                for key, record in cause["files"].items():
                    data, digest = read_bytes(record["path"])
                    if digest != record["sha256"]:
                        raise RuntimeError("timeout evidence changed before snapshot")
                    destination = directory / "evidence" / f"{index:03d}_{key}{Path(record['path']).suffix}"
                    destination.parent.mkdir(exist_ok=True)
                    with destination.open("xb") as handle:
                        handle.write(data)
                    snapshots[key] = {"path": str(destination), "sha256": digest}
                captured.append({**cause, "snapshots": snapshots})
            record = {"scope": SCOPE, "monitor_sha256": self.spec_sha, "group": group,
                      "generation": generation, "queue": str(path), "queue_sha256": loaded["sha256"],
                      "previous_queue": head["path"], "previous_queue_sha256": head["sha256"],
                      "causes": captured, "counts": dict(counts + Counter(c["task"] for c in causes)),
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "science_commands_and_outputs": "original_condition_closure_verbatim",
                      "old_failures_retained": True}
            record["sha256"] = object_hash(record)
            N.write_once(directory / "registration.json", record)
        self.launch(record)
        self.event("recovery_created", group=group["id"], generation=generation,
                   queue=record["queue"], causes=[c["task"] for c in causes], counts=record["counts"])
        return "recovering"

    def launch(self, record):
        directory = Path(record["queue"]).parent
        queue = N.load_queue(record["queue"])
        command = ["nohup", queue["python"], "-B", "-u", "-m", "regionalign.experiments.campaign_watch",
                   "run", "--queue", record["queue"], "--output", str(directory / "complete.json"),
                   "--poll-seconds", str(self.poll_seconds), "--timeout-hours", str(self.timeout_hours)]
        N.write_once(directory / "launch_intent.json", {"queue_sha256": queue["sha256"], "command": command})
        with (directory / "nohup.log").open("x") as handle:
            process = subprocess.Popen(command, cwd=queue["project"], env=N.clean_environment(),
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(self.claims[record["group"]["id"]].fileno(),))
        N.write_once(directory / "launch.json", {"pid": process.pid, "queue_sha256": queue["sha256"],
                                                 "command": command})

    def step(self):
        self.guard()
        queues, journals, states = {}, {}, {}
        result = {}
        for group in self.spec["groups"]:
            directory = self.output / "recoveries" / group["id"]
            orphaned = [str(path) for path in sorted(directory.glob("[0-9][0-9][0-9][0-9]"))
                        if path.is_dir() and not (path / "registration.json").is_file()]
            if orphaned:
                row = {"state": "initialization_unknown", "origin_queue": group["origin_queue"],
                       "run": group["run"], "condition": group["condition"],
                       "failures": [{"reason": "incomplete recovery publication retained", "paths": orphaned}]}
                result[group["id"]] = row
                if self.last.get(group["id"]) != row:
                    self.event("condition_state", group=group["id"], **row)
                    self.last[group["id"]] = row
                continue
            chain = self.chain(group)
            head_path = chain[-1]["queue"] if chain else group["origin_queue"]
            if head_path not in queues:
                queue = N.load_queue(head_path)
                queues[head_path], journals[head_path] = queue, self.journal_reader.read(queue)
                current = {}
                for task in queue["tasks"]:
                    value = phase_state(task, self.plan_hashes, self.wrapper_sha)
                    previous = journals[head_path].get(task["id"], {}).get("row", {}).get("state", "pending")
                    current[task["id"]] = value or ("partial" if previous == "complete" else previous)
                states[head_path] = current
            head, latest, current = queues[head_path], journals[head_path], states[head_path]
            tasks = [task for task in head["tasks"] if task["id"] in group["task_ids"]]
            selected = {task["id"]: current[task["id"]] for task in tasks}
            failures, causes, awaiting = [], [], []
            if all(value == "complete" for value in selected.values()):
                state = "complete_after_recovery" if chain else "complete"
            else:
                for task in tasks:
                    value = selected[task["id"]]
                    if value in ("failed", "partial", "interrupted"):
                        entry = latest.get(task["id"])
                        attempt = entry["row"].get("attempt") if entry else None
                        if value == "failed" and type(attempt) is int and attempt > 0:
                            paths = N.task_paths(head, task, attempt)
                            if not paths["controller_exit"].is_file():
                                awaiting.append(task["id"])
                                continue
                        cause = timeout_evidence(head, task, latest.get(task["id"])) if value == "failed" else None
                        if cause:
                            causes.append(cause)
                        else:
                            failures.append(failure_details(head, task, latest.get(task["id"]), value))
                if failures:
                    state = "nonrecoverable"
                elif awaiting:
                    state = "waiting_controller_exit"
                elif causes:
                    state = self.recover(group, head, causes, chain) if quiescent(tasks, selected, causes) else "waiting_condition_quiescence"
                elif chain and not (Path(head_path).parent / "launch.json").exists():
                    state = "launch_unknown"
                elif any(value == "dependency_failed" for value in selected.values()):
                    state = "nonrecoverable"
                    failures = [{"state": "dependency_failed_without_recoverable_timeout"}]
                else:
                    state = "recovering" if chain else "monitoring"
            row = {"state": state, "origin_queue": group["origin_queue"], "head_queue": head_path,
                   "run": group["run"], "condition": group["condition"], "generation": len(chain),
                   "task_state_counts": dict(Counter(selected.values())), "failures": failures}
            result[group["id"]] = row
            if self.last.get(group["id"]) != row:
                self.event("condition_state", group=group["id"], **row)
                self.last[group["id"]] = row
        return result

    def run(self):
        started = time.monotonic()
        limit = self.timeout_hours * 3600
        with self.stack:
            self.stack.enter_context(N.exclusive_lock(self.output / ".monitor.lock"))
            self.event("started", queue_count=len(self.spec["queues"]), condition_count=len(self.spec["groups"]))
            with W._signals(limit):
                try:
                    while True:
                        states = self.step()
                        if all(row["state"] in TERMINAL_GROUPS for row in states.values()):
                            complete = all(row["state"] in ("complete", "complete_after_recovery") for row in states.values())
                            result = {"scope": SCOPE, "monitor_sha256": self.spec_sha,
                                      "status": "complete" if complete else "finished_with_errors",
                                      "states": states, "exit_code": 0 if complete else 1,
                                      "completion_meaning": "canonical stage coverage only; original failures and scientific outcomes remain separate"}
                            if (self.output / "result.json").exists():
                                if N.read_json(self.output / "result.json") != result:
                                    raise RuntimeError("existing monitor result differs from current coverage")
                            else:
                                N.write_once(self.output / "result.json", result)
                            return result
                        remaining = limit - (time.monotonic() - started)
                        if remaining <= 0:
                            raise W.WatchTimeout()
                        time.sleep(min(self.poll_seconds, remaining))
                except W.WatchStopped as stopped:
                    self.event("stopped", signal=stopped.signum)
                    return {"status": "interrupted", "exit_code": 128 + stopped.signum}
                except W.WatchTimeout:
                    self.event("timeout")
                    return {"status": "timeout", "exit_code": 124}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--queue", action="append", required=True)
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--timeout-hours", type=float, default=2160)
    args = parser.parse_args(argv)
    result = Monitor(args.queue, args.output, base=args.base, poll_seconds=args.poll_seconds,
                     timeout_hours=args.timeout_hours).run()
    print(json.dumps(result, sort_keys=True), flush=True)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
