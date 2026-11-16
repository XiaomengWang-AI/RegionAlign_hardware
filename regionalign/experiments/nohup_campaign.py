"""Run a frozen dependency queue through fresh, exclusive Slurm GPU allocations.

This module schedules commands; scientific plans and result validity stay owned
by their experiment drivers. It never chooses physical GPUs or opens datasets.
``run --dry-run`` and ``status`` are read-only. ``detach`` starts ``run`` under
nohup; local and BatchMode SSH transports both submit fresh ``srun`` jobs.

Queue schema 1 requires absolute ``project``, ``state_dir`` and a ``tasks`` list.
Optional queue fields are ``python``, ``lock_dir``, ``max_workers`` (7),
``poll_seconds`` (30), and ``aging_seconds`` (300). A task requires ``id``,
``command`` (argv), ``cwd``, ``node`` (cn104/cn105/cn117), ``resources``, and a
nonempty ``completion`` list. Optional fields: ``transport`` (local/ssh),
``priority`` (higher first), ``depends_on``, ``requires``, ``lock_key`` (shared
by all shards of a condition), and ``max_allocation_attempts`` (3). Resources
require ``cpus``, ``mem_gib``, ``walltime`` (HH:MM:SS), ``partition``, and, for
the default one GPU, ``gpu_type``; ``gpus`` is 0 or 1 and ``immediate_seconds``
defaults to 30. CPU commands receive an empty CUDA_VISIBLE_DEVICES.

File conditions are {"path": absolute_path}, optionally with ``sha256``,
``json_equals`` (top-level fields), and ``json_absent`` (top-level keys).
All conditions must pass. Completion markers already present but inconsistent
are retained as partial output and never relaunched. A launched scientific
command runs once: only explicit Slurm resource rejections before the worker
starts are retried, with a frozen finite attempt limit. Interrupted submissions
are conservatively retained instead of risking duplicate orphan workers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time


NODES = {"cn104", "cn105", "cn117"}
MODULE = "regionalign.experiments.nohup_campaign"
TERMINAL = {"complete", "failed", "partial", "interrupted", "dependency_failed"}
RESOURCE_REJECTIONS = ("resources temporarily unavailable", "requested nodes are busy",
                       "immediate execution impossible", "unable to allocate resources: node is busy")
REMOTE_EXEC = ("import json,os,sys; "
               "env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_') "
               "and k not in ('CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')}; "
               "argv=json.loads(sys.argv[1]); os.execvpe(argv[0],argv,env)")


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_once(path, value):
    """Publish a complete JSON file atomically, without replacing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _absolute(value, label):
    if not isinstance(value, str) or not value or "\0" in value or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return str(Path(value).resolve())


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return value


def _file_condition(raw):
    if not isinstance(raw, dict) or set(raw) - {"path", "sha256", "json_equals", "json_absent"}:
        raise ValueError("invalid file condition")
    condition = dict(raw, path=_absolute(raw.get("path"), "condition path"))
    if "sha256" in raw and not re.fullmatch(r"[0-9a-f]{64}", str(raw["sha256"])):
        raise ValueError("invalid file-condition SHA256")
    if not isinstance(raw.get("json_equals", {}), dict):
        raise ValueError("json_equals must be an object")
    if not isinstance(raw.get("json_absent", []), list) or any(
            not isinstance(key, str) for key in raw.get("json_absent", [])):
        raise ValueError("json_absent must be a list of keys")
    return condition


def matches(condition):
    """Inspect an output or prerequisite marker without changing it."""
    path = Path(condition["path"])
    try:
        if not path.is_file() or ("sha256" in condition and digest(path) != condition["sha256"]):
            return False
        if "json_equals" in condition or "json_absent" in condition:
            value = read_json(path)
            return (isinstance(value, dict)
                    and all(key in value and value[key] == expected
                            for key, expected in condition.get("json_equals", {}).items())
                    and all(key not in value for key in condition.get("json_absent", [])))
        return True
    except (OSError, ValueError, TypeError):
        return False


def load_queue(path):
    serialized = Path(path).read_bytes()
    raw = json.loads(serialized)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("campaign queue requires schema_version 1")
    if set(raw) - {"schema_version", "project", "state_dir", "python", "lock_dir",
                   "max_workers", "poll_seconds", "aging_seconds", "tasks"}:
        raise ValueError("unknown campaign queue field")
    queue = dict(raw)
    for key in ("project", "state_dir"):
        queue[key] = _absolute(raw.get(key), key)
    project = Path(queue["project"])
    roots = ((project / "regionalign/results").resolve(), (project / "outputs").resolve())
    for key, default in (("state_dir", queue["state_dir"]),
                         ("lock_dir", str(roots[0] / ".campaign_locks"))):
        queue[key] = _absolute(raw.get(key, default), key)
        if not any(Path(queue[key]).is_relative_to(root) and Path(queue[key]) != root for root in roots):
            raise ValueError(f"{key} must be beneath project results")
    queue["python"] = _absolute(raw.get("python", sys.executable), "python")
    queue["max_workers"] = raw.get("max_workers", 7)
    if type(queue["max_workers"]) is not int or queue["max_workers"] < 1:
        raise ValueError("max_workers must be a positive integer")
    for key, default in (("poll_seconds", 30), ("aging_seconds", 300)):
        queue[key] = _positive(raw.get(key, default), key)
    if not isinstance(raw.get("tasks"), list) or not raw["tasks"]:
        raise ValueError("tasks must be a nonempty list")
    tasks = []
    for source in raw["tasks"]:
        allowed = {"id", "command", "cwd", "node", "resources", "completion", "transport",
                   "priority", "depends_on", "requires", "lock_key", "max_allocation_attempts"}
        if not isinstance(source, dict) or set(source) - allowed:
            raise ValueError("invalid task fields")
        task = dict(source)
        if (not isinstance(task.get("id"), str) or task["id"] in (".", "..")
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", task["id"])):
            raise ValueError("task id must contain only letters, digits, _, . or -")
        command = task.get("command")
        if (not isinstance(command, list) or not command
                or any(not isinstance(part, str) or "\0" in part for part in command)):
            raise ValueError("command must be a nonempty string argv list")
        _absolute(command[0], "command executable")
        task["cwd"] = _absolute(task.get("cwd"), "cwd")
        if task.get("node") not in NODES or task.get("transport", "local") not in ("local", "ssh"):
            raise ValueError("invalid node or transport")
        task["transport"] = task.get("transport", "local")
        priority = task.get("priority", 0)
        if type(priority) not in (int, float) or not math.isfinite(priority):
            raise ValueError("priority must be finite")
        task["priority"] = priority
        task["lock_key"] = task.get("lock_key", task["id"])
        if not isinstance(task["lock_key"], str) or not task["lock_key"]:
            raise ValueError("lock_key must be a nonempty string")
        task["depends_on"] = task.get("depends_on", [])
        if not isinstance(task["depends_on"], list) or any(not isinstance(x, str) for x in task["depends_on"]):
            raise ValueError("depends_on must be a list of task IDs")
        for key in ("requires", "completion"):
            values = task.get(key, [])
            if not isinstance(values, list) or (key == "completion" and not values):
                raise ValueError("completion must be nonempty; requires/completion must be lists")
            task[key] = [_file_condition(value) for value in values]
        task["max_allocation_attempts"] = task.get("max_allocation_attempts", 3)
        if type(task["max_allocation_attempts"]) is not int or task["max_allocation_attempts"] < 1:
            raise ValueError("max_allocation_attempts must be a positive integer")
        resources = dict(task.get("resources") or {})
        if set(resources) - {"cpus", "mem_gib", "walltime", "partition", "gpu_type", "gpus", "immediate_seconds"}:
            raise ValueError("unknown resource field")
        for key in ("cpus", "mem_gib"):
            if type(resources.get(key)) is not int or resources[key] < 1:
                raise ValueError(f"resources.{key} must be a positive integer")
        for key in ("partition", "gpu_type"):
            if key in resources and not re.fullmatch(r"[A-Za-z0-9_-]+", str(resources[key])):
                raise ValueError(f"invalid {key}")
        if "partition" not in resources or not re.fullmatch(r"\d{1,3}:[0-5]\d:[0-5]\d", str(resources.get("walltime", ""))):
            raise ValueError("partition and HH:MM:SS walltime are required")
        resources.setdefault("gpus", 1)
        resources.setdefault("immediate_seconds", 30)
        if type(resources["gpus"]) is not int or resources["gpus"] not in (0, 1):
            raise ValueError("each worker may allocate zero or one GPU")
        if resources["gpus"] and "gpu_type" not in resources:
            raise ValueError("GPU tasks require gpu_type")
        if type(resources["immediate_seconds"]) is not int or resources["immediate_seconds"] < 1:
            raise ValueError("immediate_seconds must be a positive integer")
        task["resources"] = resources
        tasks.append(task)
    by_id = {task["id"]: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("duplicate task ID")
    visiting, visited = set(), set()

    def visit(task_id):
        if task_id not in by_id or task_id in visiting:
            raise ValueError("unknown dependency or dependency cycle")
        if task_id in visited:
            return
        visiting.add(task_id)
        for parent in by_id[task_id]["depends_on"]:
            visit(parent)
        visiting.remove(task_id)
        visited.add(task_id)

    for task in tasks:
        visit(task["id"])
    queue["tasks"] = tasks
    queue["path"] = str(Path(path).resolve())
    queue["sha256"] = hashlib.sha256(serialized).hexdigest()
    return queue


def existing_state(task):
    if all(matches(condition) for condition in task["completion"]):
        return "complete"
    if any(Path(condition["path"]).exists() for condition in task["completion"]):
        return "partial"
    return None


def clean_environment():
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("SLURM_") and key not in ("CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")}
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    return environment


def command_for(queue, task, attempt):
    resources = task["resources"]
    command = ["srun", f"--immediate={resources['immediate_seconds']}",
               f"--partition={resources['partition']}", f"--nodelist={task['node']}",
               "--nodes=1", "--ntasks=1", f"--cpus-per-task={resources['cpus']}",
               f"--mem={resources['mem_gib']}G", f"--time={resources['walltime']}",
               "--kill-on-bad-exit=1", "--job-name=regionalign-campaign", f"--chdir={task['cwd']}"]
    if resources["gpus"]:
        command.append(f"--gres=gpu:{resources['gpu_type']}:1")
    command += [queue["python"], "-B", "-m", MODULE, "worker", "--queue", queue["path"],
                "--task", task["id"], "--attempt", str(attempt)]
    if task["transport"] == "ssh":
        remote = [queue["python"], "-B", "-c", REMOTE_EXEC, json.dumps(command)]
        return ["ssh", "-T", "-oBatchMode=yes", "-oConnectTimeout=20", task["node"], shlex.join(remote)]
    return command


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"exclusive campaign lock is held: {path}") from error
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def task_paths(queue, task, attempt):
    stem = Path(queue["state_dir"]) / "tasks" / task["id"] / str(attempt)
    return {key: stem.with_suffix(suffix) for key, suffix in
            (("log", ".log"), ("started", ".started.json"), ("exit", ".exit.json"),
             ("controller_exit", ".controller_exit.json"))}


def verify_identity(queue):
    identity = read_json(Path(queue["state_dir"]) / "queue_identity.json")
    if identity != {"queue_sha256": queue["sha256"], "queue": queue["path"]} or digest(queue["path"]) != queue["sha256"]:
        raise RuntimeError("frozen campaign queue identity changed")


def worker(queue_path, task_id, attempt):
    """Execute one argv once, only inside its assigned node/allocation and lock."""
    queue = load_queue(queue_path)
    verify_identity(queue)
    task = next(task for task in queue["tasks"] if task["id"] == task_id)
    paths = task_paths(queue, task, attempt)
    token = hashlib.sha256(task["lock_key"].encode()).hexdigest()
    code, state, reason = 1, "failed", None
    try:
        with exclusive_lock(Path(queue["lock_dir"]) / f"{token}.lock") as lock:
            if socket.gethostname().split(".")[0] != task["node"] or not os.environ.get("SLURM_JOB_ID"):
                raise RuntimeError("worker must run on the frozen node inside a Slurm allocation")
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if task["resources"]["gpus"] and (not visible or len(visible.split(",")) != 1 or visible in ("-1", "NoDevFiles")):
                raise RuntimeError("worker requires exactly one Slurm-assigned visible GPU")
            state = existing_state(task)
            if state == "complete":
                code = 0
            elif state:
                raise RuntimeError("partial completion evidence already exists")
            else:
                if not all(matches(condition) for condition in task["requires"]):
                    raise RuntimeError("worker prerequisites no longer match")
                write_once(paths["started"], {"task": task_id, "queue_sha256": queue["sha256"],
                    "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
                    "cuda_visible_devices": visible, "command": task["command"]})
                environment = dict(os.environ)
                environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                   PYTHONDONTWRITEBYTECODE="1")
                if not task["resources"]["gpus"]:
                    environment["CUDA_VISIBLE_DEVICES"] = ""
                code = subprocess.run(task["command"], cwd=task["cwd"], env=environment,
                                      stdin=subprocess.DEVNULL, check=False,
                                      pass_fds=(lock.fileno(),)).returncode
                state = "complete" if code == 0 and existing_state(task) == "complete" else "failed"
                if code == 0 and state != "complete":
                    reason = "command exited zero without matching completion evidence"
                    code = 1
    except Exception as error:
        code, state, reason = 1, "failed", f"{type(error).__name__}: {error}"
    write_once(paths["exit"], {"task": task_id, "queue_sha256": queue["sha256"],
                               "status": state, "exit_code": code, "reason": reason})
    return code


def snapshot(queue, recover=False):
    states, attempts = {}, {}
    events = Path(queue["state_dir"]) / "events.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            row = json.loads(line)
            if row["queue_sha256"] != queue["sha256"]:
                raise RuntimeError("event log belongs to a different frozen queue")
            states[row["task"]] = row["state"]
            attempts[row["task"]] = max(attempts.get(row["task"], 0), row.get("attempt", 0))
    for task in queue["tasks"]:
        current = existing_state(task)
        old = states.get(task["id"])
        if current == "complete":
            states[task["id"]] = current
        elif old == "launching":
            paths = task_paths(queue, task, attempts[task["id"]])
            if paths["exit"].exists():
                states[task["id"]] = "failed"
            elif recover:
                states[task["id"]] = "interrupted"
        elif current:
            states[task["id"]] = current
        elif old == "complete":
            states[task["id"]] = "partial"
        states.setdefault(task["id"], "pending")
    return states, attempts


class Controller:
    def __init__(self, queue_path):
        self.queue = load_queue(queue_path)
        self.states, self.attempts = snapshot(self.queue)
        self.active, self.ready_since = {}, {}

    def event(self, task, state, **fields):
        row = {"time": datetime.now(timezone.utc).isoformat(), "task": task["id"],
               "state": state, "queue_sha256": self.queue["sha256"], **fields}
        path = Path(self.queue["state_dir"]) / "events.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.states[task["id"]] = state

    def ready(self, now):
        tasks = []
        active_locks = {value["task"]["lock_key"] for value in self.active.values()}
        for task in self.queue["tasks"]:
            tid = task["id"]
            if tid in self.active or self.states.get(tid) in TERMINAL:
                continue
            dependencies = [self.states.get(parent) for parent in task["depends_on"]]
            if any(state in TERMINAL - {"complete"} for state in dependencies):
                self.event(task, "dependency_failed")
            elif all(state == "complete" for state in dependencies) and all(matches(c) for c in task["requires"]):
                self.ready_since.setdefault(tid, now)
                if task["lock_key"] not in active_locks:
                    tasks.append(task)
        return sorted(tasks, key=lambda task: (
            -(task["priority"] + (now - self.ready_since[task["id"]]) / self.queue["aging_seconds"]), task["id"]))

    def launch(self, task):
        verify_identity(self.queue)
        attempt = self.attempts.get(task["id"], 0) + 1
        self.attempts[task["id"]] = attempt
        paths = task_paths(self.queue, task, attempt)
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        handle = paths["log"].open("x")
        self.event(task, "launching", attempt=attempt, log=str(paths["log"]))
        try:
            process = subprocess.Popen(command_for(self.queue, task, attempt),
                cwd=self.queue["project"], env=clean_environment(), stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT)
        except Exception as error:
            handle.close()
            write_once(paths["controller_exit"], {"exit_code": None, "error": str(error)})
            self.event(task, "failed", attempt=attempt, reason=f"launch failed: {error}")
            return
        self.active[task["id"]] = {"task": task, "process": process, "handle": handle,
                                    "attempt": attempt, "paths": paths}

    def finish(self, tid):
        active = self.active.pop(tid)
        task, paths, code = active["task"], active["paths"], active["process"].returncode
        active["handle"].close()
        write_once(paths["controller_exit"], {"exit_code": code, "worker_started": paths["started"].exists()})
        if code == 0 and existing_state(task) == "complete":
            state = "complete"
        elif (not paths["started"].exists() and not paths["exit"].exists()
              and not existing_state(task) and code != 0
              and active["attempt"] < task["max_allocation_attempts"]
              and any(text in paths["log"].read_text(errors="replace").lower() for text in RESOURCE_REJECTIONS)):
            state = "waiting_resources"
        else:
            state = "failed"
        self.event(task, state, attempt=active["attempt"], exit_code=code, log=str(paths["log"]))

    def run(self, dry_run=False):
        if dry_run:
            return {"dry_run": True, "queue_sha256": self.queue["sha256"],
                    "tasks": [{"id": task["id"], "state": self.states.get(task["id"], "pending"),
                               "command": command_for(self.queue, task, 1)} for task in self.queue["tasks"]]}
        state = Path(self.queue["state_dir"])
        with exclusive_lock(state / ".controller.lock"):
            identity = state / "queue_identity.json"
            if not identity.exists():
                write_once(identity, {"queue": self.queue["path"], "queue_sha256": self.queue["sha256"]})
                write_once(state / "queue_snapshot.json", read_json(self.queue["path"]))
            verify_identity(self.queue)
            self.states, self.attempts = snapshot(self.queue, recover=True)
            while True:
                verify_identity(self.queue)
                for tid in list(self.active):
                    if self.active[tid]["process"].poll() is not None:
                        self.finish(tid)
                for task in self.queue["tasks"]:
                    if task["id"] not in self.active and self.states.get(task["id"]) not in TERMINAL:
                        current = existing_state(task)
                        if current:
                            self.event(task, current)
                candidates = self.ready(time.monotonic())
                for task in candidates:
                    if len(self.active) >= self.queue["max_workers"]:
                        break
                    if task["lock_key"] not in {x["task"]["lock_key"] for x in self.active.values()}:
                        self.launch(task)
                pending = [task for task in self.queue["tasks"] if self.states.get(task["id"]) not in TERMINAL]
                if not self.active:
                    if not pending:
                        break
                    if not candidates:
                        for task in pending:
                            self.event(task, "blocked", reason="prerequisite evidence or dependency missing")
                        break
                time.sleep(self.queue["poll_seconds"])
        return {"states": self.states, "all_complete": all(self.states.get(task["id"]) == "complete" for task in self.queue["tasks"])}


def detach(queue_path):
    queue = load_queue(queue_path)
    state = Path(queue["state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    # Exclusive launch markers protect an accidentally repeated detach request.
    with (state / "nohup.log").open("x") as handle:
        process = subprocess.Popen(["nohup", queue["python"], "-B", "-u", "-m", MODULE,
                                    "run", "--queue", queue["path"]],
            cwd=queue["project"], env=clean_environment(), stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    write_once(state / "controller_launch.json", {"pid": process.pid, "queue_sha256": queue["sha256"]})
    return {"pid": process.pid, "log": str(state / "nohup.log")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "detach", "status", "worker"))
    parser.add_argument("--queue", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--attempt", type=int)
    args = parser.parse_args(argv)
    if args.command == "worker":
        if not args.task or args.attempt is None or args.attempt < 1:
            parser.error("worker requires --task and positive --attempt")
        return worker(args.queue, args.task, args.attempt)
    if args.dry_run and args.command != "run":
        parser.error("--dry-run applies only to run")
    if args.command == "detach":
        result = detach(args.queue)
    elif args.command == "status":
        queue = load_queue(args.queue)
        states, attempts = snapshot(queue)
        result = {"states": states, "attempts": attempts, "queue_sha256": queue["sha256"]}
    else:
        result = Controller(args.queue).run(args.dry_run)
    print(json.dumps(result, sort_keys=True))
    return int(result.get("all_complete") is False)


if __name__ == "__main__":
    raise SystemExit(main())
