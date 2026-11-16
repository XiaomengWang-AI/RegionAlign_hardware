"""Keep a frozen campaign queue alive while external evidence is pending.

The watcher reads scheduler metadata and completion markers, never datasets or
scientific artifacts.  It delegates execution to ``nohup_campaign.Controller``.
One watcher lock per queue and the controller's own lock prevent duplicate
controllers.  Blocked queues are revisited; terminal failures are not retried.
Only an entirely complete queue publishes the requested completion sentinel.

SIGTERM/SIGINT and timeout unwind this watcher without signalling, cancelling
or killing workers.  Existing controller recovery remains conservative about
interrupted launches.  Relative output names resolve under the queue state
directory; logs append beside the completion sentinel.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time

from regionalign.experiments import nohup_campaign as N

SCOPE = "regionalign_campaign_watch_v1"
FAILURES = {"failed", "partial", "interrupted", "dependency_failed"}


class WatchStopped(BaseException):
    def __init__(self, signum):
        self.signum = signum


class WatchTimeout(BaseException):
    pass


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _sources():
    return {str(Path(__file__).resolve()): N.digest(__file__),
            str(Path(N.__file__).resolve()): N.digest(N.__file__)}


def _output_path(queue, raw):
    output = Path(raw).expanduser()
    if not output.is_absolute():
        output = Path(queue["state_dir"]) / output
    output = output.resolve()
    roots = [(Path(queue["project"]) / name).resolve()
             for name in ("regionalign/results", "outputs")]
    if not any(output != root and output.is_relative_to(root) for root in roots):
        raise ValueError("watcher output must be beneath the queue project's result roots")
    if output == Path(queue["path"]) or output.suffix != ".json":
        raise ValueError("watcher output must be a distinct JSON sentinel")
    return output


def _append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **value},
                                sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _snapshot(queue):
    states, attempts = N.snapshot(queue)
    # A valid completion marker must not erase a recorded terminal failure.
    events = Path(queue["state_dir"]) / "events.jsonl"
    latest = {}
    if events.exists():
        for line in events.read_text().splitlines():
            row = json.loads(line)
            if row["queue_sha256"] != queue["sha256"]:
                raise RuntimeError("event log belongs to a different frozen queue")
            latest[row["task"]] = row["state"]
    for task, value in latest.items():
        if value in FAILURES:
            states[task] = value
    return states, attempts


def controller_active(queue):
    """Probe the original nonblocking controller lock without taking ownership."""
    path = Path(queue["state_dir"]) / ".controller.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


@contextmanager
def _signals(timeout_seconds):
    """Interrupt sleeping or an owned Controller.run; do not signal its children."""
    previous = {signum: signal.getsignal(signum)
                for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    if old_timer != (0.0, 0.0):
        raise RuntimeError("watcher requires an independent process without an interval timer")
    def stop(signum, frame):
        raise WatchStopped(signum)
    def timeout(signum, frame):
        raise WatchTimeout()
    try:
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_watch(queue_path, output, *, poll_seconds=60, timeout_hours=168):
    """Wait or invoke the existing controller, publishing only complete evidence."""
    poll_seconds = _positive(poll_seconds, "poll seconds")
    timeout_seconds = _positive(timeout_hours, "timeout hours") * 3600
    queue = N.load_queue(queue_path)
    output = _output_path(queue, output)
    log = output.with_suffix(".events.jsonl")
    identity = {"queue": queue["path"], "queue_sha256": queue["sha256"],
                "source_sha256": _sources()}
    state = Path(queue["state_dir"])
    started = time.monotonic()
    result = None
    def event(name, **fields):
        _append(log, {"scope": SCOPE, **identity, "event": name, **fields})
    with N.exclusive_lock(state / ".watcher.lock"):
        watch_identity = state / "watcher_identity.json"
        if watch_identity.exists():
            if N.read_json(watch_identity) != identity:
                raise RuntimeError("watcher identity differs from the registered queue or sources")
        else:
            N.write_once(watch_identity, identity)
        event("started", output=str(output), poll_seconds=poll_seconds,
              timeout_hours=timeout_hours)
        try:
            with _signals(timeout_seconds):
                while True:
                    if N.digest(queue["path"]) != identity["queue_sha256"] or _sources() != identity["source_sha256"]:
                        raise RuntimeError("queue or scheduling source changed while watching")
                    if (state / "queue_identity.json").exists():
                        N.verify_identity(queue)
                    states, attempts = _snapshot(queue)
                    failed = {task: value for task, value in states.items() if value in FAILURES}
                    if failed:
                        result = {"status": "failed", "states": states, "failures": failed,
                                  "exit_code": 1}
                        event("terminal_failure", **result)
                        break
                    complete = all(states.get(task["id"]) == "complete" for task in queue["tasks"])
                    active = controller_active(queue)
                    if complete and not active:
                        sentinel = {"scope": SCOPE, **identity, "status": "complete",
                                    "task_count": len(queue["tasks"]), "states": states,
                                    "completed_at": datetime.now(timezone.utc).isoformat()}
                        if output.exists():
                            previous = N.read_json(output)
                            keys = ("scope", "queue", "queue_sha256", "source_sha256",
                                    "status", "task_count", "states")
                            if any(previous.get(key) != sentinel[key] for key in keys):
                                raise FileExistsError("refusing to replace an inconsistent completion sentinel")
                            sentinel = previous
                        else:
                            N.write_once(output, sentinel)
                        event("complete", task_count=len(queue["tasks"]), output=str(output))
                        return sentinel
                    if output.exists() and not complete:
                        raise FileExistsError("an existing sentinel cannot be replaced for an incomplete queue")
                    if not active:
                        event("controller_start", states=states)
                        controller = N.Controller(queue["path"])
                        if controller.queue["sha256"] != queue["sha256"]:
                            raise RuntimeError("queue changed before controller invocation")
                        try:
                            outcome = controller.run()
                            event("controller_return", outcome=outcome)
                        except RuntimeError as error:
                            expected = f"exclusive campaign lock is held: {state / '.controller.lock'}"
                            if str(error) != expected:
                                raise
                            event("controller_lock_race")
                        # Re-read terminal/complete states before sleeping.
                        states, attempts = _snapshot(queue)
                        if any(value in FAILURES for value in states.values()) or all(
                                states.get(task["id"]) == "complete" for task in queue["tasks"]):
                            continue
                    else:
                        event("controller_active", states=states)
                    remaining = timeout_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise WatchTimeout()
                    event("waiting", states=states, attempts=attempts)
                    time.sleep(min(poll_seconds, remaining))
        except WatchStopped as stopped:
            result = {"status": "interrupted", "signal": stopped.signum,
                      "exit_code": 128 + stopped.signum}
            event("stopped", **result)
        except WatchTimeout:
            result = {"status": "timeout", "exit_code": 124}
            event("timeout", **result)
        except BaseException as error:
            event("error", error=f"{type(error).__name__}: {error}")
            raise
    return {"scope": SCOPE, **identity, **result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--queue", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--timeout-hours", type=float, default=168)
    args = parser.parse_args(argv)
    result = run_watch(args.queue, args.output, poll_seconds=args.poll_seconds,
                       timeout_hours=args.timeout_hours)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result.get("exit_code", 0)


if __name__ == "__main__":
    raise SystemExit(main())
