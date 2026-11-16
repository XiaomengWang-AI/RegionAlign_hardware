"""CPU scheduling tests; no Slurm commands, GPUs or real scientific inputs."""
import fcntl
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from regionalign.experiments import campaign_watch as S
from regionalign.experiments import nohup_campaign as N


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.state = self.project / "regionalign/results/controller"
        self.marker = self.project / "regionalign/results/worker/done.json"
        self.done = self.state / "watch_done.json"
        self.path = self.project / "queue.json"
        self.raw = {"schema_version": 1, "project": str(self.project),
                    "state_dir": str(self.state), "python": sys.executable,
                    "poll_seconds": 0.001, "tasks": [{
                        "id": "one", "command": [sys.executable, "-B", "-c", "pass"],
                        "cwd": str(self.project), "node": "cn105",
                        "resources": {"gpus": 0, "cpus": 1, "mem_gib": 1,
                                      "walltime": "00:01:00", "partition": "das"},
                        "completion": [{"path": str(self.marker),
                                        "json_equals": {"status": "complete"}}]}]}
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.raw))

    def fake_controller(self, function):
        return SimpleNamespace(queue=N.load_queue(self.path), run=mock.Mock(side_effect=function))

    def complete(self):
        N.write_once(self.marker, {"status": "complete"})

    def watch(self, **kwargs):
        return S.run_watch(self.path, self.done, poll_seconds=0.001, **kwargs)

    def test_blocked_controller_is_revisited_when_external_evidence_arrives(self):
        prerequisite = self.project / "regionalign/results/prerequisite.json"
        self.raw["tasks"][0]["requires"] = [{"path": str(prerequisite)}]
        self.save()
        calls = []
        def run():
            calls.append(1)
            if not prerequisite.exists():
                return {"states": {"one": "blocked"}, "all_complete": False}
            self.complete()
            return {"states": {"one": "complete"}, "all_complete": True}
        controller = self.fake_controller(run)
        with (mock.patch.object(N, "Controller", return_value=controller),
              mock.patch.object(S.time, "sleep", side_effect=lambda _: N.write_once(prerequisite, {}))):
            result = self.watch()
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["queue_sha256"], N.digest(self.path))
        self.assertEqual(result["source_sha256"], S._sources())
        self.assertEqual(N.read_json(self.done), result)

    def test_an_existing_controller_is_waited_for_instead_of_duplicated(self):
        with N.exclusive_lock(self.state / ".controller.lock") as lock:
            def finish_external(_):
                self.complete()
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            with (mock.patch.object(N, "Controller") as controller,
                  mock.patch.object(S.time, "sleep", side_effect=finish_external)):
                result = self.watch()
            controller.assert_not_called()
        self.assertEqual(result["status"], "complete")

    def test_a_second_watcher_cannot_take_over_the_same_queue(self):
        with N.exclusive_lock(self.state / ".watcher.lock"):
            with mock.patch.object(N, "Controller") as controller:
                with self.assertRaisesRegex(RuntimeError, "exclusive campaign lock"):
                    self.watch()
                controller.assert_not_called()
        self.assertFalse(self.done.exists())

    def test_controller_lock_race_returns_to_waiting(self):
        message = f"exclusive campaign lock is held: {self.state / '.controller.lock'}"
        controller = self.fake_controller(mock.Mock(side_effect=RuntimeError(message)))
        with (mock.patch.object(N, "Controller", return_value=controller),
              mock.patch.object(S.time, "sleep", side_effect=lambda _: self.complete())):
            result = self.watch()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(controller.run.call_count, 1)

    def test_every_terminal_failure_stops_without_retry_or_completion_sentinel(self):
        for failure in sorted(S.FAILURES):
            with self.subTest(failure=failure):
                self.state = self.project / "regionalign/results" / failure
                self.raw["state_dir"] = str(self.state)
                self.done = self.state / "watch_done.json"
                self.save()
                controller = N.Controller(self.path)
                self.state.mkdir(parents=True)
                controller.event(controller.queue["tasks"][0], failure)
                with mock.patch.object(N, "Controller") as factory:
                    result = self.watch()
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["failures"], {"one": failure})
                self.assertEqual(result["exit_code"], 1)
                factory.assert_not_called()
                self.assertFalse(self.done.exists())

    def test_completion_markers_do_not_erase_a_recorded_command_failure(self):
        self.complete()
        controller = N.Controller(self.path)
        self.state.mkdir(parents=True)
        controller.event(controller.queue["tasks"][0], "failed")
        self.assertEqual(N.snapshot(controller.queue)[0]["one"], "complete")
        result = self.watch()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.done.exists())

    def test_sigterm_leaves_without_signalling_jobs_and_restores_handlers(self):
        previous = signal.getsignal(signal.SIGTERM)
        controller = self.fake_controller(lambda: signal.raise_signal(signal.SIGTERM))
        with (mock.patch.object(N, "Controller", return_value=controller),
              mock.patch("os.kill") as kill):
            result = self.watch()
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["exit_code"], 128 + signal.SIGTERM)
        kill.assert_not_called()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertFalse(self.done.exists())
        with N.exclusive_lock(self.state / ".watcher.lock"):
            pass

    def test_timeout_interrupts_waiting_and_never_publishes_success(self):
        previous = signal.getsignal(signal.SIGALRM)
        with (mock.patch.object(S, "controller_active", return_value=True),
              mock.patch.object(N, "Controller") as controller):
            result = S.run_watch(self.path, self.done, poll_seconds=1,
                                 timeout_hours=0.00001)
        controller.assert_not_called()
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["exit_code"], 124)
        self.assertFalse(self.done.exists())
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)

    def test_queue_drift_stops_before_a_new_controller_can_launch(self):
        controller = self.fake_controller(lambda: {"all_complete": False})
        def mutate(_):
            self.raw["tasks"][0]["priority"] = 10
            self.save()
        with (mock.patch.object(N, "Controller", return_value=controller),
              mock.patch.object(S.time, "sleep", side_effect=mutate)):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                self.watch()
        self.assertEqual(controller.run.call_count, 1)
        self.assertFalse(self.done.exists())

    def test_success_is_idempotent_and_logs_remain_append_only(self):
        self.complete()
        with mock.patch.object(N, "Controller") as controller:
            first = self.watch()
            digest = N.digest(self.done)
            log = self.done.with_suffix(".events.jsonl")
            original = log.read_bytes()
            second = self.watch()
        controller.assert_not_called()
        self.assertEqual(first, second)
        self.assertEqual(N.digest(self.done), digest)
        self.assertTrue(log.read_bytes().startswith(original))
        self.assertGreater(log.stat().st_size, len(original))

    def test_inconsistent_existing_output_is_never_overwritten(self):
        self.complete()
        N.write_once(self.done, {"status": "failed"})
        before = self.done.read_bytes()
        with self.assertRaises(FileExistsError):
            self.watch()
        self.assertEqual(self.done.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
