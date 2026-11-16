"""CPU checks for campaign admission, dependency ordering and duplicate protection."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from regionalign.experiments import nohup_campaign as S


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.state = self.project / "regionalign/results/campaign"
        self.output = self.project / "regionalign/results/science"
        self.path = self.project / "queue.json"
        self.raw = {"schema_version": 1, "project": str(self.project),
                    "python": sys.executable, "state_dir": str(self.state),
                    "poll_seconds": 0.001, "tasks": [self.task("first")]}
        self.save()

    def task(self, name, **extra):
        return {"id": name, "command": [sys.executable, "-B", "-c", "pass"],
                "cwd": str(self.project), "node": "cn105", "lock_key": "one-condition",
                "resources": {"cpus": 1, "mem_gib": 1, "walltime": "00:01:00",
                              "partition": "das", "gpu_type": "rtx_3090"},
                "completion": [{"path": str(self.output / f"{name}.json"),
                                "json_equals": {"status": "complete"},
                                "json_absent": ["error"]}], **extra}

    def save(self):
        self.path.write_text(json.dumps(self.raw))

    def identity(self, queue):
        S.write_once(self.state / "queue_identity.json",
                     {"queue": queue["path"], "queue_sha256": queue["sha256"]})

    def test_queue_rejects_cycles_duplicate_ids_and_gpu_sharing(self):
        for tasks in ([self.task("x"), self.task("x")],
                      [self.task("x", depends_on=["missing"])],
                      [self.task("x", depends_on=["y"]), self.task("y", depends_on=["x"])]):
            self.raw["tasks"] = tasks
            self.save()
            with self.assertRaises(ValueError):
                S.load_queue(self.path)
        self.raw["tasks"] = [self.task("x")]
        self.raw["tasks"][0]["resources"]["gpus"] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, "zero or one GPU"):
            S.load_queue(self.path)

    def test_dry_run_does_not_create_state_or_execute(self):
        with mock.patch.object(S.subprocess, "Popen") as popen:
            result = S.Controller(self.path).run(dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertFalse(self.state.exists())
        popen.assert_not_called()

    def test_predicates_require_frozen_identity_and_no_error_key(self):
        path = self.output / "first.json"
        S.write_once(path, {"status": "complete", "error": "retained failure"})
        condition = self.raw["tasks"][0]["completion"][0]
        self.assertFalse(S.matches(condition))
        self.assertEqual(S.existing_state(S.load_queue(self.path)["tasks"][0]), "partial")
        self.assertFalse(S.matches({"path": str(path), "sha256": "0" * 64}))
        self.assertTrue(S.matches({"path": str(path), "sha256": S.digest(path)}))

    def test_completed_output_is_skipped_and_partial_is_not_retried(self):
        S.write_once(self.output / "first.json", {"status": "complete"})
        with mock.patch.object(S.subprocess, "Popen") as popen:
            self.assertTrue(S.Controller(self.path).run()["all_complete"])
        popen.assert_not_called()
        second = self.task("second")
        self.raw["tasks"] = [second]
        self.raw["state_dir"] = str(self.state.with_name("partial_campaign"))
        self.save()
        S.write_once(self.output / "second.json", {"status": "failed"})
        with mock.patch.object(S.subprocess, "Popen") as popen:
            result = S.Controller(self.path).run()
        self.assertEqual(result["states"]["second"], "partial")
        popen.assert_not_called()

    def test_ssh_argv_quotes_shell_metacharacters_and_requests_one_slurm_gpu(self):
        self.raw["tasks"][0].update(transport="ssh", command=[sys.executable, "-c", "literal $() `cmd` ' quote"])
        self.save()
        queue = S.load_queue(self.path)
        command = S.command_for(queue, queue["tasks"][0], 1)
        self.assertEqual(command[:3], ["ssh", "-T", "-oBatchMode=yes"])
        remote = shlex.split(command[-1])
        slurm = json.loads(remote[-1])
        self.assertIn("--gres=gpu:rtx_3090:1", slurm)
        self.assertNotIn("--overlap", slurm)
        self.assertIn("--task", slurm)
        self.assertIn("first", slurm)
        self.assertEqual(queue["tasks"][0]["command"][-1], "literal $() `cmd` ' quote")
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "old", "CUDA_VISIBLE_DEVICES": "3"}):
            environment = S.clean_environment()
        self.assertNotIn("SLURM_JOB_ID", environment)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)

    def test_ready_priority_ages_and_respects_dependencies(self):
        self.raw["tasks"] = [self.task("old", priority=0), self.task("new", priority=1),
                             self.task("dependent", priority=100, depends_on=["new"])]
        self.save()
        controller = S.Controller(self.path)
        controller.ready_since["old"] = 0
        ordered = controller.ready(1000)
        self.assertEqual([task["id"] for task in ordered], ["old", "new"])
        controller.states["new"] = "complete"
        self.assertEqual(controller.ready(1000)[0]["id"], "dependent")

    def test_controller_serializes_condition_and_runs_dependency_chain(self):
        self.raw["max_workers"] = 7
        self.raw["tasks"] = [self.task("preflight"),
                             self.task("search", depends_on=["preflight"]),
                             self.task("evaluate", depends_on=["search"])]
        self.save()
        launches = []

        def launch(command, **kwargs):
            self.assertNotIn("shell", kwargs)
            tid = command[command.index("--task") + 1]
            launches.append(tid)
            S.write_once(self.output / f"{tid}.json", {"status": "complete"})
            return SimpleNamespace(returncode=0, poll=lambda: 0)

        with mock.patch.object(S.subprocess, "Popen", side_effect=launch):
            result = S.Controller(self.path).run()
        self.assertEqual(launches, ["preflight", "search", "evaluate"])
        self.assertTrue(result["all_complete"])

    def test_resource_rejections_retry_only_to_frozen_limit(self):
        def reject(_command, **kwargs):
            kwargs["stdout"].write("srun: requested nodes are busy\n")
            kwargs["stdout"].flush()
            return SimpleNamespace(returncode=1, poll=lambda: 1)

        with mock.patch.object(S.subprocess, "Popen", side_effect=reject) as popen:
            controller = S.Controller(self.path)
            result = controller.run()
        self.assertEqual(popen.call_count, 3)
        self.assertEqual(result["states"]["first"], "failed")
        self.assertEqual(len(list((self.state / "tasks/first").glob("*.controller_exit.json"))), 3)
        with mock.patch.object(S.subprocess, "Popen") as popen:
            self.assertFalse(S.Controller(self.path).run()["all_complete"])
        popen.assert_not_called()

    def test_restart_does_not_duplicate_unknown_inflight_submission(self):
        controller = S.Controller(self.path)
        self.identity(controller.queue)
        controller.event(controller.queue["tasks"][0], "launching", attempt=1)
        with mock.patch.object(S.subprocess, "Popen") as popen:
            result = S.Controller(self.path).run()
        self.assertEqual(result["states"]["first"], "interrupted")
        popen.assert_not_called()

    def test_missing_external_prerequisite_blocks_without_launch(self):
        self.raw["tasks"][0]["requires"] = [{"path": str(self.project / "missing-plan.json")}]
        self.save()
        with mock.patch.object(S.subprocess, "Popen") as popen:
            result = S.Controller(self.path).run()
        self.assertEqual(result["states"]["first"], "blocked")
        popen.assert_not_called()

    def test_worker_preserves_argv_and_hides_gpu_for_cpu_commands(self):
        task = self.raw["tasks"][0]
        task["resources"]["gpus"] = 0
        literal = "literal $() `cmd` ' quote"
        target = self.output / "first.json"
        script = ("import json,os,sys; from pathlib import Path; p=Path(sys.argv[1]); "
                  "p.parent.mkdir(parents=True,exist_ok=True); "
                  "p.write_text(json.dumps({'status':'complete','literal':sys.argv[2],"
                  "'visible':os.environ.get('CUDA_VISIBLE_DEVICES')}))")
        task["command"] = [sys.executable, "-B", "-c", script, str(target), literal]
        self.save()
        queue = S.load_queue(self.path)
        self.identity(queue)
        with mock.patch.object(S.socket, "gethostname", return_value="cn105"), \
                mock.patch.dict(os.environ, {"SLURM_JOB_ID": "cpu-test", "CUDA_VISIBLE_DEVICES": "0"}):
            code = S.worker(self.path, "first", 1)
        self.assertEqual(code, 0)
        output = S.read_json(target)
        self.assertEqual(output["literal"], literal)
        self.assertEqual(output["visible"], "")
        self.assertEqual(S.read_json(S.task_paths(queue, queue["tasks"][0], 1)["exit"])["status"], "complete")

    def test_worker_refuses_outside_allocation_and_shared_lock(self):
        queue = S.load_queue(self.path)
        self.identity(queue)
        task = queue["tasks"][0]
        with mock.patch.object(S.socket, "gethostname", return_value="cn105"), \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(S.subprocess, "run") as execute:
            self.assertEqual(S.worker(self.path, "first", 1), 1)
        execute.assert_not_called()
        lock = Path(queue["lock_dir"]) / f"{S.hashlib.sha256(task['lock_key'].encode()).hexdigest()}.lock"
        with S.exclusive_lock(lock), mock.patch.object(S.subprocess, "run") as execute:
            self.assertEqual(S.worker(self.path, "first", 2), 1)
        execute.assert_not_called()

    def test_worker_refuses_multiple_visible_gpus(self):
        queue = S.load_queue(self.path)
        self.identity(queue)
        with mock.patch.object(S.socket, "gethostname", return_value="cn105"), \
                mock.patch.dict(os.environ, {"SLURM_JOB_ID": "test", "CUDA_VISIBLE_DEVICES": "0,1"}), \
                mock.patch.object(S.subprocess, "run") as execute:
            self.assertEqual(S.worker(self.path, "first", 1), 1)
        execute.assert_not_called()

    def test_detach_uses_nohup_session_and_rejects_duplicate_launch(self):
        with mock.patch.object(S.subprocess, "Popen", return_value=SimpleNamespace(pid=123)) as popen:
            result = S.detach(self.path)
        self.assertEqual(result["pid"], 123)
        command = popen.call_args.args[0]
        self.assertEqual(command[0], "nohup")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)
        with mock.patch.object(S.subprocess, "Popen") as popen:
            with self.assertRaises(FileExistsError):
                S.detach(self.path)
        popen.assert_not_called()

    def test_allocated_command_failure_is_not_resource_retried(self):
        controller = S.Controller(self.path)
        self.identity(controller.queue)
        task = controller.queue["tasks"][0]
        paths = S.task_paths(controller.queue, task, 1)
        S.write_once(paths["started"], {"job_id": "test"})
        handle = paths["log"].open("x")
        handle.write("requested nodes are busy; CUDA out of memory\n")
        controller.active["first"] = {"task": task, "paths": paths, "handle": handle,
                                      "process": SimpleNamespace(returncode=137), "attempt": 1}
        controller.finish("first")
        self.assertEqual(controller.states["first"], "failed")

    def test_frozen_queue_drift_and_atomic_replacement_are_rejected(self):
        queue = S.load_queue(self.path)
        self.identity(queue)
        self.raw["tasks"][0]["priority"] = 1
        self.save()
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            S.verify_identity(queue)
        target = self.output / "existing.json"
        S.write_once(target, {"original": True})
        with self.assertRaises(FileExistsError):
            S.write_once(target, {"original": False})
        self.assertEqual(S.read_json(target), {"original": True})


if __name__ == "__main__":
    unittest.main()
