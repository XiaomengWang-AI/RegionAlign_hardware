"""CPU-only timeout qualification, immutable continuation and scheduler integration."""
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from regionalign.experiments import timeout_recovery as T


class TimeoutRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.base = self.project / "regionalign/results/campaign"
        self.round = self.base / "exact_checksum_round"
        self.run = self.round / "plans/baseline"
        self.run.mkdir(parents=True)
        self.source = self.project / "executing.py"
        self.source.write_text("# unchanged fixture source\n")
        plan = {"scope": "vqa_regionalign_dev", "development_only": True, "splits": ["search", "dev"],
                "sources": [{"path": "executing.py", "sha256": T.N.digest(self.source)}]}
        T.N.write_once(self.run / "plan.json", plan)
        self.plan_sha = T.N.digest(self.run / "plan.json")
        T.N.write_once(self.run / "status.json", {"plan_sha256": self.plan_sha})
        self.path = self.round / "main_queue_cn105_24g.json"
        self.raw = {"schema_version": 1, "project": str(self.project), "python": sys.executable,
                    "state_dir": str(self.round / "original_controller"), "lock_dir": str(self.round / "locks"),
                    "max_workers": 4, "poll_seconds": .001,
                    "tasks": [self.task("preflight", "preflight", "0/1"),
                              self.task("search0", "search", "0/12", ["preflight"]),
                              self.task("evaluate0", "evaluate", "0/12", ["search0"]),
                              self.task("search1", "search", "1/12", ["evaluate0"]),
                              self.task("evaluate1", "evaluate", "1/12", ["search1"])]}
        T.N.write_once(self.path, self.raw)
        self.queue = T.N.load_queue(self.path)
        self.output = self.base / "recovery"

    def task(self, name, stage, shard, depends=None, condition="tiny__int8__textvqa"):
        output = self.round / "phases/baseline" / condition / f"{name}.json"
        return {"id": name, "command": [sys.executable, "-B", "-m", "regionalign.experiments.campaign_memory",
                    "stage", "--run", str(self.run), "--", "--stage", stage, "--condition", condition,
                    "--shard", shard, "--output", str(output), "--device", "cuda:0"],
                "cwd": str(self.project), "node": "cn105", "transport": "local",
                "lock_key": f"phase:{self.plan_sha}:{condition}",
                "resources": {"cpus": 1, "mem_gib": 1, "walltime": "00:01:00", "partition": "das", "gpus": 0},
                "depends_on": depends or [], "completion": [{"path": str(output), "json_equals": {"status": "complete"}}]}

    def monitor(self):
        monitor = T.Monitor([self.path], self.output, base=self.base, poll_seconds=.001, timeout_hours=1)
        self.addCleanup(monitor.stack.close)
        return monitor

    def completion(self, task):
        identity = T.phase(task)
        T.N.write_once(identity["output"], {"schema_version": 1, "scope": "regionalign_campaign_stage_v1",
            "status": "complete", "run": str(self.run), "plan_sha256": self.plan_sha,
            "condition": identity["condition"], "stage": identity["stage"], "shard": identity["shard"],
            "wrapper_source": "regionalign/experiments/campaign_stage.py",
            "wrapper_source_sha256": T.N.digest(Path(T.__file__).with_name("campaign_stage.py")),
            "inventory": {"expected_count": 1, "present_count": 1, "missing": [], "files": [
                {"path": str(self.run / "artifacts" / f'{task["id"]}.json'), "sha256": "a" * 64, "bytes": 2,
                 "identity": {"plan_sha256": self.plan_sha, "condition": identity["condition"]}}]}})

    def failure(self, queue=None, tid="search0", *, log=None, allocated=True, attempt=1, job="1234",
                controller_exit=True, worker_exit=False, code=124):
        queue = queue or self.queue
        task = next(t for t in queue["tasks"] if t["id"] == tid)
        # A real phase can launch only after all of its prerequisites complete.
        by_id = {t["id"]: t for t in queue["tasks"]}
        def complete_ancestors(item):
            for parent in item["depends_on"]:
                complete_ancestors(by_id[parent])
                if not Path(by_id[parent]["completion"][0]["path"]).exists():
                    self.completion(by_id[parent])
        complete_ancestors(task)
        state = Path(queue["state_dir"])
        if not (state / "queue_identity.json").exists():
            T.N.write_once(state / "queue_identity.json", {"queue": queue["path"], "queue_sha256": queue["sha256"]})
        paths = T.N.task_paths(queue, task, attempt)
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        paths["log"].write_text(log if log is not None else f"slurmstepd: error: *** STEP {job}.0 ON cn105 CANCELLED AT 2026-11-11T01:00 DUE TO TIME LIMIT ***\n")
        if allocated:
            T.N.write_once(paths["started"], {"task": tid, "queue_sha256": queue["sha256"], "job_id": job,
                "host": "cn105", "command": task["command"]})
            lock = Path(queue["lock_dir"]) / f"{hashlib.sha256(task['lock_key'].encode()).hexdigest()}.lock"
            with T.N.exclusive_lock(lock):
                pass
        if controller_exit:
            T.N.write_once(paths["controller_exit"], {"exit_code": code, "worker_started": allocated})
        if worker_exit:
            T.N.write_once(paths["exit"], {"task": tid, "queue_sha256": queue["sha256"], "status": "failed",
                                          "exit_code": code, "reason": None})
        T.W._append(state / "events.jsonl", {"task": tid, "state": "failed", "attempt": attempt,
            "exit_code": code, "log": str(paths["log"]), "queue_sha256": queue["sha256"]})
        return task, paths

    def launch_stub(self, record):
        T.N.write_once(Path(record["queue"]).parent / "launch.json", {"pid": 456, "queue_sha256": record["queue_sha256"]})

    def test_strict_latest_allocated_slurm_job_timeout_not_busy_qos_oom_or_science(self):
        task, paths = self.failure()
        entry = T.journal(self.queue)[task["id"]]
        self.assertIsNotNone(T.timeout_evidence(self.queue, task, entry))
        for log in ("srun: Requested nodes are busy\n", "srun: QOS user's size and/or time limits\n",
                    "Traceback (most recent call last):\nRuntimeError: preflight failed function preservation for local\n",
                    "slurmstepd: error: *** JOB 9999 ON cn105 CANCELLED DUE TO TIME LIMIT ***\n",
                    "slurmstepd: error: *** JOB 1234 ON cn117 CANCELLED DUE TO TIME LIMIT ***\n",
                    "slurmstepd: error: *** JOB 1234 ON cn105 CANCELLED BY 42 ***\n"):
            with self.subTest(log=log):
                paths["log"].write_text(log)
                self.assertIsNone(T.timeout_evidence(self.queue, task, entry))
        paths["log"].write_text("slurmstepd: *** JOB 1234 ON cn105 CANCELLED DUE TO TIME LIMIT ***\nCUDA out of memory\n")
        self.assertIsNone(T.timeout_evidence(self.queue, task, entry))

    def test_unallocated_or_not_yet_exited_task_cannot_recover(self):
        self.failure(allocated=False)
        m = self.monitor()
        with mock.patch.object(m, "launch") as launch:
            states = m.step()
        self.assertEqual(next(iter(states.values()))["state"], "nonrecoverable")
        launch.assert_not_called()

    def test_waits_for_controller_exit_and_shared_scientific_process_lock(self):
        task, paths = self.failure(controller_exit=False)
        m = self.monitor()
        self.assertEqual(next(iter(m.step().values()))["state"], "waiting_controller_exit")
        T.N.write_once(paths["controller_exit"], {"exit_code": 124, "worker_started": True})
        group = m.spec["groups"][0]
        lock = Path(group["lock_dir"]) / f"{hashlib.sha256(group['lock_key'].encode()).hexdigest()}.lock"
        with T.N.exclusive_lock(lock), mock.patch.object(m, "launch") as launch:
            self.assertEqual(next(iter(m.step().values()))["state"], "waiting_worker_exit")
            launch.assert_not_called()
        with mock.patch.object(m, "launch", side_effect=self.launch_stub):
            self.assertEqual(next(iter(m.step().values()))["state"], "recovering")

    def test_full_original_phase_closure_resumes_existing_items_via_existing_controller(self):
        self.completion(self.queue["tasks"][0])
        artifact = self.run / "artifacts/partial.json"
        T.N.write_once(artifact, {"existing_item": "preserve bytes"})
        before_artifact, before_queue = artifact.read_bytes(), self.path.read_bytes()
        self.failure()
        m = self.monitor()
        with mock.patch.object(m, "launch", side_effect=self.launch_stub):
            m.step()
        chain = m.chain(m.spec["groups"][0])
        self.assertEqual(len(chain), 1)
        recovered = T.N.read_json(chain[0]["queue"])
        self.assertEqual(recovered["tasks"], self.raw["tasks"])
        self.assertEqual(recovered["lock_dir"], self.raw["lock_dir"])
        self.assertNotEqual(recovered["state_dir"], self.raw["state_dir"])
        launched = []
        def execute(command, **kwargs):
            tid = command[command.index("--task") + 1]
            task = next(t for t in T.N.load_queue(chain[0]["queue"])["tasks"] if t["id"] == tid)
            launched.append(tid)
            self.completion(task)
            return SimpleNamespace(returncode=0, poll=lambda: 0)
        with mock.patch.object(T.N.subprocess, "Popen", side_effect=execute):
            self.assertTrue(T.N.Controller(chain[0]["queue"]).run()["all_complete"])
        self.assertEqual(launched, ["search0", "evaluate0", "search1", "evaluate1"])
        state = next(iter(m.step().values()))
        self.assertEqual(state["state"], "complete_after_recovery")
        self.assertEqual(artifact.read_bytes(), before_artifact)
        self.assertEqual(self.path.read_bytes(), before_queue)
        self.assertEqual(T.journal(self.queue)["search0"]["row"]["state"], "failed")

    def test_recovery_limit_is_per_original_phase_not_entire_condition(self):
        self.failure(tid="search0")
        m = self.monitor()
        group = m.spec["groups"][0]
        with mock.patch.object(m, "launch", side_effect=self.launch_stub):
            m.step()
            head = T.N.load_queue(m.chain(group)[-1]["queue"])
            self.completion(next(t for t in head["tasks"] if t["id"] == "search0"))
            self.failure(head, "search1", job="2345")
            m.step()
            self.assertEqual(m.chain(group)[-1]["counts"], {"search0": 1, "search1": 1})
            head = T.N.load_queue(m.chain(group)[-1]["queue"])
            self.failure(head, "search1", job="3456")
            m.step()
            self.assertEqual(len(m.chain(group)), 3)
            head = T.N.load_queue(m.chain(group)[-1]["queue"])
            self.failure(head, "search1", job="4567")
            result = next(iter(m.step().values()))
            self.assertEqual(result["state"], "recovery_limit")
            self.assertEqual(len(m.chain(group)), 3)

    def test_function_preservation_failure_is_explicit_nonrecoverable(self):
        self.failure(tid="preflight", log="Traceback (most recent call last):\nRuntimeError: preflight failed function preservation for local\n", code=1)
        m = self.monitor()
        with mock.patch.object(m, "launch") as launch:
            row = next(iter(m.step().values()))
        self.assertEqual(row["state"], "nonrecoverable")
        self.assertIn("function preservation for local", row["failures"][0]["reason"])
        launch.assert_not_called()

    def test_partial_wrong_plan_sentinel_is_never_resumed_or_overwritten(self):
        self.failure()
        path = Path(self.queue["tasks"][2]["completion"][0]["path"])
        T.N.write_once(path, {"status": "complete", "plan_sha256": "wrong"})
        before = path.read_bytes()
        m = self.monitor()
        with mock.patch.object(m, "launch") as launch:
            row = next(iter(m.step().values()))
        self.assertEqual(row["state"], "nonrecoverable")
        self.assertEqual(path.read_bytes(), before)
        launch.assert_not_called()

    def test_malformed_declared_inventory_is_partial_without_crashing_monitor(self):
        task = self.queue["tasks"][0]
        self.completion(task)
        path = Path(task["completion"][0]["path"])
        original = T.N.read_json(path)
        wrapper = T.N.digest(Path(T.__file__).with_name("campaign_stage.py"))
        for inventory in ([], ["wrong"], {"expected_count": 1, "present_count": 1, "missing": [],
                                         "files": [{"path": {"unhashable": True}}]}):
            modified = {**original, "inventory": inventory}
            path.write_text(json.dumps(modified))
            self.assertEqual(T.phase_state(task, {str(self.run): self.plan_sha}, wrapper), "partial")

    def test_source_and_queue_hash_guards_and_no_duplicate_monitor_registration(self):
        m = self.monitor()
        self.source.write_text("# drift\n")
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            m.step()
        self.source.write_text("# unchanged fixture source\n")
        raw = copy.deepcopy(self.raw); raw["max_workers"] = 2
        self.path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(RuntimeError, "queue or scientific plan"):
            m.step()
        with self.assertRaises(ValueError):
            self.monitor()

    def test_nonprimary_final_transfer_old_round_and_empty_draft_are_rejected(self):
        for path in (self.base / "main_queue_v2.json", self.round / "followup_queue.json",
                     self.base / "official_okvqa_round/main_queue_cn117_48g.json"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "primary main"):
                T.register_sources([path], self.base)
        raw = copy.deepcopy(self.raw)
        raw["tasks"][0]["command"].append("--final")
        self.path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "primary campaign_memory"):
            T.register_sources([self.path], self.base)

    def test_single_condition_owner_and_nohup_child_inherits_coordination_lock(self):
        self.failure()
        m = self.monitor()
        with mock.patch.object(T.subprocess, "Popen", return_value=SimpleNamespace(pid=123)) as launch:
            m.step()
        kwargs = launch.call_args.kwargs
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(len(kwargs["pass_fds"]), 1)
        self.assertEqual(launch.call_args.args[0][0], "nohup")
        group = m.spec["groups"][0]
        claim = Path(group["lock_dir"]) / f"timeout_recovery_{group['id']}.lock"
        with self.assertRaisesRegex(RuntimeError, "exclusive campaign lock"):
            with T.N.exclusive_lock(claim):
                pass
        with mock.patch.object(T.subprocess, "Popen") as launch:
            m.step()
        launch.assert_not_called()

    def test_new_monitor_directory_cannot_reset_existing_condition_recovery_limit(self):
        self.failure()
        first = self.monitor()
        with mock.patch.object(first, "launch", side_effect=self.launch_stub):
            first.step()
        first.stack.close()
        second = T.Monitor([self.path], self.base / "another_recovery", base=self.base)
        self.addCleanup(second.stack.close)
        with mock.patch.object(second, "launch") as launch:
            state = next(iter(second.step().values()))
        self.assertEqual(state["state"], "owned_elsewhere")
        launch.assert_not_called()

    def test_import_opens_no_torch_model_runtime(self):
        script = "import sys;import regionalign.experiments.timeout_recovery;print('torch' in sys.modules)"
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "False")

    def test_incremental_journal_reads_only_append_and_preserves_partial_line(self):
        self.failure()
        reader = T.JournalReader()
        original = reader.read(self.queue)
        consumed = reader.payload_bytes
        self.assertIn("search0", original)
        reader.read(self.queue)
        self.assertEqual(reader.payload_bytes, consumed)
        path = Path(self.queue["state_dir"]) / "events.jsonl"
        row = {"task": "evaluate0", "state": "dependency_failed", "queue_sha256": self.queue["sha256"]}
        serialized = json.dumps(row).encode()
        with path.open("ab") as handle:
            handle.write(serialized)
        self.assertNotIn("evaluate0", reader.read(self.queue))
        with path.open("ab") as handle:
            handle.write(b"\n")
        self.assertEqual(reader.read(self.queue)["evaluate0"]["row"], row)
        self.assertEqual(reader.payload_bytes, consumed + len(serialized) + 1)

    def test_incremental_journal_rejects_truncation_replacement_and_rewrite(self):
        self.failure()
        path = Path(self.queue["state_dir"]) / "events.jsonl"
        original = path.read_bytes()
        for mode in ("truncate", "replace", "rewrite", "rewrite_and_append"):
            path.write_bytes(original)
            reader = T.JournalReader()
            reader.read(self.queue)
            if mode == "truncate":
                path.write_bytes(original[:10])
            elif mode == "replace":
                alternate = path.with_suffix(".replacement")
                alternate.write_bytes(original)
                os.replace(alternate, path)
            elif mode == "rewrite":
                path.write_bytes(original.replace(b"failed", b"FAILED"))
            else:
                path.write_bytes(original.replace(b"failed", b"FAILED") + b"{}\n")
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, "journal"):
                reader.read(self.queue)

    def test_real_cpu_child_keeps_coordination_lock_after_monitor_closes_its_fd(self):
        m = self.monitor()
        group = m.spec["groups"][0]
        self.assertTrue(m.claim(group))
        handle = m.claims[group["id"]]
        child = subprocess.Popen([sys.executable, "-B", "-c", "import time;time.sleep(.3)"],
                                 pass_fds=(handle.fileno(),))
        self.addCleanup(lambda: child.wait(timeout=5))
        m.stack.close()
        claim = Path(group["lock_dir"]) / f"timeout_recovery_{group['id']}.lock"
        with self.assertRaisesRegex(RuntimeError, "exclusive campaign lock"):
            with T.N.exclusive_lock(claim):
                pass
        child.wait(timeout=5)
        with T.N.exclusive_lock(claim):
            pass

    def test_interrupted_recovery_publication_is_explicit_and_never_overwritten(self):
        m = self.monitor()
        directory = self.output / "recoveries" / m.spec["groups"][0]["id"] / "0001"
        directory.mkdir(parents=True)
        (directory / "queue.json").write_text("incomplete retained fixture")
        with mock.patch.object(m, "launch") as launch:
            row = next(iter(m.step().values()))
        self.assertEqual(row["state"], "initialization_unknown")
        self.assertEqual((directory / "queue.json").read_text(), "incomplete retained fixture")
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
