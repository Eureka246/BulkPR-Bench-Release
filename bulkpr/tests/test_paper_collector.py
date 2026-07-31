import json

import pytest

from bulkpr.paper.collector import (
    build_collection_manifest,
    collect_manifest,
    collect_trial,
    preflight_task_fingerprints,
    validate_trial_rows,
)
from bulkpr.paper.io import write_json_atomic


def _write_trial(
    root,
    *,
    rewards,
    final_wbsr,
    fingerprint="f" * 64,
    experiment_id="a" * 16,
    exception=None,
):
    steps = []
    for index, reward in enumerate(rewards):
        name = f"batch-{index:03d}"
        verifier = root / "steps" / name / "verifier"
        verifier.mkdir(parents=True)
        steps.append(
            {
                "step_name": name,
                "verifier_result": {"rewards": {"reward": reward}},
                "exception_info": None,
            }
        )
        if index == len(rewards) - 1 and final_wbsr is not None:
            write_json_atomic(
                verifier / f"detail-{index:03d}.json",
                {
                    "schema_version": "paper-step-detail/v1",
                    "is_final_step": True,
                    "wbsr": final_wbsr,
                    "experiment_id": experiment_id,
                    "final": {
                        "score": {"merged_count": 2},
                        "relation_metrics": {
                            "metric_version": "typed_hyperedge_f1_v1",
                            "unknown_relation_count": 2,
                            "malformed_relation_count": 1,
                        },
                    },
                },
            )
    write_json_atomic(
        root / "result.json",
        {
            "task_name": "task-a",
            "task_checksum": fingerprint,
            "agent_info": {
                "name": "claude-code",
                "version": "2.1.138",
                "model_info": {"name": "model-a"},
            },
            "verifier_result": {"rewards": {"reward": sum(rewards) / len(rewards)}},
            "step_results": steps,
            "exception_info": exception,
            "agent_result": {
                "n_input_tokens": 100,
                "n_cache_tokens": 80,
                "n_output_tokens": 20,
                "cost_usd": 0.0,
            },
        },
    )
    return root


def test_collects_only_final_step_reward(tmp_path):
    trial = _write_trial(tmp_path, rewards=[1.0, 0.0], final_wbsr=0)

    row = collect_trial(trial, expected_task_fingerprint="f" * 64)

    assert row["status"] == "ok"
    assert row["wbsr"] == 0
    assert row["harbor_average_reward"] == 0.5
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 20
    assert row["relation_metric_version"] == "typed_hyperedge_f1_v1"
    assert row["unknown_relation_count"] == 2
    assert row["malformed_relation_count"] == 1


def test_collects_static_multistep_usage_when_top_level_is_null(tmp_path):
    trial = _write_trial(tmp_path, rewards=[1.0, 1.0], final_wbsr=1)
    result = json.loads((trial / "result.json").read_text())
    result["agent_result"] = None
    result["step_results"][0]["agent_result"] = {
        "n_input_tokens": 11,
        "n_cache_tokens": 3,
        "n_output_tokens": 5,
        "cost_usd": 0.1,
    }
    result["step_results"][1]["agent_result"] = {
        "n_input_tokens": 13,
        "n_cache_tokens": 4,
        "n_output_tokens": 7,
        "cost_usd": 0.2,
    }
    write_json_atomic(trial / "result.json", result)

    row = collect_trial(trial, expected_task_fingerprint="f" * 64)

    assert row["input_tokens"] == 24
    assert row["cache_tokens"] == 7
    assert row["output_tokens"] == 12
    assert row["cost_usd"] == pytest.approx(0.3)


def _write_session(dir_path, session_id):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "claude-code.txt").write_text(
        json.dumps({"type": "system", "subtype": "init", "session_id": session_id})
        + "\n"
        + json.dumps({"type": "result", "subtype": "success"})
        + "\n"
    )


def test_collect_trial_extracts_claude_code_session_ids(tmp_path):
    trial = _write_trial(tmp_path, rewards=[1.0, 1.0], final_wbsr=1)
    _write_session(trial / "steps/batch-000/agent", "sess-a")
    _write_session(trial / "steps/batch-001/agent", "sess-b")

    row = collect_trial(trial, expected_task_fingerprint="f" * 64)

    assert row["session_ids"] == ["sess-a", "sess-b"]


def test_collect_manifest_attaches_real_cost_from_session_lookup(tmp_path):
    jobs = tmp_path / "jobs"
    trial = _write_trial(jobs / "kept", rewards=[1.0, 1.0], final_wbsr=1)
    _write_session(trial / "steps/batch-000/agent", "sess-a")
    _write_session(trial / "steps/batch-001/agent", "sess-b")
    manifest = {
        "trials": [
            {
                "relative_job_path": "kept",
                "task_fingerprint": "f" * 64,
                "experiment_id": "a" * 16,
                "model": "model-a",
                "trial_index": 0,
            }
        ]
    }
    seen = {}

    def fake_lookup(session_ids):
        seen["ids"] = sorted(session_ids)
        return {
            "sess-a": {"cost_cny": 0.10, "input_tokens": 100, "output_tokens": 20, "cache_tokens": 5},
            "sess-b": {"cost_cny": 0.05, "input_tokens": 50, "output_tokens": 10, "cache_tokens": 2},
        }

    rows = collect_manifest(manifest, jobs, cost_lookup=fake_lookup)

    assert seen["ids"] == ["sess-a", "sess-b"]
    assert rows[0]["cost_cny"] == pytest.approx(0.15)
    assert rows[0]["input_tokens"] == 150
    assert rows[0]["output_tokens"] == 30
    assert rows[0]["cache_tokens"] == 7


def test_collect_manifest_missing_session_cost_leaves_cost_none(tmp_path):
    jobs = tmp_path / "jobs"
    trial = _write_trial(jobs / "kept", rewards=[1.0], final_wbsr=1)
    _write_session(trial / "steps/batch-000/agent", "sess-a")
    manifest = {
        "trials": [
            {"relative_job_path": "kept", "task_fingerprint": "f" * 64, "experiment_id": "a" * 16, "trial_index": 0}
        ]
    }

    rows = collect_manifest(manifest, jobs, cost_lookup=lambda ids: {})

    assert rows[0]["cost_cny"] is None


def test_missing_final_reward_is_infra(tmp_path):
    trial = _write_trial(tmp_path, rewards=[1.0], final_wbsr=None)

    row = collect_trial(trial, expected_task_fingerprint="f" * 64)

    assert row["status"] == "infra"
    assert row["wbsr"] is None
    assert row["infra_reason"] == "missing_final_detail"


def _mark_step_turns_exhausted(trial, step_index):
    """Modify a step so that the agent exits non-zero and the Claude stream ends with error_max_turns."""
    agent_dir = trial / "steps" / f"batch-{step_index:03d}" / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "claude-code.txt").write_text(
        json.dumps({"type": "system", "subtype": "init", "session_id": "sess-x"})
        + "\n"
        + json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True})
        + "\n"
    )
    result = json.loads((trial / "result.json").read_text())
    result["step_results"][step_index]["exception_info"] = {
        "exception_type": "NonZeroAgentExitCodeError",
        "exception_message": "Agent command failed (exit code 1)",
    }
    write_json_atomic(trial / "result.json", result)


def test_turns_exhausted_final_step_uses_verifier_score(tmp_path):
    # Turns exhausted but the final verifier gave a valid score (real smoke case):
    # record the model score as 0, not infra.
    trial = _write_trial(tmp_path, rewards=[0.0], final_wbsr=0)
    _mark_step_turns_exhausted(trial, 0)

    row = collect_trial(trial, expected_task_fingerprint="f" * 64, expected_step_count=1)

    assert row["status"] == "ok"
    assert row["infra_reason"] is None
    assert row["wbsr"] == 0
    assert row["step_rewards"] == [0.0]


def test_turns_exhausted_mid_trial_scores_zero(tmp_path):
    # Turns exhausted mid-trial (later batches never ran): record model score as 0,
    # failure_bucket=turns_exhausted.
    trial = _write_trial(tmp_path, rewards=[1.0, 0.0], final_wbsr=None)
    _mark_step_turns_exhausted(trial, 1)

    row = collect_trial(trial, expected_task_fingerprint="f" * 64, expected_step_count=3)

    assert row["status"] == "ok"
    assert row["infra_reason"] is None
    assert row["wbsr"] == 0
    assert row["failure_bucket"] == "turns_exhausted"


def test_non_turns_exhausted_step_exception_stays_infra(tmp_path):
    # A step exception without error_max_turns evidence remains infra
    # (Docker/environment crashes must not be mixed in as model 0 scores).
    trial = _write_trial(tmp_path, rewards=[0.0], final_wbsr=0)
    result = json.loads((trial / "result.json").read_text())
    result["step_results"][0]["exception_info"] = {
        "exception_type": "NonZeroAgentExitCodeError",
        "exception_message": "Agent command failed (exit code 1)",
    }
    write_json_atomic(trial / "result.json", result)

    row = collect_trial(trial, expected_task_fingerprint="f" * 64, expected_step_count=1)

    assert row["status"] == "infra"
    assert row["infra_reason"] == "step_exception"


def test_duplicate_trial_key_fails():
    row = {
        "task_name": "task-a",
        "model": "model-a",
        "trial_index": 0,
        "status": "ok",
    }
    with pytest.raises(ValueError, match="duplicate trial"):
        validate_trial_rows([row, dict(row)])


def test_wrong_task_fingerprint_fails_loudly(tmp_path):
    trial = _write_trial(tmp_path, rewards=[1.0], final_wbsr=1)
    with pytest.raises(ValueError, match="fingerprint"):
        collect_trial(trial, expected_task_fingerprint="0" * 64)


def test_wrong_experiment_identity_fails_loudly(tmp_path):
    trial = _write_trial(tmp_path, rewards=[1.0], final_wbsr=1)

    with pytest.raises(ValueError, match="experiment identity"):
        collect_trial(trial, expected_experiment_id="b" * 16)


def test_preflight_fingerprints_require_oracle_nop_agreement(tmp_path):
    oracle = _write_trial(
        tmp_path / "oracle", rewards=[1.0], final_wbsr=1, fingerprint="a" * 64
    )
    nop = _write_trial(
        tmp_path / "nop", rewards=[0.0], final_wbsr=0, fingerprint="a" * 64
    )
    manifest = {
        "trials": [
            {"task_name": "task-a", "agent": "oracle", "trial_dir": str(oracle)},
            {"task_name": "task-a", "agent": "nop", "trial_dir": str(nop)},
        ]
    }

    assert preflight_task_fingerprints(manifest) == {"task-a": "a" * 64}

    result = json.loads((nop / "result.json").read_text())
    result["task_checksum"] = "b" * 64
    write_json_atomic(nop / "result.json", result)
    with pytest.raises(ValueError, match="checksum differs"):
        preflight_task_fingerprints(manifest)


def test_collect_manifest_reads_only_frozen_relative_paths(tmp_path):
    jobs = tmp_path / "jobs"
    _write_trial(jobs / "kept", rewards=[1.0], final_wbsr=1)
    _write_trial(jobs / "ignored", rewards=[0.0], final_wbsr=0)
    manifest = {
        "trials": [
            {
                "relative_job_path": "kept",
                "task_fingerprint": "f" * 64,
                "experiment_id": "a" * 16,
                "trial_index": 0,
            }
        ]
    }

    rows = collect_manifest(manifest, jobs)

    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "a" * 16
    assert rows[0]["trial_index"] == 0


def test_collect_manifest_rejects_actual_model_mismatch(tmp_path):
    jobs = tmp_path / "jobs"
    _write_trial(jobs / "kept", rewards=[1.0], final_wbsr=1)
    manifest = {
        "trials": [
            {
                "relative_job_path": "kept",
                "task_fingerprint": "f" * 64,
                "experiment_id": "a" * 16,
                "task_name": "task-a",
                "model": "model-b",
                "trial_index": 0,
            }
        ]
    }

    with pytest.raises(ValueError, match="actual model"):
        collect_manifest(manifest, jobs)


def test_collect_manifest_rejects_actual_agent_version_mismatch(tmp_path):
    jobs = tmp_path / "jobs"
    _write_trial(jobs / "kept", rewards=[1.0], final_wbsr=1)
    manifest = {
        "trials": [
            {
                "relative_job_path": "kept",
                "task_fingerprint": "f" * 64,
                "experiment_id": "a" * 16,
                "task_name": "task-a",
                "model": "model-a",
                "agent": "claude-code",
                "agent_version": "9.9.9",
                "trial_index": 0,
            }
        ]
    }

    with pytest.raises(ValueError, match="actual agent_version"):
        collect_manifest(manifest, jobs)


def test_collection_manifest_maps_helper_state_to_frozen_matrix(tmp_path):
    run = tmp_path / "state/runs/run-1"
    jobs = tmp_path / "jobs"
    write_json_atomic(run / "meta.json", {"run_id": "run-1", "trials_total": 2})
    for suffix in ("", "__1"):
        trial_id = f"task-a__claude-code__model-a{suffix}"
        write_json_atomic(
            run / "trials" / trial_id / "meta.json",
            {
                "trial_id": trial_id,
                "task_name": "task-a",
                "model": "model-a",
                "agent": "claude-code",
                "harbor_job_name": f"run-1__{trial_id}",
            },
        )
    tasks = [
        {
            "task_name": "task-a",
            "experiment_id": "a" * 16,
            "repo_id": "attrs",
            "num_steps": 2,
            "task_fingerprint": "f" * 64,
        }
    ]
    matrix = [
        {
            "experiment_id": "a" * 16,
            "repo_id": "attrs",
            "cohort": "primary",
            "paper_status": "ready",
        }
    ]

    manifest = build_collection_manifest(
        run,
        jobs_root=jobs,
        task_entries=tasks,
        matrix_rows=matrix,
        task_tree_sha256="d" * 64,
        expected_models=["model-a"],
        expected_repeats=2,
        expected_agent="claude-code",
        expected_agent_version="2.1.138",
    )

    assert [trial["trial_index"] for trial in manifest["trials"]] == [0, 1]
    assert manifest["trials"][0]["relative_job_path"].startswith("run-1__task-a")
    assert manifest["trials"][0]["matrix"] == matrix[0]
    assert manifest["trials"][0]["task_fingerprint"] == "f" * 64
    assert manifest["task_tree_sha256"] == "d" * 64


def test_collection_manifest_rejects_wrong_task_distribution(tmp_path):
    run = tmp_path / "state/runs/run-1"
    jobs = tmp_path / "jobs"
    write_json_atomic(run / "meta.json", {"run_id": "run-1", "trials_total": 2})
    for suffix in ("", "__1"):
        trial_id = f"task-a__claude-code__model-a{suffix}"
        write_json_atomic(
            run / "trials" / trial_id / "meta.json",
            {
                "trial_id": trial_id,
                "task_name": "task-a",
                "model": "model-a",
                "agent": "claude-code",
                "harbor_job_name": f"run-1__{trial_id}",
            },
        )
    tasks = [
        {
            "task_name": name,
            "experiment_id": value * 16,
            "repo_id": "attrs",
            "num_steps": 1,
            "task_fingerprint": value * 64,
        }
        for name, value in (("task-a", "a"), ("task-b", "b"))
    ]
    matrix = [
        {
            "experiment_id": value * 16,
            "repo_id": "attrs",
            "cohort": "primary",
            "paper_status": "ready",
        }
        for value in ("a", "b")
    ]

    with pytest.raises(ValueError, match="trial inventory"):
        build_collection_manifest(
            run,
            jobs_root=jobs,
            task_entries=tasks,
            matrix_rows=matrix,
            task_tree_sha256="d" * 64,
            expected_models=["model-a"],
            expected_repeats=1,
            expected_agent="claude-code",
            expected_agent_version="2.1.138",
        )


def test_collection_manifest_rejects_matrix_row_that_drifted_from_task(tmp_path):
    run = tmp_path / "state/runs/run-1"
    jobs = tmp_path / "jobs"
    write_json_atomic(run / "meta.json", {"run_id": "run-1", "trials_total": 1})
    trial_id = "task-a__claude-code__model-a"
    write_json_atomic(
        run / "trials" / trial_id / "meta.json",
        {
            "trial_id": trial_id,
            "task_name": "task-a",
            "model": "model-a",
            "agent": "claude-code",
            "harbor_job_name": f"run-1__{trial_id}",
        },
    )
    task = {
        "task_name": "task-a",
        "experiment_id": "a" * 16,
        "repo_id": "attrs",
        "episode_id": None,
        "K": 4,
        "variant": "no_deferral",
        "prompt_condition": "generic",
        "order_name": "default",
        "num_steps": 1,
        "task_fingerprint": "f" * 64,
    }
    matrix = {
        "experiment_id": "a" * 16,
        "repo_id": "attrs",
        "episode_id": None,
        "K": 8,
        "variant": "no_deferral",
        "prompt_condition": "generic",
        "order_name": "default",
        "cohort": "primary",
        "paper_status": "ready",
    }

    with pytest.raises(ValueError, match="matrix row differs from frozen task"):
        build_collection_manifest(
            run,
            jobs_root=jobs,
            task_entries=[task],
            matrix_rows=[matrix],
            task_tree_sha256="d" * 64,
            expected_models=["model-a"],
            expected_repeats=1,
            expected_agent="claude-code",
            expected_agent_version="2.1.138",
        )
