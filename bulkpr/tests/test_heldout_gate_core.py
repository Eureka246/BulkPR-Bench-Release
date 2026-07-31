"""Shared gate_core tests: unified dict return / flock concurrency / confirm pre-registration
allow-listing / fingerprint closure including gate_core itself.
Shared by all three adapters; py adapter is injected via gate_py.PY_ADAPTER."""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HELDOUT = os.path.join(ROOT, "bulkpr", "heldout")
sys.path.insert(0, HELDOUT)

import gate_core as gc
import gate_py as gp


def _params(tmp_path, **kw):
    return {"episode_id": "core-test", "transcript_path": str(tmp_path / "t.json"),
            "diff_dir": None, **kw}


FAKE_ADAPTER = {
    "classify": lambda rc, rep, wit, sigs: ("GREEN", None, None),
    "run_suite": lambda params, scope, ids: (0, {}, False),
    "evidence": lambda rep, wit: {},
    "fingerprint_inputs": lambda params: [],
    "fingerprint_code_files": [],
    "expected_red_signatures": lambda params, ids: (),
    "protocol_manifest": {"gate_protocol_version": "fake-1"},
}


# ---- Unified return shape (interface pin test) ----
def test_gate_returns_unified_dict(tmp_path):
    g = gc.make_gate(_params(tmp_path), FAKE_ADAPTER,
                     raw=lambda i, h, s: {"result": "GREEN", "failure_stage": None,
                                          "reason": None, "evidence": {}})
    rec = g(["P1"])
    assert isinstance(rec, dict)
    for key in ("result", "failure_stage", "reason", "evidence"):
        assert key in rec
    assert rec["result"] == "GREEN"


def test_cached_hit_also_returns_dict(tmp_path):
    calls = []

    def raw(i, h, s):
        calls.append(1)
        return {"result": "RED", "failure_stage": "assertion", "reason": "x",
                "evidence": {}}

    g = gc.make_gate(_params(tmp_path), FAKE_ADAPTER, raw=raw)
    r1 = g(["P1"])
    r2 = g(["P1"])
    assert len(calls) == 1
    assert r1["result"] == r2["result"] == "RED"
    assert "seconds" in r2 and "ts" in r2       # cache hit returns a complete record


def test_infra_retry_then_fail_loud(tmp_path):
    seq = [{"result": "INFRA", "reason": "flake", "failure_stage": "infra",
            "evidence": None},
           {"result": "GREEN", "failure_stage": None, "reason": None,
            "evidence": {}}]
    calls = []

    def raw(i, h, s):
        calls.append(1)
        return dict(seq[min(len(calls) - 1, 1)])

    g = gc.make_gate(_params(tmp_path), FAKE_ADAPTER, raw=raw)
    assert g(["P1"])["result"] == "GREEN" and len(calls) == 2

    g2 = gc.make_gate({**_params(tmp_path), "transcript_path":
                       str(tmp_path / "t2.json")}, FAKE_ADAPTER,
                      raw=lambda i, h, s: {"result": "INFRA", "reason": "x",
                                           "failure_stage": "infra",
                                           "evidence": None})
    with pytest.raises(RuntimeError, match="INFRA twice"):
        g2(["P1"])


def test_bypass_cache_not_recorded(tmp_path):
    calls = []

    def raw(i, h, s):
        calls.append(1)
        return {"result": "GREEN", "failure_stage": None, "reason": None,
                "evidence": {}}

    g = gc.make_gate(_params(tmp_path), FAKE_ADAPTER, raw=raw)
    g(["P1"])
    g(["P1"], bypass_cache=True)
    assert len(calls) == 2
    assert len(g.transcript["calls"]) == 1


# ---- flock: concurrent writes from multiple processes / instances do not overwrite each other ----
def test_concurrent_gates_both_keys_survive(tmp_path):
    p = _params(tmp_path)

    def mk(result_key):
        def raw(i, h, s):
            time.sleep(0.05)     # create an overlapping write window
            return {"result": "GREEN", "failure_stage": None,
                    "reason": result_key, "evidence": {}}
        return gc.make_gate(p, FAKE_ADAPTER, raw=raw)

    g1, g2 = mk("one"), mk("two")
    t1 = threading.Thread(target=lambda: g1(["A"]))
    t2 = threading.Thread(target=lambda: g2(["B"]))
    t1.start(); t2.start(); t1.join(); t2.join()
    t = json.load(open(p["transcript_path"]))
    keys = set(t["calls"])
    assert any("ids=A" in k for k in keys) and any("ids=B" in k for k in keys)


# ---- transcript fingerprint semantics ----
def test_transcript_fingerprint_mismatch_rejected(tmp_path):
    p = _params(tmp_path)
    json.dump({"episode_id": "core-test", "truth_fingerprint": "OLD", "calls": {}},
              open(p["transcript_path"], "w"))
    with pytest.raises(RuntimeError, match="truth_fingerprint"):
        gc.make_gate(p, FAKE_ADAPTER,
                     raw=lambda i, h, s: {"result": "GREEN", "failure_stage": None,
                                          "reason": None, "evidence": {}})


# ---- Fingerprint closure: gate_core's own bytes must contribute to cache invalidation ----
def test_fingerprint_covers_gate_core_itself(tmp_path, monkeypatch):
    diffs = tmp_path / "diffs"
    diffs.mkdir()
    (diffs / "p1.diff").write_text("fake\n")
    params = _params(tmp_path, diff_dir=str(diffs))
    fp1 = gc.truth_fingerprint(params, FAKE_ADAPTER)
    # Tamper with the gate_core source hash input: monkeypatch the hash function to return a
    # different value for gate_core.py
    orig = gc._ast_or_bytes_hash

    def tampered(path):
        if os.path.basename(path) == "gate_core.py":
            return "0" * 64
        return orig(path)

    monkeypatch.setattr(gc, "_ast_or_bytes_hash", tampered)
    assert gc.truth_fingerprint(params, FAKE_ADAPTER) != fp1


def test_fingerprint_covers_adapter_code_files(tmp_path):
    diffs = tmp_path / "diffs"
    diffs.mkdir()
    (diffs / "p1.diff").write_text("fake\n")
    extra = tmp_path / "helper.go"
    extra.write_text("package main\n")
    params = _params(tmp_path, diff_dir=str(diffs))
    ad = {**FAKE_ADAPTER, "fingerprint_code_files": [str(extra)]}
    fp1 = gc.truth_fingerprint(params, ad)
    extra.write_text("package main // changed\n")
    assert gc.truth_fingerprint(params, ad) != fp1


# ---- confirm tier verdict (no new signatures allowed + pre-registration allow-listing) ----
BASE_SIGS = [["a.py", "compile", "bad syntax"]]


def test_confirm_no_new_ok():
    r = gc.confirm_verdict(BASE_SIGS, BASE_SIGS)
    assert r["ok"] and not r["new_signatures"]


def test_confirm_baseline_disappeared_ok():
    r = gc.confirm_verdict(BASE_SIGS, [])
    assert r["ok"]         # an error disappearing from the baseline should not cause a failure


def test_confirm_new_sig_blocked():
    cur = BASE_SIGS + [["b.go", "compile", "undefined: X"]]
    r = gc.confirm_verdict(BASE_SIGS, cur)
    assert not r["ok"]
    assert r["blocked"] == [["b.go", "compile", "undefined: X"]]


def test_confirm_prereg_allows_expected_build_red():
    cur = BASE_SIGS + [["b.go", "compile", "undefined: X"]]
    allow = [{"state_id": "confirm:singleton:a01", "red_channel": "build",
              "signatures": [["b.go", "compile", "undefined: X"]]}]
    r = gc.confirm_verdict(BASE_SIGS, cur, prereg_allow=allow,
                           state_id="confirm:singleton:a01")
    assert r["ok"] and r["allowed"] == [["b.go", "compile", "undefined: X"]]
    # a different state_id is not allowed
    r2 = gc.confirm_verdict(BASE_SIGS, cur, prereg_allow=allow,
                            state_id="confirm:singleton:a02")
    assert not r2["ok"]


def test_confirm_prereg_channel_machine_check():
    with pytest.raises(ValueError, match="build/typecheck"):
        gc.confirm_verdict(BASE_SIGS, BASE_SIGS,
                           prereg_allow=[{"state_id": "s", "red_channel":
                                          "assertion", "signatures": []}])


# ---- py adapter through the full gate_core pipeline ----
def test_py_adapter_shape():
    for key in ("classify", "run_suite", "evidence", "fingerprint_inputs",
                "fingerprint_code_files", "expected_red_signatures",
                "protocol_manifest"):
        assert key in gp.PY_ADAPTER
    # py adapter has no compile-type RED channel: expected_red_signatures is always empty
    assert gp.PY_ADAPTER["expected_red_signatures"]({}, ["P1"]) == ()


def test_py_classify_accepts_signature_param():
    # classify four-parameter contract; py accepts sigs but ignores them, verdict table unchanged
    v, stage, reason = gp.classify(0, None, [], ())
    assert v == "INFRA"


# ---- End-to-end: py adapter runs against a real mini repo (gate() dict shape) ----
@pytest.fixture()
def mini(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def val():\n    return 1\n")
    (repo / "test_mod.py").write_text(
        "from mod import val\n\ndef test_val():\n    assert val() == 1\n")

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t",
                               "-c", "user.email=t@t", *a],
                              capture_output=True, text=True, check=True)

    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD").stdout.strip()
    diffs = tmp_path / "diffs"
    diffs.mkdir()
    lines = ["def test_ok():", "    assert True"]
    (diffs / "ok.diff").write_text(
        "diff --git a/test_ok.py b/test_ok.py\nnew file mode 100644\n"
        "--- /dev/null\n+++ b/test_ok.py\n"
        f"@@ -0,0 +1,{len(lines)} @@\n" + "".join(f"+{l}\n" for l in lines))
    return {"episode_id": "mini", "repo_path": str(repo), "base_commit": base,
            "diff_dir": str(diffs), "report_tmpdir": str(tmp_path),
            "pytest_bin": os.path.join(os.path.dirname(sys.executable), "pytest"),
            "truth_scope_testpaths": None, "timeout_seconds": 30,
            "transcript_path": str(tmp_path / "transcript.json"),
            "witnesses": {}, "pytest_config_files": ()}


def test_py_gate_e2e_dict_and_transcript(mini):
    g = gp.make_gate(mini)
    rec = g(["ok"])
    assert rec["result"] == "GREEN"
    assert "evidence" in rec and rec["evidence"]["failed"] == 0
    # transcript written to disk; cache hit returns the same shape
    rec2 = g(["ok"])
    assert rec2["result"] == "GREEN"


# ---- orchestrate x gate_core end-to-end (interface alignment pin test) ----
def test_orchestrate_consumes_gate_core_dict(tmp_path):
    sys.path.insert(0, HELDOUT)
    import accept_pool_generic as ap
    gold = {"prs": ["a", "b"], "constraints": []}
    sim = ap.make_sim_gate(gold)
    states = [{"state_id": "base", "kind": "base", "ids": [],
               "include_hidden": False, "scope": "full"},
              {"state_id": "singleton:a", "kind": "singleton", "ids": ["a"],
               "include_hidden": False, "scope": "full"}]
    g = gc.make_gate(_params(tmp_path), FAKE_ADAPTER,
                     raw=lambda i, h, s: {"result": "GREEN", "failure_stage": None,
                                          "reason": None, "evidence": {}})

    def real_gate(ids, include_hidden, scope="scoped"):
        return g(ids, include_hidden, scope)

    rep = ap.orchestrate(states, sim, real_gate, smoke_n=1)
    assert rep["ok"] is True


def test_orchestrate_rejects_legacy_string_gate(tmp_path):
    import accept_pool_generic as ap
    states = [{"state_id": "base", "kind": "base", "ids": [],
               "include_hidden": False, "scope": "full"}]
    with pytest.raises(TypeError, match="unified dict"):
        ap.orchestrate(states, lambda ids, ih: "GREEN",
                       lambda ids, ih, scope="scoped": "GREEN", smoke_n=1)


# ---------- confirm pre-registration tightening ----------
def test_confirm_prereg_requires_state_id():
    cur = BASE_SIGS + [["b.go", "compile", "undefined: X"]]
    allow = [{"state_id": "confirm:singleton:a01", "red_channel": "build",
              "signatures": [["b.go", "compile", "undefined: X"]]}]
    # pre-registration table present but no state_id supplied = caller error, raises
    # (wholesale allow-listing the entire table is not permitted)
    with pytest.raises(ValueError, match="state_id"):
        gc.confirm_verdict(BASE_SIGS, cur, prereg_allow=allow)


def test_validate_confirm_prereg_against_states():
    states = [{"state_id": "confirm:singleton:a01", "kind": "singleton",
               "ids": ["a01"], "scope": "full"}]
    sim_expect = {"confirm:singleton:a01": "RED"}
    red_channels = {"confirm:singleton:a01": "build"}
    ok_entry = [{"state_id": "confirm:singleton:a01", "red_channel": "build",
                 "signatures": [["b.go", "compile", "undefined: X"]]}]
    gc.validate_confirm_prereg(ok_entry, states, sim_expect, red_channels)
    with pytest.raises(ValueError, match="non-existent"):
        gc.validate_confirm_prereg(
            [{**ok_entry[0], "state_id": "confirm:ghost"}],
            states, sim_expect, red_channels)
    with pytest.raises(ValueError, match="not RED"):
        gc.validate_confirm_prereg(ok_entry, states,
                                   {"confirm:singleton:a01": "GREEN"},
                                   red_channels)
    with pytest.raises(ValueError, match="red_channel"):
        gc.validate_confirm_prereg(ok_entry, states, sim_expect,
                                   {"confirm:singleton:a01": "assertion"})
