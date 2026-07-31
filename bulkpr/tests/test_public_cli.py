"""Behaviour tests for the public `bulkpr` command-line interface.

The focus is not just "it runs" — it is a handful of properties that must hold without ambiguity:
a mismatch must exit non-zero; things that were not checked must not be reported as checked;
missing required arguments must produce an immediate error; and a failure a reader has been told
to expect must arrive as a message rather than as a traceback.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

from bulkpr.cli import main
from bulkpr.public import checksums


@pytest.fixture
def tree(tmp_path):
    """A minimal release tree containing only what verify needs."""
    root = tmp_path / "tree"
    (root / "data" / "reference").mkdir(parents=True)
    (root / "bulkpr").mkdir()
    (root / "bulkpr" / "wbsr.py").write_text("# scorer\n")
    (root / "data" / "reference" / "reproducibility.json").write_text(
        json.dumps({"schema_version": "release-reproducibility/v1"}))
    return root


# ---------- verify checksums ----------

def test_verify_checksums_passes_on_a_clean_tree(tree, capsys):
    checksums.write_checksums(tree)
    assert main(["verify", "checksums", "--root", str(tree)]) == 0
    assert "PASS checksums" in capsys.readouterr().out


def test_verify_checksums_fails_after_tampering(tree, capsys):
    checksums.write_checksums(tree)
    (tree / "bulkpr" / "wbsr.py").write_text("# tampered\n")
    assert main(["verify", "checksums", "--root", str(tree)]) == 1
    assert "FAIL checksums" in capsys.readouterr().out


def test_verify_checksums_fails_on_an_extra_file(tree):
    checksums.write_checksums(tree)
    (tree / "extra.txt").write_text("hi\n")
    assert main(["verify", "checksums", "--root", str(tree)]) == 1


def test_score_needs_somewhere_to_read_results_from(tree, capsys):
    """`--jobs` is required: without a results directory there is nothing to score;
    silently emitting an empty table is not acceptable.

    The full behaviour of `score` is tested end-to-end in `test_public_score.py`."""
    with pytest.raises(SystemExit):
        main(["score", "--root", str(tree)])
    assert "--jobs" in capsys.readouterr().err


def test_verify_all_runs_every_check(tree, capsys):
    """`verify all` must run both checks, not stop at the first one."""
    checksums.write_checksums(tree)
    main(["verify", "all", "--root", str(tree)])
    out = capsys.readouterr().out
    assert "checksums" in out and "reference" in out


# ---------- build: expected failures must read as messages ----------

def test_build_reports_a_hash_mismatch_as_a_message_not_a_traceback(
        tree, tmp_path, capsys, monkeypatch):
    """`docs/REPRODUCE.md` tells the reader that a moved upstream, a shallow clone or a
    different git version makes snapshot hashes mismatch — all 18 at once, in the git-version
    case. That is a documented, expected outcome, so it has to arrive as a sentence they can
    act on. A raw traceback reads like the tool is broken rather than like their input is."""
    from bulkpr.paper.taskgen import FrozenDataMismatchError
    from bulkpr.public import reference as reference_module

    def mismatch(*args, **kwargs):
        raise FrozenDataMismatchError("snapshot hash mismatch for attrs: aaaa != bbbb")

    monkeypatch.setattr(reference_module, "build_public_task_tree", mismatch)
    code = main(["build", "--root", str(tree), "--clones", str(tmp_path),
                 "--snapshots", str(tmp_path), "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 1
    assert "snapshot hash mismatch for attrs" in captured.err
    assert "Traceback" not in captured.err


def test_build_does_not_hide_an_unexpected_value_error(tree, tmp_path, monkeypatch):
    """A programming error needs its traceback; calling it a data mismatch hides the bug."""
    from bulkpr.public import reference as reference_module

    def broken(*args, **kwargs):
        raise ValueError("internal invariant failed")

    monkeypatch.setattr(reference_module, "build_public_task_tree", broken)
    with pytest.raises(ValueError, match="internal invariant failed"):
        main(["build", "--root", str(tree), "--clones", str(tmp_path),
              "--snapshots", str(tmp_path), "--out", str(tmp_path / "out")])


# ---------- fetch: progress and errors must read in the order they happened ----------

def test_fetch_progress_lines_come_before_the_errors_they_explain(tree, tmp_path):
    """Progress goes to stdout, failures to stderr. Python block-buffers stdout when it is a
    pipe and leaves stderr unbuffered, so anyone redirecting both to one file or one terminal
    pipe used to see every error first and every progress line afterwards — it looks as though
    the command died before it started. Which repository a failure belongs to is only readable
    from the ordering, so the ordering is part of the output being correct."""
    (tree / "data" / "repos.json").write_text(json.dumps({"repos": [
        {"repo_id": "does-not-exist", "base_commit": "0" * 40,
         "upstream_url": str(tmp_path / "no-such-repo")},
    ]}))
    # Run it the way a reader would — as a real process with both streams joined — but make
    # the package importable from wherever pytest happens to have been started.
    import bulkpr.cli
    package_root = pathlib.Path(bulkpr.cli.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "bulkpr.cli", "fetch",
         "--root", str(tree), "--dest", str(tmp_path / "clones")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300,
        env={**os.environ, "PYTHONPATH": str(package_root)},
    )
    merged = result.stdout.decode(errors="replace")
    assert "[fetch] does-not-exist" in merged, merged
    assert "FAILED" in merged, merged
    assert merged.index("[fetch] does-not-exist") < merged.index("FAILED"), merged


def test_verify_reference_says_when_the_tree_hash_was_not_checked(tree, capsys):
    """Things that were not checked must not be reported as checked."""
    (tree / "data" / "pools").mkdir(parents=True)
    main(["verify", "reference", "--root", str(tree)])
    combined = capsys.readouterr()
    assert "task tree hash NOT checked" in combined.out or \
           "no public pool" in (combined.out + combined.err).lower() or \
           "FAIL reference" in combined.out
