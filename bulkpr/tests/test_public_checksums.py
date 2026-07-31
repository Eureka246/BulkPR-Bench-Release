"""`CHECKSUMS.sha256`: can be written, verified, and must report corruption clearly."""
import pytest

from bulkpr.public import checksums


def _tree(root):
    (root / "data").mkdir(parents=True)
    (root / "data" / "a.json").write_text('{"x": 1}\n')
    (root / "b.py").write_text("print(1)\n")
    return root


def test_write_then_verify_passes(tmp_path):
    root = _tree(tmp_path)
    checksums.write_checksums(root)
    result = checksums.verify_checksums(root)
    assert result["ok"] is True
    assert result["checked"] == 2
    assert result["problems"] == []


def test_checksums_file_itself_is_not_listed(tmp_path):
    """The checksum file must not include itself, or the digest can never match."""
    root = _tree(tmp_path)
    checksums.write_checksums(root)
    text = (root / checksums.CHECKSUMS_NAME).read_text()
    assert checksums.CHECKSUMS_NAME not in text


def test_verify_catches_modified_file(tmp_path):
    root = _tree(tmp_path)
    checksums.write_checksums(root)
    (root / "b.py").write_text("print(2)\n")
    result = checksums.verify_checksums(root)
    assert result["ok"] is False
    assert any(p["path"] == "b.py" and p["kind"] == "changed"
               for p in result["problems"])


def test_verify_catches_missing_file(tmp_path):
    root = _tree(tmp_path)
    checksums.write_checksums(root)
    (root / "b.py").unlink()
    result = checksums.verify_checksums(root)
    assert result["ok"] is False
    assert any(p["path"] == "b.py" and p["kind"] == "missing"
               for p in result["problems"])


def test_verify_catches_extra_file(tmp_path):
    """An extra file that is not in the manifest should also be flagged — the release tree
    must not contain anything outside the manifest."""
    root = _tree(tmp_path)
    checksums.write_checksums(root)
    (root / "sneaky.txt").write_text("hi\n")
    result = checksums.verify_checksums(root)
    assert result["ok"] is False
    assert any(p["path"] == "sneaky.txt" and p["kind"] == "unlisted"
               for p in result["problems"])


def test_output_is_deterministic_and_sorted(tmp_path):
    root = _tree(tmp_path)
    (root / "z.txt").write_text("z\n")
    (root / "a.txt").write_text("a\n")
    checksums.write_checksums(root)
    first = (root / checksums.CHECKSUMS_NAME).read_bytes()
    (root / checksums.CHECKSUMS_NAME).unlink()
    checksums.write_checksums(root)
    assert (root / checksums.CHECKSUMS_NAME).read_bytes() == first
    paths = [line.split("  ", 1)[1] for line in first.decode().splitlines()]
    assert paths == sorted(paths)


def test_format_is_sha256sum_compatible(tmp_path):
    """The output must be directly usable by `sha256sum -c CHECKSUMS.sha256`."""
    import hashlib
    root = _tree(tmp_path)
    checksums.write_checksums(root)
    for line in (root / checksums.CHECKSUMS_NAME).read_text().splitlines():
        digest, path = line.split("  ", 1)
        assert len(digest) == 64
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest


def test_symlink_is_refused(tmp_path):
    """The release tree must not contain symlinks; they should be rejected when writing the manifest."""
    root = _tree(tmp_path)
    (root / "link.py").symlink_to(root / "b.py")
    with pytest.raises(checksums.ChecksumError):
        checksums.write_checksums(root)


# --- checksums must not break after a normal run ---------------------------------------------
#
# Running a command from the release tree causes Python to write __pycache__ entries.
# If those were treated as "extra files", `bulkpr verify checksums` would fail under
# normal usage. They are runtime artifacts, not content — they are explicitly excluded
# and must never be included during assembly.

def test_pycache_created_by_running_the_code_does_not_break_verification(tmp_path):
    root = _tree(tmp_path)
    checksums.write_checksums(root)
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "b.cpython-314.pyc").write_bytes(b"\x00compiled\x00")
    (root / "data" / "__pycache__").mkdir()
    (root / "data" / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    result = checksums.verify_checksums(root)
    assert result["ok"] is True, result["problems"]


def test_pycache_is_not_listed_in_the_manifest(tmp_path):
    root = _tree(tmp_path)
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "b.cpython-314.pyc").write_bytes(b"\x00")
    checksums.write_checksums(root)
    assert "__pycache__" not in (root / checksums.CHECKSUMS_NAME).read_text()


def test_a_real_extra_file_is_still_caught(tmp_path):
    """Excluding __pycache__ must not weaken the 'unlisted extra file' check."""
    root = _tree(tmp_path)
    checksums.write_checksums(root)
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (root / "sneaky.py").write_text("import os\n")
    result = checksums.verify_checksums(root)
    assert result["ok"] is False
    assert [p["path"] for p in result["problems"]] == ["sneaky.py"]


def test_verify_ignores_the_git_directory(tmp_path):
    """A cloned tree contains .git/, which is not release content.

    Without excluding it, the user's very first command, `bulkpr verify all` (which the
    README instructs them to run), would immediately fail with "900+ unlisted files".
    """
    (tmp_path / "a.txt").write_text("hello\n")
    checksums.write_checksums(tmp_path)
    git_dir = tmp_path / ".git"
    (git_dir / "refs" / "remotes" / "origin").mkdir(parents=True)
    (git_dir / "refs" / "remotes" / "origin" / "HEAD").write_text("ref: x\n")
    (git_dir / "config").write_text("[core]\n")
    result = checksums.verify_checksums(tmp_path)
    assert result["ok"], result["problems"]


def test_verify_ignores_the_pytest_cache(tmp_path):
    """Both README and REPRODUCE instruct users to run pytest inside the tree, which
    creates a .pytest_cache/ directory afterward.

    Without excluding it, following the documentation's first instruction would cause
    the documentation's next instruction, `bulkpr verify all`, to fail.
    """
    (tmp_path / "a.txt").write_text("hello\n")
    checksums.write_checksums(tmp_path)
    cache = tmp_path / ".pytest_cache" / "v" / "cache"
    cache.mkdir(parents=True)
    (cache / "nodeids").write_text("[]\n")
    (tmp_path / ".pytest_cache" / "CACHEDIR.TAG").write_text("Signature\n")
    result = checksums.verify_checksums(tmp_path)
    assert result["ok"], result["problems"]


def test_verify_still_reports_other_unlisted_files(tmp_path):
    """Negative control: excluding .git and .pytest_cache must not suppress reports of other unlisted files."""
    (tmp_path / "a.txt").write_text("hello\n")
    checksums.write_checksums(tmp_path)
    (tmp_path / "sneaky.txt").write_text("extra\n")
    result = checksums.verify_checksums(tmp_path)
    assert not result["ok"]
    assert any(p["path"] == "sneaky.txt" for p in result["problems"])
