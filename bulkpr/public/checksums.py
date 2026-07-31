"""Write and verify `CHECKSUMS.sha256` for the release tree.

The format is compatible with `sha256sum -c` (`<64-hex-digits>  <relative-path>`), so
third parties can verify without installing any of our tooling.

Three invariants:
1. **The manifest file itself is not listed in the manifest**, otherwise it can never match
   after being written.
2. **Extra files also count as mismatches** (`unlisted`) — the release tree should contain
   nothing outside the manifest; checking only "are listed files still present" would miss
   files that were injected.
3. **Symlinks are rejected outright**: they can bring tree-external content into the release
   artifact and evade content-based verification.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

CHECKSUMS_NAME = "CHECKSUMS.sha256"

# Python writes bytecode caches on first run. These are runtime artifacts, not release
# content; if listed in the manifest, running any command and then verifying would fail
# with "unlisted files".
#
# `.git` is similar and more important: the release is distributed as a git repo, so
# anyone who clones it gets a `.git/` directory in the tree. Without exclusion, the very
# first command the README asks users to run — `bulkpr verify all` — would immediately
# report 900+ "unlisted files", hitting every user right at first impression.
#
# `.pytest_cache` is the same: both the README and REPRODUCE instruct users to run
# `python -m pytest` inside the tree; afterwards `.pytest_cache/` appears in the tree,
# and the next `bulkpr verify all` goes red — following the docs breaks the docs' own check.
#
# These three are explicitly excluded, and **only these three** — any other extra file still
# counts as a mismatch.
# (`*.egg-info/` left by `pip install -e .` is not listed here: installing the package
# into the tree is a user choice, not a required step in our docs; REPRODUCE explains
# how to handle it.)
_RUNTIME_ARTIFACTS = {"__pycache__", ".git", ".pytest_cache"}


class ChecksumError(Exception):
    """Checksum cannot be computed."""


def _walk(root: Path) -> list[Path]:
    """List all regular files in the tree (excluding the manifest itself), in deterministic path order."""
    out = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root)
        if _RUNTIME_ARTIFACTS.intersection(rel.parts):
            continue
        if path.is_symlink():
            raise ChecksumError(
                f"symlink is not allowed in a release tree: {rel.as_posix()}")
        if not path.is_file():
            continue
        if rel.as_posix() == CHECKSUMS_NAME:
            continue
        out.append(path)
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_checksums(root) -> str:
    """Compute the checksum manifest text for the entire tree."""
    root = Path(root)
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in _walk(root)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_checksums(root) -> int:
    """Write the manifest into the tree root and return the number of files covered."""
    root = Path(root)
    text = build_checksums(root)
    (root / CHECKSUMS_NAME).write_text(text)
    return len(text.splitlines())


def verify_checksums(root) -> dict:
    """Recompute each checksum and compare against the manifest. Missing, changed, and extra files all count as mismatches."""
    root = Path(root)
    manifest = root / CHECKSUMS_NAME
    if not manifest.is_file():
        raise ChecksumError(f"{CHECKSUMS_NAME} not found under {root}")

    expected: dict[str, str] = {}
    for lineno, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, path = line.split("  ", 1)
        except ValueError as exc:
            raise ChecksumError(f"{CHECKSUMS_NAME}:{lineno}: cannot parse line") from exc
        expected[path] = digest

    problems = []
    actual = {p.relative_to(root).as_posix() for p in _walk(root)}
    for path, digest in sorted(expected.items()):
        target = root / path
        if not target.is_file():
            problems.append({"path": path, "kind": "missing"})
        elif sha256_file(target) != digest:
            problems.append({"path": path, "kind": "changed"})
    for path in sorted(actual - set(expected)):
        problems.append({"path": path, "kind": "unlisted"})

    return {"ok": not problems, "checked": len(expected), "problems": problems}
