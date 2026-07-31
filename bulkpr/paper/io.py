"""Deterministic file operations shared across paper experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical_json(value):
    """Return UTF-8 JSON with a fixed key order and fixed whitespace."""
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def sha256_json(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path, value):
    _write_bytes_atomic(path, canonical_json(value))


def write_jsonl_atomic(path, values):
    _write_bytes_atomic(path, b"".join(canonical_json(value) for value in values))


def write_text_atomic(path, text):
    _write_bytes_atomic(path, text.encode("utf-8"))


def _write_bytes_atomic(path, data):
    """Write to a temp file in the same directory, then atomically replace the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def resolve_under(root, relative):
    """Resolve a path inside root; rejects absolute paths, .., and symlinks that escape root."""
    root = Path(root).resolve()
    relative = Path(relative)
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes root: {relative}")
    return path
