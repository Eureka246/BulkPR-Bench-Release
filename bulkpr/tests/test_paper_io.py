import hashlib

import pytest

from bulkpr.paper.io import (
    canonical_json,
    read_json,
    resolve_under,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_jsonl_atomic,
)


def test_canonical_json_is_order_independent():
    expected = b'{"a":[2],"b":1}\n'

    assert canonical_json({"b": 1, "a": [2]}) == expected
    assert canonical_json({"a": [2], "b": 1}) == expected
    assert sha256_json({"b": 1, "a": [2]}) == hashlib.sha256(expected).hexdigest()


def test_atomic_json_round_trip_and_file_hash(tmp_path):
    path = tmp_path / "nested" / "value.json"
    value = {"ключ": True, "items": [2, 1]}

    write_json_atomic(path, value)

    assert read_json(path) == value
    assert path.read_bytes() == canonical_json(value)
    assert sha256_file(path) == hashlib.sha256(canonical_json(value)).hexdigest()
    assert not list(path.parent.glob("*.tmp"))


def test_atomic_jsonl_uses_one_canonical_object_per_line(tmp_path):
    path = tmp_path / "rows.jsonl"

    write_jsonl_atomic(path, [{"b": 2, "a": 1}, {"row": 2}])

    assert path.read_bytes() == b'{"a":1,"b":2}\n{"row":2}\n'


@pytest.mark.parametrize("relative", ["../outside", "/tmp/outside"])
def test_resolve_under_rejects_escape(tmp_path, relative):
    with pytest.raises(ValueError, match="escapes root"):
        resolve_under(tmp_path, relative)


def test_resolve_under_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes root"):
        resolve_under(tmp_path, "link/value.json")
