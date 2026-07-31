"""Serialisation robustness for lone surrogates (exposed by yaml pool builds).

Real repos (e.g. eemeli/yaml) use bare surrogates as test names (`test('\\uDEAD', ...)`).
The vitest reporter writes them into structured JSON; when Python's `json.load` reads them
back, they become lone surrogate code points. Downstream `json.dump(ensure_ascii=False)`
(observation packages / scout_report / transcript / feature hash) cannot encode them as
UTF-8 and crashes. gate_core.escape_lone_surrogates replaces lone surrogates with
reversible `\\uXXXX` ASCII text at the entry points for vitest reports and gh diffs.
Verdict semantics are unchanged; for repos with no surrogates this is a byte-for-byte no-op.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HELDOUT = os.path.join(ROOT, "bulkpr", "heldout")
sys.path.insert(0, HELDOUT)

import gate_core as gc


def test_lone_low_surrogate_escaped():
    assert gc.escape_lone_surrogates("\udead") == "\\udead"


def test_lone_high_and_reversed_pair_escaped():
    # reversed "pair" = two lone surrogates (low before high), each escaped individually
    assert gc.escape_lone_surrogates("\udf06\ud834") == "\\udf06\\ud834"


def test_valid_astral_char_untouched():
    # valid non-BMP characters are single code points in Python str (not surrogate pairs) → left unchanged
    assert gc.escape_lone_surrogates("\U0001D306") == "\U0001D306"  # 𝌆
    assert gc.escape_lone_surrogates("\U0001F600") == "\U0001F600"  # 😀


def test_ascii_and_non_ascii_untouched():
    assert gc.escape_lone_surrogates("hello") == "hello"
    assert gc.escape_lone_surrogates("café €") == "café €"


def test_recursive_over_dict_list_and_keys():
    obj = {"a\udeadb": ["x", "y\udf06", {"k": "\udcff"}], "clean": 1}
    out = gc.escape_lone_surrogates(obj)
    assert out == {"a\\udeadb": ["x", "y\\udf06", {"k": "\\udcff"}], "clean": 1}
    # key check: result can be serialised to UTF-8 (the original object cannot)
    json.dumps(out, ensure_ascii=False).encode("utf-8")


def test_idempotent():
    once = gc.escape_lone_surrogates("t\udeade")
    twice = gc.escape_lone_surrogates(once)
    assert once == twice == "t\\udeade"


def test_non_str_scalars_passthrough():
    assert gc.escape_lone_surrogates(3) == 3
    assert gc.escape_lone_surrogates(None) is None
    assert gc.escape_lone_surrogates(True) is True


def test_pertest_key_dict_becomes_utf8_serializable():
    # reproduce the real per_test composite-key crash: project|file|fullName|id|kind
    per = {
        "yaml:default|tests/doc/stringify.ts|unpaired surrogate \udead|42_0|runtime":
            "passed",
        "yaml:default|tests/doc/stringify.ts|clean name|42_1|runtime": "passed",
    }
    # the raw dict cannot be written to disk as UTF-8
    try:
        json.dumps(per, ensure_ascii=False).encode("utf-8")
        raised = False
    except UnicodeEncodeError:
        raised = True
    assert raised, "a raw lone-surrogate key must not be UTF-8 encodable"
    safe = gc.escape_lone_surrogates(per)
    json.dumps(safe, ensure_ascii=False).encode("utf-8")
    assert len(safe) == 2  # keys remain distinct (escaping is reversible, preserving uniqueness)
