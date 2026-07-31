"""relation schema v2 (native contract table + canonical ID + legacy-compat acceptance table).

Coverage: per-type contract, arity normalization, namespace ID, SUPERSEDES direction,
legacy alias mapping, malformed classification counts, active rules, ref merging.
"""
import pytest

from bulkpr.paper import relation_schema_v2 as rs

POOL_IDS = {f"PR-{i:02d}" for i in range(1, 11)}
FP_A = "a" * 60
FP_B = "b" * 60


def _native(**kw):
    base = {
        "agent_relation_ref": "rel-1",
        "type": "CONFLICT",
        "members": ["PR-01", "PR-02"],
        "dependent": None,
        "prerequisite": None,
        "replacement": None,
        "superseded": None,
        "visibility": "hidden",
        "confidence": 0.8,
        "status": "confirmed",
        "evidence_refs": [],
    }
    base.update(kw)
    return base


# ---------- per-type contract table ----------

def test_conflict_arity_exactly_two_ok():
    out = rs.normalize_native([_native()], POOL_IDS, FP_A)
    assert len(out.atoms) == 1
    assert out.atoms[0].family == "CONFLICT"


def test_conflict_singleton_malformed():
    out = rs.normalize_native([_native(members=["PR-01"])], POOL_IDS, FP_A)
    assert not out.atoms
    assert out.malformed_by_reason.get("bad_arity") == 1


def test_conflict_three_members_normalizes_to_high_order():
    out = rs.normalize_native([_native(members=["PR-01", "PR-02", "PR-03"])], POOL_IDS, FP_A)
    assert out.atoms[0].family == "HIGH_ORDER_CONFLICT"


def test_must_reject_single_member():
    out = rs.normalize_native(
        [_native(type="MUST_REJECT", members=["PR-05"])], POOL_IDS, FP_A
    )
    assert out.atoms[0].family == "MUST_REJECT"
    out2 = rs.normalize_native(
        [_native(type="MUST_REJECT", members=["PR-05", "PR-06"])], POOL_IDS, FP_A
    )
    assert out2.malformed_by_reason.get("bad_arity") == 1


def test_depends_on_uses_role_fields_members_must_be_null():
    ok = _native(type="DEPENDS_ON", members=None, dependent="PR-03", prerequisite="PR-04")
    out = rs.normalize_native([ok], POOL_IDS, FP_A)
    assert out.atoms[0].family == "DEPENDS_ON"
    assert out.atoms[0].roles == ("PR-03", "PR-04")
    bad = _native(type="DEPENDS_ON", members=["PR-03", "PR-04"], dependent="PR-03", prerequisite="PR-04")
    out2 = rs.normalize_native([bad], POOL_IDS, FP_A)
    assert out2.malformed_by_reason.get("unexpected_field") == 1


def test_depends_on_self_loop_malformed():
    bad = _native(type="DEPENDS_ON", members=None, dependent="PR-03", prerequisite="PR-03")
    out = rs.normalize_native([bad], POOL_IDS, FP_A)
    assert out.malformed_by_reason.get("self_loop") == 1


def test_set_type_with_role_fields_malformed():
    bad = _native(dependent="PR-01")
    out = rs.normalize_native([bad], POOL_IDS, FP_A)
    assert out.malformed_by_reason.get("unexpected_field") == 1


def test_unknown_endpoint_malformed():
    out = rs.normalize_native([_native(members=["PR-01", "PR-99"])], POOL_IDS, FP_A)
    assert out.malformed_by_reason.get("unknown_pr") == 1


# ---------- canonical ID ----------

def test_canonical_id_namespace():
    a = _native()
    id_a = rs.normalize_native([a], POOL_IDS, FP_A).atoms[0].canonical_id
    id_b = rs.normalize_native([a], POOL_IDS, FP_B).atoms[0].canonical_id
    assert id_a != id_b


def test_canonical_id_stable_and_member_order_free():
    a = _native(members=["PR-01", "PR-02"])
    b = _native(members=["PR-02", "PR-01"])
    id_a = rs.normalize_native([a], POOL_IDS, FP_A).atoms[0].canonical_id
    id_b = rs.normalize_native([b], POOL_IDS, FP_A).atoms[0].canonical_id
    assert id_a == id_b
    assert id_a == rs.normalize_native([a], POOL_IDS, FP_A).atoms[0].canonical_id


def test_supersedes_direction_distinct_ids():
    fwd = _native(type="SUPERSEDES", members=None, replacement="PR-01", superseded="PR-02")
    rev = _native(type="SUPERSEDES", members=None, replacement="PR-02", superseded="PR-01")
    id_f = rs.normalize_native([fwd], POOL_IDS, FP_A).atoms[0].canonical_id
    id_r = rs.normalize_native([rev], POOL_IDS, FP_A).atoms[0].canonical_id
    assert id_f != id_r


# ---------- native state machine and ref merging ----------

def test_native_retract_deactivates():
    r1 = _native(agent_relation_ref="r1", status="confirmed")
    r2 = _native(agent_relation_ref="r1", status="retracted")
    out = rs.normalize_native([r1, r2], POOL_IDS, FP_A)
    assert not [a for a in out.atoms if a.active]


def test_ref_merge_any_active_wins():
    r1 = _native(agent_relation_ref="r1", status="retracted")
    r2 = _native(agent_relation_ref="r2", status="hypothesis")  # same canonical atom
    out = rs.normalize_native([r1, r2], POOL_IDS, FP_A)
    actives = [a for a in out.atoms if a.active]
    assert len(actives) == 1


# ---------- legacy acceptance table ----------

def test_legacy_source_target_maps_to_depends_on():
    rec = {"type": "DEPENDS_ON", "source": "PR-03", "target": "PR-04", "reason": "x"}
    out = rs.normalize_legacy([rec], POOL_IDS, FP_A)
    atom = out.atoms[0]
    assert atom.family == "DEPENDS_ON"
    assert atom.roles == ("PR-03", "PR-04")  # dependent=source, prerequisite=target
    assert atom.active and atom.status is None


def test_legacy_new_old_maps_to_supersedes():
    rec = {"type": "SUPERSEDES", "new": "PR-01", "old": "PR-02"}
    out = rs.normalize_legacy([rec], POOL_IDS, FP_A)
    assert out.atoms[0].family == "SUPERSEDES"
    assert out.atoms[0].roles == ("PR-01", "PR-02")


def test_legacy_prs_alias_for_members():
    rec = {"type": "ALL_OR_NONE", "prs": ["PR-05", "PR-06"]}
    out = rs.normalize_legacy([rec], POOL_IDS, FP_A)
    assert out.atoms[0].family == "ALL_OR_NONE"


def test_legacy_singleton_conflict_malformed():
    rec = {"type": "CONFLICT", "members": ["PR-01"], "reason": "sus"}
    out = rs.normalize_legacy([rec], POOL_IDS, FP_A)
    assert not out.atoms
    assert out.malformed_by_reason.get("bad_arity") == 1


def test_legacy_missing_target_malformed():
    rec = {"type": "DEPENDS_ON", "source": "PR-03", "reason": "x"}
    out = rs.normalize_legacy([rec], POOL_IDS, FP_A)
    assert out.malformed_by_reason.get("missing_endpoint") == 1


def test_legacy_nonstring_endpoint_is_malformed_not_crash():
    # Directed relation endpoint written as list (real agent output) → marked malformed, must not crash
    rec = {"type": "DEPENDS_ON", "source": ["PR-03", "PR-04"], "target": "PR-04"}
    out = rs.normalize_legacy([rec], POOL_IDS, FP_A)
    assert not out.atoms
    assert out.malformed_by_reason.get("unknown_pr") == 1


def test_legacy_alias_conflict_malformed():
    rec = {"type": "DEPENDS_ON", "source": "PR-03", "target": "PR-04",
           "dependent": "PR-05", "prerequisite": "PR-04"}
    out = rs.normalize_legacy([rec], POOL_IDS, FP_A)
    assert out.malformed_by_reason.get("alias_conflict") == 1


def test_legacy_non_object_malformed():
    out = rs.normalize_legacy([["PR-01", "PR-02"]], POOL_IDS, FP_A)
    assert out.malformed_by_reason.get("not_object") == 1


def test_legacy_direction_null_depends_via_members():
    # Old protocol sometimes only provides members for DEPENDS_ON: direction unknown,
    # atom preserved but roles=None
    rec = {"type": "DEPENDS_ON", "members": ["PR-03", "PR-04"]}
    out = rs.normalize_legacy([rec], POOL_IDS, FP_A)
    assert len(out.atoms) == 1
    assert out.atoms[0].roles is None
    assert out.direction_unknown_count == 1


def test_e_hat_nonempty_on_compat_dirty_ledger():
    ledger = (
        [{"type": "CONFLICT", "members": ["PR-01"]}] * 3        # dirty: singleton member
        + [{"type": "CONFLICT", "members": ["PR-01", "PR-02"], "confidence": 0.9}]
        + [{"type": "DEPENDS_ON", "source": "PR-03", "target": "PR-04"}]
        + [{"type": "DEPENDS_ON", "source": "PR-05"}]           # dirty: missing target
        + [["not", "an", "object"]]                              # dirty: not an object
    )
    out = rs.normalize_legacy(ledger, POOL_IDS, FP_A)
    actives = [a for a in out.atoms if a.active]
    assert len(actives) == 2
    assert sum(out.malformed_by_reason.values()) == 5


def test_legacy_nonstring_members_are_malformed_not_crash():
    # Real agent output (caught in rehearsal): members contains dict elements → malformed count, no crash
    out = rs.normalize_legacy(
        [{"type": "CONFLICT", "members": [{"pr": "PR-01"}, "PR-02"]},
         {"type": "CONFLICT", "members": ["PR-01", "PR-02"]}],
        {"PR-01", "PR-02"}, "f" * 60)
    assert len(out.atoms) == 1
    assert out.malformed_by_reason.get("bad_members") == 1
