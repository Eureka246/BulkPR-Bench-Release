# bulkpr/tests/test_relation_metrics.py
"""Tests for the typed hyperedge metrics layer."""
import sys, pathlib
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # bulkpr/
import relation_metrics as rm


def _gold_full():
    """Small gold fixture covering all six constraint types (with visibility)."""
    return {
        "repo_id": "toy", "prs": ["A", "B", "C", "D", "E", "F", "G"],
        "constraints": [
            {"type": "forbidden_set", "members": ["A", "B"], "visibility": "hidden"},
            {"type": "high_order_conflict", "members": ["C", "D", "E"], "visibility": "public"},
            {"type": "depends_on", "source": "F", "target": "C", "visibility": "public"},
            {"type": "all_or_none_group", "members": ["D", "E"], "visibility": "public"},
            {"type": "duplicate_group", "members": ["A", "C"], "visibility": "public"},
            {"type": "supersedes", "new": "F", "old": "G", "visibility": "public"},
        ],
        "must_hold": [{"pr": "G", "visibility": "public"}],
    }


# ---------- Gold atoms ----------
def test_typed_gold_hyperedges_full_language():
    atoms, vis = rm.typed_gold_hyperedges(_gold_full())
    assert ("CONFLICT", frozenset({"A", "B"})) in atoms
    assert ("CONFLICT", frozenset({"C", "D", "E"})) in atoms      # high_order → CONFLICT
    assert ("DEPENDS_ON", "F", "C") in atoms
    assert ("ALL_OR_NONE", frozenset({"D", "E"})) in atoms        # old layer dropped; new layer must have it
    assert ("DUPLICATE", frozenset({"A", "C"})) in atoms
    assert ("SUPERSEDES", "F", "G") in atoms
    assert ("FORCED_REJECT", "G") in atoms                        # must_hold → unary atom
    assert len(atoms) == 7
    assert vis[("CONFLICT", frozenset({"A", "B"}))] == "hidden"
    assert vis[("ALL_OR_NONE", frozenset({"D", "E"}))] == "public"


def test_require_set_maps_to_all_or_none():
    g = {"prs": ["A", "B"], "constraints": [
        {"type": "require_set", "members": ["A", "B"], "visibility": "public"}], "must_hold": []}
    atoms, _ = rm.typed_gold_hyperedges(g)
    assert atoms == {("ALL_OR_NONE", frozenset({"A", "B"}))}


# ---------- Predicted atoms (synonym normalisation + unknown/malformed counts) ----------
def test_pred_synonyms_normalized():
    sub = {"relations": [
        {"type": "FORBIDDEN_SET", "members": ["A", "B"]},
        {"type": "REQUIRE_SET", "members": ["D", "E"]},
        {"type": "duplicate_group", "members": ["A", "C"]},
        {"type": "DEPENDS_ON", "source": "F", "target": "C"},
    ]}
    atoms, unknown, malformed = rm.typed_pred_hyperedges(sub)
    assert ("CONFLICT", frozenset({"A", "B"})) in atoms
    assert ("ALL_OR_NONE", frozenset({"D", "E"})) in atoms
    assert ("DUPLICATE", frozenset({"A", "C"})) in atoms
    assert ("DEPENDS_ON", "F", "C") in atoms
    assert unknown == 0 and malformed == 0


def test_pred_unknown_and_malformed_counted():
    sub = {"relations": [
        {"type": "TOTALLY_MADE_UP", "members": ["A", "B"]},   # unknown type
        {"type": "CONFLICT", "members": ["A"]},               # too few members
        {"type": "FORCED_REJECT", "members": ["A", "B"]},     # unary type given two members
    ]}
    atoms, unknown, malformed = rm.typed_pred_hyperedges(sub)
    assert atoms == set()
    assert unknown == 1 and malformed == 2


# ---------- F1 report family ----------
def test_exact_f1_perfect_roundtrip():
    g = _gold_full()
    sub = {"relations": rm.gold_to_relations(g)}
    rep = rm.typed_hyperedge_f1_report(g, sub, S=set())
    assert rep["exact"]["f1"] == 1.0
    assert rep["all_edge"]["f1"] == 1.0            # all_edge is an alias for exact
    assert rep["unknown_relation_count"] == 0 and rep["malformed_relation_count"] == 0


def test_macro_family_conventions():
    g = {"prs": ["A", "B"], "constraints": [
        {"type": "forbidden_set", "members": ["A", "B"], "visibility": "public"}], "must_hold": []}
    # gold has only CONFLICT; prediction hallucinates a DUPLICATE
    sub = {"relations": [{"type": "DUPLICATE", "members": ["A", "B"]}]}
    rep = rm.typed_hyperedge_f1_report(g, sub, S=set())
    fam = rep["macro_by_family"]
    assert fam["CONFLICT"] == 0.0        # gold present, pred absent → 0
    assert fam["DUPLICATE"] == 0.0       # gold empty, pred non-empty → hallucination penalised = 0 (not None)
    assert fam["DEPENDS_ON"] is None     # both empty → None
    assert fam["ALL_OR_NONE"] is None and fam["FORCED_REJECT"] is None and fam["SUPERSEDES"] is None
    assert fam["macro_f1"] == 0.0        # None entries excluded from average


def test_projected_atom_f1_expands_hyperedge():
    g = {"prs": ["C", "D", "E"], "constraints": [
        {"type": "high_order_conflict", "members": ["C", "D", "E"], "visibility": "public"}],
        "must_hold": []}
    # Prediction names only the pair {C,D} → exact 0; projected layer gets partial credit (1 of 3 pairs)
    sub = {"relations": [{"type": "CONFLICT", "members": ["C", "D"]}]}
    rep = rm.typed_hyperedge_f1_report(g, sub, S=set())
    assert rep["exact"]["f1"] == 0.0
    p = rep["projected_atom_f1"]
    assert p["tp"] == 1 and p["fn"] == 2 and p["fp"] == 0
    assert p["f1"] == pytest.approx(2 * 1 / (2 * 1 + 0 + 2))


def test_projected_unary_kept_as_atom():
    g = {"prs": ["G"], "constraints": [], "must_hold": [{"pr": "G", "visibility": "public"}]}
    sub = {"relations": [{"type": "FORCED_REJECT", "members": ["G"]}]}
    rep = rm.typed_hyperedge_f1_report(g, sub, S=set())
    assert rep["projected_atom_f1"]["tp"] == 1 and rep["projected_atom_f1"]["f1"] == 1.0


def test_public_gold_recall_only_recall():
    g = _gold_full()                                  # 1 hidden edge out of 7
    sub = {"relations": rm.gold_to_relations(g, only_visibility="public")}
    rep = rm.typed_hyperedge_f1_report(g, sub, S=set())
    assert rep["public_gold_recall"] == 1.0           # all public edges recalled
    # Correctly predicting a hidden edge must not penalise this metric (it is recall-only, no precision)
    sub2 = {"relations": rm.gold_to_relations(g)}     # includes hidden
    assert rm.typed_hyperedge_f1_report(g, sub2, S=set())["public_gold_recall"] == 1.0


def test_action_consistent_f1_and_adherence():
    g = {"prs": ["A", "B", "C"], "constraints": [
        {"type": "forbidden_set", "members": ["A", "B"], "visibility": "public"}], "must_hold": []}
    sub = {"relations": [{"type": "CONFLICT", "members": ["A", "B"]}]}   # TP=1
    ok = rm.typed_hyperedge_f1_report(g, sub, S={"A"})                   # correct action (not both merged)
    bad = rm.typed_hyperedge_f1_report(g, sub, S={"A", "B"})             # wrong action (both merged)
    assert ok["action_consistent_f1"]["f1"] == 1.0
    assert ok["action_consistent_f1"]["tp_action_adherence_rate"] == 1.0
    assert bad["action_consistent_f1"]["tp"] == 0                        # TP with wrong action → becomes FN
    assert bad["action_consistent_f1"]["f1"] == 0.0
    assert bad["action_consistent_f1"]["tp_action_adherence_rate"] == 0.0


def test_within_batch_layer():
    g = {"prs": ["A", "B", "C", "D"], "constraints": [
        {"type": "forbidden_set", "members": ["A", "B"], "visibility": "public"},
        {"type": "forbidden_set", "members": ["A", "C"], "visibility": "public"}], "must_hold": []}
    part = [["A", "B"], ["C", "D"]]
    sub = {"relations": [{"type": "CONFLICT", "members": ["A", "B"]}]}
    rep = rm.typed_hyperedge_f1_report(g, sub, S=set(), partition=part)
    assert rep["within_batch"]["tp"] == 1 and rep["within_batch"]["fn"] == 0   # {A,C} cross-batch, not in layer
    assert rep["exact"]["fn"] == 1


# ---------- Compiler (one flip test per family) ----------
@pytest.mark.parametrize("rels,probe,expect_flip_to_unsafe", [
    ([{"type": "CONFLICT", "members": ["A", "B"]}], {"A", "B"}, True),
    ([{"type": "DEPENDS_ON", "source": "A", "target": "B"}], {"A"}, True),
    ([{"type": "ALL_OR_NONE", "members": ["A", "B"]}], {"A"}, True),
    ([{"type": "FORCED_REJECT", "members": ["A"]}], {"A"}, True),
    ([{"type": "DUPLICATE", "members": ["A", "B"]}], {"A", "B"}, True),
    ([{"type": "SUPERSEDES", "new": "A", "old": "B"}], {"B"}, True),
])
def test_compiler_family_flips_safety(rels, probe, expect_flip_to_unsafe):
    from wbsr import check_safe
    prs = ["A", "B", "C"]
    compiled, unknown, malformed = rm.compile_predicted_relations(rels, prs)
    assert unknown == 0 and malformed == 0
    empty_compiled, _, _ = rm.compile_predicted_relations([], prs)
    assert check_safe(empty_compiled, probe)[0] is True          # empty graph: probe is safe
    assert check_safe(compiled, probe)[0] is not expect_flip_to_unsafe   # compiled graph flips safety


def test_compiler_membership_and_malformed():
    compiled, unknown, malformed = rm.compile_predicted_relations(
        [{"type": "CONFLICT", "members": ["A", "ZZZ"]},          # member outside the PR universe
         {"type": "NONSENSE", "members": ["A", "B"]}], ["A", "B"])
    assert compiled["constraints"] == [] and compiled["must_hold"] == []
    assert malformed == 1 and unknown == 1


# ---------- Probe family ----------
def test_generate_probes_dedup_counts_and_digest():
    g = {"repo_id": "toy", "prs": ["A", "B", "C"], "constraints": [
        {"type": "forbidden_set", "members": ["A", "B"], "visibility": "public"}],
        "must_hold": [], "oracle": {"objective": "max_cardinality_unit_pr"}}
    probes, meta = rm.generate_probe_subsets(g)
    assert meta["generated_count"] > meta["unique_count"]        # duplicates (e.g. empty set) removed
    assert frozenset() in probes and frozenset({"A", "C"}) in probes   # cross-constrained × free pair present
    probes2, meta2 = rm.generate_probe_subsets(g)
    assert meta["sha256"] == meta2["sha256"] and probes == probes2      # deterministic


# ---------- Probe agreement rate ----------
def test_probe_agreement_confusion_and_balanced():
    g = {"prs": ["A", "B"], "constraints": [
        {"type": "forbidden_set", "members": ["A", "B"], "visibility": "public"}], "must_hold": []}
    pred, _, _ = rm.compile_predicted_relations([], ["A", "B"])          # empty prediction
    probes = [frozenset({"A"}), frozenset({"B"}), frozenset({"A", "B"})]
    out = rm.behavioral_probe_agreement(g, pred, probes)
    assert out["confusion"] == {"gold_safe_pred_safe": 2, "gold_safe_pred_unsafe": 0,
                                "gold_unsafe_pred_safe": 1, "gold_unsafe_pred_unsafe": 0}
    assert out["agreement"] == pytest.approx(2 / 3)
    assert out["recall_safe"] == 1.0 and out["recall_unsafe"] == 0.0
    assert out["balanced_accuracy"] == pytest.approx(0.5)
    assert out["disagreements"] == [{"probe": ["A", "B"], "gold_safe": False, "pred_safe": True}]


def test_probe_agreement_class_absent_balanced_none():
    g = {"prs": ["A"], "constraints": [], "must_hold": []}
    pred, _, _ = rm.compile_predicted_relations([], ["A"])
    out = rm.behavioral_probe_agreement(g, pred, [frozenset(), frozenset({"A"})])
    assert out["agreement"] == 1.0
    assert out["recall_unsafe"] is None and out["balanced_accuracy"] is None
