# bulkpr/tests/test_batch_oracle.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest  # noqa: E402
import batch_oracle  # noqa: E402


def test_components_split_by_shared_members():
    gold = {"prs": ["A", "B", "C", "D", "E"], "must_hold": [], "constraints": [
        {"type": "forbidden_set", "members": ["A", "B"]},
        {"type": "depends_on", "source": "C", "target": "D"}]}
    free, forced, comps = batch_oracle._components(gold)
    assert set(free) == {"E"}
    assert forced == set()
    assert sorted(sorted(m) for m, _ in comps) == [["A", "B"], ["C", "D"]]


def test_components_forced_zero_excluded_from_free():
    gold = {"prs": ["A", "B"], "must_hold": [{"pr": "B"}], "constraints": []}
    free, forced, comps = batch_oracle._components(gold)
    assert set(free) == {"A"} and forced == {"B"} and comps == []


import wbsr  # noqa: E402


def _pool():
    # A/B co-required (safety); C forbidden with D; E depends on F (source=E dependent, target=F prereq); G free
    return {"prs": ["A", "B", "C", "D", "E", "F", "G"], "must_hold": [], "constraints": [
        {"type": "all_or_none_group", "members": ["A", "B"], "reason": "safety"},
        {"type": "forbidden_set", "members": ["C", "D"]},
        {"type": "depends_on", "source": "E", "target": "F"}]}


def test_optk_all_in_one_batch_equals_optn():
    gold = _pool()
    assert batch_oracle.opt_k_clairvoyant(gold, [gold["prs"]]) == wbsr.solve_oracle_proof(gold)["opt"]


def test_optk_split_pair_and_forward_dep():
    gold = _pool()
    # A(b0)/B(b1) split → co-required pair cannot merge (lose 2); C(b0)/D(b1) conflict still merges one;
    # F(b0)/E(b1) → E requires F first, F is in an earlier batch = forward, both can merge; G free
    part = [["A", "F", "C", "G"], ["B", "D", "E"]]
    assert wbsr.solve_oracle_proof(gold)["opt"] == 6      # A+B + max(C,D) + E+F + G = 2+1+2+1
    assert batch_oracle.opt_k_clairvoyant(gold, part) == 4  # 0 + 1 + 2 + 1 (only lose the co-required pair)


def test_optk_backward_dep_loses_dependent():
    gold = _pool()
    # E(b0) requires F(b1) → prerequisite F arrives later = backward → dependent E cannot merge
    # (when E's batch is processed, F is not yet in trunk and cannot be backtracked); but
    # prerequisite F has no dependencies of its own, so merging it alone is safe → F can merge solo.
    # Only E (1 PR) is lost, not the whole pair; A/B in same batch can merge (2); C/D merge one (1); G free (1).
    # A common mistake is ==4 ("E and F both cannot merge, lose 2"): backward dependency only
    # loses the dependent (E), not the prerequisite (F); F can merge solo → correct answer is ==5.
    part = [["E", "A", "B", "G"], ["F", "C", "D"]]
    assert batch_oracle.opt_k_clairvoyant(gold, part) == 5  # A+B(2)+max(C,D)(1)+F solo(1)+G(1)


def test_reasons_split_all_or_none():
    gold = _pool()
    part = [["A", "F", "C", "G"], ["B", "D", "E"]]       # A/B split across batches; F before E = forward
    reasons = batch_oracle.known_unreachable_reasons(gold, part)
    kinds = {r["reason"] for r in reasons}
    assert "split_all_or_none_group" in kinds
    assert "backward_dependency" not in kinds


def test_reasons_backward_dependency():
    gold = _pool()
    part = [["E", "A", "B", "G"], ["F", "C", "D"]]        # E before F, E depends on F → backward
    reasons = batch_oracle.known_unreachable_reasons(gold, part)
    assert any(r["reason"] == "backward_dependency" for r in reasons)


def test_reachability_full_batch():
    gold = _pool()
    r = batch_oracle.reachability(gold, [gold["prs"]])
    assert r["reachable"] is True
    assert r["opt_n"] == 6 and r["opt_k"] == 6 and r["structural_regret"] == 0
    assert r["known_unreachable_reasons"] == []


def test_reachability_split_pool():
    gold = _pool()
    r = batch_oracle.reachability(gold, [["A", "F", "C", "G"], ["B", "D", "E"]])
    assert r["reachable"] is False
    assert r["opt_n"] == 6 and r["opt_k"] == 4 and r["structural_regret"] == 2
    assert any(x["reason"] == "split_all_or_none_group" for x in r["known_unreachable_reasons"])


def test_dep_direction_helper():
    c = {"type": "depends_on", "source": "E", "target": "F"}   # E=dependent, F=prerequisite
    assert batch_oracle._dep_direction(c, {"E": 0, "F": 1}) == "backward"  # prereq in a later batch
    assert batch_oracle._dep_direction(c, {"E": 1, "F": 0}) == "forward"   # prereq in an earlier batch
    assert batch_oracle._dep_direction(c, {"E": 0, "F": 0}) == "forward"   # same batch: rolling can merge both
    assert batch_oracle._dep_direction({"type": "forbidden_set", "members": ["A", "B"]}, {"A": 0, "B": 1}) is None
    assert batch_oracle._dep_direction(c, {"E": 0}) is None                 # missing one member → undecidable


def test_witness_matches_count_and_is_feasible_order():
    gold = _pool()   # A/B all_or_none, C/D forbidden, E depends_on F, G free (opt=6)
    part = [["A", "F", "C", "G"], ["B", "D", "E"]]
    count, witness, order = batch_oracle.opt_k_clairvoyant_witness(gold, part)
    assert count == batch_oracle.opt_k_clairvoyant(gold, part) == 4     # behavior unchanged
    assert len(witness) == count
    assert wbsr._feasible(witness, gold["constraints"])                 # final state is feasible
    if "E" in witness and "F" in witness:                              # topological order: F before E
        assert order.index("F") < order.index("E")
    assert set(order) == set(witness)


def test_witness_all_one_batch_equals_optn():
    gold = _pool()
    count, witness, _ = batch_oracle.opt_k_clairvoyant_witness(gold, [gold["prs"]])
    assert count == wbsr.solve_oracle_proof(gold)["opt"] == 6
    assert wbsr.check_safe(gold, witness)[0]


# ---------- buffered oracle tests ----------
def _gold_coreq():
    return {"repo_id": "cq", "prs": ["A", "B", "C", "D"], "must_hold": [], "constraints": [
        {"type": "all_or_none_group", "members": ["A", "B"], "reason": "safety", "visibility": "public"}]}


def test_buffered_degenerate_T0_and_B0():
    gold = _gold_coreq(); part = [["A", "C"], ["B", "D"]]     # A/B split across batches → without deferral only free PRs can merge
    base = batch_oracle.opt_k_clairvoyant(gold, part)
    assert batch_oracle.opt_k_clairvoyant_buffered(gold, part, B=5, T=0) == base
    assert batch_oracle.opt_k_clairvoyant_buffered(gold, part, B=0, T=5) == base


def test_buffered_recovers_split_coreq():
    gold = _gold_coreq(); part = [["A", "C"], ["B", "D"]]
    base = batch_oracle.opt_k_clairvoyant(gold, part)         # A/B split across batches, cannot merge → 2 (C,D free)
    buf = batch_oracle.opt_k_clairvoyant_buffered(gold, part, B=2, T=1)
    assert base == 2 and buf == 4                             # buffering recovers A/B as well


def test_buffered_monotone_and_bounded():
    gold = _gold_coreq(); part = [["A", "C"], ["B", "D"]]
    opt_n = batch_oracle.reachability(gold, part)["opt_n"]
    vals = [batch_oracle.opt_k_clairvoyant_buffered(gold, part, B=2, T=t) for t in (0, 1, 2)]
    assert vals[0] <= vals[1] <= vals[2] <= opt_n
    assert batch_oracle.opt_k_clairvoyant(gold, part) <= vals[-1] <= opt_n


def test_buffered_schedule_count_matches():
    gold = _gold_coreq(); part = [["A", "C"], ["B", "D"]]
    total, sched = batch_oracle.opt_k_clairvoyant_buffered_schedule(gold, part, B=2, T=1)
    assert total == batch_oracle.opt_k_clairvoyant_buffered(gold, part, B=2, T=1)
    assert len(sched) == len(part)                           # one step per batch


def test_buffered_large_universe_solves_not_raises():
    """The main buffered backend solves exactly for |U|>10 without raising (strengthened from old guard test)."""
    # 12 co-required pairs → |U|=24 > MAX_BUFFERED_REFERENCE_U
    prs, cons, first, second = [], [], [], []
    for i in range(12):
        a, b = f"a{i}", f"b{i}"
        prs += [a, b]; first.append(a); second.append(b)
        cons.append({"type": "all_or_none_group", "members": [a, b], "reason": "safety"})
    gold = {"prs": prs, "constraints": cons}
    part = [first, second]
    got = batch_oracle.opt_k_clairvoyant_buffered(gold, part, B=3, T=1)   # must not raise
    assert got == 2 * 3                                                   # B=3 → saves 3 pairs
    # the reference implementation still protects itself with raise for |U|>10 (guard belongs to reference)
    with pytest.raises(ValueError, match="MAX_BUFFERED_REFERENCE_U"):
        batch_oracle._buffered_dp_reference(gold, part, B=3, T=1)


def test_reachability_buffered_fields():
    gold = _gold_coreq(); part = [["A", "C"], ["B", "D"]]
    r = batch_oracle.reachability_buffered(gold, part, B=2, T=1)
    assert r["opt_n"] == 4 and r["opt_k"] == 2 and r["opt_k_buffered"] == 4
    assert r["reachable_buffered"] is True
    assert r["buffer_recovered"] == 2                     # buffering net-recovered A and B
    assert r["structural_regret_buffered"] == 0


# ---------- information-layer tests ----------
def _gold_info():
    # cross-batch unobservable forbidden(A,B) + same-batch unobservable forbidden(C,D) + free E,F
    return {"repo_id": "info", "prs": ["A", "B", "C", "D", "E", "F"], "must_hold": [], "constraints": [
        {"type": "forbidden_set", "members": ["A", "B"], "visibility": "hidden", "inferability": "unobservable"},
        {"type": "forbidden_set", "members": ["C", "D"], "visibility": "hidden", "inferability": "unobservable"},
    ]}


def test_info_hazards_cross_batch_only_and_missing_raises():
    gold = _gold_info()
    part = [["A", "C", "D", "E"], ["B", "F"]]                 # A/B cross-batch; C/D same batch
    hz = batch_oracle.info_hazards(gold, part)
    assert len(hz) == 1 and set(hz[0]["members"]) == {"A", "B"}   # only cross-batch pairs go into hazards
    assert batch_oracle.info_hazard_count(gold, part) == 1
    # missing inferability → raises
    bad = {"repo_id": "x", "prs": ["A", "B"], "must_hold": [], "constraints": [
        {"type": "forbidden_set", "members": ["A", "B"], "visibility": "hidden"}]}
    with pytest.raises(ValueError):
        batch_oracle.info_hazards(bad, [["A"], ["B"]])


def test_positional_vs_overall_info_clean_split():
    gold = _gold_info()
    part = [["A", "C", "D", "E"], ["B", "F"]]                 # C/D same batch unobservable
    assert batch_oracle.positional_info_clean(gold, part) is False   # A/B cross-batch unobservable
    # still dirty at same batch: move A/B to same batch → positional clean, overall still dirty (C/D same batch unobservable)
    part2 = [["A", "B", "E"], ["C", "D", "F"]]
    assert batch_oracle.positional_info_clean(gold, part2) is True
    assert batch_oracle.overall_info_clean(gold) is False


def test_is_info_annotated():
    assert batch_oracle.is_info_annotated(_gold_info()) is True
    plain = {"repo_id": "p", "prs": ["A", "B"], "must_hold": [], "constraints": [
        {"type": "forbidden_set", "members": ["A", "B"], "visibility": "hidden"}]}
    assert batch_oracle.is_info_annotated(plain) is False     # missing inferability


def test_info_touched_variant_aligned():
    # buffered witness includes members of the recovered unobservable pair → touched count aligned by variant
    gold = {"repo_id": "t", "prs": ["A", "B", "C"], "must_hold": [], "constraints": [
        {"type": "all_or_none_group", "members": ["A", "B"], "reason": "safety",
         "visibility": "public", "inferability": "unobservable"}]}
    part = [["A", "C"], ["B"]]
    nd = batch_oracle.info_touched_pr_count(gold, part, variant="no_deferral")
    bf = batch_oracle.info_touched_pr_count(gold, part, variant="buffered", B=2, T=1)
    assert nd == 0 and bf == 2                                # no-deferral witness does not include A/B; buffered does
