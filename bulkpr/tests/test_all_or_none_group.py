# bulkpr/tests/test_all_or_none_group.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wbsr  # noqa: E402


def _gold(constraints, prs, must_hold=None):
    return {"prs": prs, "constraints": constraints, "must_hold": must_hold or []}


def test_all_or_none_half_merge_infeasible():
    cons = [{"type": "all_or_none_group", "members": ["A", "B"], "reason": "safety"}]
    assert wbsr._feasible(["A"], cons) is False      # half-merged = infeasible
    assert wbsr._feasible(["A", "B"], cons) is True   # both members merged
    assert wbsr._feasible([], cons) is True           # neither merged


def test_all_or_none_check_safe_mirror():
    gold = _gold([{"type": "all_or_none_group", "members": ["A", "B"], "reason": "migration"}], ["A", "B"])
    ok_half, viol = wbsr.check_safe(gold, {"A"})
    assert ok_half is False and viol
    ok_both, _ = wbsr.check_safe(gold, {"A", "B"})
    assert ok_both is True


def test_all_or_none_oracle_pairs_both():
    gold = _gold([{"type": "all_or_none_group", "members": ["A", "B"], "reason": "safety"}], ["A", "B", "C"])
    assert wbsr.solve_oracle_proof(gold)["opt"] == 3   # A+B (both members) + C (free)


def test_require_set_still_alias():
    cons = [{"type": "require_set", "members": ["A", "B"]}]
    assert wbsr._feasible(["A"], cons) is False
    assert wbsr._feasible(["A", "B"], cons) is True


def test_opt_order_acyclic_ok():
    gold = {"prs": ["A", "B"], "must_hold": [],
            "constraints": [{"type": "depends_on", "source": "A", "target": "B"}]}
    assert wbsr.opt_admits_executable_order(gold) is True


def test_opt_order_cycle_detected():
    gold = {"prs": ["A", "B"], "must_hold": [], "constraints": [
        {"type": "depends_on", "source": "A", "target": "B"},
        {"type": "depends_on", "source": "B", "target": "A"}]}
    # A/B mutually dependent → oracle witness set {A,B} is feasible but has no topological order
    assert wbsr.opt_admits_executable_order(gold) is False
