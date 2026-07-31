"""Repo-level BCa bootstrap + Holm correction + degenerate-case rules + formal full-grid gate."""
import random

import pytest

from bulkpr.paper import stats_v2 as sv


def test_bca_ci_covers_true_mean_on_normal_samples():
    rng = random.Random(20260718)
    hits = 0
    for _ in range(20):
        xs = [rng.gauss(0.5, 0.1) for _ in range(20)]
        lo, hi = sv.bca_ci(xs, draws=2000, seed=7)
        assert lo <= hi
        if lo <= 0.5 <= hi:
            hits += 1
    assert hits >= 18  # rough check: 95% CI should cover the true mean in >=90% of draws


def test_bca_deterministic_given_seed():
    xs = [0.1, 0.4, 0.35, 0.8, 0.55]
    assert sv.bca_ci(xs, draws=500, seed=42) == sv.bca_ci(xs, draws=500, seed=42)


def test_all_equal_samples_degenerate_ci():
    assert sv.bca_ci([0.3, 0.3, 0.3], draws=200, seed=1) == (0.3, 0.3)


def test_two_point_sample_bounded_and_deterministic():
    # acceleration denominator = 0 only for all-equal samples (already caught by the [x,x] rule);
    # this fallback branch is purely defensive.
    # Two-point sample verifies BCa stays in bounds, does not raise, and is deterministic.
    xs = [0.2, 0.8]
    lo, hi = sv.bca_ci(xs, draws=500, seed=9)
    assert 0.2 <= lo <= hi <= 0.8
    assert (lo, hi) == sv.bca_ci(xs, draws=500, seed=9)


def test_wilcoxon_all_zero_diffs_p_one():
    assert sv.wilcoxon_signed_rank([0.0, 0.0, 0.0]) == 1.0


def test_wilcoxon_detects_consistent_difference():
    diffs = [0.1, 0.12, 0.08, 0.15, 0.09, 0.11, 0.1, 0.13]
    assert sv.wilcoxon_signed_rank(diffs) < 0.05


def test_holm_hand_computed():
    adjusted = sv.holm_correction({"ab": 0.01, "ac": 0.04, "bc": 0.03})
    assert adjusted["ab"] == pytest.approx(0.03)
    assert adjusted["bc"] == pytest.approx(0.06)
    assert adjusted["ac"] == pytest.approx(0.06)


def test_paired_repo_bootstrap_diff():
    a = {"r1": 0.5, "r2": 0.6, "r3": 0.4}
    b = {"r1": 0.2, "r2": 0.3, "r3": 0.1}
    res = sv.paired_repo_bootstrap_diff(a, b, draws=1000, seed=3)
    assert res["mean_diff"] == pytest.approx(0.3)
    assert res["ci"][0] <= 0.3 <= res["ci"][1]
    assert res["n_repos"] == 3
    assert res["wins"] == 3


def test_formal_grid_missing_cell_fails():
    rows = [{"matrix": {"repo_id": "chi"}, "model": "m1", "trial_index": 0}]
    with pytest.raises(sv.GridError):
        sv.validate_formal_grid(rows, expected_repos=["chi", "zod"],
                                models=["m1"], required_trial_indices=[0])
    ok_rows = [
        {"matrix": {"repo_id": r}, "model": "m1", "trial_index": i}
        for r in ("chi", "zod") for i in (0,)
    ]
    sv.validate_formal_grid(ok_rows, expected_repos=["chi", "zod"],
                            models=["m1"], required_trial_indices=[0])


def test_pilot_pairwise_reports_repo_set():
    a = {"r1": 0.5, "r2": 0.6}
    b = {"r1": 0.2, "r3": 0.1}
    res = sv.paired_repo_bootstrap_diff(a, b, draws=200, seed=5)
    assert res["repo_ids"] == ["r1"]     # pairwise deletion reports the matched repo set explicitly
    assert res["n_repos"] == 1
