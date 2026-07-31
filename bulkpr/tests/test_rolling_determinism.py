"""Cross-process determinism: rolling baselines with a fixed seed must produce identical results
regardless of PYTHONHASHSEED.

Background: the `random` baseline uses a fixed seed, but its sampling output was fed into
a Python `set`. Set iteration order depends on hash randomisation, so the same seed can
produce different PR selections in different processes.
This test runs the same input in **subprocesses** varying only PYTHONHASHSEED and compares
digests one by one.
"""

import hashlib
import json
import os
import subprocess
import sys

BUILDERS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Five distinct hash seeds; `0` disables randomisation, the rest are arbitrary fixed values.
HASH_SEEDS = ("0", "1", "12345", "67890", "424242")

# Child process script: build a large enough gold fixture, run all six baselines once each,
# and serialise the results into a digest.
CHILD_SCRIPT = r"""
import hashlib, json, sys

sys.path.insert(0, sys.argv[1])
import rolling

PRS = ["pr-%02d" % i for i in range(24)]
GOLD = {
    "repo_id": "det",
    "prs": list(PRS),
    "must_hold": [],
    "constraints": [
        {"type": "all_or_none_group", "members": ["pr-00", "pr-09"],
         "reason": "safety", "visibility": "public"},
        {"type": "all_or_none_group", "members": ["pr-03", "pr-14"],
         "reason": "safety", "visibility": "hidden"},
        {"type": "forbidden_set", "members": ["pr-05", "pr-11"], "visibility": "public"},
        {"type": "forbidden_set", "members": ["pr-07", "pr-18"], "visibility": "hidden"},
        {"type": "forbidden_set", "members": ["pr-02", "pr-21"], "visibility": "hidden"},
        {"type": "depends_on", "source": "pr-13", "target": "pr-04", "visibility": "public"},
        {"type": "depends_on", "source": "pr-20", "target": "pr-08", "visibility": "hidden"},
    ],
}
PARTITION = [PRS[i:i + 4] for i in range(0, len(PRS), 4)]
SEED = 20260728
B, T = 4, 16


def _jsonable(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _run(name):
    if name == "random":
        return rolling.run_rolling(GOLD, PARTITION, rolling.random_strategy(SEED))
    if name == "random-buffered":
        return rolling.run_rolling(GOLD, PARTITION, rolling.random_buffered_strategy(SEED),
                                   variant="buffered", B=B, T=T)
    if name == "greedy-ci":
        return rolling.run_rolling(GOLD, PARTITION, rolling.greedy_ci_strategy)
    if name == "greedy-ci-buffered":
        return rolling.run_rolling(GOLD, PARTITION, rolling.greedy_ci_buffered_strategy,
                                   variant="buffered", B=B, T=T)
    if name == "merge-all":
        return rolling.run_rolling(GOLD, PARTITION, rolling.merge_all_strategy)
    if name == "clairvoyant":
        return rolling.run_rolling(GOLD, PARTITION,
                                   rolling.clairvoyant_strategy(GOLD, PARTITION))
    if name == "clairvoyant-buffered":
        return rolling.run_rolling(
            GOLD, PARTITION,
            rolling.clairvoyant_buffered_strategy(GOLD, PARTITION, B, T),
            variant="buffered", B=B, T=T)
    raise ValueError(name)


NAMES = ["random", "random-buffered", "greedy-ci", "greedy-ci-buffered",
         "merge-all", "clairvoyant", "clairvoyant-buffered"]

out = {}
for name in NAMES:
    res = _run(name)
    out[name] = {
        "final_merged": sorted(res.final_merged),
        "merge_plan": _jsonable(res.merge_plan),
        "per_batch": _jsonable(res.per_batch),
        "pending_final": sorted(res.pending_final),
        "all_prefix_safe": res.all_prefix_safe,
        "first_failure_batch": res.first_failure_batch,
        "first_public_rejection_batch": res.first_public_rejection_batch,
        "public_rejection_count": res.public_rejection_count,
        "public_ci_query_count": res.public_ci_query_count,
        "public_ci_query_reject_count": res.public_ci_query_reject_count,
        "score": _jsonable(rolling.score_rolling(GOLD, res)),
    }

digests = {name: hashlib.sha256(
    json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    for name, payload in out.items()}
print(json.dumps(digests, sort_keys=True))
"""


def _digests_under(hash_seed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    proc = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT, BUILDERS_DIR],
        capture_output=True, text=True, env=env, check=False,
    )
    assert proc.returncode == 0, f"child failed (PYTHONHASHSEED={hash_seed}):\n{proc.stderr}"
    return json.loads(proc.stdout)


def test_rolling_baselines_identical_across_hash_seeds():
    """All six baselines produce byte-identical results across 5 different PYTHONHASHSEED values."""
    per_seed = {seed: _digests_under(seed) for seed in HASH_SEEDS}
    reference_seed = HASH_SEEDS[0]
    reference = per_seed[reference_seed]

    unstable = {}
    for name in reference:
        seen = {seed: digests[name] for seed, digests in per_seed.items()}
        if len(set(seen.values())) != 1:
            unstable[name] = seen
    assert not unstable, (
        "these baselines differ across PYTHONHASHSEED values (the seed is fixed, "
        "so a result was fed through an unordered set somewhere): "
        + json.dumps(unstable, indent=2, sort_keys=True)
    )


def test_rolling_context_orders_pending_by_defer_order():
    """pending / available are ordered by defer submission order, not via a set.

    The defined order is FIFO: deferred PRs appear first in the order they were deferred;
    current-batch PRs follow in batch order. Previously a `set` was used here,
    making the order accidental and seed-dependent across processes.
    """
    gold = {
        "repo_id": "ctx", "prs": ["p1", "p2", "p3", "p4"], "must_hold": [],
        "constraints": [],
    }
    partition = [["p1", "p2"], ["p3", "p4"]]
    rolling = rolling_module()
    state = rolling.initial_rolling_state(gold, partition, variant="buffered", B=4, T=16)
    state = rolling.advance_rolling_state(gold, state, {"merge": [], "defer": ["p2", "p1"]})
    assert state["pending"] == ["p2", "p1"]
    ctx = rolling.rolling_context(gold, state)
    assert ctx["available"] == ["p2", "p1", "p3", "p4"]

    # Reversing the defer order reverses the result — the ordering truly comes from the proposal,
    # not just coincidentally matching sorted order.
    state2 = rolling.initial_rolling_state(gold, partition, variant="buffered", B=4, T=16)
    state2 = rolling.advance_rolling_state(gold, state2, {"merge": [], "defer": ["p1", "p2"]})
    assert state2["pending"] == ["p1", "p2"]
    assert rolling.rolling_context(gold, state2)["available"] == ["p1", "p2", "p3", "p4"]


def rolling_module():
    sys.path.insert(0, BUILDERS_DIR)
    import rolling  # noqa: PLC0415

    return rolling
