"""bulkpr flaky pre-filter "random order" plugin v1.

Deterministic shuffle with a fixed seed: sorts by sha256(seed‖nodeid) — same seed gives
the same order (reproducible), different seeds give different orders (genuine reordering).
Seed is read from $BULKPR_SHUFFLE_SEED; if unset, order is unchanged.
Zero third-party dependencies (the protocol requires "random order"; reversed order does
not qualify — caught in adversarial review M1).
"""
import hashlib
import os


def pytest_collection_modifyitems(config, items):
    seed = os.environ.get("BULKPR_SHUFFLE_SEED")
    if not seed:
        return
    items.sort(key=lambda it: hashlib.sha256((seed + it.nodeid).encode()).hexdigest())
