"""Strict parsing of the public experiment registry and matrix configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


_COHORTS = {"primary", "extension", "dev"}
_STATUSES = {"ready", "provisional", "dev_only"}
_SOURCE_KINDS = {"heldout", "openclaw"}
_ADAPTERS = {
    "python311",
    "python314",
    "go125",
    "go126",
    "node-zod",
    "node-openclaw",
    "node-vercel-ai",
    "node-yaml",
    "bun-opencode",
}


def _relative_path(value, field, *, allow_none=False):
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a relative path without '..': {value}")
    return value


# In the frozen experiment registry, paths use the old directory name `builders/` from before this package
# was renamed. The registry file is a frozen experiment input and must not change by even one byte;
# instead, the old prefix is mapped to the current directory name **at path resolution time**.
# The mapping is applied only when constructing filesystem paths; the strings stored in the registry
# are preserved verbatim, byte-for-byte identical to the frozen file.
_LEGACY_PACKAGE_PREFIX = "builders/"
_PACKAGE_PREFIX = "bulkpr/"


def map_package_prefix(relative):
    """Replace the pre-rename `builders/` prefix from the frozen experiment registry with the current package directory `bulkpr/`.

    Call this only when constructing filesystem paths; the string stored in the registry is left
    unchanged, byte-for-byte identical to the frozen file.
    """
    if relative.startswith(_LEGACY_PACKAGE_PREFIX):
        return _PACKAGE_PREFIX + relative[len(_LEGACY_PACKAGE_PREFIX):]
    return relative


@dataclass(frozen=True)
class PoolEntry:
    repo_id: str
    cohort: str
    paper_status: str
    source_kind: str
    pool_dir: str | None
    repo_cache: str
    order_beacon: str | None
    snapshot: str
    language_adapter: str

    @classmethod
    def from_dict(cls, data):
        repo_id = data.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError("repo_id must be a non-empty string")
        cohort = data.get("cohort")
        paper_status = data.get("paper_status")
        source_kind = data.get("source_kind")
        adapter = data.get("language_adapter")
        if cohort not in _COHORTS:
            raise ValueError(f"unknown cohort for {repo_id}: {cohort}")
        if paper_status not in _STATUSES:
            raise ValueError(f"unknown paper_status for {repo_id}: {paper_status}")
        if source_kind not in _SOURCE_KINDS:
            raise ValueError(f"unknown source_kind for {repo_id}: {source_kind}")
        if adapter not in _ADAPTERS:
            raise ValueError(f"unknown language_adapter for {repo_id}: {adapter}")

        heldout = source_kind == "heldout"
        pool_dir = _relative_path(data.get("pool_dir"), "pool_dir", allow_none=not heldout)
        order_beacon = _relative_path(
            data.get("order_beacon"), "order_beacon", allow_none=not heldout
        )
        return cls(
            repo_id=repo_id,
            cohort=cohort,
            paper_status=paper_status,
            source_kind=source_kind,
            pool_dir=pool_dir,
            repo_cache=_relative_path(data.get("repo_cache"), "repo_cache"),
            order_beacon=order_beacon,
            snapshot=_relative_path(data.get("snapshot"), "snapshot"),
            language_adapter=adapter,
        )


@dataclass(frozen=True)
class Registry:
    pools: tuple[PoolEntry, ...]

    def by_repo(self, repo_id):
        for pool in self.pools:
            if pool.repo_id == repo_id:
                return pool
        raise KeyError(repo_id)


@dataclass(frozen=True)
class MatrixConfig:
    trial_count: int
    prompt_conditions: tuple[str, ...]
    random_order_count: int
    random_ks: tuple[int, ...]
    buffer_B: int
    buffer_T: int
    buffered_oracle_B: tuple[int, ...]
    buffered_oracle_T: tuple[int, ...]
    models: tuple[str, ...]


def load_registry(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != "paper-registry/v1":
        raise ValueError("registry schema_version must be paper-registry/v1")
    pools = tuple(PoolEntry.from_dict(item) for item in data.get("pools", []))
    ids = [pool.repo_id for pool in pools]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate repo_id in paper registry")
    if not pools:
        raise ValueError("paper registry must contain at least one pool")
    return Registry(pools)


def _positive_int(value, field):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _positive_int_tuple(values, field):
    result = tuple(_positive_int(value, field) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def load_matrix_config(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != "paper-matrix/v1":
        raise ValueError("matrix schema_version must be paper-matrix/v1")
    prompts = tuple(data["prompt_conditions"])
    if prompts != ("generic", "checklist"):
        raise ValueError("prompt_conditions must be generic, checklist in that order")
    buffer_cfg = data["buffered_agent"]
    oracle_cfg = data["buffered_oracle_grid"]
    models = tuple(data.get("models", []))
    if any(not isinstance(model, str) or not model for model in models):
        raise ValueError("models entries must be non-empty strings")
    return MatrixConfig(
        trial_count=_positive_int(data["trial_count"], "trial_count"),
        prompt_conditions=prompts,
        random_order_count=_positive_int(data["random_order_count"], "random_order_count"),
        random_ks=_positive_int_tuple(data["random_ks"], "random_ks"),
        buffer_B=_positive_int(buffer_cfg["B"], "buffered_agent.B"),
        buffer_T=_positive_int(buffer_cfg["T"], "buffered_agent.T"),
        buffered_oracle_B=_positive_int_tuple(oracle_cfg["B"], "buffered_oracle_grid.B"),
        buffered_oracle_T=_positive_int_tuple(oracle_cfg["T"], "buffered_oracle_grid.T"),
        models=models,
    )
