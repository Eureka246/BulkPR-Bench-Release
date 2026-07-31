"""Base images must be pinned by digest, not by tag.

A tag moves. `node:22-bookworm-slim` picks up new minor versions, and even a
fully-qualified tag like `python:3.14-bookworm` is republished. Anyone building
these tasks months from now would then be running a different container from the
one the results came out of, with nothing in the output to say so.
"""
import re

import pytest

from bulkpr.paper import taskgen


FROM_LINE = re.compile(r"^FROM (?P<image>\S+)$", re.MULTILINE)
DIGEST = re.compile(r"^[a-z0-9./_-]+:[\w.-]+@sha256:[0-9a-f]{64}$")


def _dockerfile_for(adapter, repo="attrs"):
    return taskgen._dockerfile({"repo_id": repo, "language_adapter": adapter})


ADAPTERS = [("python314", "networkx"), ("python311", "attrs"), ("go125", "chi"),
            ("go126", "adk-go"), ("node-zod", "zod"),
            ("node-vercel-ai", "vercel-ai"), ("node-openclaw", "openclaw"),
            ("node-yaml", "yaml"), ("bun-opencode", "opencode")]


@pytest.mark.parametrize("adapter,repo", ADAPTERS)
def test_every_adapter_emits_a_digest_pinned_from_line(adapter, repo):
    matches = FROM_LINE.findall(_dockerfile_for(adapter, repo))
    assert matches, "no FROM line at all"
    for image in matches:
        assert DIGEST.match(image), f"{adapter}: {image} is not digest-pinned"


def test_the_digest_table_covers_every_image_the_generator_can_pick():
    used = {image for adapter, repo in ADAPTERS
            for image in FROM_LINE.findall(_dockerfile_for(adapter, repo))}
    assert used == {taskgen.pinned_base_image(tag)
                    for tag in taskgen.BASE_IMAGE_DIGESTS}


def test_an_unregistered_image_fails_loudly_instead_of_falling_back_to_the_tag():
    with pytest.raises(ValueError, match="no pinned digest"):
        taskgen.pinned_base_image("ubuntu:24.04")


def test_every_registered_digest_is_a_full_sha256():
    for tag, digest in taskgen.BASE_IMAGE_DIGESTS.items():
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest), tag
