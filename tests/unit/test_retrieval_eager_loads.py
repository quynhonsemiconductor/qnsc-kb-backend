"""Retrieval must eager-load every Article relationship its consumers read.

The async session cannot lazy-load. A relationship missing from the retrieval loader
options is therefore not a slow path — it is `MissingGreenlet: greenlet_spawn has not been
called` at request time, which surfaces as "AI generation failed" with no clue why.

This happened: `Article.sources` was removed as unused because permissions reads it as
`getattr(article, "sources", [])`, so a search for `.sources` found nothing. That is what
this test exists to catch, by reading the attribute names out of the permission code
itself rather than trusting a hand-maintained list.
"""
from __future__ import annotations

import inspect
import re

from src.domain import permissions
from src.repositories.chunk import RETRIEVAL_LOAD_OPTIONS

#: Relationships the response formatter reads directly (see search_service.search).
FORMATTER_RELATIONSHIPS = {"owner", "parent_chunk", "child_chunks"}


def _loaded_attribute_names() -> set[str]:
    """The relationship names the retrieval statements eager-load."""
    names: set[str] = set()
    for option in RETRIEVAL_LOAD_OPTIONS:
        for element in option.context:
            path = getattr(element, "path", None) or ()
            for entry in path:
                key = getattr(entry, "key", None)
                if isinstance(key, str):
                    names.add(key)
    return names


def _article_attributes_read_by_permissions() -> set[str]:
    """Every `getattr(article, "...")` and `article.x` in the permission module."""
    source = inspect.getsource(permissions)
    quoted = set(re.findall(r'getattr\(\s*article\s*,\s*["\'](\w+)["\']', source))
    dotted = set(re.findall(r'\barticle\.(\w+)', source))
    return quoted | dotted


def test_every_article_relationship_permissions_reads_is_eager_loaded():
    loaded = _loaded_attribute_names()
    # Only relationships matter; scalar columns come back with the row.
    from src.models.article import Article

    relationships = {rel.key for rel in Article.__mapper__.relationships}
    required = _article_attributes_read_by_permissions() & relationships

    assert required, "the permission module must read at least one relationship"
    missing = required - loaded
    assert not missing, (
        f"retrieval does not eager-load {sorted(missing)}, which "
        "PermissionService reads on every retrieved chunk. The async session cannot "
        "lazy-load, so this raises MissingGreenlet at request time."
    )


def test_sources_is_eager_loaded():
    """Named explicitly because it is the one that was removed as 'unused'."""
    assert "sources" in _loaded_attribute_names()


def test_the_response_formatter_relationships_are_eager_loaded():
    loaded = _loaded_attribute_names()

    assert FORMATTER_RELATIONSHIPS <= loaded, sorted(FORMATTER_RELATIONSHIPS - loaded)


def test_the_options_are_shared_not_duplicated_per_statement():
    """Both statements use one definition; duplicating them is how sources got dropped."""
    from src.repositories import chunk as chunk_module

    source = inspect.getsource(chunk_module.ChunkRepository.hybrid_search)

    assert source.count("RETRIEVAL_LOAD_OPTIONS") == 2
    assert "selectinload(" not in source, "loader options belong in one shared tuple"
