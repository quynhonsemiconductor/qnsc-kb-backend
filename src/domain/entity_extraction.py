"""Best-effort entity & relationship extraction for one document.

Unlike `auto_tagging.py`/`department_routing.py` -- which run during draft review and
produce a per-document suggestion a reviewer can edit before publish -- this module runs
ONCE per published article, from `governance.py::approve_draft`, alongside the
contradiction check and structured-metadata extraction it sits next to there. The result
is not a field on one draft; it is merged into a graph shared by every article the
tenant has ever published (see `graph_service.apply_extraction`), so it belongs after
publish is durable, not before it, the same way `structured_metadata` only ever runs
once per article rather than on every draft revision.

Deliberately never raises and never blocks publish: a broken or empty extraction here
must not be why an otherwise-valid draft could not be approved.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

#: Past this a reply is being asked to re-outline the whole document rather than name
#: its subject, and the prompt cost keeps climbing with little retrieval benefit.
MAX_ENTITIES = 15
MAX_RELATIONSHIPS = 20

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)

#: A closed vocabulary, not because the LLM cannot invent others, but because a graph
#: whose node types keep growing per-document is not a schema a reviewer or a UI filter
#: can reason about (the same "let the LLM decide the schema" trap the GraphRAG research
#: for this feature warned produces noise). An entity of a real but uncovered kind still
#: gets in as "concept" or "other" -- covered by _ALLOWED_ENTITY_TYPES below -- rather
#: than being dropped.
_ALLOWED_ENTITY_TYPES = {
    "person",
    "organization",
    "system",
    "policy",
    "location",
    "product",
    "concept",
    "other",
}

_SYSTEM_PROMPT = """You extract a knowledge graph from one company knowledge-base document.

Return ONLY a JSON object in this exact shape, nothing else -- no explanation, no
markdown code fence:
{"entities": [{"name": "...", "type": "...", "description": "..."}],
 "relationships": [{"source": "...", "relation": "...", "target": "...", "description": "..."}]}

Rules:
- Use up to 15 entities and 20 relationships.
- type must be exactly one of: person, organization, system, policy, location, product,
  concept, other.
- Every "source" and "target" in relationships must exactly match a "name" from entities.
- relation is a short lowercase snake_case verb phrase (e.g. "reports_to", "depends_on",
  "part_of", "owns", "supersedes", "approved_by"). Keep it factual and specific to the
  document -- do not invent a relationship the text does not support.
- description is one short sentence, or empty.
- Only extract named, specific entities the document is actually about. Skip generic or
  common nouns that are not a real named thing.
"""


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class ExtractedRelationship:
    source: str
    relation: str
    target: str
    description: str


@dataclass(frozen=True)
class ExtractionResult:
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


def _clean_name(raw: object) -> str | None:
    value = re.sub(r"\s+", " ", str(raw).strip())
    if not value or len(value) > 200:
        return None
    return value


def _clean_relation(raw: object) -> str | None:
    value = re.sub(r"[^a-z0-9_]+", "_", str(raw).strip().lower()).strip("_")
    if not value or len(value) > 100:
        return None
    return value


def _clean_description(raw: object) -> str:
    return re.sub(r"\s+", " ", str(raw or "").strip())[:500]


def normalize_entity_name(value: str) -> str:
    """Accent- and case-insensitive key used to dedupe entities per tenant.

    Same NFKD-fold convention `auto_tagging.py` uses for catalogue matching and
    `articles.py` uses for tag comparisons, applied here to entity names so "An toan lao
    dong" and "An toàn lao động" merge into one node instead of two.
    """
    return re.sub(
        r"\s+",
        " ",
        "".join(
            ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch)
        )
        .strip()
        .casefold(),
    )


def _parse_entities(raw_entities: object) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    seen: set[str] = set()
    if not isinstance(raw_entities, list):
        return entities
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        name = _clean_name(item.get("name"))
        entity_type = str(item.get("type") or "").strip().lower()
        if not name or entity_type not in _ALLOWED_ENTITY_TYPES:
            continue
        key = normalize_entity_name(name)
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            ExtractedEntity(name=name, type=entity_type, description=_clean_description(item.get("description")))
        )
        if len(entities) >= MAX_ENTITIES:
            break
    return entities


def _parse_relationships(
    raw_relationships: object, known_keys: set[str]
) -> list[ExtractedRelationship]:
    relationships: list[ExtractedRelationship] = []
    seen_edges: set[tuple[str, str, str]] = set()
    if not isinstance(raw_relationships, list):
        return relationships
    for item in raw_relationships:
        if not isinstance(item, dict):
            continue
        source = _clean_name(item.get("source"))
        target = _clean_name(item.get("target"))
        relation = _clean_relation(item.get("relation"))
        if not source or not target or not relation:
            continue
        source_key, target_key = normalize_entity_name(source), normalize_entity_name(target)
        # A relation naming something outside this same reply's `entities` has nothing
        # to attach to; accepting it would mean silently inventing a bare entity for it.
        if source_key not in known_keys or target_key not in known_keys or source_key == target_key:
            continue
        edge_key = (source_key, relation, target_key)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        relationships.append(
            ExtractedRelationship(
                source=source,
                relation=relation,
                target=target,
                description=_clean_description(item.get("description")),
            )
        )
        if len(relationships) >= MAX_RELATIONSHIPS:
            break
    return relationships


async def extract_entities_and_relationships(title: str, body_md: str) -> ExtractionResult:
    """Best-effort graph extraction for one document. Returns [] rather than raising.

    Every failure mode (no provider configured, the call fails, the reply is not
    parseable JSON, the shape is wrong) is swallowed and logged, mirroring
    `auto_tagging.suggest_tags_for_document`.
    """
    # Imported here, not at module load, so a test can monkeypatch
    # `src.domain.llm_client.complete`/`resolve_provider` the same way every other
    # LLM-calling module in this codebase is tested.
    from src.core.config import settings
    from src.domain.llm_client import complete, resolve_provider

    empty = ExtractionResult(entities=[], relationships=[])
    if not settings.GRAPH_EXTRACTION_ENABLED:
        return empty
    provider = resolve_provider()
    if not provider:
        return empty

    try:
        answer, _tokens, _model, _provider_name = await complete(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"TITLE: {title}\n\nCONTENT:\n{body_md[:8000]}",
                },
            ],
            timeout=settings.GRAPH_EXTRACTION_TIMEOUT_SECONDS,
            # Hidden reasoning would spend the token budget before the JSON appears --
            # same reasoning as department_routing.py's LLM call.
            thinking=False,
            max_tokens=1500,
        )
    except Exception as exc:
        logger.warning("Entity extraction call failed", error=str(exc))
        return empty

    cleaned = _CODE_FENCE_RE.sub("", answer.strip()).strip()
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "Entity extraction returned unparseable JSON", error=str(exc), reply=cleaned[:200]
        )
        return empty
    if not isinstance(payload, dict):
        logger.warning("Entity extraction returned a non-object JSON payload", reply=cleaned[:200])
        return empty

    entities = _parse_entities(payload.get("entities"))
    known_keys = {normalize_entity_name(entity.name) for entity in entities}
    relationships = _parse_relationships(payload.get("relationships"), known_keys)
    return ExtractionResult(entities=entities, relationships=relationships)
