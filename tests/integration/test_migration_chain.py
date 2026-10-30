from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_migrations_have_one_current_head():
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "migrations" / "alembic.ini"))
    script = ScriptDirectory.from_config(config)
    # Pinned on purpose: a new revision must be a deliberate act, and a second head
    # means one was chained off the wrong parent. Bumped for 20260912_81, the last of
    # five migrations chained in sequence off 20260911_76 (the tenant knowledge graph):
    # the pgvector column realigned to 1024 for multilingual-e5-large-instruct (77), a
    # re-index after adding contextual chunk headers (78), conflict_records.contradiction_type
    # (79), approval_rules.risk_tiers/version plus the approval_rule_versions history
    # table (80), and article_chunks.content_hash for skip-if-unchanged re-embedding (81).
    assert script.get_heads() == ["20261030_89"]
