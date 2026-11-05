"""Mark the lessons AI assistants already saved as lessons (articles.type = 'LESSON')

Lessons used to be ordinary articles of type REFERENCE with the tag `session-lesson`, published only after a reviewer
approved them. They are now their own kind of article (src/models/article.py `LESSON_TYPE`), published at once, and
search and the assistant can be limited to reviewed documents or to lessons by that type. This labels the ones that
already exist, so they are treated like the new ones. Drafts still waiting for review are unchanged: when approved they
become articles of the type they were submitted with.

Idempotent. The downgrade sets them back to REFERENCE, the type they were approved with.

Revision ID: 20261104_93
Revises: 20261104_92
Create Date: 2026-11-04
"""

from alembic import op
from sqlalchemy import inspect

revision = "20261104_93"
down_revision = "20261104_92"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = set(inspect(op.get_bind()).get_table_names())
    if not {"articles", "article_tags"} <= tables:
        return
    op.execute(
        "UPDATE articles SET type = 'LESSON' "
        "WHERE type <> 'LESSON' "
        "AND id IN (SELECT article_id FROM article_tags WHERE tag = 'session-lesson')"
    )


def downgrade() -> None:
    tables = set(inspect(op.get_bind()).get_table_names())
    if not {"articles", "article_tags"} <= tables:
        return
    # The rows upgrade() converted carry the lesson tag, and so does every lesson saved since: the two cannot be told
    # apart, so a downgrade after new lessons exist turns those into REFERENCE too. Only tagged rows are touched.
    op.execute(
        "UPDATE articles SET type = 'REFERENCE' "
        "WHERE type = 'LESSON' "
        "AND id IN (SELECT article_id FROM article_tags WHERE tag = 'session-lesson')"
    )
