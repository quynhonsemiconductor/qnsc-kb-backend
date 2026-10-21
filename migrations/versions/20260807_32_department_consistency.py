"""reconcile department master data and article search permissions

Revision ID: 20260807_32
Revises: 20260807_31
Create Date: 2026-08-07
"""

from alembic import op

revision = "20260807_32"
down_revision = "20260807_31"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "UPDATE users SET dept = NULL WHERE dept IS NOT NULL AND btrim(dept) = ''"
    )
    op.execute(
        "UPDATE pending_drafts SET dept = NULL WHERE dept IS NOT NULL AND btrim(dept) = ''"
    )
    op.execute(
        "UPDATE gaps SET dept = NULL WHERE dept IS NOT NULL AND btrim(dept) = ''"
    )
    op.execute("""
        INSERT INTO departments (id, created_at, updated_at, company_domain, name, active)
        SELECT gen_random_uuid(), now(), now(), source.company_domain, source.name, TRUE
        FROM (
            SELECT company_domain, min(btrim(dept)) AS name
            FROM (
                SELECT company_domain, dept FROM users WHERE dept IS NOT NULL AND btrim(dept) <> ''
                UNION ALL
                SELECT company_domain, dept FROM articles WHERE dept IS NOT NULL AND btrim(dept) <> ''
                UNION ALL
                SELECT company_domain, dept FROM pending_drafts WHERE dept IS NOT NULL AND btrim(dept) <> ''
                UNION ALL
                SELECT company_domain, dept FROM gaps WHERE dept IS NOT NULL AND btrim(dept) <> ''
            ) values_to_seed
            GROUP BY company_domain, lower(btrim(dept))
        ) source
        WHERE NOT EXISTS (
            SELECT 1 FROM departments d
            WHERE d.company_domain = source.company_domain
              AND lower(d.name) = lower(source.name)
        )
    """)
    op.execute("""
        UPDATE users u SET dept = d.name
        FROM departments d
        WHERE d.company_domain = u.company_domain
          AND lower(d.name) = lower(btrim(u.dept))
          AND u.dept IS NOT NULL
    """)
    op.execute("""
        UPDATE articles a SET dept = d.name
        FROM departments d
        WHERE d.company_domain = a.company_domain
          AND lower(d.name) = lower(btrim(a.dept))
    """)
    op.execute("""
        UPDATE pending_drafts p SET dept = d.name
        FROM departments d
        WHERE d.company_domain = p.company_domain
          AND lower(d.name) = lower(btrim(p.dept))
          AND p.dept IS NOT NULL
    """)
    op.execute("""
        UPDATE gaps g SET dept = d.name
        FROM departments d
        WHERE d.company_domain = g.company_domain
          AND lower(d.name) = lower(btrim(g.dept))
          AND g.dept IS NOT NULL
    """)
    op.execute("""
        INSERT INTO access_groups (id, created_at, updated_at, name, company_domain, bitmask_position)
        SELECT gen_random_uuid(), now(), now(), missing.name, missing.company_domain,
               current_max.max_position + missing.row_number
        FROM (
            SELECT d.company_domain,
                   'dept_' || lower(d.name) AS name,
                   row_number() OVER (PARTITION BY d.company_domain ORDER BY d.name) AS row_number
            FROM departments d
            WHERE d.active = TRUE
              AND NOT EXISTS (
                  SELECT 1 FROM access_groups ag
                  WHERE ag.company_domain = d.company_domain
                    AND lower(ag.name) = lower('dept_' || d.name)
              )
        ) missing
        JOIN (
            SELECT company_domain, coalesce(max(bitmask_position), -1) AS max_position
            FROM access_groups
            GROUP BY company_domain
        ) current_max ON current_max.company_domain = missing.company_domain
        WHERE current_max.max_position + missing.row_number < 62
    """)
    op.execute("""
        INSERT INTO article_access (article_id, group_id)
        SELECT a.id, ag.id
        FROM articles a
        JOIN access_groups ag
          ON ag.company_domain = a.company_domain
         AND lower(ag.name) = lower('dept_' || a.dept)
        WHERE a.sensitivity <> 'public'
          AND a.dept IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM article_access existing
              WHERE existing.article_id = a.id
          )
        ON CONFLICT DO NOTHING
    """)
    op.execute("""
        UPDATE article_chunks c
        SET department_id = a.dept,
            sensitivity = a.sensitivity,
            visibility = a.sensitivity,
            access_group_bitmap = CASE
                WHEN a.sensitivity = 'public' THEN 1
                ELSE coalesce((
                    SELECT bit_or(1::bigint << ag.bitmask_position)
                    FROM article_access aa
                    JOIN access_groups ag ON ag.id = aa.group_id
                    WHERE aa.article_id = a.id
                ), 0)
            END
        FROM articles a
        WHERE a.id = c.article_id
    """)


def downgrade() -> None:
    pass
