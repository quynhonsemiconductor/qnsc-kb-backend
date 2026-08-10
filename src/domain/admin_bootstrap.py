"""Create the first global administrator at startup.

A migrated database has no users, and there is no way in through the API to fix that:
the only unauthenticated route is POST /login, POST /users requires the `user.manage`
permission, and there is no signup route. Without a bootstrap step a fresh deployment is
unreachable by everyone, including whoever deployed it.

scripts/create_admin.py does the same thing as a deliberate one-off. This runs it
automatically instead, on every API start, so `docker compose up` yields an account you
can log in with.

THREE THINGS KEEP THAT FROM BEING A BACK DOOR:

  1. It fires ONLY when the deployment has no global administrator at all. Not "no
     account with this address" — deleting the seeded admin on purpose must not be undone
     by the next restart, and an operator who has already created their own admin must
     not silently acquire a second one.

  2. The development default password is REJECTED in production. Settings.
     validate_production raises before the app serves anything, so a production
     deployment either sets BOOTSTRAP_ADMIN_PASSWORD to something of its own or does not
     start. There is no path where a public deployment comes up with a password that is
     written down in this repository.

  3. It is switchable. BOOTSTRAP_ADMIN_ENABLED=false skips it entirely, for a deployment
     that provisions identities some other way.

Concurrency: several API replicas can start at once and all see an empty table. The
unique constraint on email settles it — the loser catches IntegrityError, rolls back, and
carries on, because by then the account it wanted exists.
"""
from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.config import DEVELOPMENT_DEFAULT_PASSWORD, settings
from src.core.security import get_password_hash
from src.models.rbac import Role
from src.models.user import User

logger = structlog.get_logger()


async def _global_administrator_exists(db: AsyncSession) -> bool:
    """Whether any identity already holds the global Admin role.

    Mirrors AuthorizationService.is_global_administrator: the role that bypasses tenant
    RLS is the one with company_domain = NULL, so a company-scoped Admin does not count
    and must not suppress the bootstrap.
    """
    found = await db.scalar(
        select(User.id)
        .join(User.roles)
        .where(Role.name == "Admin", Role.company_domain.is_(None))
        .limit(1)
    )
    return found is not None


async def ensure_bootstrap_admin(db: AsyncSession) -> User | None:
    """Create the configured administrator if the deployment has none.

    Returns the created user, or None when nothing was done. Expects bootstrap_rbac to
    have run already — it is what creates the global Admin role this attaches — and a
    session whose RLS context is the global-admin bypass, which is how init_db calls it.
    """
    if not settings.BOOTSTRAP_ADMIN_ENABLED:
        return None

    if await _global_administrator_exists(db):
        return None

    email = settings.BOOTSTRAP_ADMIN_EMAIL.strip().lower()
    if "@" not in email:
        logger.error(
            "BOOTSTRAP_ADMIN_EMAIL is not an email address; no administrator created",
            email=email,
        )
        return None

    admin_role = await db.scalar(
        select(Role)
        .where(Role.name == "Admin", Role.company_domain.is_(None))
        .options(selectinload(Role.permissions))
    )
    if admin_role is None:
        # bootstrap_rbac creates this unconditionally, so its absence means the RBAC
        # bootstrap did not run or did not commit. Creating the user anyway would leave
        # an account that cannot administer anything and would then suppress every later
        # attempt, because the check above only looks for the role.
        logger.error("Global Admin role is missing; no administrator created")
        return None

    user = User(
        email=email,
        name=settings.BOOTSTRAP_ADMIN_NAME.strip() or "Admin",
        password_hash=get_password_hash(settings.BOOTSTRAP_ADMIN_PASSWORD),
        company_domain=email.rsplit("@", 1)[1],
        # Kept in step with the attached role: `role` is the legacy scalar column several
        # queries still read, and the relationship below is what authorization evaluates.
        role="Admin",
        active=True,
    )
    db.add(user)

    try:
        await db.flush()
        user.roles = [admin_role]
        await db.commit()
    except IntegrityError:
        # Another replica won the race and created it first. Nothing to repair.
        await db.rollback()
        logger.info("Bootstrap administrator already created by another process", email=email)
        return None

    logger.info("Created bootstrap administrator", email=email, user_id=str(user.id))
    if settings.BOOTSTRAP_ADMIN_PASSWORD == DEVELOPMENT_DEFAULT_PASSWORD:
        logger.warning(
            "Bootstrap administrator is using the development default password; "
            "change it before this deployment is reachable by anyone else",
            email=email,
        )
    return user
