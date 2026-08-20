from typing import AsyncGenerator
from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from sqlalchemy import event, text
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from src.core.config import settings
from src.models import User
from src.repositories.user import UserRepository
from src.domain.rbac import AuthorizationService, bootstrap_rbac
from src.domain.admin_bootstrap import ensure_bootstrap_admin
from src.domain.llm_config import load_runtime_config

engine = create_async_engine(
    settings.DATABASE_URL,
    connect_args={
        "timeout": 5,
        "command_timeout": 5,
    },
)
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine, class_=AsyncSession)

# OAuth2 scheme point to login route
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login", auto_error=False)

async def init_db() -> None:
    # Runtime schema creation and ALTER statements are intentionally forbidden.
    # The deployment entrypoint runs Alembic before the application starts.
    if settings.AUTO_CREATE_SCHEMA:
        raise RuntimeError("AUTO_CREATE_SCHEMA is disabled; run Alembic migrations before starting the application")
    async with SessionLocal() as db:
        # Startup bootstrap is an internal maintenance operation.  It must
        # remain able to seed roles after production RLS has been migrated.
        await set_database_context(db, None, True)
        await bootstrap_rbac(db)
        # After bootstrap_rbac, which is what creates the global Admin role this attaches,
        # and inside the same global-admin RLS context — the identity policies FORCE row
        # security on the owner too, so without the bypass above the insert would be
        # filtered rather than rejected and the startup would report success having
        # written nothing.
        await ensure_bootstrap_admin(db)
        await load_runtime_config(db)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            # No RESET of app.* here any more, deliberately. The tenant context is
            # TRANSACTION-local (see set_database_context), so it is gone the moment this
            # session's last transaction ends — a pooled connection cannot carry it to the
            # next request. The old reset was what turned a post-commit connection swap
            # into a request running with no context at all.
            await session.rollback()


# The tenant context, re-applied at the start of EVERY transaction.
#
# `true` is the last argument to each set_config: these are TRANSACTION-local, not
# session-local. That one letter is the whole fix.
#
# Session-scoped values live on the CONNECTION, and an AsyncSession releases its
# connection back to the pool on commit and takes one again for the next statement — a
# swap measured on 11 of 12 runs. get_db() then RESETs app.* before returning a
# connection, correctly, so no request inherits another's tenant. The two mechanisms
# fought each other: after any commit the session could pick up a cleaned connection and
# every following statement ran with NO context. On a FORCEd-RLS table that means reads
# return nothing — /ai/ask surfaced it as "Could not refresh instance", but a plain SELECT
# after a commit would simply have returned zero rows and said nothing at all.
#
# Transaction-local values cannot leak, because they die with the transaction. That makes
# get_db's RESET block unnecessary, which is why it is gone.
_TENANT_SQL = text(
    "SELECT set_config('app.company_domain', :company_domain, true),"
    " set_config('app.global_admin', :global_admin, true),"
    " set_config('app.global_article_access', :global_article_access, true),"
    " set_config('app.global_identity_access', :global_identity_access, true),"
    " set_config('app.global_connector_access', :global_connector_access, true),"
    " set_config('app.global_governance_access', :global_governance_access, true),"
    " set_config('app.user_id', :user_id, true)"
)

TENANT_CONTEXT_KEY = "tenant_context"


@event.listens_for(Session, "after_begin")
def _reapply_tenant_context(session: Session, transaction, connection) -> None:
    """Re-issue the session's tenant context whenever a transaction begins.

    Including the implicit transaction that follows a commit, on whatever pooled
    connection it lands on. Without this, transaction-local settings would be correct but
    would vanish at the first commit; with it, they are correct for every statement and
    no call site has to remember anything.

    Sessions that never set a context — the migrator, workers, startup — are skipped, so
    they keep failing closed under RLS rather than silently acquiring someone else's.
    """
    context = session.info.get(TENANT_CONTEXT_KEY)
    if context:
        connection.execute(_TENANT_SQL, context)


async def set_database_context(
    db: AsyncSession,
    company_domain: str | None,
    global_admin: bool = False,
    user_id: str | None = None,
    global_article_access: bool = False,
    global_identity_access: bool = False,
    global_connector_access: bool = False,
    global_governance_access: bool = False,
) -> None:
    """Bind this session to a tenant for the rest of its life.

    Stored on the session AND applied now: stored so `_reapply_tenant_context` can restore
    it after every commit, applied so the transaction already open picks it up.
    """
    context = {
        "company_domain": company_domain or "",
        "global_admin": "true" if global_admin else "false",
        "global_article_access": "true" if global_article_access else "false",
        "global_identity_access": "true" if global_identity_access else "false",
        "global_connector_access": "true" if global_connector_access else "false",
        "global_governance_access": "true" if global_governance_access else "false",
        "user_id": user_id or "",
    }
    # Test doubles stand in for the session in several unit tests and only record the
    # statements they are given; they have no sync_session to carry `info`. Applying the
    # context still works for them, which is what those tests assert — only the
    # re-application after a commit needs the real thing.
    info = getattr(getattr(db, "sync_session", None), "info", None)
    if info is not None:
        info[TENANT_CONTEXT_KEY] = context
    await db.execute(_TENANT_SQL, context)

async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str | None = Depends(oauth2_scheme),
    access_token_cookie: str | None = Cookie(default=None, alias="access_token"),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token = token or access_token_cookie
    if not token:
        raise credentials_exception
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        if payload.get("type") != "access":
            # Access endpoints accept access tokens only; refresh/state tokens
            # are rejected even though they share the signing key.
            raise credentials_exception
        email = payload.get("sub")
        if not isinstance(email, str):
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    email = email.strip().lower()
    if "@" not in email:
        raise credentials_exception
    # A signed token has already authenticated this domain.  Set it before
    # loading the user so identity and membership RLS policies do not require
    # a temporary global bypass.
    await set_database_context(db, email.rsplit("@", 1)[1])
    user_repo = UserRepository(db)
    user = await user_repo.get_by_email(email)
    if user is None:
        raise credentials_exception
    if not user.active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This account has been deactivated")
    if int(payload.get("av", 0)) != user.auth_version:
        raise credentials_exception
    if not user.roles:
        await bootstrap_rbac(db)
        user = await user_repo.get_by_email(email)
    await set_database_context(
        db,
        user.company_domain,
        AuthorizationService.is_global_administrator(user),
        str(user.id),
        AuthorizationService.has_global_article_access(user),
        AuthorizationService.has_global_identity_management(user),
        AuthorizationService.has_global_connector_management(user),
        AuthorizationService.has_global_governance_access(user),
    )
    return user

def require_permission(permission: str, scope: str = "company"):
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not AuthorizationService.has_permission(current_user, permission, requested_scope=scope):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Missing permission: {permission}")
        return current_user
    return permission_checker
