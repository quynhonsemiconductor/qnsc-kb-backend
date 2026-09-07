import uuid
from datetime import timedelta
from fastapi import HTTPException, status
from src.core.security import verify_password, get_password_hash, create_access_token, create_refresh_token
from src.core.config import settings
from src.models.user import User
from src.repositories.user import UserRepository

# One bcrypt comparison happens on every authentication attempt, including the ones where
# no account exists. Returning 401 before hashing made the unknown-email path finish in
# microseconds while a real address paid the full bcrypt cost, which is a remotely
# measurable oracle for "does this address have an account here". The hash is computed
# once at import so the equalizing comparison does not add a per-request key derivation.
_ABSENT_ACCOUNT_PASSWORD_HASH = get_password_hash(uuid.uuid4().hex)


class AuthService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    async def authenticate_user(self, email: str, password: str) -> User:
        user = await self.user_repo.get_by_email(email)
        if not user:
            verify_password(password, _ABSENT_ACCOUNT_PASSWORD_HASH)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    async def register_user(self, email: str, name: str, password: str, dept: str | None = None, role: str = "Staff", allow_privileged_role: bool = False) -> User:
        email = email.strip().lower()
        company_domain = email.rsplit("@", 1)[-1] if "@" in email else "local"
        allowed_domains = {item.strip().lower() for item in settings.ALLOWED_EMAIL_DOMAINS.split(",") if item.strip()}
        if allowed_domains and company_domain not in allowed_domains:
            raise HTTPException(status_code=403, detail="Use an approved company email address")
        if role not in {"Admin", "CEO", "Reviewer", "Staff"}:
            raise HTTPException(status_code=422, detail="Invalid role")
        existing = await self.user_repo.get_by_email(email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
            
        hashed_password = get_password_hash(password)
        user = User(
            email=email,
            name=name,
            password_hash=hashed_password,
            company_domain=company_domain,
            dept=dept,
            role=role if allow_privileged_role else "Staff"
        )
        
        # Audience membership is department membership. Public content is
        # readable through `Article.sensitivity == "public"`, so there is no
        # separate everyone-group to seed here.
        from src.domain.departments import normalize_department_name
        dept = normalize_department_name(dept)
        if dept:
            department = await self.user_repo.get_department_by_name(dept, company_domain)
            if department:
                user.departments.append(department)

        return await self.user_repo.create(user)

    def create_token(self, user: User) -> str:
        # Save email as subject
        expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        return create_access_token(subject=user.email, expires_delta=expires, auth_version=user.auth_version)

    def create_refresh_token(self, user: User) -> str:
        return create_refresh_token(subject=user.email, auth_version=user.auth_version)
