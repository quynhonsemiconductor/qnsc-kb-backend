import uuid
from datetime import timedelta
from fastapi import HTTPException, status
from src.core.security import verify_password, get_password_hash, create_access_token, create_refresh_token
from src.core.config import settings
from src.models.user import User
from src.repositories.user import UserRepository

class AuthService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    async def authenticate_user(self, email: str, password: str) -> User:
        user = await self.user_repo.get_by_email(email)
        if not user:
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
        
        # Seed access group mapping
        # Automatically assign every user to a "public" access group (index 0) if it exists,
        # or create it if it doesn't exist yet (very useful for local bootstrapping!).
        from src.models.user import AccessGroup
        from src.domain.departments import lock_company_access_groups, normalize_department_name
        dept = normalize_department_name(dept)
        # Registration is also used by the public signup flow. Serialize both
        # public-group creation and department-group bit allocation.
        await lock_company_access_groups(self.user_repo.db, company_domain)
        public_group = await self.user_repo.get_group_by_name("public", company_domain)
        if not public_group:
            public_group = AccessGroup(name="public", company_domain=company_domain, bitmask_position=0)
            public_group = await self.user_repo.create_group(public_group, commit=False)
            
        user.groups.append(public_group)
        
        # If user is in a department, auto create/assign department group too (e.g. at bit position based on dept name length or random)
        if dept:
            dept_group_name = f"dept_{dept.lower()}"
            dept_group = await self.user_repo.get_group_by_name(dept_group_name, company_domain)
            if not dept_group:
                # Find max bitmask position and increment
                all_groups = await self.user_repo.get_all_groups(company_domain)
                max_pos = max([g.bitmask_position for g in all_groups]) if all_groups else 0
                dept_group = AccessGroup(name=dept_group_name, company_domain=company_domain, bitmask_position=max_pos + 1)
                dept_group = await self.user_repo.create_group(dept_group, commit=False)
            user.groups.append(dept_group)

        return await self.user_repo.create(user)

    def create_token(self, user: User) -> str:
        # Save email as subject
        expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        return create_access_token(subject=user.email, expires_delta=expires, auth_version=user.auth_version)

    def create_refresh_token(self, user: User) -> str:
        return create_refresh_token(subject=user.email, auth_version=user.auth_version)
