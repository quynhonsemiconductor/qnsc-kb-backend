# Provision the first global Admin

A freshly migrated database has no users, and kb is invite-only: nobody can sign in until
one exists. `scripts/provision_admin.py` creates or repairs that account and grants the
system **Admin** role (the one with `company_domain = NULL`). Re-running it is safe.

The script sets the same tenant context an Admin request gets before it touches the
database. Without that context the identity RLS policies refuse the INSERT under the
application role (`kb_app`) with `new row violates row-level security policy for table
"users"` — so run the script as deployed, not with a hand-written SQL bypass.

## Production (k3s), SSO-only

Prod admins sign in with Entra, so the bootstrap account should have no usable password.
`--no-password` generates a random password, hashes it, and discards the plaintext; it is
never printed or stored, and any `ADMIN_PASSWORD` in the environment is ignored.

```bash
kubectl -n kb-prod exec -i deploy/api -- env ADMIN_EMAIL=you@qnsc.vn ADMIN_NAME="Your Name" python scripts/provision_admin.py --no-password
```

The address must be the one Entra returns for the person, and the `--no-password` account
can only sign in through SSO. To give a local or non-SSO account a password instead, set
`ADMIN_PASSWORD` (12+ characters) and omit the flag:

```bash
kubectl -n kb-prod exec -i deploy/api -- env ADMIN_EMAIL=you@qnsc.vn ADMIN_NAME="Your Name" ADMIN_PASSWORD='<at least 12 chars>' python scripts/provision_admin.py
```

The script resets the password and re-grants Admin on every run, and bumps
`auth_version`, so the account's existing sessions end.

## Verify

```bash
kubectl -n kb-prod exec -i deploy/api -- env ADMIN_EMAIL=you@qnsc.vn python - <<'PY'
import asyncio
from sqlalchemy import text
from src.api.deps import SessionLocal, set_database_context

async def main():
    async with SessionLocal() as db:
        await set_database_context(db, "qnsc.vn", global_admin=True)
        row = (await db.execute(text(
            "SELECT u.email, r.name, r.company_domain FROM users u "
            "JOIN user_roles ur ON ur.user_id = u.id "
            "JOIN roles r ON r.id = ur.role_id WHERE u.email = :email"
        ), {"email": "you@qnsc.vn"})).first()
        print(row)

asyncio.run(main())
PY
```

Expect `('you@qnsc.vn', 'Admin', None)` — a global role, not a company-scoped one.
