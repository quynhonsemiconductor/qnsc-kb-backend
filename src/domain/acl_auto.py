"""Resolve provider permission principals without an administrator mapping each one.

A connector document carries the provider's own access list (SharePoint, OneDrive, Google
Drive). The knowledge base only ever NARROWS its internal policy with it, but every principal
on that list that the system could not recognise blocked approval until someone mapped it by
hand: a library with a dozen groups, sharing links and people who never signed in meant a dozen
decisions per source, most of them with an obvious answer.

This module makes the obvious answers itself, in the order a person would:

* a person (`user`) is the KB account with the same e-mail address, read from the provider's
  e-mail, a SharePoint login name (``i:0#.f|membership|name@x.com``), or an id/name that is one.
  A person with no KB account cannot open anything here, so they grant nothing and are no longer
  a blocker; the next sync picks them up as soon as an account exists.
* a group is the department whose name it carries once the SharePoint role suffix (Members,
  Owners, Visitors) and the word for "department" are set aside, but only when exactly one
  department matches. Anything ambiguous or unmatched stays for an administrator, with the
  closest department offered as a suggestion.
* ``domain`` ("everyone in the organisation") and an organisation/anonymous sharing link mean
  the whole company, so they stand for every department.
* an application, a device, any other sharing link, and an opaque `unknown` principal are not
  people the KB could ever authenticate, so they grant nothing and do not block.

An explicit mapping always wins: this only looks at what is still unresolved. Nothing here
widens what the internal policy allows, since the provider ACL is intersected with it
afterwards; the worst a wrong match can do is pick the wrong subset of that policy, and
`CONNECTOR_ACL_AUTO_RESOLVE=false` turns the whole thing off.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.models.user import Department, User

USER_TYPES = {"user", "siteUser"}
GROUP_TYPES = {"group", "siteGroup"}
#: Not people the KB can authenticate, so they grant nothing and need no decision.
IGNORED_TYPES = {"application", "device", "unknown"}
#: Whole-organisation principals.
COMPANY_TYPES = {"domain"}
#: Sharing links that anyone in (or beyond) the organisation can open.
COMPANY_LINKS = {"link:organization", "link:anonymous"}

_EMAIL = re.compile(r"[A-Za-z0-9._%+'-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_ROLE_TOKENS = {
    "members",
    "member",
    "owners",
    "owner",
    "visitors",
    "visitor",
    "thanh",
    "vien",
    "chu",
    "so",
    "huu",
    "khach",
}
_ROLE_PHRASES = ("thanh vien", "chu so huu", "khach truy cap", "nguoi truy cap")
_DEPARTMENT_WORDS = {
    "phong",
    "ban",
    "bo",
    "phan",
    "department",
    "dept",
    "team",
    "group",
    "nhom",
    "division",
}


def fold(text: str) -> str:
    """Lower-case, without diacritics (đ included), punctuation as spaces, runs of space collapsed."""
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    stripped = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", stripped.lower()).split())


def normalize_group_name(name: str) -> str:
    """A provider group's name as the department it stands for.

    ``"Nhân sự Members"`` and ``"Phòng Nhân sự - Owners"`` both become ``"nhan su"``. Only a
    role suffix at the END is removed, and only words from a short fixed list.
    """
    text = fold(name)
    for phrase in _ROLE_PHRASES:
        text = text.replace(phrase, " ")
    tokens = text.split()
    while tokens and tokens[-1] in _ROLE_TOKENS:
        tokens.pop()
    return " ".join(token for token in tokens if token not in _DEPARTMENT_WORDS)


def normalize_department_name(name: str) -> str:
    return " ".join(
        token for token in fold(name).split() if token not in _DEPARTMENT_WORDS
    )


def emails_of(principal: dict[str, Any]) -> list[str]:
    """Every e-mail address a principal carries, lower-cased, best source first.

    The provider's own `principal_email` comes first, then a login name such as
    ``i:0#.f|membership|name@x.com`` (only the part after the last ``|`` is read), then the id
    (Google uses the address as the id), then the display name.
    """
    found: list[str] = []
    for key in ("principal_email", "principal_login", "principal_id", "principal_name"):
        raw = str(principal.get(key) or "")
        if not raw:
            continue
        for match in _EMAIL.finditer(
            raw.rsplit("|", 1)[-1] if key == "principal_login" else raw
        ):
            address = match.group(0).lower()
            if address not in found:
                found.append(address)
    return found


def match_department(
    group_name: str, departments: Iterable[tuple[str, str]]
) -> str | None:
    """The id of the one department whose name equals the group's, or None.

    Ambiguity is a refusal: two departments that normalise to the same name, or a group whose
    name reduces to nothing, are left for an administrator.
    """
    wanted = normalize_group_name(group_name)
    if not wanted:
        return None
    matches = [
        department_id
        for department_id, name in departments
        if normalize_department_name(name) == wanted
    ]
    return matches[0] if len(matches) == 1 else None


def suggest_department(
    group_name: str, departments: Iterable[tuple[str, str]], *, threshold: float = 0.62
) -> str | None:
    """The closest department by name, as a suggestion only (never applied automatically)."""
    wanted = normalize_group_name(group_name)
    if not wanted:
        return None
    best: tuple[float, str | None] = (0.0, None)
    for department_id, name in departments:
        candidate = normalize_department_name(name)
        if not candidate:
            continue
        score = SequenceMatcher(None, wanted, candidate).ratio()
        if wanted in candidate or candidate in wanted:
            score = max(score, 0.8)
        if score > best[0]:
            best = (score, department_id)
    return best[1] if best[0] >= threshold else None


def principal_key(principal_type: str, principal_id: str) -> str:
    return f"{principal_type}:{principal_id}"


@dataclass
class AutoResolution:
    #: provider principal id -> KB user id, for people matched by e-mail.
    users: dict[str, str] = field(default_factory=dict)
    #: department ids granted by auto-matched groups and whole-company principals.
    department_ids: set[str] = field(default_factory=set)
    #: `type:id` -> how it was resolved, for every principal this module resolved.
    resolved: dict[str, dict[str, str]] = field(default_factory=dict)
    #: provider people with no KB account: they grant nothing and do not block.
    without_account: set[str] = field(default_factory=set)
    #: `type:id` of principals that grant nothing and do not block.
    ignored: set[str] = field(default_factory=set)

    def keys(self) -> set[str]:
        return set(self.resolved) | self.ignored


def enabled(connector: Any) -> bool:
    if not settings.CONNECTOR_ACL_AUTO_RESOLVE:
        return False
    config = getattr(connector, "config_json", None) or {}
    return bool(config.get("acl_auto_resolve", True))


async def company_departments(
    db: AsyncSession, company_domain: str
) -> list[tuple[str, str]]:
    rows = await db.execute(
        select(Department.id, Department.name).where(
            Department.company_domain == company_domain, Department.active.is_(True)
        )
    )
    return [(str(identifier), str(name)) for identifier, name in rows.all()]


async def users_by_email(
    db: AsyncSession, company_domain: str, emails: set[str]
) -> dict[str, str]:
    if not emails:
        return {}
    rows = await db.execute(
        select(User.id, func.lower(User.email)).where(
            func.lower(User.email).in_(emails),
            User.company_domain == company_domain,
            User.active.is_(True),
        )
    )
    return {str(email): str(identifier) for identifier, email in rows.all()}


async def auto_resolve(
    db: AsyncSession,
    connector: Any,
    principals: list[dict[str, Any]],
    *,
    already_resolved: set[tuple[str, str]],
    matched_users: dict[str, str],
) -> AutoResolution:
    """Resolve what is still unresolved. `already_resolved` and `matched_users` come from the
    explicit mappings and the identity table, which this never overrides."""
    result = AutoResolution()
    if not enabled(connector):
        return result
    pending = [
        item
        for item in principals
        if item.get("principal_id")
        and (str(item.get("principal_type") or "unknown"), str(item["principal_id"]))
        not in already_resolved
    ]
    if not pending:
        return result
    company = str(connector.company_domain)

    # Whole-organisation principals, and ones that grant nothing, need no lookup at all.
    wants_company = False
    for item in pending:
        kind, identifier = str(item.get("principal_type") or "unknown"), str(
            item["principal_id"]
        )
        key = principal_key(kind, identifier)
        if kind in COMPANY_TYPES or identifier in COMPANY_LINKS:
            result.resolved[key] = {
                "via": "company",
                "name": str(item.get("principal_name") or ""),
            }
            wants_company = True
        elif kind in IGNORED_TYPES or (
            kind == "link" and identifier not in COMPANY_LINKS
        ):
            result.ignored.add(key)

    # People: the KB account with the same e-mail.
    people = [item for item in pending if str(item.get("principal_type")) in USER_TYPES]
    unmatched_people = [
        item for item in people if str(item["principal_id"]) not in matched_users
    ]
    wanted: set[str] = set()
    for item in unmatched_people:
        wanted.update(emails_of(item))
    accounts = await users_by_email(db, company, wanted)
    for item in unmatched_people:
        identifier = str(item["principal_id"])
        hit = next(
            (accounts[address] for address in emails_of(item) if address in accounts),
            None,
        )
        key = principal_key(str(item.get("principal_type")), identifier)
        if hit:
            result.users[identifier] = hit
            result.resolved[key] = {
                "via": "email",
                "name": str(item.get("principal_name") or ""),
                "user_id": hit,
            }
        else:
            result.without_account.add(identifier)

    # Groups: the one department with the same name.
    groups = [
        item for item in pending if str(item.get("principal_type")) in GROUP_TYPES
    ]
    departments: list[tuple[str, str]] = []
    if groups or wants_company:
        departments = await company_departments(db, company)
    for item in groups:
        target = match_department(str(item.get("principal_name") or ""), departments)
        if target:
            result.department_ids.add(target)
            result.resolved[
                principal_key(str(item["principal_type"]), str(item["principal_id"]))
            ] = {
                "via": "name",
                "name": str(item.get("principal_name") or ""),
                "department_id": target,
            }
    if wants_company:
        result.department_ids.update(identifier for identifier, _ in departments)
    return result
