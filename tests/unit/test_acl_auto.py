"""Provider permission principals are resolved without an administrator mapping each one.

Before, every principal the system did not already recognise (a group, a person who had not
signed in here, a sharing link, an application) blocked approval of every document that carried
it until someone mapped it by hand. The rules here make the obvious decisions, and these tests
pin both halves of the contract: what is resolved, and -- more important -- what is NOT, so a
wrong guess cannot quietly widen access.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from src.core.config import settings
from src.domain.acl_auto import (
    auto_resolve,
    emails_of,
    fold,
    match_department,
    normalize_department_name,
    normalize_group_name,
    suggest_department,
)
from src.domain.cloud_sync import _save_permissions
from src.domain.connector_adapters import (
    _dedupe_principals,
    sharepoint_permission_principals,
)
from src.models.connectors import ExternalDocument
from src.models.ops import Connector

HR, FIN, IT1, IT2 = (str(uuid.uuid4()) for _ in range(4))
ALICE = str(uuid.uuid4())


# ── names ──────────────────────────────────────────────────────────────────────────────────


def test_fold_removes_diacritics_including_d_stroke():
    assert fold("Phòng Đào tạo – Kế toán!") == "phong dao tao ke toan"


@pytest.mark.parametrize(
    "group,expected",
    [
        ("Nhân sự Members", "nhan su"),
        ("Phòng Nhân sự - Owners", "nhan su"),
        ("Kế toán Visitors", "ke toan"),
        ("Nhân sự Thành viên", "nhan su"),
        ("IT Team Members", "it"),
        ("Members", ""),
        (
            "Nhân sự Members Archive",
            "nhan su members archive",
        ),  # a role word in the MIDDLE is part of the name
    ],
)
def test_a_group_name_reduces_to_the_department_it_stands_for(group, expected):
    assert normalize_group_name(group) == expected


def test_department_names_drop_the_word_for_department():
    assert normalize_department_name("Phòng Nhân sự") == "nhan su"
    assert normalize_department_name("Ban Giám đốc") == "giam doc"


DEPARTMENTS = [(HR, "Phòng Nhân sự"), (FIN, "Kế toán")]


def test_a_group_matches_the_one_department_with_its_name():
    assert match_department("Nhân sự Members", DEPARTMENTS) == HR
    assert match_department("KẾ TOÁN OWNERS", DEPARTMENTS) == FIN


def test_no_match_when_nothing_or_more_than_one_department_fits():
    assert match_department("Marketing Members", DEPARTMENTS) is None
    assert (
        match_department("IT Members", [(IT1, "IT"), (IT2, "Phòng IT")]) is None
    )  # two departments reduce to "it"
    assert match_department("Members", DEPARTMENTS) is None  # reduces to nothing


def test_the_closest_department_is_suggested_but_never_applied():
    assert suggest_department("Nhan su HCM", DEPARTMENTS) == HR
    assert suggest_department("Totally different", DEPARTMENTS) is None


# ── e-mail ─────────────────────────────────────────────────────────────────────────────────


def test_an_email_is_found_in_the_providers_email_login_name_id_or_display_name():
    assert emails_of({"principal_email": "Alice@Corp.com"}) == ["alice@corp.com"]
    assert emails_of({"principal_login": "i:0#.f|membership|Alice@Corp.com"}) == [
        "alice@corp.com"
    ]
    assert emails_of({"principal_id": "bob@corp.com"}) == ["bob@corp.com"]
    assert emails_of(
        {"principal_id": "guid", "principal_name": "Carol <carol@corp.com>"}
    ) == ["carol@corp.com"]
    assert emails_of({"principal_id": "guid", "principal_name": "Alice Nguyen"}) == []


def test_the_providers_own_email_outranks_the_one_in_a_display_name():
    found = emails_of(
        {"principal_email": "real@corp.com", "principal_name": "fake@elsewhere.com"}
    )
    assert found[0] == "real@corp.com"


# ── auto_resolve ───────────────────────────────────────────────────────────────────────────


class Result:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class Db:
    """Answers the two lookups acl_auto makes, by what the statement selects from."""

    def __init__(self, users=None, departments=None):
        self.users = users or []
        self.departments = departments if departments is not None else DEPARTMENTS
        self.statements: list[str] = []

    async def execute(self, statement):
        sql = str(statement)
        self.statements.append(sql)
        if "FROM departments" in sql:
            assert "departments.company_domain" in sql and "departments.active" in sql
            return Result(
                [(uuid.UUID(identifier), name) for identifier, name in self.departments]
            )
        assert "FROM users" in sql
        assert (
            "users.company_domain" in sql and "users.active" in sql
        ), "the KB account must be in this tenant and active"
        return Result(
            [(uuid.UUID(identifier), email) for identifier, email in self.users]
        )


def connector(**config):
    return Connector(
        id=uuid.uuid4(),
        company_domain="acme.test",
        system="sharepoint",
        config_json=config or None,
    )


def resolve(principals, db=None, *, already=None, matched=None, conn=None):
    return asyncio.run(
        auto_resolve(
            db or Db(),
            conn or connector(),
            principals,
            already_resolved=already or set(),
            matched_users=matched or {},
        )
    )


def test_a_person_is_matched_to_the_account_with_the_same_email():
    result = resolve(
        [
            {
                "principal_type": "user",
                "principal_id": "graph-1",
                "principal_name": "Alice",
                "principal_email": "Alice@acme.test",
            }
        ],
        Db(users=[(ALICE, "alice@acme.test")]),
    )
    assert result.users == {"graph-1": ALICE}
    assert result.resolved["user:graph-1"]["via"] == "email"
    assert not result.without_account


def test_a_person_with_no_account_grants_nothing_and_does_not_block():
    result = resolve(
        [
            {
                "principal_type": "user",
                "principal_id": "graph-2",
                "principal_email": "ghost@acme.test",
            }
        ],
        Db(users=[]),
    )
    assert (
        result.users == {}
        and result.without_account == {"graph-2"}
        and "user:graph-2" not in result.resolved
    )


def test_a_site_user_is_matched_through_the_claims_login_name():
    result = resolve(
        [
            {
                "principal_type": "siteUser",
                "principal_id": "7",
                "principal_login": "i:0#.f|membership|alice@acme.test",
            }
        ],
        Db(users=[(ALICE, "alice@acme.test")]),
    )
    assert result.users == {"7": ALICE}


def test_a_group_is_matched_to_its_department_by_name():
    result = resolve(
        [
            {
                "principal_type": "group",
                "principal_id": "g1",
                "principal_name": "Nhân sự Members",
            }
        ]
    )
    assert result.department_ids == {HR}
    assert result.resolved["group:g1"] == {
        "via": "name",
        "name": "Nhân sự Members",
        "department_id": HR,
    }


def test_an_unmatched_group_is_left_for_an_administrator():
    result = resolve(
        [
            {
                "principal_type": "group",
                "principal_id": "g2",
                "principal_name": "Marketing",
            }
        ]
    )
    assert (
        result.department_ids == set()
        and "group:g2" not in result.resolved
        and "group:g2" not in result.ignored
    )


def test_a_group_with_an_ambiguous_name_is_left_alone():
    result = resolve(
        [
            {
                "principal_type": "group",
                "principal_id": "g3",
                "principal_name": "IT Members",
            }
        ],
        Db(departments=[(IT1, "IT"), (IT2, "Phòng IT")]),
    )
    assert result.department_ids == set() and not result.resolved


@pytest.mark.parametrize(
    "principal",
    [
        {"principal_type": "domain", "principal_id": "acme.test"},
        {"principal_type": "link", "principal_id": "link:organization"},
        {"principal_type": "link", "principal_id": "link:anonymous"},
    ],
)
def test_whole_organisation_principals_stand_for_every_department(principal):
    result = resolve([principal])
    assert result.department_ids == {HR, FIN}
    assert (
        result.resolved[f"{principal['principal_type']}:{principal['principal_id']}"][
            "via"
        ]
        == "company"
    )


@pytest.mark.parametrize(
    "principal",
    [
        {"principal_type": "application", "principal_id": "app-1"},
        {"principal_type": "device", "principal_id": "dev-1"},
        {"principal_type": "unknown", "principal_id": "opaque"},
        {"principal_type": "link", "principal_id": "link:users"},
    ],
)
def test_principals_that_can_never_sign_in_grant_nothing_and_do_not_block(principal):
    result = resolve([principal], Db())
    assert (
        f"{principal['principal_type']}:{principal['principal_id']}" in result.ignored
    )
    assert result.department_ids == set() and result.users == {}


def test_an_explicit_mapping_is_never_overridden():
    result = resolve(
        [
            {
                "principal_type": "group",
                "principal_id": "g1",
                "principal_name": "Nhân sự Members",
            }
        ],
        already={("group", "g1")},
    )
    assert result.department_ids == set() and not result.resolved


def test_a_person_the_identity_table_already_matched_is_left_alone():
    result = resolve(
        [
            {
                "principal_type": "user",
                "principal_id": "graph-1",
                "principal_email": "alice@acme.test",
            }
        ],
        Db(users=[(ALICE, "alice@acme.test")]),
        already={("user", "graph-1")},
        matched={"graph-1": "someone-else"},
    )
    assert result.users == {} and not result.resolved


def test_nothing_is_looked_up_when_there_is_nothing_to_resolve():
    db = Db()
    resolve([], db)
    resolve(
        [{"principal_type": "user", "principal_id": "no-address"}], db
    )  # no e-mail -> no user query
    resolve([{"principal_type": "application", "principal_id": "app"}], db)
    assert db.statements == []


def test_the_global_switch_turns_it_all_off(monkeypatch):
    monkeypatch.setattr(settings, "CONNECTOR_ACL_AUTO_RESOLVE", False)
    result = resolve(
        [
            {"principal_type": "domain", "principal_id": "acme.test"},
            {
                "principal_type": "group",
                "principal_id": "g",
                "principal_name": "Nhân sự",
            },
        ]
    )
    assert not result.resolved and not result.ignored and result.department_ids == set()


def test_a_connector_can_opt_out():
    result = resolve(
        [{"principal_type": "domain", "principal_id": "acme.test"}],
        conn=connector(acl_auto_resolve=False),
    )
    assert not result.resolved


# ── through the real sync ─────────────────────────────────────────────────────────────────


class SyncDb:
    """What `_save_permissions` asks for: mappings, identities, then acl_auto's two lookups."""

    def __init__(self, users=None, departments=None):
        self.inner = Db(users, departments)
        self.added = []

    async def execute(self, statement):
        sql = str(statement)
        if (
            "permission_snapshots" in sql
            or "external_group_mappings" in sql
            or "external_identities" in sql
        ):
            return SimpleNamespace(
                scalars=lambda: SimpleNamespace(all=lambda: []),
                scalar_one_or_none=lambda: None,
            )
        return await self.inner.execute(statement)

    def add(self, item):
        self.added.append(item)

    async def flush(self):
        return None


def sync(permissions, db):
    conn = connector()
    document = ExternalDocument(
        id=uuid.uuid4(),
        connector_id=conn.id,
        corpus_id="lib",
        external_id="f",
        name="policy.md",
    )
    asyncio.run(_save_permissions(db, conn, document, permissions))
    return document.metadata_json


def test_a_typical_sharepoint_acl_needs_no_mapping_at_all():
    metadata = sync(
        [
            {
                "principal_type": "group",
                "principal_id": "g1",
                "principal_name": "Nhân sự Members",
                "role": "read",
            },
            {
                "principal_type": "user",
                "principal_id": "graph-1",
                "principal_name": "Alice",
                "principal_email": "alice@acme.test",
                "role": "write",
            },
            {
                "principal_type": "user",
                "principal_id": "graph-2",
                "principal_name": "Ghost",
                "principal_email": "ghost@acme.test",
                "role": "read",
            },
            {"principal_type": "link", "principal_id": "link:users", "role": "read"},
            {"principal_type": "application", "principal_id": "app-1", "role": "read"},
        ],
        SyncDb(users=[(ALICE, "alice@acme.test")]),
    )
    assert metadata["mapped_department_ids"] == [HR]
    assert metadata["mapped_source_user_ids"] == [ALICE]
    # Nothing is left to block approval.
    assert metadata["unmapped_group_ids"] == []
    assert metadata["unmapped_source_user_ids"] == []
    assert metadata["unmapped_principal_ids"] == []
    assert metadata["source_users_without_account"] == ["graph-2"]
    assert metadata["auto_resolved"]["group:g1"]["via"] == "name"


def test_what_cannot_be_resolved_still_blocks_so_nothing_is_widened():
    metadata = sync(
        [
            {
                "principal_type": "group",
                "principal_id": "g9",
                "principal_name": "Marketing Members",
                "role": "read",
            }
        ],
        SyncDb(),
    )
    assert metadata["unmapped_group_ids"] == ["g9"]
    assert metadata["mapped_department_ids"] == []


def test_an_organisation_wide_link_alone_gives_the_whole_company_access():
    metadata = sync(
        [
            {
                "principal_type": "link",
                "principal_id": "link:organization",
                "role": "read",
            }
        ],
        SyncDb(),
    )
    assert sorted(metadata["mapped_department_ids"]) == sorted([HR, FIN])
    assert metadata["unmapped_principal_ids"] == []


# ── what SharePoint tells us ───────────────────────────────────────────────────────────────


def test_the_sharepoint_adapter_keeps_the_email_and_login_name():
    principals = sharepoint_permission_principals(
        {
            "roles": ["read"],
            "grantedToV2": {
                "user": {
                    "id": "u1",
                    "displayName": "Alice",
                    "email": "alice@acme.test",
                },
                "siteUser": {
                    "id": "7",
                    "displayName": "Bob",
                    "loginName": "i:0#.f|membership|bob@acme.test",
                },
            },
        }
    )
    by_id = {item["principal_id"]: item for item in principals}
    assert by_id["u1"]["principal_email"] == "alice@acme.test"
    assert by_id["7"]["principal_login"] == "i:0#.f|membership|bob@acme.test"


def test_the_same_person_in_two_identity_sets_keeps_the_email_one_of_them_had():
    merged = _dedupe_principals(
        [
            {
                "principal_type": "user",
                "principal_id": "u1",
                "principal_name": "Alice",
                "role": "read",
            },
            {
                "principal_type": "user",
                "principal_id": "u1",
                "principal_name": "",
                "role": "write",
                "principal_email": "alice@acme.test",
            },
        ]
    )
    assert len(merged) == 1 and merged[0]["principal_email"] == "alice@acme.test"


# ── the administrator's ACL panel agrees with the sync ───────────────────────────────────


class PanelResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows

    def scalars(self):
        return self


class PanelDb:
    def __init__(self, principals, snapshots, users=None):
        self.principals, self.snapshots, self.users = principals, snapshots, users or []

    async def execute(self, statement):
        sql = str(statement)
        if "FROM external_acl_principals" in sql:
            return PanelResult(self.principals)
        if "permission_snapshots.permissions_json" in sql:
            return PanelResult(self.snapshots)
        if "external_group_mappings" in sql or "external_identities" in sql:
            return PanelResult([])
        if "FROM departments" in sql:
            return PanelResult(
                [(uuid.UUID(identifier), name) for identifier, name in DEPARTMENTS]
            )
        assert "FROM users" in sql
        return PanelResult(
            [(uuid.UUID(identifier), email) for identifier, email in self.users]
        )


def principal(kind, identifier, name=None, role="read"):
    return SimpleNamespace(
        principal_type=kind, principal_id=identifier, principal_name=name, role=role
    )


def panel(monkeypatch, principals, snapshots, users=None):
    from src.api.routers import connectors as router

    conn = connector()

    async def lookup(db, connector_id, user):
        return conn

    monkeypatch.setattr(router, "_connector_for_user", lookup)
    return asyncio.run(
        router.list_acl_principals(
            conn.id,
            SimpleNamespace(id=uuid.uuid4()),
            PanelDb(principals, snapshots, users),
        )
    )


def test_the_panel_shows_how_each_principal_was_resolved_and_what_is_left(monkeypatch):
    rows = panel(
        monkeypatch,
        [
            principal("group", "g1", "Nhân sự Members"),
            principal("group", "g2", "Nhan su HCM"),
            principal("user", "graph-1", "Alice"),
            principal("user", "graph-2", "Ghost"),
            principal("link", "link:organization", "Sharing link (organization)"),
            principal("application", "app-1", "Bot"),
        ],
        [
            [
                {
                    "principal_type": "user",
                    "principal_id": "graph-1",
                    "principal_email": "alice@acme.test",
                }
            ]
        ],
        users=[(ALICE, "alice@acme.test")],
    )
    by_id = {row["principal_id"]: row for row in rows}
    assert by_id["g1"]["mapping_status"] == "auto" and by_id["g1"]["auto_via"] == "name"
    assert by_id["g1"]["auto_department_name"] == "Phòng Nhân sự"
    assert (
        by_id["graph-1"]["mapping_status"] == "auto"
        and by_id["graph-1"]["internal_user_id"] == ALICE
    )
    assert by_id["graph-2"]["mapping_status"] == "no_account"
    assert (
        by_id["link:organization"]["mapping_status"] == "auto"
        and by_id["link:organization"]["auto_via"] == "company"
    )
    assert by_id["app-1"]["mapping_status"] == "ignored"
    # An unmatched group stays unresolved, with the nearest department offered, not applied.
    assert by_id["g2"]["mapping_status"] == "unmapped"
    assert by_id["g2"]["suggested_department_id"] == HR


def test_a_group_with_a_loose_name_is_only_ever_a_suggestion(monkeypatch):
    rows = panel(monkeypatch, [principal("group", "g2", "Nhan su HCM")], [])
    assert rows[0]["mapping_status"] == "unmapped"
    assert (
        rows[0]["suggested_department_id"] == HR
        and rows[0]["suggested_department_name"] == "Phòng Nhân sự"
    )
