"""A SharePoint ACL must resolve to principals a person can recognise.

The ACL screen for one library showed **57 unresolved principals**, 51 of them typed
`unknown` and identified only by a bare GUID, each captioned "Requires an approved
provider identity mapping". Every document from that source was blocked from approval
until an administrator mapped all of them — a decision nobody can make, because nothing
on the screen said who or what any of those GUIDs were.

Two faults produced that.

Graph reports a grant in four shapes and only the two SINGULAR ones were read:
`grantedToV2` and `grantedTo`. The plural `grantedToIdentitiesV2` and
`grantedToIdentities` — which is what SharePoint uses for most library permissions —
were missed entirely. A permission carrying only those fell through to the fallback and
was recorded as `unknown`, with the PERMISSION ENTRY's own id standing in for a principal
id. That identifier names nobody, differs on every file, and can never map to anything,
so a single library manufactured dozens of permanently unmappable blockers.

And nothing stored what the provider CALLS a principal, so even a correctly resolved one
could only be displayed as a GUID.

`principal_id` is what a mapping is keyed on, so these tests are mostly about it being a
real, stable identifier rather than an accident of one API response.
"""
from __future__ import annotations

import pytest

from src.domain.connector_adapters import sharepoint_permission_principals as principals


def test_the_plural_identity_list_is_read():
    """The regression: this shape used to yield one `unknown` GUID and nothing else."""
    result = principals(
        {
            "id": "permission-entry-id",
            "roles": ["read"],
            "grantedToIdentitiesV2": [
                {"siteGroup": {"id": "12", "displayName": "Site Members"}},
                {"user": {"id": "u-1", "displayName": "Tai Quach"}},
            ],
        }
    )

    assert [(item["principal_type"], item["principal_id"]) for item in result] == [
        ("group", "12"),
        ("user", "u-1"),
    ]
    # And crucially not the permission entry's own id, which identifies nobody.
    assert all(item["principal_id"] != "permission-entry-id" for item in result)


def test_the_provider_name_is_captured():
    """Without this the screen can only show a GUID, which is not something anyone can
    make an access decision about."""
    result = principals(
        {"roles": ["read"], "grantedToV2": {"group": {"id": "g1", "displayName": "HR Team"}}}
    )
    assert result[0]["principal_name"] == "HR Team"


def test_an_email_stands_in_when_there_is_no_display_name():
    result = principals(
        {"roles": ["read"], "grantedToV2": {"user": {"id": "u1", "email": "tai@qnsc.vn"}}}
    )
    assert result[0]["principal_name"] == "tai@qnsc.vn"


@pytest.mark.parametrize("key", ["grantedToV2", "grantedTo"])
def test_the_singular_shapes_still_work(key):
    result = principals({"roles": ["write"], key: {"user": {"id": "u-9"}}})
    assert result == [
        {"principal_type": "user", "principal_id": "u-9", "principal_name": "", "role": "write"}
    ]


@pytest.mark.parametrize("key", ["grantedToIdentitiesV2", "grantedToIdentities"])
def test_both_plural_shapes_are_read(key):
    result = principals({"roles": ["read"], key: [{"group": {"id": "g-9"}}]})
    assert [item["principal_id"] for item in result] == ["g-9"]


@pytest.mark.parametrize(
    "key,expected",
    [
        ("group", "group"),
        ("siteGroup", "group"),
        ("user", "user"),
        ("siteUser", "user"),
        ("application", "application"),
        ("device", "device"),
    ],
)
def test_identity_kinds_are_classified(key, expected):
    """siteGroup is a group and siteUser is a user; an application is neither, and
    pretending otherwise would map a service principal to a person's access group."""
    result = principals({"roles": ["read"], "grantedToV2": {key: {"id": "x"}}})
    assert result[0]["principal_type"] == expected


def test_a_sharing_link_is_named_for_what_it_is():
    """A link is a real grant and must not be dropped -- an ACL that looks narrower than
    it is would let an unsafe approval through. But it is keyed by its scope, which is
    stable across every file, so one decision covers the library instead of one per
    document."""
    result = principals({"id": "perm-1", "roles": ["read"], "link": {"scope": "anonymous"}})
    assert result == [
        {
            "principal_type": "link",
            "principal_id": "link:anonymous",
            "principal_name": "Sharing link (anonymous)",
            "role": "read",
        }
    ]


def test_the_same_link_scope_is_the_same_principal_on_every_file():
    """The whole point of not keying on the permission id."""
    first = principals({"id": "perm-1", "roles": ["read"], "link": {"scope": "organization"}})
    second = principals({"id": "perm-2", "roles": ["read"], "link": {"scope": "organization"}})
    assert first[0]["principal_id"] == second[0]["principal_id"]


def test_an_invitation_is_keyed_by_email():
    result = principals(
        {"id": "p", "roles": ["write"], "invitation": {"email": "Guest@Example.com"}}
    )
    assert result[0]["principal_id"] == "guest@example.com"
    assert result[0]["principal_type"] == "user"


def test_something_unrecognised_is_still_recorded():
    """Kept deliberately. Dropping an entry nobody understands would make the ACL look
    narrower than it is, which is the one outcome worse than an awkward screen."""
    result = principals({"id": "mystery", "roles": ["read"]})
    assert result == [
        {"principal_type": "unknown", "principal_id": "mystery", "principal_name": "", "role": "read"}
    ]


def test_an_entry_with_nothing_to_identify_it_is_dropped():
    assert principals({"roles": ["read"]}) == []


def test_a_principal_named_twice_is_returned_once():
    """Graph repeats an identity across the singular and plural fields, and the storage
    layer has a uniqueness constraint on (snapshot, type, id)."""
    from src.domain.connector_adapters import _dedupe_principals

    result = _dedupe_principals(
        principals(
            {
                "roles": ["read"],
                "grantedToV2": {"user": {"id": "u1"}},
                "grantedToIdentitiesV2": [{"user": {"id": "u1", "displayName": "Tai"}}],
            }
        )
    )

    assert len(result) == 1
    # The name is kept from whichever copy actually carried one.
    assert result[0]["principal_name"] == "Tai"


def test_merging_keeps_the_widest_role():
    """Losing a role would understate what a principal can do at the provider."""
    from src.domain.connector_adapters import _dedupe_principals

    result = _dedupe_principals(
        [
            {"principal_type": "user", "principal_id": "u1", "principal_name": "", "role": "read"},
            {"principal_type": "user", "principal_id": "u1", "principal_name": "", "role": "write"},
        ]
    )
    assert result[0]["role"] == "read,write"
