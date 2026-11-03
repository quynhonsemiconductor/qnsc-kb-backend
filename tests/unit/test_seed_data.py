from scripts.seed_data import seed_user_definitions


def test_seed_data_covers_every_mvp_role_without_credentials_in_source():
    definitions = seed_user_definitions("acme.test")

    assert {item["role"] for item in definitions} == {"Admin", "CEO", "Reviewer", "Staff"}
    assert all(item["email"].endswith("@acme.test") for item in definitions)
