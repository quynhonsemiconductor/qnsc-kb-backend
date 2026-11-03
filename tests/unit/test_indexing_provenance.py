from src.domain.indexing import _child_chunk_metadata


def test_child_metadata_is_bound_to_its_parent_spec():
    """Indexing must not read metadata from a later parent after the loop."""
    parent_specs = [
        {"parent_text": "Policy table", "chunk_type": "table", "heading": "Retention", "children": ["30 days"]},
        {"parent_text": "Required steps", "chunk_type": "list", "heading": "Procedure", "children": ["Validate request"]},
    ]

    pending_children = []
    for parent_spec in parent_specs:
        parent_id = object()
        parent_chunk_type, parent_heading = _child_chunk_metadata(parent_spec, "")
        for child_text in parent_spec["children"]:
            pending_children.append((child_text, None, parent_id, parent_chunk_type, parent_heading))

    assert [(item[3], item[4]) for item in pending_children] == [
        ("table", "Retention"),
        ("list", "Procedure"),
    ]
