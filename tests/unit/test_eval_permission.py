from src.api.routers.governance import _permission_leakage_detected


def test_eval_permission_check_accepts_citations_from_authorized_retrieval():
    retrieved = [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}]
    citations = [{"chunk_id": "chunk-2"}]

    assert _permission_leakage_detected(retrieved, citations) is False


def test_eval_permission_check_flags_citation_outside_authorized_retrieval():
    retrieved = [{"chunk_id": "chunk-1"}]
    citations = [{"chunk_id": "chunk-1"}, {"chunk_id": "restricted-chunk"}]

    assert _permission_leakage_detected(retrieved, citations) is True


def test_eval_permission_check_ignores_malformed_metadata():
    retrieved = [{"chunk_id": "chunk-1"}]
    citations = [{"title": "forged"}, "invalid"]

    assert _permission_leakage_detected(retrieved, citations) is False
