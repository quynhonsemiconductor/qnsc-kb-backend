from types import SimpleNamespace

from src.rag.reranker import is_definition_query, rerank_chunks, score_retrieval_text


def _chunk(text: str, section: str = "Section 1") -> SimpleNamespace:
    return SimpleNamespace(
        chunk_text=text,
        parent_chunk=SimpleNamespace(text=text, section_ref=section),
        article=SimpleNamespace(title="Lecture-2-Verilog-2018-19"),
    )


def test_definition_query_is_detected_in_english_and_vietnamese() -> None:
    assert is_definition_query("What is Verilog?")
    assert is_definition_query("Verilog là gì?")
    assert not is_definition_query("How do I write a Verilog module?")


def test_definition_passage_beats_reference_passage() -> None:
    definition = _chunk(
        "Verilog is a hardware description language (HDL) used to represent digital hardware."
    )
    references = _chunk(
        "Some helpful documents and references about Verilog: https://example.com/a "
        "https://example.com/b",
        section="References",
    )

    ranked = rerank_chunks("Verilog là gì?", [references, definition], limit=2)

    assert ranked[0] is definition
    assert score_retrieval_text("Verilog là gì?", definition.chunk_text) > score_retrieval_text(
        "Verilog là gì?", references.chunk_text, section="References"
    )
