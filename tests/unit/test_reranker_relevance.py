from src.rag.reranker import normalize_query, score_retrieval_text


def test_stopword_only_query_has_no_retrieval_signal():
    assert normalize_query("what is") == ""
    assert score_retrieval_text("what is", "A completely unrelated passage") == 0


def test_unrelated_short_query_does_not_match_arbitrary_passage():
    assert score_retrieval_text("hi", "Lecture 8: Clock Tree Synthesis") < 0.12


def test_relevant_terms_receive_positive_relevance():
    assert score_retrieval_text("clock tree synthesis", "Clock tree synthesis balances clock path delay") >= 0.12
