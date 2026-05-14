import pytest
from sentence_transformers import SentenceTransformer
import numpy as np


class TestModelInitialization:
    def test_create_model(self):
        from core.llm import Model
        m = Model(model_name="granite4.1:3b", embeddings_model="all-MiniLM-L6-v2")
        assert m.model_name == "granite4.1:3b"
        assert m.embeddings_model == "all-MiniLM-L6-v2"
        assert m.embeddings is None
        assert m.vectorstore is None
        assert m.retriever is None
        assert m.llm is None


class TestDirectMatch:
    def test_exact_question_returns_direct_match(self, model, direct_match_question, direct_match_answer):
        result = model.query(direct_match_question)
        assert result["direct_match"] is True
        assert result["score"] >= 0.90
        assert result["response"].strip() == direct_match_answer.strip()


class TestFallback:
    def test_no_index_returns_error(self):
        from core.llm import Model
        m = Model(
            model_name="granite4.1:3b",
            embeddings_model="all-MiniLM-L6-v2",
            index_path="data/nonexistent_index",
        )
        result = m.query("any question")
        assert "error" in result


class TestRetrieval:
    def test_paraphrase_retrieves_correct_faq(self, model, paraphrase_question):
        model.load_retriever()
        results = model.vectorstore.similarity_search_with_relevance_scores(paraphrase_question, k=1)
        doc, score = results[0]
        assert score > 0
        assert "disabilit" in doc.page_content.lower()

    def test_gibberish_returns_low_score(self, model):
        result = model.query("asdfghjkl qwerty zxcvbnm")
        assert result["score"] < 0.1


class TestLLMRephrase:
    def test_paraphrase_uses_llm_not_direct_match(self, model, paraphrase_question):
        result = model.query(paraphrase_question)
        assert result["direct_match"] is False
        assert result["score"] < 0.90

    def test_llm_response_is_semantically_similar(self, model, paraphrase_question, expected_paraphrase_answer):
        result = model.query(paraphrase_question)
        llm_response = result["response"]

        sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        emb1 = sim_model.encode(llm_response)
        emb2 = sim_model.encode(expected_paraphrase_answer)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        assert similarity > 0.70, f"LLM response is not semantically similar enough (cosine sim: {similarity:.3f})"
