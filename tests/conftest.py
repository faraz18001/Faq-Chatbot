import pandas as pd
import pytest
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from core.llm import Model


@pytest.fixture(scope="session")
def faq_data():
    return pd.read_csv("data/faq_data.csv")


@pytest.fixture(scope="session")
def model():
    return Model(
        model_name="granite4.1:3b",
        embeddings_model="all-MiniLM-L6-v2",
    )


@pytest.fixture(scope="session")
def direct_match_question():
    return "What is Spectra Dashboard?"


@pytest.fixture(scope="session")
def direct_match_answer():
    return "Spectra Dashboards provides a central location for users to access, interact and analyze up-to-date information so they can make smarter, data-driven decisions. It provides interactive visualizations to find answers and handle large amounts of business data."


@pytest.fixture(scope="session")
def paraphrase_question():
    return "How can someone use the Spectra Dashboard for business data?"


@pytest.fixture(scope="session")
def expected_paraphrase_answer():
    return "Spectra Dashboards provides a central location for users to access, interact and analyze up-to-date information so they can make smarter, data-driven decisions. It provides interactive visualizations to find answers and handle large amounts of business data."
