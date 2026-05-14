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
    return "Does UT Dallas provide services for students with disabilities?"


@pytest.fixture(scope="session")
def direct_match_answer():
    df = pd.read_csv("data/faq_data.csv")
    return str(df.iloc[0]["Answering"]).strip()


@pytest.fixture(scope="session")
def paraphrase_question():
    return "Can disabled students get help at UTD?"


@pytest.fixture(scope="session")
def expected_paraphrase_answer():
    df = pd.read_csv("data/faq_data.csv")
    return str(df.iloc[0]["Answering"]).strip()
