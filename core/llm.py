import os

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


class Model:
    def __init__(
        self, model_name: str, embeddings_model: str, index_path: str = "data/faq_index"
    ) -> None:
        self.model_name = model_name
        self.embeddings_model = embeddings_model
        self.index_path = index_path
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None

    def get_embeddings(self):
        if self.embeddings is not None:
            return self.embeddings
        print(f"Loading HuggingFaceEmbeddings ({self.embeddings_model})...")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
        return self.embeddings

    def load_retriever(self):
        if self.retriever is not None:
            return self.retriever
        if self.embeddings is None:
            self.get_embeddings()
        if os.path.exists(self.index_path):
            print(f"Loading FAISS vectorstore from {self.index_path}...")
            self.vectorstore = FAISS.load_local(
                self.index_path, self.embeddings, allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever()
            return self.retriever
        print(f"Warning: {self.index_path} not found.")
        return None

    def get_llm(self):
        if self.llm is not None:
            return self.llm
        print(f"Initializing Ollama LLM ({self.model_name})...")
        self.llm = Ollama(model=self.model_name)
        return self.llm

    def query(self, question: str):
        if self.retriever is None:
            self.load_retriever()
        if self.llm is None:
            self.get_llm()
        if self.retriever is None:
            return {"error": "Vectorstore not loaded. Please run ingestion first."}

        results = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=1
        )

        if not results:
            return {
                "response": "I couldn't find an answer to your question in the FAQ.",
                "score": 0,
            }

        best_doc, score = results[0]
        original_answer = best_doc.metadata.get("answer", "")

        if score >= 0.90:
            return {
                "score": float(score),
                "response": original_answer,
                "direct_match": True,
            }

        prompt = PromptTemplate(
            input_variables=["answer", "question"],
            template="rephrase this FAQ answer naturally to address the question '{question}': {answer}",
        )
        formatted_prompt = prompt.format(question=question, answer=original_answer)
        llm_output = self.llm.invoke(formatted_prompt)

        return {"score": float(score), "response": llm_output, "direct_match": False}


if __name__ == "__main__":
    m = Model(model_name="granite4.1:3b", embeddings_model="all-MiniLM-L6-v2")
    result = m.query("Does UT Dallas provide services for students with disabilities??")
    print(result)
