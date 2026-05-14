import os
from langchain_community.vectorstores import FAISS

def load_retriever(embeddings, index_path="data/faq_index"):
    """Loads the FAISS index from the specified path."""
    if os.path.exists(index_path):
        print(f"Loading FAISS vectorstore from {index_path}...")
        return FAISS.load_local(
            index_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
    print(f"Warning: {index_path} not found.")
    return None
