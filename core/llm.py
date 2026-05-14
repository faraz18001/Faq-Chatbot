from langchain_community.llms import Ollama

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """Loads and returns the HuggingFace embeddings model."""
    print("Loading HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")





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





def get_llm():
    """Initializes and returns the Ollama LLM."""
    print("Initializing Ollama LLM (granite4.1:3b)...")
    return Ollama(model="granite4.1:3b")




res=get_llm()

print(res.invoke("Who is the president of united states?"))




