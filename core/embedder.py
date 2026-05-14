from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """Loads and returns the HuggingFace embeddings model."""
    print("Loading HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
