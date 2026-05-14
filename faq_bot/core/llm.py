from langchain_community.llms import Ollama

def get_llm():
    """Initializes and returns the Ollama LLM."""
    print("Initializing Ollama LLM (granite4.1:3b)...")
    return Ollama(model="granite4.1:3b")
