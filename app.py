from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os

app = FastAPI(title="FAQ Chatbot API")

class QuestionRequest(BaseModel):
    question: str

# Global variables to hold models
embeddings = None
vectorstore = None
llm = None

@app.on_event("startup")
def load_models():
    global embeddings, vectorstore, llm
    
    print("Loading HuggingFaceEmbeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Loading FAISS vectorstore...")
    if os.path.exists("faq_index"):
        vectorstore = FAISS.load_local(
            "faq_index", 
            embeddings,
            allow_dangerous_deserialization=True  # Required since LangChain 0.0.300+
        )
    else:
        print("Warning: faq_index directory not found. Please run ingest.py first.")
        
    print("Initializing Ollama LLM (granite4.1:3b)...")
    llm = Ollama(model="granite4.1:3b")
    print("Startup complete.")

@app.post("/ask")
def ask_question(req: QuestionRequest):
    if not vectorstore:
        return {"error": "Vectorstore not loaded."}
        
    query = req.question
    
    # Retrieve top 1 nearest neighbor with score
    # Note: similarity_search_with_relevance_scores returns a normalized score (1.0 = identical)
    results = vectorstore.similarity_search_with_relevance_scores(query, k=1)
    
    if not results:
        return {"response": "I couldn't find an answer to your question in the FAQ."}
        
    best_doc, score = results[0]
    original_answer = best_doc.metadata.get("answer", "")
    
    print(f"Retrieved Score: {score:.4f}")
    
    if score >= 0.90:
        print("Score >= 0.90, returning direct answer.")
        return {
            "score": float(score),
            "response": original_answer,
            "direct_match": True
        }
    else:
        print("Score < 0.90, rephrasing with LLM...")
        prompt = PromptTemplate(
            input_variables=["answer", "question"],
            template="rephrase this FAQ answer naturally to address the question '{question}': {answer}"
        )
        
        formatted_prompt = prompt.format(question=query, answer=original_answer)
        llm_output = llm.invoke(formatted_prompt)
        
        return {
            "score": float(score),
            "response": llm_output,
            "direct_match": False
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
