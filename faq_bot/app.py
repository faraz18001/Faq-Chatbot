from fastapi import FastAPI
from core.embedder import get_embeddings
from core.retriever import load_retriever
from core.llm import get_llm
from pipeline.rag import RAGPipeline
from api.routes import router, set_pipeline

app = FastAPI(title="FAQ Chatbot API")

@app.on_event("startup")
def startup_event():
    # Initialize components
    embeddings = get_embeddings()
    retriever = load_retriever(embeddings, index_path="data/faq_index")
    llm = get_llm()
    
    # Initialize pipeline
    pipeline = RAGPipeline(retriever, llm)
    set_pipeline(pipeline)
    print("Startup complete.")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
