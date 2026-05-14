import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_data():
    print("Loading data from faq_data.csv...")
    df = pd.read_csv("faq_data.csv")
    
    print("Creating LangChain documents...")
    documents = []
    for _, row in df.iterrows():
        # Store Question in page_content for embedding
        # Store Answer in metadata for retrieval
        doc = Document(
            page_content=str(row["Question"]),
            metadata={"answer": str(row["Answering"])}
        )
        documents.append(doc)
    
    print("Initializing HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print("Saving FAISS index to 'faq_index' directory...")
    vectorstore.save_local("faq_index")
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()
