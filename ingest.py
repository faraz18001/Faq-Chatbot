import pandas as pd
from langchain_core.documents import Document
from core.llm import Model
from langchain_community.vectorstores import FAISS

def ingest_data(csv_path="data/faq_data.csv", output_path="data/faq_index"):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Creating LangChain documents...")
    documents = [
        Document(
            page_content=str(row["Question"]),
            metadata={"answer": str(row["Answering"])}
        )
        for _, row in df.iterrows()
    ]
    
    # Use Model class for embeddings
    m = Model(model_name="granite4.1:3b", embeddings_model="all-MiniLM-L6-v2")
    embeddings = m.get_embeddings()
    
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print(f"Saving FAISS index to {output_path}...")
    vectorstore.save_local(output_path)
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()
