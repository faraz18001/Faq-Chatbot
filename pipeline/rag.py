from langchain.prompts import PromptTemplate

class RAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["answer", "question"],
            template="rephrase this FAQ answer naturally to address the question '{question}': {answer}"
        )

    def answer(self, question: str):
        if not self.retriever:
            return {"error": "Vectorstore not loaded. Please run ingestion first."}
            
        # Retrieve top 1 nearest neighbor
        results = self.retriever.similarity_search_with_relevance_scores(question, k=1)
        
        if not results:
            return {"response": "I couldn't find an answer to your question in the FAQ.", "score": 0}
            
        best_doc, score = results[0]
        original_answer = best_doc.metadata.get("answer", "")
        
        if score >= 0.90:
            return {
                "score": float(score),
                "response": original_answer,
                "direct_match": True
            }
        else:
            formatted_prompt = self.prompt.format(question=question, answer=original_answer)
            llm_output = self.llm.invoke(formatted_prompt)
            
            return {
                "score": float(score),
                "response": llm_output,
                "direct_match": False
            }
