from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
pipeline_instance = None

class QuestionRequest(BaseModel):
    question: str

def set_pipeline(pipeline):
    global pipeline_instance
    pipeline_instance = pipeline

@router.post("/ask")
def ask_question(req: QuestionRequest):
    if not pipeline_instance:
        return {"error": "Pipeline not initialized"}
    return pipeline_instance.answer(req.question)
