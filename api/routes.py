from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
model_instance = None

class QuestionRequest(BaseModel):
    question: str

def set_model(model):
    global model_instance
    model_instance = model

@router.post("/ask")
def ask_question(req: QuestionRequest):
    if not model_instance:
        return {"error": "Model not initialized"}
    return model_instance.query(req.question)
