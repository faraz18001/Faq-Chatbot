from fastapi import FastAPI
from core.llm import Model
from api.routes import router, set_model

app = FastAPI(title="FAQ Chatbot API")

@app.on_event("startup")
def startup_event():
    # Initialize components using the consolidated Model class
    model = Model(model_name="granite4.1:3b", embeddings_model="all-MiniLM-L6-v2")
    
    # Initialize pipeline (Model class handles logic now)
    set_model(model)
    print("Startup complete.")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
