import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form , APIRouter , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ingest import ingest_file
from chat import get_answer


app = FastAPI()

app.add_middleware(CORSMiddleware,allow_origins=["*"], allow_credentials=True,allow_methods=["*"],allow_headers=["*"])


class ChatRequest(BaseModel):
    user_query: str
    chat_history: list
    chat_id: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), chat_id: str = Form(...)):
    """
    Receives a file from React, saves it locally, and runs the ingestion pipeline.
    """
    # 1. THE GATEKEEPER: Check extension before saving
    ext = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = {".pdf", ".docx", ".pptx", ".txt", ".md"}
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {ext}. Allowed types: {allowed_extensions}"
        )

    # 2. SAVE THE FILE
    os.makedirs("./temp_uploades", exist_ok=True)
    file_path = os.path.join("./temp_uploades", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer) 

    # 3. PROCESS WITH SAFETIES
    try:
        result = ingest_file(file_path, chat_id)
    except Exception as e:
        # If MarkItDown or LanceDB crashes, delete the temp file first, THEN throw the error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    # 4. CLEAN UP ON SUCCESS
    if os.path.exists(file_path):
        os.remove(file_path)

    return result

@app.post("/chat")
async def chat_with_document(request: ChatRequest):
    """
    Receives a question from React, queries LanceDB, and returns the AI's answer.
    """
 
    answer = get_answer(request.user_query,request.chat_history,request.chat_id)

    return {"answer":answer}
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)