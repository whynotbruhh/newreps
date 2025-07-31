from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    source_chunks: list = [] # To optionally show which chunks were used