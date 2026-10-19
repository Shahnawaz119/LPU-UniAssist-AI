import uuid
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends

from app.core.config import settings
from app.services.rag_service import rag_service
from app.services.memory_service import memory_service
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    question: str = Field(..., description="The user's question about university services or support.")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID for multi-turn tracking.")

class SourceItem(BaseModel):
    title: str
    source: str
    section: str
    similarity: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []
    session_id: str
    error: bool = False

class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    has_api_key: bool
    docs_indexed: int

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    try:
        result = rag_service.process_query(session_id=session_id, question=request.question)
        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            session_id=result["session_id"],
            error=result.get("error", False)
        )
    except Exception as e:
        logger.exception("Unexpected error in chat endpoint: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_endpoint():
    count = vector_store.collection.count()
    has_key = bool(settings.OPENROUTER_API_KEY and settings.OPENROUTER_API_KEY.strip() != "" and settings.OPENROUTER_API_KEY != "your_openrouter_api_key")
    return HealthResponse(
        status="healthy",
        provider="openrouter",
        model=settings.OPENROUTER_MODEL,
        has_api_key=has_key,
        docs_indexed=count
    )

@router.delete("/chat/{session_id}")
async def clear_session_endpoint(session_id: str):
    memory_service.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}

@router.post("/seed")
async def seed_knowledge_base_endpoint(force: bool = False):
    vector_store.seed_knowledge_base(force=force)
    return {"status": "success", "count": vector_store.collection.count()}
