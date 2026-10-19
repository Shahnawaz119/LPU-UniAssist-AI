import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router as api_router
from app.services.vector_store import vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("uniassist")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing UniAssist AI University Helpdesk...")
    logger.info("LLM Provider: OpenRouter (Base URL: %s, Model: %s)", settings.OPENROUTER_BASE_URL, settings.OPENROUTER_MODEL)
    try:
        vector_store.seed_knowledge_base()
        logger.info("ChromaDB initialized with %d knowledge base chunks", vector_store.collection.count())
    except Exception as e:
        logger.error("Failed to seed knowledge base at startup: %s", e)
    yield
    logger.info("Shutting down UniAssist AI.")

app = FastAPI(
    title="UniAssist AI — University Helpdesk",
    description="Grounded University Helpdesk RAG API powered by OpenRouter LLM and ChromaDB",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS - allow frontend to communicate securely with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {
        "app": "UniAssist AI",
        "description": "University Helpdesk RAG Assistant",
        "provider": "OpenRouter",
        "model": settings.OPENROUTER_MODEL,
        "docs": "/docs"
    }
