import logging
from typing import Dict, Any, List, Optional

from app.core.config import settings
from app.services.memory_service import memory_service
from app.services.query_service import query_service
from app.services.vector_store import vector_store
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

UNKNOWN_FALLBACK = "I couldn't find that information in the university helpdesk knowledge base."
OUT_OF_SCOPE_FALLBACK = "I'm the University Helpdesk Assistant, so I can only help with questions related to university services and support."

class RAGService:
    """
    Coordinates the University Helpdesk RAG pipeline:
    User Question -> Conversation Memory -> Query Understanding -> Embedding ->
    ChromaDB -> Top-K Relevant Documents -> Relevance Threshold -> Retrieved Context ->
    OpenRouter LLM -> Grounded Answer -> Sources -> React Frontend
    """

    def __init__(self, vector_store_instance: Optional[Any] = None):
        self._vector_store = vector_store_instance

    @property
    def vector_store(self):
        return self._vector_store if self._vector_store is not None else vector_store

    def process_query(self, session_id: str, question: str) -> Dict[str, Any]:
        """
        Executes end-to-end RAG workflow for a user question.
        Guarantees that every factual university answer is grounded in retrieved knowledge.
        """
        clean_question = question.strip()
        if not clean_question:
            return {
                "answer": "Please ask a question about university services or support.",
                "sources": [],
                "session_id": session_id,
            }

        # 1. Out-of-Scope Pre-check
        if query_service.is_out_of_scope(clean_question):
            logger.info("Query identified as out-of-scope: '%s'", clean_question)
            memory_service.add_turn(session_id, clean_question, OUT_OF_SCOPE_FALLBACK, [])
            return {
                "answer": OUT_OF_SCOPE_FALLBACK,
                "sources": [],
                "session_id": session_id,
            }

        # 2. Conversation Memory
        history = memory_service.get_history(session_id, limit=5)

        # 3. Query Understanding (coreference & follow-up resolution)
        search_query = query_service.contextualize_query(clean_question, history)

        # 4. Embedding & ChromaDB Retrieval (Top-K)
        top_k = settings.TOP_K
        candidates = self.vector_store.query(search_query, top_k=top_k)

        # 5. Relevance Threshold Filtering
        threshold = settings.RELEVANCE_THRESHOLD
        relevant_docs = [doc for doc in candidates if doc["similarity"] >= threshold]

        logger.info(
            "Retrieved %d candidates, %d passed threshold (>= %.2f) for query: '%s'",
            len(candidates),
            len(relevant_docs),
            threshold,
            clean_question,
        )

        # If no documents meet the relevance threshold, do NOT call LLM with general knowledge
        if not relevant_docs:
            logger.info("No documents passed relevance threshold. Returning unknown fallback.")
            memory_service.add_turn(session_id, clean_question, UNKNOWN_FALLBACK, [])
            return {
                "answer": UNKNOWN_FALLBACK,
                "sources": [],
                "session_id": session_id,
            }

        # 6. Format Retrieved Context & Collect Sources
        context_blocks = []
        sources = []
        seen_sources = set()

        for idx, doc in enumerate(relevant_docs, start=1):
            context_blocks.append(f"--- Document [{idx}] ---\n{doc['text']}")
            source_key = (doc.get("source"), doc.get("section"))
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "title": doc.get("title", "University Guide"),
                    "source": doc.get("source", "Helpdesk Knowledge Base"),
                    "section": doc.get("section", "General"),
                    "similarity": doc.get("similarity", 0.0),
                })

        retrieved_context = "\n\n".join(context_blocks)

        # 7. Call Isolated OpenRouter LLM Service
        try:
            answer = llm_service.generate_answer(
                context=retrieved_context,
                conversation_history=history,
                user_question=clean_question,
            )
        except ValueError as e:
            logger.error("Configuration error in LLM service: %s", e)
            return {
                "answer": f"Configuration Error: {str(e)}",
                "sources": sources,
                "session_id": session_id,
                "error": True,
            }
        except RuntimeError as e:
            logger.error("Error executing LLM call: %s", e)
            return {
                "answer": f"LLM Provider Error: {str(e)}",
                "sources": sources,
                "session_id": session_id,
                "error": True,
            }

        # 8. Record Turn in Memory
        memory_service.add_turn(session_id, clean_question, answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
        }

rag_service = RAGService()
