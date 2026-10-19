import pytest
from unittest.mock import MagicMock, patch
import openai

from app.core.config import settings
from app.services.llm_service import LLMService, SYSTEM_PROMPT, generate_answer
from app.services.vector_store import VectorStore
from app.services.memory_service import MemoryService
from app.services.query_service import QueryService
from app.services.rag_service import (
    RAGService,
    UNKNOWN_FALLBACK,
    OUT_OF_SCOPE_FALLBACK
)

@pytest.fixture
def test_vector_store(tmp_path):
    """VectorStore fixture pointing to temporary test directory."""
    vs = VectorStore(persist_dir=str(tmp_path / "test_chroma"))
    vs.seed_knowledge_base(force=True)
    return vs

@pytest.fixture
def test_rag(test_vector_store):
    """RAGService fixture configured with test vector store."""
    rag = RAGService(vector_store_instance=test_vector_store)
    return rag

def test_system_prompt_enforces_grounding():
    """Verify system prompt contains the required strict grounding rules."""
    assert "You are a University Helpdesk Assistant." in SYSTEM_PROMPT
    assert "Answer ONLY using the retrieved knowledge base context." in SYSTEM_PROMPT
    assert "I couldn't find that information in the university helpdesk knowledge base." in SYSTEM_PROMPT
    assert "I'm the University Helpdesk Assistant, so I can only help with questions related to university services and support." in SYSTEM_PROMPT
    assert "Never invent:" in SYSTEM_PROMPT

def test_missing_api_key_handling():
    """Verify missing API key raises clear ValueError."""
    llm = LLMService(api_key="", base_url="https://openrouter.ai/api/v1")
    with pytest.raises(ValueError) as excinfo:
        llm.generate_answer(
            context="Some context",
            conversation_history=None,
            user_question="When is tuition due?"
        )
    assert "OPENROUTER_API_KEY is not set" in str(excinfo.value)

def test_openrouter_api_failure():
    """Verify OpenRouter network or API failures are caught and handled."""
    llm = LLMService(api_key="test_dummy_key", base_url="https://openrouter.ai/api/v1")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.APIConnectionError(request=MagicMock())
    llm._client = mock_client

    with pytest.raises(RuntimeError) as excinfo:
        llm.generate_answer(
            context="Tuition is due August 15.",
            conversation_history=None,
            user_question="When is tuition due?"
        )
    assert "Could not connect to OpenRouter API" in str(excinfo.value)

def test_out_of_scope_question():
    """Verify out-of-scope question returns standard response without calling LLM."""
    rag = RAGService()
    session_id = "test_out_of_scope_session"
    result = rag.process_query(session_id, "Can you teach me how to bake a chocolate chip cake?")
    
    assert result["answer"] == OUT_OF_SCOPE_FALLBACK
    assert result["sources"] == []
    assert result["session_id"] == session_id

def test_unknown_question_below_relevance_threshold(test_rag):
    """Verify question with no relevant knowledge returns standard fallback."""
    session_id = "test_unknown_session"
    result = test_rag.process_query(session_id, "What is the policy for space shuttle docking on campus?")
    
    assert result["answer"] == UNKNOWN_FALLBACK
    assert result["sources"] == []

def test_normal_faq_question_with_retrieval(test_rag):
    """Verify normal FAQ question retrieves knowledge context and generates answer."""
    session_id = "test_faq_session"
    question = "When is the tuition fee payment deadline for Fall semester?"

    mock_llm_response = (
        "The Odd Semester (Fall) regular fee payment due date is usually July 15 to July 31."
    )

    with patch("app.services.rag_service.llm_service.generate_answer", return_value=mock_llm_response) as mock_gen:
        result = test_rag.process_query(session_id, question)

        assert mock_gen.called
        call_args = mock_gen.call_args[1]
        assert "July 15" in call_args["context"]
        assert result["answer"] == mock_llm_response
        assert len(result["sources"]) > 0
        assert any("tuition" in s["source"].lower() for s in result["sources"])

def test_follow_up_question_using_memory(test_rag):
    """Verify follow-up question uses conversation memory to retrieve relevant antecedent context."""
    session_id = "test_memory_session"
    
    # Turn 1: Ask about library
    turn1_question = "What are the hours for Central Library?"
    with patch("app.services.rag_service.llm_service.generate_answer", return_value="The library is open 8:00 AM to 10:00 PM."):
        test_rag.process_query(session_id, turn1_question)

    # Turn 2: Follow-up using pronoun 'it'
    turn2_question = "Where is it located?"
    mock_turn2_response = "Central Library is located in Block 36 and Block 38."
    with patch("app.services.rag_service.llm_service.generate_answer", return_value=mock_turn2_response) as mock_gen:
        result = test_rag.process_query(session_id, turn2_question)
        
        assert mock_gen.called
        call_args = mock_gen.call_args[1]
        # Context should contain library information resolved from turn 1
        assert "library" in call_args["context"].lower()
        assert any("library" in s["source"].lower() for s in result["sources"])

def test_grounding_verification_no_bypassing(test_rag):
    """Verify retrieval is never bypassed and general knowledge questions don't hallucinate."""
    # When context is empty or irrelevant, the RAG service MUST NOT send question freely
    with patch("app.services.rag_service.llm_service.generate_answer") as mock_gen:
        result = test_rag.process_query("test_grounding", "Who was the Roman Emperor in 100 AD?")
        # LLM should NOT be called because query is either out of scope or below relevance threshold
        assert not mock_gen.called
        assert result["answer"] in [UNKNOWN_FALLBACK, OUT_OF_SCOPE_FALLBACK]
