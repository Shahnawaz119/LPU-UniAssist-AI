import logging
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI, OpenAIError, AuthenticationError, APIConnectionError, RateLimitError
from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a University Helpdesk Assistant.
Answer ONLY using the retrieved knowledge base context.
Conversation history is used only to understand references and follow-up questions.
If the answer is not present in the retrieved context, respond exactly with:
'I couldn't find that information in the university helpdesk knowledge base.'
Never invent:
university policies
fees
deadlines
contact information
eligibility requirements
procedures
facilities
academic rules
If the question is unrelated to university support, respond:
'I'm the University Helpdesk Assistant, so I can only help with questions related to university services and support.'"""

class LLMService:
    """
    Isolated service for calling OpenRouter API via OpenAI-compatible SDK.
    The rest of the application must NOT directly call OpenRouter.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
    ):
        self._api_key = api_key
        self._model = model or settings.OPENROUTER_MODEL
        self._base_url = base_url or settings.OPENROUTER_BASE_URL
        self._site_url = site_url or settings.OPENROUTER_SITE_URL
        self._app_name = app_name or settings.OPENROUTER_APP_NAME
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """
        Lazily initializes the OpenAI-compatible client configured for OpenRouter.
        """
        api_key = self._api_key if self._api_key is not None else settings.OPENROUTER_API_KEY
        if not api_key or api_key.strip() == "" or api_key == "your_openrouter_api_key":
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Please provide a valid OPENROUTER_API_KEY in your backend/.env file."
            )

        if self._client is None:
            default_headers = {}
            site_url = self._site_url or settings.OPENROUTER_SITE_URL
            app_name = self._app_name or settings.OPENROUTER_APP_NAME

            if site_url:
                default_headers["HTTP-Referer"] = site_url
            if app_name:
                default_headers["X-Title"] = app_name

            self._client = OpenAI(
                base_url=self._base_url or settings.OPENROUTER_BASE_URL,
                api_key=api_key,
                default_headers=default_headers if default_headers else None,
            )
        return self._client

    def _format_conversation_history(
        self, conversation_history: Optional[Union[List[Dict[str, str]], str]]
    ) -> str:
        """
        Formats conversation history into readable text.
        """
        if not conversation_history:
            return "None"
        if isinstance(conversation_history, str):
            return conversation_history.strip() if conversation_history.strip() else "None"
        
        formatted = []
        for turn in conversation_history:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted) if formatted else "None"

    def generate_answer(
        self,
        context: str,
        conversation_history: Optional[Union[List[Dict[str, str]], str]],
        user_question: str,
    ) -> str:
        """
        Generates a strictly grounded answer to the user question using OpenRouter API.

        Prioritizes retrieved knowledge over conversational history and world knowledge.
        """
        client = self._get_client()
        model_name = self._model or settings.OPENROUTER_MODEL

        formatted_history = self._format_conversation_history(conversation_history)
        clean_context = context.strip() if context and context.strip() else "[NO CONTEXT RETRIEVED]"

        user_content = (
            f"Retrieved Context:\n{clean_context}\n\n"
            f"Conversation History:\n{formatted_history}\n\n"
            f"Current Question:\n{user_question}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            logger.info("Calling OpenRouter with model=%s", model_name)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,  # Zero temperature for deterministic grounding
            )

            choice = response.choices[0]
            answer = choice.message.content
            if answer:
                return answer.strip()
            return "I couldn't find that information in the university helpdesk knowledge base."

        except AuthenticationError as e:
            logger.error("OpenRouter authentication error: %s", e)
            raise RuntimeError(f"OpenRouter authentication failed: {e}") from e
        except RateLimitError as e:
            logger.error("OpenRouter rate limit exceeded: %s", e)
            raise RuntimeError("OpenRouter rate limit exceeded. Please try again later.") from e
        except APIConnectionError as e:
            logger.error("OpenRouter network connection error: %s", e)
            raise RuntimeError(f"Could not connect to OpenRouter API: {e}") from e
        except OpenAIError as e:
            logger.error("OpenRouter API error: %s", e)
            raise RuntimeError(f"OpenRouter API error: {e}") from e


# Global singleton instance for use by rag_service
llm_service = LLMService()

def generate_answer(
    context: str,
    conversation_history: Optional[Union[List[Dict[str, str]], str]],
    user_question: str,
) -> str:
    """
    Convenience function exposing generate_answer directly.
    """
    return llm_service.generate_answer(context, conversation_history, user_question)
