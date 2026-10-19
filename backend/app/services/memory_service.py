import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryService:
    """
    Manages in-memory conversation history and session context for multi-turn conversations.
    """

    def __init__(self, max_history_per_session: int = 10):
        self.max_history = max_history_per_session
        self._sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_turn(
        self,
        session_id: str,
        user_question: str,
        assistant_answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Appends a conversational turn to the session history.
        """
        turn = {
            "timestamp": datetime.utcnow().isoformat(),
            "user": user_question,
            "assistant": assistant_answer,
            "sources": sources or [],
        }
        self._sessions[session_id].append(turn)
        # Keep within max history limit
        if len(self._sessions[session_id]) > self.max_history:
            self._sessions[session_id] = self._sessions[session_id][-self.max_history:]

    def get_history(self, session_id: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Returns history formatted as role/content pairs for prompt formatting.
        """
        turns = self._sessions.get(session_id, [])[-limit:]
        history: List[Dict[str, str]] = []
        for turn in turns:
            history.append({"role": "user", "content": turn["user"]})
            history.append({"role": "assistant", "content": turn["assistant"]})
        return history

    def get_raw_turns(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Returns raw turns for session inspection.
        """
        return list(self._sessions.get(session_id, []))

    def clear_session(self, session_id: str):
        """
        Clears conversation history for a session.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Cleared session history for %s", session_id)

memory_service = MemoryService()
