import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Typical out-of-scope keywords or intents that have no relation to university operations
OUT_OF_SCOPE_PATTERNS = [
    r"\b(recipe|cook|bake|cake|cookies|dinner|pizza)\b",
    r"\b(world cup|olympics|super bowl|premier league|nba)\b",
    r"\b(weather in (paris|tokyo|london|new york|california))\b",
    r"\b(capital of (france|germany|italy|spain|japan|canada))\b",
    r"\b(write a poem|write code|invert a binary tree|python script|leetcode)\b",
    r"\b(who is the president of|who won the 19\d\d|who won the 20\d\d)\b",
    r"\b(movie recommendations|tell me a joke|write a song)\b",
]

# Educational / campus topics that are definitely in-scope
CAMPUS_KEYWORDS = [
    "university", "college", "campus", "tuition", "fee", "fees", "admission", "admissions",
    "deadline", "course", "courses", "class", "classes", "register", "registration", "drop",
    "withdraw", "withdrawal", "library", "wifi", "wi-fi", "internet", "it support", "helpdesk",
    "dorm", "housing", "residence", "meal plan", "dining", "scholarship", "financial aid",
    "fafsa", "health", "clinic", "counseling", "caps", "registrar", "bursar", "advisor",
    "gpa", "transcript", "credit", "credits", "study room", "parking", "id card",
    "lpu", "lpunest", "ums", "rms", "dsw", "unimall", "uni-mall", "unihospital", "uni-hospital",
    "bh", "gh", "hostel", "mess", "phagwara", "jalandhar", "cgpa", "sgpa", "mte", "ete",
    "ca", "attendance", "debarred", "dl", "duty leave", "netreg", "lms",
    "placement", "placements", "internship", "internships", "job", "jobs", "package", "packages",
    "salary", "lpa", "dcs", "cpe", "pehchan", "super dream", "dream company", "recruiters",
    "recruitment", "interview", "interviews", "resume", "ojt", "noc", "club", "clubs",
    "fest", "fests", "youth vibe", "one india", "one world"
]

class QueryService:
    """
    Analyzes and contextualizes user queries using conversation history.
    Detects out-of-scope queries and rewrites follow-up queries for enhanced retrieval.
    """

    def is_out_of_scope(self, query: str) -> bool:
        """
        Determines if a query is clearly unrelated to university support.
        """
        query_lower = query.lower().strip()

        # Check if matches known out-of-scope patterns first
        for pattern in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, query_lower):
                return True

        # Check if contains explicit campus keywords (whole word matching)
        for kw in CAMPUS_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
                return False

        return False

    def contextualize_query(
        self,
        query: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Enhances follow-up queries with antecedent context from recent turns.
        E.g., "Where is it located?" after asking about "John Sterling Memorial Library"
        becomes "Where is John Sterling Memorial Library located?"
        """
        if not conversation_history:
            return query

        query_lower = query.lower()
        pronouns = [" it ", " it?", " its ", " there ", " that ", " this ", " they ", " them "]
        has_pronoun = any(p in f" {query_lower} " for p in pronouns) or query_lower.startswith(("where is it", "how much is it", "when does it"))

        if not has_pronoun:
            return query

        # Find the last user question to extract context subject
        last_user_turns = [turn["content"] for turn in conversation_history if turn.get("role") == "user"]
        if not last_user_turns:
            return query

        last_question = last_user_turns[-1]
        logger.info("Contextualizing query '%s' using previous turn '%s'", query, last_question)

        # Combine previous topic context with current question for vector search
        return f"{last_question} - {query}"

query_service = QueryService()
