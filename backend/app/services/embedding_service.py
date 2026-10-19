import math
import hashlib
import logging
from typing import List, Set
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or", "is",
    "are", "was", "were", "when", "what", "where", "how", "why", "who", "do",
    "does", "did", "can", "could", "would", "should", "my", "your", "his", "her",
    "their", "our", "i", "you", "he", "she", "it", "its", "they", "them", "be",
    "been", "being", "have", "has", "had", "with", "from", "by", "about", "into",
    "through", "during", "before", "after", "above", "below", "up", "down", "in",
    "out", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "don", "should", "now", "will", "shall"
}

def get_stems(w: str) -> Set[str]:
    """Generates prefix, suffix, and morphological variants to match plurals, verb forms, and affixes."""
    stems = {w}
    if len(w) > 4:
        stems.add(w[:4])
        stems.add(w[:5])
        stems.add(w[:6])
    if w.endswith("s") and len(w) > 3:
        stems.add(w[:-1])
    if w.endswith("es") and len(w) > 4:
        stems.add(w[:-2])
    if w.endswith("ing") and len(w) > 5:
        stems.add(w[:-3])
    if w.endswith("ed") and len(w) > 4:
        stems.add(w[:-2])
    if w.endswith("ment") and len(w) > 6:
        stems.add(w[:-4])
    if w.endswith("ments") and len(w) > 7:
        stems.add(w[:-5])
    if w.endswith("tion") and len(w) > 6:
        stems.add(w[:-4])
    if w.endswith("tions") and len(w) > 7:
        stems.add(w[:-5])
    if w.endswith("al") and len(w) > 5:
        stems.add(w[:-2])
    if w.endswith("ic") and len(w) > 5:
        stems.add(w[:-2])
    # Common academic & campus prefixes (e.g. reappear -> appear, prefinal -> final)
    if w.startswith("re") and len(w) > 5:
        stems.add(w[2:])
    if w.startswith("pre") and len(w) > 6:
        stems.add(w[3:])
    if w.startswith("non") and len(w) > 6:
        stems.add(w[3:])
    if w.startswith("sub") and len(w) > 6:
        stems.add(w[3:])
    return stems

class SmartSemanticEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    High-performance, 100% deterministic local semantic embedding function.
    Computes L2-normalized feature vectors across 4096 dimensions with morphological
    sub-word stemming, prefix detachment, sub-linear term weighting, and stopword filtration.
    Completely self-contained, offline, and free from external/Gemini dependencies.
    """

    def __init__(self, dim: int = 4096):
        self.dim = dim

    @classmethod
    def name(cls) -> str:
        return "smart_semantic_local_v3"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @classmethod
    def build_from_config(cls, config: dict):
        return cls(dim=config.get("dim", 4096))

    def __call__(self, input: Documents) -> Embeddings:
        results = []
        for text in input:
            vec = [0.0] * self.dim

            # 1. Capture hyphen-joined words (e.g., 're-appear' -> 'reappear', 'wi-fi' -> 'wifi')
            raw_tokens = text.lower().split()
            combined_tokens = []
            for t in raw_tokens:
                if "-" in t:
                    joined = "".join(c for c in t if c.isalnum())
                    if len(joined) > 2:
                        combined_tokens.append(joined)

            # 2. Split standard punctuation
            clean = text.lower()
            for ch in [".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", "-", "_", "/", "\\", "\"", "'", "`", "*", "#", "$", "@", "%"]:
                clean = clean.replace(ch, " ")
            all_words = clean.split() + combined_tokens

            # 3. Frequency count with sub-linear saturation
            tf_counts = {}
            for w in all_words:
                if len(w) > 1 and w not in STOPWORDS:
                    for stem in get_stems(w):
                        tf_counts[stem] = tf_counts.get(stem, 0) + 1

            for stem, count in tf_counts.items():
                idx = int(hashlib.md5(stem.encode("utf-8")).hexdigest(), 16) % self.dim
                weight = 1.0 + math.log(count)
                vec[idx] += weight

            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([round(x / norm, 6) for x in vec])
        return results


class EmbeddingService:
    """
    Local embedding service that computes text embeddings.
    Zero external API calls. Never uses Gemini API.
    """

    def __init__(self):
        self._embedding_fn = SmartSemanticEmbeddingFunction()
        logger.info("Initialized local deterministic embedding service (4096 dimensions with morphological stemming)")

    @property
    def embedding_function(self) -> EmbeddingFunction[Documents]:
        return self._embedding_fn

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_function(texts)

embedding_service = EmbeddingService()
