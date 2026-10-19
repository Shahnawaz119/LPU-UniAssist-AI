import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

COLLECTION_NAME = "university_helpdesk"

class VectorStore:
    """
    Manages ChromaDB vector collection for university knowledge base documents.
    """

    def __init__(self, persist_dir: Optional[str] = None, embedding_fn: Optional[Any] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._embedding_fn = embedding_fn
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            fn = self._embedding_fn if self._embedding_fn is not None else embedding_service.embedding_function
            try:
                self._collection = self.client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=fn,
                    metadata={"hnsw:space": "cosine"}
                )
            except ValueError as e:
                if "Embedding function conflict" in str(e):
                    logger.warning("Embedding function conflict in collection '%s'. Resetting collection.", COLLECTION_NAME)
                    try:
                        self.client.delete_collection(COLLECTION_NAME)
                    except Exception:
                        pass
                    self._collection = self.client.create_collection(
                        name=COLLECTION_NAME,
                        embedding_function=fn,
                        metadata={"hnsw:space": "cosine"}
                    )
                else:
                    raise
        return self._collection

    def chunk_document(self, content: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Splits a markdown document into meaningful semantic sections and granular bullet topics.
        Ensures both high recall for broad inquiries and pinpoint retrieval for specific questions.
        """
        lines = content.split("\n")
        doc_title = ""
        current_section = "General"
        current_lines = []
        chunks = []

        # Find main h1 title
        for line in lines:
            if line.startswith("# "):
                doc_title = line.replace("#", "").strip()
                break
        if not doc_title:
            doc_title = Path(source_file).stem.replace("_", " ").title()

        chunk_index = 0

        def add_section_and_bullets(sec_title: str, sec_lines: List[str]):
            nonlocal chunk_index
            if not sec_lines:
                return

            text_block = "\n".join(sec_lines).strip()
            if len(text_block) > 40:
                chunks.append({
                    "id": f"{Path(source_file).stem}_{chunk_index}",
                    "text": f"Document: {doc_title}\nSection: {sec_title}\n\n{text_block}",
                    "metadata": {
                        "source": source_file,
                        "document_title": doc_title,
                        "section": sec_title,
                        "chunk_id": chunk_index
                    }
                })
                chunk_index += 1

            # Extract granular bullet topics (- **Topic**: details)
            bullet_re = re.compile(r"^[*-]\s+\*\*([^*]+)\*\*:\s*(.*)")
            current_bullet_title = None
            current_bullet_lines = []

            def flush_bullet():
                nonlocal chunk_index, current_bullet_title, current_bullet_lines
                if current_bullet_title and current_bullet_lines:
                    b_text = "\n".join(current_bullet_lines).strip()
                    if len(b_text) > 30:
                        full_sec = f"{sec_title} — {current_bullet_title}"
                        chunks.append({
                            "id": f"{Path(source_file).stem}_{chunk_index}",
                            "text": f"Document: {doc_title}\nSection: {full_sec}\n\n{b_text}",
                            "metadata": {
                                "source": source_file,
                                "document_title": doc_title,
                                "section": full_sec,
                                "chunk_id": chunk_index
                            }
                        })
                        chunk_index += 1
                current_bullet_title = None
                current_bullet_lines = []

            for l in sec_lines:
                match = bullet_re.match(l)
                if match:
                    flush_bullet()
                    current_bullet_title = match.group(1).strip()
                    current_bullet_lines.append(l)
                elif current_bullet_title and (l.startswith("  ") or l.startswith("\t")):
                    current_bullet_lines.append(l)
                elif current_bullet_title and not l.strip():
                    continue
                else:
                    flush_bullet()

            flush_bullet()

        for line in lines:
            if line.startswith("## "):
                # Finish previous section
                if current_lines:
                    add_section_and_bullets(current_section, current_lines)
                    current_lines = []
                current_section = line.replace("##", "").strip()
                current_lines.append(line)
            else:
                current_lines.append(line)

        # Remaining lines
        if current_lines:
            add_section_and_bullets(current_section, current_lines)

        return chunks

    def seed_knowledge_base(self, force: bool = False):
        """
        Loads all markdown files in KNOWLEDGE_BASE_DIR into the ChromaDB collection.
        """
        kb_path = Path(settings.KNOWLEDGE_BASE_DIR)
        if not kb_path.exists():
            logger.warning("Knowledge base directory %s does not exist", kb_path)
            return

        current_count = self.collection.count()
        if current_count > 0 and not force:
            logger.info("ChromaDB collection '%s' already has %d documents. Skipping seeding.", COLLECTION_NAME, current_count)
            return

        if force and current_count > 0:
            logger.info("Force re-seeding: resetting collection '%s'", COLLECTION_NAME)
            self.client.delete_collection(COLLECTION_NAME)
            self._collection = self.client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_service.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )

        all_ids = []
        all_texts = []
        all_metadatas = []

        for md_file in kb_path.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            chunks = self.chunk_document(content, md_file.name)
            for c in chunks:
                all_ids.append(c["id"])
                all_texts.append(c["text"])
                all_metadatas.append(c["metadata"])

        if all_ids:
            logger.info("Adding %d chunks to ChromaDB collection '%s'", len(all_ids), COLLECTION_NAME)
            self.collection.add(
                ids=all_ids,
                documents=all_texts,
                metadatas=all_metadatas
            )
            logger.info("Successfully indexed %d chunks into ChromaDB", len(all_ids))

    def query(self, query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB collection and returns matched documents with similarity score.
        """
        # Ensure knowledge base is seeded
        if self.collection.count() == 0:
            self.seed_knowledge_base()

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        matched = []
        for i in range(len(documents)):
            dist = distances[i] if distances else 0.0
            # For cosine distance (in [0, 2]), similarity is 1.0 - distance
            # Clamped between 0 and 1
            similarity = max(0.0, min(1.0, 1.0 - dist))
            matched.append({
                "id": ids[i] if ids else f"doc_{i}",
                "text": documents[i],
                "metadata": metadatas[i] if metadatas else {},
                "source": metadatas[i].get("source", "University Knowledge Base") if metadatas else "University Knowledge Base",
                "title": metadatas[i].get("document_title", "University Guide") if metadatas else "University Guide",
                "section": metadatas[i].get("section", "General") if metadatas else "General",
                "distance": dist,
                "similarity": round(similarity, 4)
            })

        return matched

vector_store = VectorStore()
