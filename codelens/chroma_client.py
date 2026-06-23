import time
from typing import Any

import chromadb

from codelens.config import CHROMA_BATCH, CHROMA_PATH, COLLECTION_NAME
from codelens.logger import get_logger
from codelens.models import Chunk

logger = get_logger(__name__)


class ChromaDBClient:
    def __init__(self, path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def upsert_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        total = len(chunks)
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]

        metadatas: list[dict[str, Any]] = []
        for c in chunks:
            chunk_id_parts = c.chunk_id.rsplit(":", 2)
            rel_path = (
                chunk_id_parts[0] if len(chunk_id_parts) == 3 else c.file_path
            )
            metadatas.append(
                {
                    "file_path_abs": c.file_path,
                    "file_path": rel_path,
                    "chunk_type": c.chunk_type,
                    "name": c.name,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "docstring": c.docstring,
                    "file_modified_at": c.file_modified_at,
                }
            )

        for i in range(0, total, CHROMA_BATCH):
            batch_end = min(i + CHROMA_BATCH, total)
            self.collection.upsert(
                ids=ids[i:batch_end],
                embeddings=[embeddings[j].tolist() for j in range(i, batch_end)],
                metadatas=metadatas[i:batch_end],
                documents=documents[i:batch_end],
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        if include is None:
            include = ["documents", "metadatas", "distances"]
        return self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.count()),
            include=include,
        )

    def get_file_mtimes(self) -> dict[str, float]:
        result = self.collection.get(include=["metadatas"])
        mtimes: dict[str, float] = {}
        if result and result.get("metadatas"):
            for meta in result["metadatas"]:
                fp = meta.get("file_path_abs", "")
                mtime = meta.get("file_modified_at", 0.0)
                if fp:
                    if fp not in mtimes or mtime > mtimes[fp]:
                        mtimes[fp] = mtime
        return mtimes

    def remove_file_chunks(self, file_path_abs: str) -> None:
        result = self.collection.get(where={"file_path_abs": file_path_abs})
        if result and result.get("ids"):
            self.collection.delete(ids=result["ids"])

    def query_simple(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.count()),
            include=["metadatas"],
        )

    def measure_search(self, query_vector: list[float], top_k: int = 5):
        start = time.time()
        results = self.search(query_vector, top_k=top_k)
        elapsed = time.time() - start
        return results, elapsed
