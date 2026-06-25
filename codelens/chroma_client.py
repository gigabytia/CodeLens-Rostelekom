import json
import os
import time
from typing import Any

import chromadb

from codelens.config import CHROMA_BATCH, CHROMA_PATH, COLLECTION_NAME, MTIME_CACHE_PATH
from codelens.logger import get_logger
from codelens.models import Chunk, SearchHit, parse_chunk_id

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

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        total = len(chunks)
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]

        metadatas: list[dict[str, Any]] = []
        for c in chunks:
            parsed = parse_chunk_id(c.chunk_id)
            rel_path = parsed[0] if parsed else c.file_path
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
                embeddings=[e.tolist() for e in embeddings[i:batch_end]],
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
            n_results=min(top_k, max(self.count(), 1)),
            include=include,
        )

    def search_hits(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchHit]:
        raw = self.search(query_vector, top_k=top_k)
        hits: list[SearchHit] = []

        ids = (raw.get("ids") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        for chunk_id, meta, doc, dist in zip(ids, metas, docs, dists):
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    name=meta.get("name", "unknown"),
                    file_path=meta.get("file_path", "?"),
                    chunk_type=meta.get("chunk_type", ""),
                    start_line=int(meta.get("start_line", 0)),
                    end_line=int(meta.get("end_line", 0)),
                    docstring=meta.get("docstring", ""),
                    content=doc,
                    relevance_pct=round((1.0 - dist) * 100, 1),
                )
            )
        return hits

    def query_simple(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(self.count(), 1)),
            include=["metadatas"],
        )

    def get_file_mtimes(self) -> dict[str, float]:
        if not os.path.isfile(MTIME_CACHE_PATH):
            return {}
        try:
            with open(MTIME_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def save_file_mtimes(self, mtimes: dict[str, float]) -> None:
        try:
            with open(MTIME_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(mtimes, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def remove_file_chunks(self, file_path_abs: str) -> None:
        result = self.collection.get(where={"file_path_abs": file_path_abs})
        if result and result.get("ids"):
            self.collection.delete(ids=result["ids"])

    def remove_file_chunks_batch(self, paths: list[str]) -> None:
        for path in paths:
            self.remove_file_chunks(path)

    def measure_search(self, query_vector: list[float], top_k: int = 5):
        t0 = time.perf_counter()
        results = self.search(query_vector, top_k=top_k)
        elapsed = time.perf_counter() - t0
        return results, elapsed
