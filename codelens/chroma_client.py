import json
import os
import re
import time
from typing import Any

import chromadb

from codelens.config import CHROMA_BATCH, CHROMA_PATH, COLLECTION_NAME, HYBRID_WEIGHT, MTIME_CACHE_PATH
from codelens.logger import get_logger
from codelens.models import Chunk, SearchHit, parse_chunk_id

logger = get_logger(__name__)

_HAS_BM25 = False
try:
    from rank_bm25 import BM25Okapi

    _HAS_BM25 = True
except ImportError:
    logger.warning("rank_bm25 не установлен. Гибридный поиск недоступен.")


def _bm25_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9_]", " ", text)
    return text.split()


class ChromaDBClient:
    def __init__(self, path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25_index = None
        self._bm25_chunk_ids: list[str] = []
        self._bm25_count = -1

    def count(self) -> int:
        return self.collection.count()

    def _rebuild_bm25(self) -> None:
        if not _HAS_BM25:
            return
        count = self.count()
        if count == self._bm25_count and self._bm25_index is not None:
            return
        try:
            all_data = self.collection.get(include=["documents"])
            docs = all_data.get("documents") or []
            ids = all_data.get("ids") or []
            tokenized = [_bm25_tokenize(d) for d in docs]
            self._bm25_index = BM25Okapi(tokenized)
            self._bm25_chunk_ids = ids
            self._bm25_count = len(docs)
            logger.debug("BM25 индекс перестроен: %d документов", len(docs))
        except Exception as e:
            logger.warning("Ошибка перестроения BM25: %s", e)
            self._bm25_index = None

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
        self._bm25_count = -1

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

    def hybrid_search_hits(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchHit]:
        raw = self.search(query_vector, top_k=top_k * 3)
        ids_v = (raw.get("ids") or [[]])[0]
        metas_v = (raw.get("metadatas") or [[]])[0]
        docs_v = (raw.get("documents") or [[]])[0]
        dists_v = (raw.get("distances") or [[]])[0]

        vec_scores: dict[str, float] = {}
        vec_meta: dict[str, tuple] = {}
        for cid, meta, doc, dist in zip(ids_v, metas_v, docs_v, dists_v):
            vec_scores[cid] = 1.0 - dist
            vec_meta[cid] = (meta, doc)

        bm25_scores: dict[str, float] = {}
        if _HAS_BM25:
            self._rebuild_bm25()
            if self._bm25_index is not None:
                tokens = _bm25_tokenize(query_text)
                scores = self._bm25_index.get_scores(tokens)
                max_score = max(scores) if scores.size > 0 else 1.0
                max_score = max_score if max_score > 0 else 1.0
                for cid, sc in zip(self._bm25_chunk_ids, scores):
                    bm25_scores[cid] = sc / max_score if max_score > 0 else 0.0

        all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
        w = HYBRID_WEIGHT
        combined: list[tuple[str, float]] = []
        for cid in all_ids:
            vs = vec_scores.get(cid, 0.0)
            bs = bm25_scores.get(cid, 0.0)
            combined.append((cid, w * vs + (1.0 - w) * bs))

        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[:top_k]

        hits: list[SearchHit] = []
        for cid, score in combined:
            if cid in vec_meta:
                meta, doc = vec_meta[cid]
            else:
                meta = {"name": "?", "file_path": "?", "chunk_type": "", "start_line": 0, "end_line": 0, "docstring": ""}
                doc = ""
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    name=meta.get("name", "?"),
                    file_path=meta.get("file_path", "?"),
                    chunk_type=meta.get("chunk_type", ""),
                    start_line=int(meta.get("start_line", 0)),
                    end_line=int(meta.get("end_line", 0)),
                    docstring=meta.get("docstring", ""),
                    content=doc,
                    relevance_pct=round(score * 100, 1),
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
        self._bm25_count = -1

    def remove_file_chunks_batch(self, paths: list[str]) -> None:
        for path in paths:
            self.remove_file_chunks(path)

    def measure_search(self, query_vector: list[float], top_k: int = 5):
        t0 = time.perf_counter()
        results = self.search(query_vector, top_k=top_k)
        elapsed = time.perf_counter() - t0
        return results, elapsed
