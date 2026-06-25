from typing import List, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    file_path: str
    chunk_type: str
    name: str
    start_line: int
    end_line: int
    docstring: str
    content: str
    imports: str
    embedding_text: str
    file_modified_at: float = 0.0


class Question(BaseModel):
    question_id: str
    query: str
    correct_chunk_ids: list[str]
    difficulty: str = "unknown"
    language: str = "unknown"
    category: str = "unknown"


class SearchResult(BaseModel):
    question_id: str
    top_5_chunks: List[str] = Field(default_factory=list)


class OllamaRequest(BaseModel):
    model: str
    prompt: str
    system: str = ""
    stream: bool = False


class SearchHit(BaseModel):
    chunk_id: str
    name: str
    file_path: str
    chunk_type: str
    start_line: int
    end_line: int
    docstring: str
    content: str
    relevance_pct: float


class EvalRow(BaseModel):
    question_id: str
    query: str
    correct: int
    matched: int
    score: float
    top_5_chunks: List[str] = Field(default_factory=list)


def parse_chunk_id(chunk_id: str) -> Optional[tuple[str, str, int]]:
    parts = chunk_id.rsplit(":", 2)
    if len(parts) != 3:
        return None
    path, name, line_str = parts
    try:
        lineno = int(line_str)
    except ValueError:
        return None
    return path, name, lineno
