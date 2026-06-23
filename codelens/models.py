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
    top_5_chunks: list[str] = Field(default_factory=list)

class OllamaRequest(BaseModel):
    model: str
    prompt: str
    system: str = ""
    stream: bool = False