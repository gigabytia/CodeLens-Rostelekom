import os
from pathlib import Path


def _get_int(name: str, default: str) -> int:
    val = os.getenv(name, default)
    try:
        return int(val)
    except ValueError:
        return int(default)


def _get_float(name: str, default: str) -> float:
    val = os.getenv(name, default)
    try:
        return float(val)
    except ValueError:
        return float(default)


class Settings:
    def __init__(self):
        BASE_DIR = Path(__file__).resolve().parent.parent
        self.chroma_path: str = os.getenv("CODELENS_CHROMA_PATH", str(BASE_DIR / "chroma_db"))
        self.collection_name: str = os.getenv("CODELENS_COLLECTION_NAME", "code_chunks")
        self.batch_size: int = _get_int("CODELENS_BATCH_SIZE", "8")
        self.chroma_batch: int = _get_int("CODELENS_CHROMA_BATCH", "500")
        self.embedding_model_name: str = os.getenv("CODELENS_EMBEDDING_MODEL", "BAAI/bge-m3")
        self.search_prefix: str = os.getenv(
            "CODELENS_SEARCH_PREFIX",
            "Represent this sentence for searching relevant passages: ",
        )
        self.model_cache_path: str = os.getenv("CODELENS_MODEL_CACHE", str(BASE_DIR / "model_cache"))
        self.ollama_url: str = os.getenv("CODELENS_OLLAMA_URL", "http://localhost:11434")
        self.ollama_model: str = os.getenv("CODELENS_OLLAMA_MODEL", "mistral:7b")
        self.score_line_tolerance: int = _get_int("CODELENS_LINE_TOLERANCE", "2")
        self.mtime_cache_path: str = os.getenv("CODELENS_MTIME_CACHE", str(BASE_DIR / ".mtime_cache.json"))
        self.hybrid_weight: float = _get_float("CODELENS_HYBRID_WEIGHT", "0.5")
        self.bm25_cache_path: str = os.getenv("CODELENS_BM25_CACHE", str(BASE_DIR / ".bm25_index.json"))


settings = Settings()

CHROMA_PATH = settings.chroma_path
COLLECTION_NAME = settings.collection_name
BATCH_SIZE = settings.batch_size
CHROMA_BATCH = settings.chroma_batch
EMBEDDING_MODEL_NAME = settings.embedding_model_name
SEARCH_PREFIX = settings.search_prefix
MODEL_CACHE_PATH = settings.model_cache_path
OLLAMA_URL = settings.ollama_url
OLLAMA_MODEL = settings.ollama_model
SCORE_LINE_TOLERANCE = settings.score_line_tolerance
MTIME_CACHE_PATH = settings.mtime_cache_path
HYBRID_WEIGHT = settings.hybrid_weight
BM25_CACHE_PATH = settings.bm25_cache_path

LANG_EXTENSIONS = {".py": "python", ".java": "java", ".js": "javascript", ".ts": "typescript"}
