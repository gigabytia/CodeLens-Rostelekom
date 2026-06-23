import ast
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from codelens.chroma_client import ChromaDBClient
from codelens.config import BATCH_SIZE, EMBEDDING_MODEL_NAME, IMPORT_CONTEXT_LINES, MODEL_CACHE_PATH
from codelens.logger import get_logger
from codelens.models import Chunk

logger = get_logger(__name__)


def load_model():
    from sentence_transformers import SentenceTransformer

    if os.path.isdir(MODEL_CACHE_PATH):
        return SentenceTransformer(MODEL_CACHE_PATH)
    logger.info("Локальный кеш модели не найден. Модель будет скачана")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def extract_imports(source_lines: list[str], n: int = IMPORT_CONTEXT_LINES) -> str:
    return "\n".join(source_lines[:n])


def collect_py_files(directory: str) -> list[str]:
    py_files: list[str] = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".py"):
                py_files.append(os.path.join(root, fname))
    return sorted(py_files)


def parse_file(file_path: str, repo_root: str) -> list[Chunk]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError as e:
        logger.error("Не удалось прочитать %s: %s", file_path, e)
        return []

    source_lines = source.splitlines()
    imports_context = extract_imports(source_lines)

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        logger.error("ошибка в %s: строка %s", file_path, e.lineno)
        return []

    rel = Path(file_path).resolve().relative_to(repo_root).as_posix()
    chunks: list[Chunk] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            chunks.append(_build_chunk(node, rel, source_lines, imports_context, "class"))
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(
                        _build_chunk(
                            child, rel, source_lines, imports_context, "method", class_name=node.name
                        )
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(
                _build_chunk(node, rel, source_lines, imports_context, "function")
            )

    return chunks


def _build_chunk(
    node: ast.AST,
    rel: str,
    source_lines: list[str],
    imports_context: str,
    chunk_type: str,
    class_name: str | None = None,
) -> Chunk:
    name = node.name  # type: ignore[attr-defined]
    start_line = node.lineno  # type: ignore[attr-defined]
    end_line = getattr(node, "end_lineno", start_line) or start_line
    content = "\n".join(source_lines[start_line - 1 : end_line])
    docstring = ast.get_docstring(node) or ""

    if class_name:
        full_name = f"{class_name}.{name}"
    else:
        full_name = name

    chunk_id = f"{rel}:{full_name}:{start_line}"
    embedding_text = (
        f"# imports context\n{imports_context}\n\n"
        f"{chunk_type} {full_name}\n{docstring}\n{content}"
    )

    return Chunk(
        chunk_id=chunk_id,
        file_path="",
        chunk_type=chunk_type,
        name=full_name,
        start_line=start_line,
        end_line=end_line,
        docstring=docstring,
        content=content,
        imports=imports_context,
        embedding_text=embedding_text,
    )


def index_directory(directory: str, full_reindex: bool = False) -> None:
    start_time = time.time()

    scan_root = Path(directory).resolve()
    if not scan_root.is_dir():
        logger.error("директория '%s' не найдена.", directory)
        sys.exit(1)

    repo_root = str(Path(directory).resolve().parent)
    logger.info("repo_root: %s, scan_root: %s", repo_root, scan_root)

    model = load_model()
    db = ChromaDBClient()

    py_files = collect_py_files(directory)
    if not py_files:
        logger.warning("В директории '%s' не найдено .py файлов.", directory)
        sys.exit(0)

    logger.info("Найдено .py файлов: %s", len(py_files))
    existing_mtimes = {} if full_reindex else db.get_file_mtimes()

    files_to_process: list[str] = []
    skipped_count = 0
    for fp in py_files:
        try:
            current_mtime = os.path.getmtime(fp)
        except OSError:
            continue
        if not full_reindex and fp in existing_mtimes:
            if current_mtime <= existing_mtimes[fp]:
                skipped_count += 1
                continue
        files_to_process.append(fp)

    new_count = len(files_to_process)
    all_chunks: list[Chunk] = []

    for fp in tqdm(files_to_process, desc="Парсинг файлов", unit="файл"):
        try:
            current_mtime = os.path.getmtime(fp)
        except OSError:
            continue
        db.remove_file_chunks(fp)

        chunks = parse_file(fp, repo_root)
        for chunk in chunks:
            chunk.file_path = fp
            chunk.file_modified_at = current_mtime
        all_chunks.extend(chunks)

    if not all_chunks:
        elapsed = time.time() - start_time
        logger.info(
            "Обработано файлов: %s Чанков: 0. Время: %.1f сек.",
            len(py_files),
            elapsed,
        )
        return

    logger.info("Итого чанков: %s", len(all_chunks))
    texts = [c.embedding_text for c in all_chunks]
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    db.upsert_chunks(all_chunks, embeddings)

    elapsed = time.time() - start_time
    logger.info(
        "Обработано файлов: %s новых: %s, пропущено: %s. Чанков: %s. Время: %.1f сек.",
        len(py_files),
        new_count,
        skipped_count,
        len(all_chunks),
        elapsed,
    )
