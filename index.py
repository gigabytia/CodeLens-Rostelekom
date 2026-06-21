import ast
import os
import sys
import time
import argparse
from pathlib import Path
import chromadb
from tqdm import tqdm

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "code_chunks"
BATCH_SIZE = 32
IMPORT_CONTEXT_LINES = 10

def load_model():
    from sentence_transformers import SentenceTransformer

    model_path = "./model_cache"
    if os.path.isdir(model_path):
        return SentenceTransformer(model_path)
    else:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model

def extract_imports(source_lines: list[str], n: int = IMPORT_CONTEXT_LINES) -> str:
    return "\n".join(source_lines[:n])

def parse_file(file_path: str, repo_root: str) -> list[dict]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError as e:
        print(f"Не удалось прочитать {file_path}: {e}")
        return []

    source_lines = source.splitlines()
    imports_context = extract_imports(source_lines)

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        print(f"ошибка в {file_path}: строка {e.lineno}")
        return []

    rel = Path(file_path).resolve().relative_to(repo_root).as_posix()
    chunks = []

    def make_chunk(node, chunk_type, class_name=None):
        name = node.name
        start_line = node.lineno
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
        return {
            "chunk_id": chunk_id,
            "file_path": file_path,
            "chunk_type": chunk_type,
            "name": full_name,
            "start_line": start_line,
            "end_line": end_line,
            "docstring": docstring,
            "content": content,
            "imports": imports_context,
            "embedding_text": embedding_text,
        }

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            chunks.append(make_chunk(node, "class"))
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(make_chunk(child, "method", class_name=node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(make_chunk(node, "function"))

    return chunks

def collect_py_files(directory: str) -> list[str]:
    py_files = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".py"):
                py_files.append(os.path.join(root, fname))
    return sorted(py_files)

def get_existing_file_mtimes(collection) -> dict[str, float]:
    result = collection.get(include=["metadatas"])
    mtimes = {}
    if result and result["metadatas"]:
        for meta in result["metadatas"]:
            fp = meta.get("file_path_abs", "")
            mtime = meta.get("file_modified_at", 0.0)
            if fp:
                if fp not in mtimes or mtime > mtimes[fp]:
                    mtimes[fp] = mtime
    return mtimes


def remove_chunks_for_file(collection, file_path: str):
    result = collection.get(
        where={"file_path_abs": file_path},
    )
    if result and result["ids"]:
        collection.delete(ids=result["ids"])


def index_directory(directory: str, full_reindex: bool = False):
    start_time = time.time()
    if not os.path.isdir(directory):
        print(f"директория '{directory}' не найдена.")
        sys.exit(1)

    repo_root = str(Path(directory).resolve())
    model = load_model()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    py_files = collect_py_files(directory)
    if not py_files:
        print(f"В директории '{directory}' не найдено .py файлов.")
        sys.exit(0)

    print(f"Найдено .py файлов: {len(py_files)}")
    existing_mtimes = {} if full_reindex else get_existing_file_mtimes(collection)
    files_to_process = []
    skipped_count = 0
    for fp in py_files:
        try:
            current_mtime = os.path.getmtime(fp)
        except OSError:
            continue
        if not full_reindex and fp in existing_mtimes:
            saved_mtime = existing_mtimes[fp]
            if current_mtime <= saved_mtime:
                skipped_count += 1
                continue
        files_to_process.append(fp)
    new_count = len(files_to_process)
    all_chunks = []
    for fp in tqdm(files_to_process, desc="Парсинг файлов", unit="файл"):
        try:
            current_mtime = os.path.getmtime(fp)
        except OSError:
            continue
        remove_chunks_for_file(collection, fp)

        chunks = parse_file(fp, repo_root)
        for chunk in chunks:
            chunk["file_modified_at"] = current_mtime
        all_chunks.extend(chunks)

    if not all_chunks:
        elapsed = time.time() - start_time
        print(
            f"Обработано файлов: {len(py_files)} "
            f"Чанков: 0. Время: {elapsed:.1f} сек."
        )
        return

    print(f"Итого чанков: {len(all_chunks)}")
    texts = [chunk["embedding_text"] for chunk in all_chunks]
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    ids = [chunk["chunk_id"] for chunk in all_chunks]
    metadatas = []
    documents = []
    for chunk in all_chunks:
        chunk_id_parts = chunk["chunk_id"].rsplit(":", 2)
        rel_path = chunk_id_parts[0] if len(chunk_id_parts) == 3 else chunk["file_path"]
        meta = {
            "file_path_abs": chunk["file_path"],
            "file_path": rel_path,
            "chunk_type": chunk["chunk_type"],
            "name": chunk["name"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "docstring": chunk["docstring"],
            "file_modified_at": chunk["file_modified_at"],
        }
        metadatas.append(meta)
        documents.append(chunk["content"])
    CHROMA_BATCH = 500
    total = len(ids)
    print(f"Сохранение в ChromaDB {total} чанков")
    for i in tqdm(range(0, total, CHROMA_BATCH), desc="Запись в ChromaDB", unit="батч"):
        batch_end = min(i + CHROMA_BATCH, total)
        collection.upsert(
            ids=ids[i:batch_end],
            embeddings=[embeddings[j].tolist() for j in range(i, batch_end)],
            metadatas=metadatas[i:batch_end],
            documents=documents[i:batch_end],
        )

    elapsed = time.time() - start_time
    print(
        f"Обработано файлов: {len(py_files)} "
        f"новых: {new_count}, пропущено: {skipped_count}."
        f"Чанков: {len(all_chunks)}. Время: {elapsed:.1f} сек."
    )

def generate_results():
    """Generate results.json from eval_questions.json using the ChromaDB index."""
    import json

    eval_path = "eval_questions.json"
    if not os.path.isfile(eval_path):
        print(f"Файл {eval_path} не найден.")
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    model = load_model()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        print("ChromaDB пуста. Сначала выполните индексацию.")
        sys.exit(1)

    results = []
    for item in tqdm(questions, desc="Генерация results.json"):
        qid = item.get("question_id", "")
        query = item.get("query", "")
        if not qid or not query:
            continue

        query_vector = model.encode([query], show_progress_bar=False)
        query_results = collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=min(5, collection.count()),
            include=["metadatas"],
        )

        top5 = []
        if query_results and query_results["metadatas"] and query_results["metadatas"][0]:
            for meta in query_results["metadatas"][0]:
                top5.append(
                    f"{meta['file_path']}:{meta['name']}:{meta['start_line']}"
                )
                if len(top5) >= 5:
                    break

        results.append({
            "question_id": qid,
            "top_5_chunks": top5,
        })

    output_path = "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(results)} результатов в {output_path}")

    # Calculate Precision@5 (same formula as score.py)
    total_q = len(results)
    total_score = 0.0
    gt_index = {q["question_id"]: q for q in questions}

    for entry in results:
        qid = entry["question_id"]
        top5 = entry["top_5_chunks"]
        gt = gt_index.get(qid, {})
        correct = gt.get("correct_chunk_ids", [])

        if not correct:
            continue

        seen = []
        for c in top5:
            if c not in seen:
                seen.append(c)
        top5_dedup = seen

        matched = 0
        used_refs = set()
        for pred in top5_dedup:
            for i, ref in enumerate(correct):
                p_parts = pred.rsplit(":", 2)
                r_parts = ref.rsplit(":", 2)
                if len(p_parts) != 3 or len(r_parts) != 3:
                    continue
                p_path, p_name, p_line = p_parts
                r_path, r_name, r_line = r_parts
                try:
                    p_ln = int(p_line)
                    r_ln = int(r_line)
                except ValueError:
                    continue
                if i not in used_refs and p_path == r_path and p_name == r_name and abs(p_ln - r_ln) <= 2:
                    matched += 1
                    used_refs.add(i)
                    break

        score = matched / min(5, len(correct))
        total_score += score

    mean_score = total_score / total_q if total_q > 0 else 0.0
    print(f"Precision@5 (предварительный): {mean_score:.3f}")
    print(f"Запустите: python score.py --predictions results.json --questions {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Индексация Python-кода для семантического поиска"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="./gymhero",
        help="Путь к папке gymhero (внешняя) для индексации",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Принудительная полная переиндексация (игнорировать кеш)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Сгенерировать results.json и вычислить Precision@5",
    )
    args = parser.parse_args()

    if args.eval:
        generate_results()
    else:
        index_directory(args.directory, full_reindex=args.full)