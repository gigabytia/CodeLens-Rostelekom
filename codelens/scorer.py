import json
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from codelens.chroma_client import ChromaDBClient
from codelens.config import SCORE_LINE_TOLERANCE, SEARCH_PREFIX
from codelens.indexer import load_model
from codelens.logger import get_logger

logger = get_logger(__name__)


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


def chunks_match(predicted: str, reference: str, tolerance: int = SCORE_LINE_TOLERANCE) -> bool:
    p = parse_chunk_id(predicted)
    r = parse_chunk_id(reference)
    if p is None or r is None:
        return False
    p_path, p_name, p_line = p
    r_path, r_name, r_line = r
    return p_path == r_path and p_name == r_name and abs(p_line - r_line) <= tolerance


def generate_results(eval_path: str = "eval_questions.json", output_path: str = "results.json") -> None:
    if not Path(eval_path).is_file():
        logger.error("Файл %s не найден.", eval_path)
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    model = load_model()
    db = ChromaDBClient()

    if db.count() == 0:
        logger.error("ChromaDB пуста. Сначала выполните индексацию.")
        sys.exit(1)

    results: list[dict] = []
    for item in tqdm(questions, desc="Генерация results.json"):
        qid = item.get("question_id", "")
        query = item.get("query", "")
        if not qid or not query:
            continue

        query_emb = model.encode([SEARCH_PREFIX + query], show_progress_bar=False)
        query_results = db.query_simple(query_emb[0].tolist(), top_k=5)

        top5: list[str] = []
        if query_results and query_results.get("metadatas") and query_results["metadatas"][0]:
            for meta in query_results["metadatas"][0]:
                top5.append(f"{meta['file_path']}:{meta['name']}:{meta['start_line']}")
                if len(top5) >= 5:
                    break

        results.append({"question_id": qid, "top_5_chunks": top5})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Сохранено %s результатов в %s", len(results), output_path)

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

        seen: list[str] = []
        for c in top5:
            if c not in seen:
                seen.append(c)
        top5_dedup = seen

        matched = 0
        used_refs: set[int] = set()
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
                if (
                    i not in used_refs
                    and p_path == r_path
                    and p_name == r_name
                    and abs(p_ln - r_ln) <= SCORE_LINE_TOLERANCE
                ):
                    matched += 1
                    used_refs.add(i)
                    break

        score = matched / min(5, len(correct))
        total_score += score

    mean_score = total_score / total_q if total_q > 0 else 0.0
    logger.info("Precision@5 (предварительный): %.3f", mean_score)
    logger.info(
        "Запустите: python score.py --predictions %s --questions %s",
        output_path,
        eval_path,
    )


def generate_results_ui(
    questions: list[dict],
    model,
    db: ChromaDBClient,
    progress_bar=None,
) -> list[dict]:
    results: list[dict] = []

    for idx, item in enumerate(questions):
        qid = item.get("question_id", f"q_{idx+1:02d}")
        query = item.get("query", "")

        query_emb = model.encode([SEARCH_PREFIX + query], show_progress_bar=False)
        query_results = db.query_simple(query_emb[0].tolist(), top_k=5)

        top5: list[str] = []
        if query_results and query_results.get("metadatas") and query_results["metadatas"][0]:
            for meta in query_results["metadatas"][0]:
                top5.append(f"{meta['file_path']}:{meta['name']}:{meta['start_line']}")
                if len(top5) >= 5:
                    break

        results.append({"question_id": qid, "top_5_chunks": top5})

        if progress_bar is not None:
            progress_bar.progress((idx + 1) / len(questions))

    return results


def evaluate_results(
    results: list[dict], questions: list[dict]
) -> tuple[float, list[dict]]:
    gt_index = {q["question_id"]: q for q in questions}
    total_q = len(results)
    total_score = 0.0
    table_rows: list[dict] = []

    for entry in results:
        qid = entry["question_id"]
        top5 = entry["top_5_chunks"]
        correct = entry.get("correct_chunk_ids", [])

        gt = gt_index.get(qid, {})
        correct = gt.get("correct_chunk_ids", [])

        if not correct:
            continue

        seen: list[str] = []
        for c in top5:
            if c not in seen:
                seen.append(c)
        top5_dedup = seen

        matched = 0
        used_refs: set[int] = set()
        for pred in top5_dedup:
            for i, ref in enumerate(correct):
                if i not in used_refs and chunks_match(pred, ref):
                    matched += 1
                    used_refs.add(i)
                    break

        score = matched / min(5, len(correct))
        total_score += score

        table_rows.append({
            "question_id": qid,
            "correct": len(correct),
            "matched": matched,
            "score": score,
            "top_5_chunks": top5,
        })

    mean_score = total_score / total_q if total_q > 0 else 0.0
    return mean_score, table_rows
