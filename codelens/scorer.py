import json
import sys
from pathlib import Path

from tqdm import tqdm

from codelens.chroma_client import ChromaDBClient
from codelens.config import SCORE_LINE_TOLERANCE, SEARCH_PREFIX
from codelens.indexer import load_model
from codelens.logger import get_logger
from codelens.models import EvalRow, SearchResult, parse_chunk_id

logger = get_logger(__name__)


def chunks_match(predicted: str, reference: str, tolerance: int = SCORE_LINE_TOLERANCE) -> bool:
    p = parse_chunk_id(predicted)
    r = parse_chunk_id(reference)
    if p is None or r is None:
        return False
    p_path, p_name, p_line = p
    r_path, r_name, r_line = r
    return p_path == r_path and p_name == r_name and abs(p_line - r_line) <= tolerance


def _retrieve_top5(questions: list[dict], model, db: ChromaDBClient) -> list[dict]:
    queries = [SEARCH_PREFIX + q.get("query", "") for q in questions]
    query_embs = model.encode(queries, show_progress_bar=True)
    results: list[dict] = []
    for item, qe in zip(questions, query_embs):
        qid = item.get("question_id", "?")
        query_results = db.collection.query(
            query_embeddings=[qe.tolist()],
            n_results=min(5, db.count()),
            include=["metadatas"],
        )
        top5: list[str] = []
        if query_results and query_results.get("metadatas") and query_results["metadatas"][0]:
            for meta in query_results["metadatas"][0]:
                top5.append(f"{meta['file_path']}:{meta['name']}:{meta['start_line']}")
                if len(top5) >= 5:
                    break
        results.append({"question_id": qid, "top_5_chunks": top5})
    return results


def compute_precision_at_5(top5: list[str], correct: list[str]) -> float:
    if not correct:
        return 0.0
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
    return matched / min(5, len(correct)) if correct else 0.0


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

    results = _retrieve_top5(questions, model, db)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Сохранено %s результатов в %s", len(results), output_path)

    gt_index = {q["question_id"]: q for q in questions}
    total_p5 = 0.0
    total_q = len(results)

    for entry in results:
        qid = entry["question_id"]
        correct = gt_index.get(qid, {}).get("correct_chunk_ids", [])
        total_p5 += compute_precision_at_5(entry["top_5_chunks"], correct)

    logger.info("Precision@5 (внутренняя, matched / min(5, правильные)): %.3f", total_p5 / total_q if total_q else 0.0)
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
    results = _retrieve_top5(questions, model, db)
    if progress_bar is not None:
        progress_bar.progress(1.0)
    return results


def evaluate_results(
    results: list[dict], questions: list[dict]
) -> tuple[float, list[EvalRow]]:
    gt_index = {q["question_id"]: q for q in questions}
    total_q = len(results)
    total_p5 = 0.0
    table_rows: list[EvalRow] = []

    for entry in results:
        qid = entry["question_id"]
        top5 = entry["top_5_chunks"]
        gt = gt_index.get(qid, {})
        correct = gt.get("correct_chunk_ids", [])
        query = gt.get("query", "")

        p5 = compute_precision_at_5(top5, correct) if correct else 0.0
        total_p5 += p5

        matched = round(p5 * min(5, len(correct))) if correct else 0

        table_rows.append(EvalRow(
            question_id=qid,
            query=query,
            correct=len(correct),
            matched=matched,
            score=p5,
            top_5_chunks=top5,
        ))

    mean_p5 = total_p5 / total_q if total_q > 0 else 0.0
    return mean_p5, table_rows
