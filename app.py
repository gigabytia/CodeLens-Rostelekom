import json
import os
import time

import pandas as pd
import streamlit as st

from codelens.chroma_client import ChromaDBClient
from codelens.config import EMBEDDING_MODEL_NAME, MODEL_CACHE_PATH, RELEVANCE_THRESHOLD, SEARCH_PREFIX
from codelens.ollama_client import OllamaClient, SYSTEM_PROMPT
from codelens.scorer import chunks_match, generate_results_ui, parse_chunk_id

st.set_page_config(layout="wide")


@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    if os.path.isdir(MODEL_CACHE_PATH):
        return SentenceTransformer(MODEL_CACHE_PATH)
    st.warning("Локальный кеш модели не найден. Модель будет скачана")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_chromadb():
    return ChromaDBClient()


def search_code(query: str, top_k: int = 5):
    db = load_chromadb()
    model = load_embedding_model()

    if db.count() == 0:
        return [], 0.0

    start = time.time()
    query_vector = model.encode([SEARCH_PREFIX + query], show_progress_bar=False)
    results, elapsed = db.measure_search(query_vector[0].tolist(), top_k=top_k)
    return results, elapsed


def check_ollama(url: str) -> bool:
    client = OllamaClient(url)
    return client.is_available_sync()


def ask_ollama(query: str, fragments: str, url: str, model_name: str) -> str:
    user_prompt = (
        f"Вопрос разработчика: {query}\n\n"
        f"Найденные фрагменты кода:\n{fragments}\n\n"
        f"Ответ:"
    )
    client = OllamaClient(url)
    return client.generate_sync(model_name, user_prompt)


def render_ollama_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ollama (RAG)")

    ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
    ollama_model = st.sidebar.text_input("Модель", value="mistral:7b")

    if "ollama_available" not in st.session_state:
        st.session_state.ollama_available = check_ollama(ollama_url)

    if st.sidebar.button("Проверить подключение"):
        if check_ollama(ollama_url):
            st.sidebar.success("Ollama подключена!")
            st.session_state.ollama_available = True
        else:
            st.sidebar.error("Ollama недоступна")
            st.session_state.ollama_available = False

    return ollama_url, ollama_model, st.session_state.ollama_available


def render_search_page(top_k, rag_mode, ollama_url, ollama_model, ollama_available):
    query = st.text_input(
        "Введите запрос",
        placeholder="Введите вопрос на русском или английском...",
        label_visibility="collapsed",
    )

    search_button = st.button("Найти", type="primary")

    if search_button and query.strip():
        results, elapsed = search_code(query.strip(), top_k=top_k)

        if not results or not results.get("ids") or not results["ids"][0]:
            st.warning(
                "Ничего не найдено. Убедитесь, что база проиндексирована (python index.py <путь>)."
            )
            return

        filtered_indices = [
            i
            for i, dist in enumerate(results["distances"][0])
            if (1 - dist) * 100 >= RELEVANCE_THRESHOLD
        ]

        if not filtered_indices:
            st.warning(
                f"Не найдено релевантных фрагментов кода (релевантность < {RELEVANCE_THRESHOLD}%). "
                "Попробуйте переформулировать вопрос."
            )
            return

        n_results = len(filtered_indices)
        st.success(f"Найдено {n_results} результатов за {elapsed:.2f} сек.")

        for i in filtered_indices:
            meta = results["metadatas"][0][i]
            doc = results["documents"][0][i]
            dist = results["distances"][0][i]

            score = round((1 - dist) * 100, 1)
            name = meta.get("name", "unknown")
            file_path = meta.get("file_path", "?")
            start_line = meta.get("start_line", 0)
            end_line = meta.get("end_line", 0)
            chunk_type = meta.get("chunk_type", "")
            docstring = meta.get("docstring", "")

            with st.expander(
                f"#{i+1} {name} ({file_path}:{start_line}) - {score}%",
                expanded=(i == 0),
            ):
                st.code(doc, language="python")
                st.caption(f"{file_path} | строки {start_line}-{end_line} | {chunk_type}")
                if docstring:
                    st.info(docstring)
                st.metric("Релевантность", f"{score}%")

        if rag_mode and ollama_available:
            st.markdown("---")
            st.subheader("Ответ ИИ")

            fragments_parts = []
            for i in range(n_results):
                meta = results["metadatas"][0][i]
                doc = results["documents"][0][i]
                fname = meta.get("name", "unknown")
                fpath = meta.get("file_path", "?")
                ds = meta.get("docstring", "")
                header = f"--- {fname} ({fpath})"
                if ds:
                    header += f" - {ds}"
                header += " ---"
                fragments_parts.append(f"{header}\n{doc}\n")

            fragments_text = "\n".join(fragments_parts)

            with st.spinner("Генерация ответа через Ollama."):
                answer = ask_ollama(query, fragments_text, ollama_url, ollama_model)

            st.markdown(answer)
        elif rag_mode and not ollama_available:
            st.info("RAG-режим включён, но Ollama недоступна. Запустите: ollama serve")

    elif search_button and not query.strip():
        st.warning("Введите поисковый запрос.")


def render_chat_page(top_k, ollama_url, ollama_model, ollama_available):
    st.header("Чат с код-базой")
    st.caption(
        "Задавайте вопросы о проекте. Система ищет релевантный код "
        "и отвечает с учётом истории диалога."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if st.session_state.chat_messages and st.button(
        "Очистить историю", type="secondary"
    ):
        st.session_state.chat_messages = []
        st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if (
                msg["role"] == "assistant"
                and "references" in msg
                and msg["references"]
            ):
                for ref in msg["references"]:
                    label = f"{ref['name']} ({ref['file_path']}:{ref['start_line']})"
                    if ref.get("docstring"):
                        label += f" - {ref['docstring']}"
                    with st.expander(label):
                        st.code(ref["content"], language="python")

    if prompt := st.chat_input("Введите вопрос о кодовой базе..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        references = []
        fragments_text = ""

        with st.spinner("Поиск релевантного кода."):
            results, elapsed = search_code(prompt, top_k=top_k)

        if results and results.get("ids") and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                dist = results["distances"][0][i]
                score = (1 - dist) * 100
                if score < RELEVANCE_THRESHOLD:
                    continue
                meta = results["metadatas"][0][i]
                doc = results["documents"][0][i]
                ds = meta.get("docstring", "")
                ref = {
                    "name": meta.get("name", "unknown"),
                    "file_path": meta.get("file_path", "?"),
                    "start_line": meta.get("start_line", 0),
                    "content": doc,
                    "docstring": ds,
                    "score": round(score, 1),
                }
                references.append(ref)
                header = f"--- {ref['name']} ({ref['file_path']})"
                if ds:
                    header += f" - {ds}"
                header += " ---"
                fragments_text += f"{header}\n{doc}\n"

        with st.chat_message("assistant"):
            if not ollama_available:
                answer = (
                    "Ollama недоступна."
                    "Проверьте подключение в боковой панели или запустите ollama serve."
                )
                st.warning(answer)
            elif not fragments_text:
                answer = (
                    f"Релевантный код не найден (все результаты ниже порога {RELEVANCE_THRESHOLD}%). "
                    "Попробуйте переформулировать вопрос."
                )
                st.warning(answer)
            else:
                history_msgs = st.session_state.chat_messages[:-1]
                context_parts = []
                for h in history_msgs[-4:]:
                    role = "Пользователь" if h["role"] == "user" else "Ассистент"
                    context_parts.append(f"{role}: {h['content']}")
                context_text = "\n".join(context_parts)

                chat_user_prompt = (
                    f"История диалога:\n{context_text}\n\n"
                    f"Вопрос разработчика: {prompt}\n\n"
                    f"Найденные фрагменты кода:\n{fragments_text}\n\n"
                    f"Ответ:"
                )

                with st.spinner("Генерация ответа."):
                    client = OllamaClient(ollama_url)
                    answer = client.generate_sync(
                        ollama_model, chat_user_prompt, system=SYSTEM_PROMPT
                    )
                    st.markdown(answer)

            if references:
                for ref in references:
                    label = f"{ref['name']} ({ref['file_path']}:{ref['start_line']})"
                    if ref.get("docstring"):
                        label += f" - {ref['docstring']}"
                    with st.expander(label):
                        st.code(ref["content"], language="python")

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer,
            "references": references,
        })


def generate_results_json():
    eval_path = "eval_questions.json"
    if not os.path.isfile(eval_path):
        st.error("Файл eval_questions.json не найден.")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    model = load_embedding_model()
    db = load_chromadb()

    if db.count() == 0:
        st.error("ChromaDB пуста. Сначала выполните индексацию.")
        return

    progress_bar = st.progress(0)
    results = generate_results_ui(questions, model, db, progress_bar)

    output_path = "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    st.success(f"Сохранено {len(results)} результатов в {output_path}")

    mean_score, _ = evaluate_results_local(results, questions)
    st.metric("Precision@5 (расчёт как в score.py)", f"{mean_score:.3f}")


def evaluate_results_local(results: list[dict], questions: list[dict]) -> tuple[float, list[dict]]:
    from codelens.scorer import evaluate_results
    return evaluate_results(results, questions)


def render_metrics_page():
    st.header("Precision@5 - оценка качества поиска")

    eval_path = "eval_questions.json"
    if not os.path.isfile(eval_path):
        st.info("Файл eval_questions.json не найден. Создайте его в корне проекта.")
        return

    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        st.error(f"Ошибка чтения eval_questions.json: {e}")
        return

    if not questions:
        st.warning("Файл eval_questions.json пуст.")
        return

    model = load_embedding_model()
    db = load_chromadb()

    if db.count() == 0:
        st.warning(
            "ChromaDB пуста. Сначала выполните индексацию (python index.py <путь>)."
        )
        return

    if st.button("Сгенерировать results.json (для score.py)"):
        generate_results_json()

    total = len(questions)
    results = generate_results_ui(questions, model, db)
    mean_score, table_rows = evaluate_results_local(results, questions)

    total_possible = sum(min(5, r["correct"]) for r in table_rows) if table_rows else 0
    precision = sum(r.get("matched", 0) for r in table_rows) / total_possible * 100 if total_possible > 0 else 0.0
    st.metric("Precision@5 (сырая)", f"{precision:.1f}%")

    if table_rows:
        st.metric("Precision@5 (score.py формула)", f"{mean_score:.3f}")
        st.write(f"Всего вопросов: {total}")

        display_rows = []
        for r in table_rows:
            query_text = ""
            for q in questions:
                if q.get("question_id") == r["question_id"]:
                    query_text = q.get("query", "")
                    break
            display_rows.append({
                "question_id": r["question_id"],
                "query": query_text[:60] + "..." if len(query_text) > 60 else query_text,
                "correct": r["correct"],
                "matched": r["matched"],
                "score": f"{r['score']:.2f}",
                "top_5_chunks": "\n".join(r["top_5_chunks"]) if r["top_5_chunks"] else "-",
            })

        df = pd.DataFrame(display_rows)
        st.dataframe(df, use_container_width=True)


def main():
    st.title("CodeLens - поиск по коду")

    page = st.sidebar.radio("Навигация", ["Поиск", "Чат", "Precision@5"])

    top_k = st.sidebar.slider(
        "Количество результатов", min_value=1, max_value=10, value=5
    )

    ollama_url, ollama_model, ollama_available = render_ollama_sidebar()

    rag_mode = st.sidebar.checkbox(
        "Режим RAG (ответ на основе кода)",
        value=False,
        disabled=(page == "Чат"),
        help=None if page != "Чат" else "В режиме чата ИИ-ответ всегда включён",
    )

    if page == "Precision@5":
        render_metrics_page()
    elif page == "Чат":
        render_chat_page(top_k, ollama_url, ollama_model, ollama_available)
    else:
        render_search_page(top_k, rag_mode, ollama_url, ollama_model, ollama_available)


if __name__ == "__main__":
    main()
