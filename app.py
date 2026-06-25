import json
import os
import time

import pandas as pd
import streamlit as st

from codelens.chroma_client import ChromaDBClient
from codelens.config import EMBEDDING_MODEL_NAME, MODEL_CACHE_PATH, SEARCH_PREFIX
from codelens.models import SearchHit
from codelens.ollama_client import SYSTEM_PROMPT, OllamaClient
from codelens.scorer import evaluate_results, generate_results_ui

st.set_page_config(layout="wide")

HISTORY_CONTEXT_LINES = 4


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


def search_code(query: str, top_k: int = 5) -> tuple[list[SearchHit], float]:
    db = load_chromadb()
    model = load_embedding_model()

    if db.count() == 0:
        return [], 0.0

    t0 = time.perf_counter()
    query_vector = model.encode([SEARCH_PREFIX + query], show_progress_bar=False)
    hits = db.search_hits(query_vector[0].tolist(), top_k=top_k)
    elapsed = time.perf_counter() - t0
    return hits, elapsed


def _hits_to_fragments(hits: list[SearchHit]) -> str:
    parts = []
    for h in hits:
        header = f"--- {h.name} ({h.file_path})"
        if h.docstring:
            header += f" - {h.docstring}"
        header += " ---"
        parts.append(f"{header}\n{h.content}\n")
    return "\n".join(parts)


def render_ollama_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ollama (RAG)")

    ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
    ollama_model = st.sidebar.text_input("Модель", value="mistral:7b")

    if "ollama_available" not in st.session_state:
        st.session_state.ollama_available = False

    if st.sidebar.button("Проверить подключение"):
        client = OllamaClient(ollama_url)
        if client.is_available_sync():
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
        hits, elapsed = search_code(query.strip(), top_k=top_k)

        if not hits:
            st.warning(
                "Ничего не найдено. Убедитесь, что база проиндексирована (python index.py <путь>)."
            )
            return

        st.success(f"Найдено {len(hits)} результатов за {elapsed:.2f} сек.")

        for i, hit in enumerate(hits):
            with st.expander(
                f"#{i+1} `{hit.name}` - {hit.file_path}:{hit.start_line} - {hit.relevance_pct}%",
                expanded=(i == 0),
            ):
                st.code(hit.content, language="python")
                cols = st.columns(3)
                cols[0].caption(f"{hit.file_path}")
                cols[1].caption(f"строки {hit.start_line}-{hit.end_line} | {hit.chunk_type}")
                cols[2].metric("Релевантность", f"{hit.relevance_pct}%")
                if hit.docstring:
                    st.info(hit.docstring)

        if rag_mode and ollama_available:
            st.markdown("---")
            st.subheader("Ответ ИИ")
            fragments = _hits_to_fragments(hits)
            with st.spinner("Генерация ответа через Ollama."):
                client = OllamaClient(ollama_url)
                user_prompt = (
                    f"Вопрос разработчика: {query}\n\n"
                    f"Найденные фрагменты кода:\n{fragments}\n\n"
                    f"Ответ:"
                )
                answer = client.generate_sync(ollama_model, user_prompt)
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
            hits, elapsed = search_code(prompt, top_k=top_k)

        if hits:
            for hit in hits:
                ref = {
                    "name": hit.name,
                    "file_path": hit.file_path,
                    "start_line": hit.start_line,
                    "content": hit.content,
                    "docstring": hit.docstring,
                    "score": hit.relevance_pct,
                }
                references.append(ref)
            fragments_text = _hits_to_fragments(hits)

        with st.chat_message("assistant"):
            if not ollama_available:
                answer = (
                    "Ollama недоступна."
                    "Проверьте подключение в боковой панели или запустите ollama serve."
                )
                st.warning(answer)
            elif not fragments_text:
                answer = (
                    "Релевантный код не найден. Попробуйте переформулировать вопрос."
                )
                st.warning(answer)
            else:
                history_msgs = st.session_state.chat_messages[:-1]
                context_parts = []
                for h in history_msgs[-HISTORY_CONTEXT_LINES:]:
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

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        run_eval = st.button("Запустить оценку", type="primary")

    with col_btn2:
        save_json = st.button("Сохранить results.json")

    if not run_eval and not save_json:
        st.info("Нажмите **Запустить оценку** для расчёта Precision@5.")
        return

    progress_bar = st.progress(0, text="Обработка вопросов…")
    raw_results = generate_results_ui(questions, model, db, progress_bar)
    progress_bar.empty()

    if save_json:
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(raw_results, f, ensure_ascii=False, indent=2)
        st.success(f"Сохранено {len(raw_results)} результатов → `results.json`")

    mean_score, rows = evaluate_results(raw_results, questions)

    m1, m2, m3 = st.columns(3)
    m1.metric("Precision@5", f"{mean_score:.3f}")
    m2.metric("Вопросов", len(questions))
    total_matched = sum(r.matched for r in rows)
    total_possible = sum(min(5, r.correct) for r in rows)
    raw_pct = total_matched / total_possible * 100 if total_possible else 0.0
    m3.metric("Хитов из возможных", f"{total_matched}/{total_possible} ({raw_pct:.1f}%)")

    if rows:
        display = []
        for r in rows:
            q_text = next(
                (q.get("query", "") for q in questions if q.get("question_id") == r.question_id),
                "",
            )
            display.append({
                "ID": r.question_id,
                "Запрос": (q_text[:70] + "…" if len(q_text) > 70 else q_text),
                "Эталонов": r.correct,
                "Найдено": r.matched,
                "Score": f"{r.score:.2f}",
                "Top-5 чанки": " | ".join(r.top_5_chunks) if r.top_5_chunks else "-",
            })
        st.dataframe(pd.DataFrame(display), use_container_width=True)


def main():
    st.title("CodeLens - поиск по коду")

    page = st.sidebar.radio(
        "Навигация",
        ["Поиск", "Чат", "Precision@5"],
    )

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

    db = load_chromadb()
    count = db.count()
    st.sidebar.markdown("---")
    if count > 0:
        st.sidebar.success(f"В базе {count} чанков")
    else:
        st.sidebar.warning("База пуста - запустите `python index.py <путь>`")

    if page == "Precision@5":
        render_metrics_page()
    elif page == "Чат":
        render_chat_page(top_k, ollama_url, ollama_model, ollama_available)
    else:
        render_search_page(top_k, rag_mode, ollama_url, ollama_model, ollama_available)


if __name__ == "__main__":
    main()
