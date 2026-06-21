import json
import os
import time
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    layout="wide",
)

@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    model_path = "./model_cache"
    if os.path.isdir(model_path):
        return SentenceTransformer(model_path)
    else:
        st.warning("Локальный кеш модели не найден. Модель будет скачана")
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_chromadb():
    import chromadb

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="code_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    return collection

def search_code(query: str, top_k: int = 5):
    model = load_embedding_model()
    collection = load_chromadb()

    if collection.count() == 0:
        return [], 0.0

    start = time.time()
    query_vector = model.encode([query], show_progress_bar=False)
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    elapsed = time.time() - start
    return results, elapsed

def check_ollama(url: str) -> bool:
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


SYSTEM_PROMPT = (
    "Ты ассистент, помогающий разработчику разобраться в кодовой базе на Python.\n"
    "СТРОГИЕ ПРАВИЛА - нарушать нельзя:\n"
    "1. Отвечай на основе предоставленных фрагментов кода.\n"
    "2. Если спрашивают о функции или классе, которых НЕТ в фрагментах - скажи прямо: 'Функция/класс [название] не найдена в кодовой базе."
    "3. Не придумывай, не угадывай, не используй общие знания о Python.\n"
    "4. Если вопрос не про код - скажи: 'Я помогаю только с вопросами по кодовой базе.'\n"
    "5. Отвечай кратко и по делу."
)

def ask_ollama(query: str, fragments: str, url: str, model_name: str) -> str:
    user_prompt = (
        f"Вопрос разработчика: {query}\n\n"
        f"Найденные фрагменты кода:\n{fragments}\n\n"
        f"Ответ:"
    )
    try:
        response = requests.post(
            f"{url}/api/generate",
            json={
                "model": model_name,
                "system": SYSTEM_PROMPT,
                "prompt": user_prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "Нет ответа от модели.")
    except requests.exceptions.ConnectionError:
        return "Ошибка подключения к Ollama. Проверьте, что сервер запущен."
    except requests.exceptions.Timeout:
        return "Таймаут ответа Ollama. Попробуйте ещё раз."
    except Exception as e:
        return f"Ошибка Ollama: {e}"

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

        if not results or not results["ids"] or not results["ids"][0]:
            st.warning("Ничего не найдено. Убедитесь, что база проиндексирована (python index.py <путь>).")
            return

        RELEVANCE_THRESHOLD = 10
        filtered_indices = [
            i for i, dist in enumerate(results["distances"][0])
            if (1 - dist) * 100 >= RELEVANCE_THRESHOLD
        ]

        if not filtered_indices:
            st.warning(f"Не найдено релевантных фрагментов кода (релевантность < {RELEVANCE_THRESHOLD}%). Попробуйте переформулировать вопрос.")
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

        RELEVANCE_THRESHOLD = 10
        if results and results["ids"] and results["ids"][0]:
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
                    try:
                        response = requests.post(
                            f"{ollama_url}/api/generate",
                            json={
                                "model": ollama_model,
                                "system": SYSTEM_PROMPT,
                                "prompt": chat_user_prompt,
                                "stream": False,
                            },
                            timeout=120,
                        )
                        response.raise_for_status()
                        answer = response.json().get("response", "Нет ответа от модели.")
                        st.markdown(answer)
                    except requests.exceptions.ConnectionError:
                        answer = "Ошибка подключения к Ollama. Сервер не отвечает."
                        st.warning(answer)
                    except requests.exceptions.Timeout:
                        answer = "Таймаут ответа Ollama. Попробуйте ещё раз."
                        st.warning(answer)
                    except Exception as e:
                        answer = f"Ошибка Ollama: {e}"
                        st.warning(answer)

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


def parse_chunk_id(chunk_id: str):
    """Parse a chunk_id into (path, name, lineno). Returns None on invalid format."""
    parts = chunk_id.rsplit(":", 2)
    if len(parts) != 3:
        return None
    path, name, line_str = parts
    try:
        lineno = int(line_str)
    except ValueError:
        return None
    return path, name, lineno


def chunks_match(predicted: str, reference: str, tolerance: int = 2) -> bool:
    """Check if two chunk_ids match within ±tolerance lines (matches score.py logic)."""
    p = parse_chunk_id(predicted)
    r = parse_chunk_id(reference)
    if p is None or r is None:
        return False
    p_path, p_name, p_line = p
    r_path, r_name, r_line = r
    return (p_path == r_path and p_name == r_name and abs(p_line - r_line) <= tolerance)


def generate_results_json():
    """Generate results.json from eval_questions.json matching score.py format."""
    eval_path = "eval_questions.json"
    if not os.path.isfile(eval_path):
        st.error("Файл eval_questions.json не найден.")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    model = load_embedding_model()
    collection = load_chromadb()

    if collection.count() == 0:
        st.error("ChromaDB пуста. Сначала выполните индексацию.")
        return

    results = []
    progress_bar = st.progress(0)

    for idx, item in enumerate(questions):
        qid = item.get("question_id", f"q_{idx+1:02d}")
        query = item.get("query", "")

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
        progress_bar.progress((idx + 1) / len(questions))

    output_path = "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    st.success(f"Сохранено {len(results)} результатов в {output_path}")

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
                if i not in used_refs and chunks_match(pred, ref):
                    matched += 1
                    used_refs.add(i)
                    break

        score = matched / min(5, len(correct))
        total_score += score

    mean_score = total_score / total_q if total_q > 0 else 0.0
    st.metric("Precision@5 (расчёт как в score.py)", f"{mean_score:.3f}")


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
    collection = load_chromadb()

    if collection.count() == 0:
        st.warning("ChromaDB пуста. Сначала выполните индексацию (python index.py <путь>).")
        return

    if st.button("Сгенерировать results.json (для score.py)"):
        generate_results_json()

    total = len(questions)
    total_matched = 0
    table_rows = []

    for item in questions:
        qid = item.get("question_id", "")
        query = item.get("query", "")
        correct_ids = item.get("correct_chunk_ids", [])
        if not qid or not query or not correct_ids:
            continue

        query_vector = model.encode([query], show_progress_bar=False)
        results = collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=min(5, collection.count()),
            include=["metadatas"],
        )

        top5 = []
        if results and results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                top5.append(
                    f"{meta['file_path']}:{meta['name']}:{meta['start_line']}"
                )
                if len(top5) >= 5:
                    break

        seen = []
        for c in top5:
            if c not in seen:
                seen.append(c)
        top5_dedup = seen

        matched = 0
        used_refs = set()
        for pred in top5_dedup:
            for i, ref in enumerate(correct_ids):
                if i not in used_refs and chunks_match(pred, ref):
                    matched += 1
                    used_refs.add(i)
                    break

        score = matched / min(5, len(correct_ids))
        total_matched += matched

        table_rows.append({
            "question_id": qid,
            "query": query[:60] + "..." if len(query) > 60 else query,
            "correct": len(correct_ids),
            "matched": matched,
            "score": f"{score:.2f}",
            "top_5_chunks": "\n".join(top5) if top5 else "-",
        })

    precision = (total_matched / total * 100) if total > 0 else 0.0
    st.metric("Precision@5 (сырая)", f"{precision:.1f}%")

    if table_rows:
        mean_score = sum(float(r["score"]) for r in table_rows) / len(table_rows)
        st.metric("Precision@5 (score.py формула)", f"{mean_score:.3f}")
        st.write(f"Всего вопросов: {total}")

        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True)

def main():
    st.title("CodeLens - поиск по коду")

    page = st.sidebar.radio("Навигация", ["Поиск", "Чат", "Precision@5"])

    top_k = st.sidebar.slider("Количество результатов", min_value=1, max_value=10, value=5)

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