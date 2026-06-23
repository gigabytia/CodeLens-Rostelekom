## Установка

### Через uv (рекомендуется)

```bash
pip install uv
uv sync
```

### Через pip

```bash
pip install -r requirements.txt
```

---

## Запуск

### Быстрый старт (индексация + веб-интерфейс)

```bash
python start.py [путь к папке]
```

### Если не сработал быстрый запуск

1. **Скачать модель эмбеддингов:**

```bash
python setup.py
```

2. **Проиндексировать код:**

```bash
python index.py ./gymhero/gymhero
```
3. **RAG-режим (опционально)**

```bash
ollama serve
ollama pull mistral:7b
```

4. **Запустить веб-интерфейс:**

```bash
streamlit run app.py
```

---

## Оценка качества

После индексации сгенерировать `results.json`:

```bash
python index.py ./gymhero/gymhero --eval
```

Запустить скорер из датасета:

```bash
python score.py --predictions results.json --questions eval_questions.json
```