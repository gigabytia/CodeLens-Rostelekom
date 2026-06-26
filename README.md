# CodeLens

Семантический поиск по кодовой базе на Python, Java и JavaScript. Векторные эмбеддинги + ChromaDB + Streamlit.

## Архитектура

Код лежит в папке codelens/, разбит на модули:

- config.py - все настройки, которые тянутся из переменных окружения с префиксом CODELENS_. Если переменных нет, подставляются дефолтные значения (пути до ChromaDB, кеша моделей и прочее).
- models.py - Pydantic-модели: Chunk (фрагмент кода), SearchHit (результат поиска),
  EvalRow (строка с оценкой), Question, OllamaRequest.
- chroma_client.py - обёртка над ChromaDB:
  - добавляет чанки батчами по 500 штук (upsert),
  - выдет результаты поиска - список SearchHit с релевантностью в процентах,
  - query_simple - простой запрос метаданных
  - замеряет время поиска через perf_counter
  - инкрементальная индексация через .mtime_cache.json
- indexer.py - AST-парсер. ast.iter_child_nodes обходит дерево,
  собирает функции, классы, методы. Импорты берет через AST.
  Эмбеддинги через sentence-transformers (BAAI/bge-m3) батчами.
- scorer.py - Precision@5. Сначала _retrieve_top5() делает эмбеддинги
  батчем для всех вопросов разом, потом compute_precision_at_5() считает
  совпадения с учётом ±2 строк и дедупликации.
  Формула: matched / min(5, правильные) - совпадает с официальным score.py из датасета.
- ollama_client.py - клиент к Ollama. Асинхронный httpx.AsyncClient,
  для Streamlit.
- ollama_manager.py - запуск/остановка ollama serve, скачивание модели,
  CREATE_NO_WINDOW на Windows.
- app.py - Streamlit. Три страницы: Поиск, Чат, Precision@5.
- index.py - CLI: python index.py <путь> [--full | --eval]
- start.py - быстрый старт: проверяет зависимости, скачивает модель,
  индексирует, запускает Streamlit.
- setup.py - для скачивания модель эмбеддингов.
- ts_parser.py - парсинг Java и JavaScript через tree-sitter (приоритет) или regex (фолбек).
  Поддерживаются .java, .js, .ts. Если tree-sitter не установлен или не скомпилировался, 
  работает regex - ищет class, function,
  interface по регулярным выражениям и считает {} для границ блоков.

Векторная БД: ChromaDB (persistent, хранится в chroma_db/).

Гибридный поиск: векторный (ChromaDB + BGE-M3) + полнотекстовый (BM25 через rank-bm25).
Результаты склеиваются с весом HYBRID_WEIGHT (0.5 по умолчанию, настраивается через CODELENS_HYBRID_WEIGHT).
BM25 цепляет точные совпадения имён функций/классов, векторный - смысл запроса.
В интерфейсе включается чекбоксом "Гибридный поиск".
Модель эмбеддингов: BAAI/bge-m3, кешируется в model_cache/.

## Установка

```
pip install -r requirements.txt
```

Или через uv:

```
pip install uv
uv sync
```

## Запуск

```
python start.py ./gymhero/gymhero
```

Пошагово:

1. Скачать модель: python setup.py
2. Проиндексировать: python index.py ./gymhero/gymhero
3. Запустить: streamlit run app.py

RAG-режим (опционально):

```
ollama serve
ollama pull mistral:7b
```

Оценка качества:

```
python index.py ./gymhero/gymhero --eval
python score.py --predictions results.json --questions eval_questions.json
```

Важно: передавать всегда внутреннюю папку (./gymhero/gymhero), а не внешнюю (./gymhero). Иначе chunk_id будут вида:
gymhero/gymhero/security.py вместо gymhero/security.py.

## Стратегия чункования

Под чанком понимается ровно одна функция или один класс. Это минимальная самодостаточная единица кода: у неё есть имя, docstring
и чёткая зона ответственности. Если нарезать по строкам, получится каша из обрывков, которые ничего не значат по отдельности.
А если индексировать целый файл, эмбеддинг размажется на десяток разных сущностей и потеряет фокус.

Методы индексируются отдельно от класса. Идентификатор чанка собирается по шаблону

```
путь/файла.py:ClassName.method_name:номер_строки.
```

Так можно искать и конкретный метод, и класс целиком - найдётся и то и другое.
Чтобы поиск понимал, откуда функция берёт зависимости, к её эмбеддингу добавляются импорты - первые N строк.
Для дедупликации: если у функции нет docstring, эмбеддинг строится только из её кода, без повторов. Если docstring есть, она включается
в текст, но без избыточности.
При оценке точности допускается попадание в +-2 строки. Если chunk_id указывает на строку 25, а поиск вернул строку 26 - считаем, как попадание.
Это страхует от мелких смещений.

## Примеры запросов

### 1. "Как создать токен доступа"

Находит:
- #1 create_access_token - gymhero/security.py:12 - 56.0%
- #2 Token - gymhero/schemas/auth.py:6 - 52.7%
- #3 get_current_user - gymhero/api/dependencies.py:58 - 51.7%
- #4 login_for_access_token - gymhero/api/routes/auth.py:19 - 50.0%
- #5 get_password_hash - gymhero/security.py:52 - 49.6%

Ответ ИИ режима:

Для создания токена доступа можно воспользоваться функцией create_access_token из файла security.py. Эта функция принимает два параметра — subject и expires_delta. Здесь subject — это субъект, для которого создается токен (например, идентификатор пользователя). Второй параметр expires_delta можно использовать для указания срока действия токена в виде timedelta.

В приведенном выше коде присутствует функция get_password_hash, которая генерирует хэш пароля, что может быть полезно для хранения паролей безопасно.

Чтобы получить токен доступа для конкретного пользователя, можно воспользоваться функцией login_for_access_token из файла auth.py. Эта функция проводит авторизацию пользователя по его почте и паролю, после чего создает токен доступа с помощью функции create_access_token, используя в качестве субъекта идентификатор авторизованного пользователя.

Надеюсь, что данные ответы помогут вам разобраться в кодовой базе. Если возникнут какие-то вопросы или непонятности, пожалуйста, обращайтесь!

### 2. "Получить все тренировочные планы"

Находит:
- #1 get_all_training_plans - gymhero/api/routes/training_plan.py:34 - 54.4%
- #2 get_all_training_plans_for_owner - gymhero/api/routes/training_plan.py:59 - 53.0%
- #3 TrainingPlanCRUD.get_training_units_in_training_plan - gymhero/crud/training_plan.py:62 - 52.8%
- #4 TrainingPlan.__repr__ - gymhero/models/training_plan.py:42 - 52.4%
- #5 TrainingPlanUpdate - gymhero/schemas/training_plan.py:18 - 52.4%

Ответ ИИ режима:

Получить все тренировочные планы можно с помощью функции get_all_training_plans в файле gymhero/api/routes/training_plan.py. Эта функция использует базовую CRUD-функцию training_plan_crud.get_many, которая выполняет поиск всех тренировочных планов с использованием пагинации. Однако эта функция предполагает, что вызывается пользователем с правами суперпользователя, так как в ней проверяется Depends(get_current_superuser). Если нужно получить тренировочные планы для владельца, то лучше воспользоваться функцией get_all_training_plans_for_owner в том же файле. Она также использует пагинацию и требует от пользователя иметь активный статус, проверяя Depends(get_current_active_user).

### 3. "Поиск пользователя по email"

Находит:
- #1 UserCRUDRepository.get_user_by_email - gymhero/crud/user.py:11 - 56.0%
- #2 User.__repr__ - gymhero/models/user.py:20 - 54.0%
- #3 UserCRUDRepository - gymhero/crud/user.py:10 - 53.4%
- #4 fetch_user_by_email - gymhero/api/routes/user.py:70 - 49.5%
- #5 UserCreate - gymhero/schemas/user.py:12 - 49.3%

Ответ ИИ режима:

Пользователь можно получить с помощью функции get_user_by_email в классе UserCRUDRepository. Эта функция принимает объект сессии базы данных (db) и email пользователя в качестве аргументов, и возвращает объект User или None, если пользователь не найден.

Вы можете использовать эту функцию для поиска пользователя, например, так:

```
db = Session()  # получить объект сессии базы данных
user = user_crud.get_user_by_email(db, 'example@example.com')  # найти пользователя с email 'example@example.com'
```

### 4. "Обработка ошибок авторизации"

Находит:
- #1 get_current_user - gymhero/api/dependencies.py:58 - 48.0%
- #2 verify_password - gymhero/security.py:38 - 47.8%
- #3 UserCRUDRepository.authenticate_user - gymhero/crud/user.py:67 - 47.2%
- #4 _get_credential_exception - gymhero/exceptions.py:5 - 46.6%
- #5 UserInDB - gymhero/schemas/user.py:23 - 46.2%

Ответ ИИ режима:

Код get_current_user() из файла dependencies.py запрашивает текущего пользователя из базы данных, используя сеанс и токен авторизации. Если такого пользователя не найдено, функция выбрасывает исключение HTTPException.

В случае ошибки авторизации может помочь изменение обработки ошибок в коде authenticate_user() в файле crud/user.py. Вместо возвращения None, можно выбрасывать исключение HTTPException с соответствующим статусом-кодом и подробностями, например:

```
def authenticate_user(self, db: Session, email: str, password: str) -> Optional[User]:
    user = self.get_user_by_email(db, email)
    if not user:
        raise _get_credential_exception(status_code=status.HTTP_404_NOT_FOUND, details="User not found")
    if not verify_password(password, user.hashed_password):
        raise _get_credential_exception(status_code=status.HTTP_401_UNAUTHORIZED, details="Invalid credentials")
    return user
```
В этом случае при неправильном пароле или отсутствии пользователя будет бросано исключение HTTPException.

### 5. "Как открывается сессия базы данных"

Находит:
- #1 get_db - gymhero/database/db.py:11 - 54.0%
- #2 get_ctx_db - gymhero/database/db.py:32 - 52.7%
- #3 CRUDRepository.get_one - gymhero/crud/base.py:33 - 51.5%
- #4 CRUDRepository.get_many - gymhero/crud/base.py:53 - 50.1%
- #5 fetch_body_parts - gymhero/api/routes/body_part.py:24 - 49.8%

Ответ ИИ режима:

Для открытия сессии базы данных в предоставленной кодовой базе используется функция get_db() и контекстный менеджер get_ctx_db(database_url). Функция get_db() возвращает генератор, который дает доступ к объекту сессии базы данных при каждом вызове next(). Контекстный менеджер get_ctx_db(database_url) создает сессию базы данных и автоматически закрывает её после выхода из блока with. Для работы с данными в базе данных можно использовать класс CRUDRepository, который предоставляет методы для работы с отдельными записями и списками записей. Например, для получения одной записи используется метод get_one(db, *args, **kwargs), а для получения списка — get_many(db, *args, skip: int = 0, limit: int = 100, **kwargs). Для выполнения запросов в базу данных можно использовать функцию fetch_body_parts, которая использует сессию базы данных и параметры пагинации для получения списка записей по body parts.