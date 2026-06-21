Для запуска:

python start.py [путь к файлу для индексации]

Эта команда установит все необходимые зависимости, проверит подключение ollama и запустит streamlit

Если start.py не сработал:
    Для установки зависимостей:
    1. pip install -r requirements.txt
    2. python setup.py

    Запуск:
        Для RAG режима (можно без него):
        1. ollama serve
        2. ollama pull mistral:7b
    3. python index.py ./codebase_python
    4. streamlit run app.py

(В финальной версии мы допишем readme, покачто здесь только инструкция к запуску)