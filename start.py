import subprocess
import sys
import os

from codelens.indexer import load_model
from codelens.ollama_manager import OllamaManager

MODEL_CACHE_PATH = "./model_cache"


def run(cmd: list[str], check: bool = True) -> bool:
    print(f"{' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=check)
        return True
    except FileNotFoundError:
        print(f"Команда не найдена: {cmd[0]}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Команда завершилась с кодом {e.returncode}")
        return False


def check_python_deps() -> bool:
    try:
        import streamlit  # noqa: F401
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError as e:
        print(f"Не установлена зависимость: {e.name}")
        print("Выполните: pip install -r requirements.txt")
        return False


def check_model_cache() -> bool:
    return os.path.isdir(MODEL_CACHE_PATH) and os.path.isfile(
        os.path.join(MODEL_CACHE_PATH, "config.json")
    )


def download_model():
    try:
        model = load_model()
        model.save(MODEL_CACHE_PATH)
        print(f"Модель сохранена в {MODEL_CACHE_PATH}")
        return True
    except Exception as e:
        print(f"Не удалось скачать модель: {e}")
        return False


def main():
    codebase = sys.argv[1] if len(sys.argv) > 1 else "./gymhero/gymhero"
    full_index = "--full" in sys.argv

    if not check_python_deps():
        sys.exit(1)
    print("Python-зависимости установлены")

    if not check_model_cache():
        print("Модель не найдена, будет скачана.")
        if not download_model():
            print("Не удалось скачать модель")
            sys.exit(1)
    else:
        print("Модель в кеше")

    print(f"\nИндексация кодовой базы: {codebase}")
    if not os.path.isdir(codebase):
        print(f"Папка '{codebase}' не найдена")
        print(f"Укажите путь: python start.py <путь_к_папке>")
        sys.exit(1)

    index_cmd = [sys.executable, "index.py", codebase]
    if full_index:
        index_cmd.append("--full")
    if not run(index_cmd):
        print("Ошибка индексации")

    manager = OllamaManager()
    if manager.is_installed():
        if not manager.is_running():
            manager.start()
        manager.ensure_model()
        print("RAG-режим доступен в веб-интерфейсе\n")
    else:
        print("Ollama не установлена. Для установки: https://ollama.com/download")
        print("RAG-режим будет недоступен.\n")

    print("Запуск веб-интерфейса Streamlit.")
    print("Откроется браузер с адресом http://localhost:8501")
    print("Для остановки нажмите Ctrl+C")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\nПрограмма остановлена.")


if __name__ == "__main__":
    main()
