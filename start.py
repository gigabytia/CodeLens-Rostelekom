import subprocess
import sys
import os
import time
import platform

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
        import streamlit
        import chromadb
        import sentence_transformers
        return True
    except ImportError as e:
        print(f"Не установлена зависимость: {e.name}")
        print("Выполните: pip install -r requirements.txt")
        return False


def check_model_cache() -> bool:
    return os.path.isdir("./model_cache") and os.path.isfile("./model_cache/config.json")


def try_ollama() -> subprocess.Popen | None:

    result = subprocess.run(
        ["ollama", "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Ollama не установлена.")
        print("Для установки: https://ollama.com/download")
        print("RAG-режим будет недоступен.\n")
        return None

    check = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if check.returncode == 0:
        print("Ollama запущена")
    else:
        popen_kw = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if platform.system() == "Windows":
            popen_kw["creationflags"] = 0x00000008  # CREATE_NO_WINDOW

        try:
            proc = subprocess.Popen(["ollama", "serve"], **popen_kw)
            time.sleep(3)

            verify = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if verify.returncode == 0:
                print("Ollama запущена")
            else:
                return None
        except Exception as e:
            print(f"Не удалось запустить Ollama: {e}")
            print("RAG-режим будет недоступен.\n")
            return None

    list_result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if "mistral" in list_result.stdout.lower():
        print("mistral:7b уже скачана")
    else:
        print("Скачивание mistral:7b")
        pull_ok = run(["ollama", "pull", "mistral:7b"], check=False)
        if pull_ok:
            print("mistral:7b готова")
        else:
            print("Не удалось скачать mistral:7b")
            print("Выполните вручную: ollama pull mistral:7b")

    print("RAG-режим доступен в веб-интерфейсе\n")
    return None


def main():
    codebase = sys.argv[1] if len(sys.argv) > 1 else "./gymhero"
    full_index = "--full" in sys.argv

    if not check_python_deps():
        sys.exit(1)
    print("Python-зависимости установлены")

    if not check_model_cache():
        print("Модель не найдена, модель будет скачана.")
        if not run([sys.executable, "setup.py"]):
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
        print("Ошибка индексации, но продолжаю запуск...")

    try:
        try_ollama()
    except Exception as e:
        print(f"Ollama недоступна: {e}")
        print("RAG-режим будет отключен в интерфейсе.\n")

    print("Запуск веб-интерфейса Streamlit.")
    print("Откроется браузер с адресом http://localhost:8501")
    print("Для остановки нажмите Ctrl+C")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\nПрограмма остановлена.")


if __name__ == "__main__":
    main()
