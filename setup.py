import sys

def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers не установлен.")
        print("Выполните: pip install sentence-transformers")
        sys.exit(1)
    print("Скачивание BAAI/bge-m3")
    model = SentenceTransformer("BAAI/bge-m3")
    model.save("./model_cache")
    print("Модель сохранена в ./model_cache")
    print("python index.py <путь_к_папке>")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()