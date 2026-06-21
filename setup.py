import sys

def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers не установлен.")
        print("Выполните: pip install sentence-transformers")
        sys.exit(1)
    print("Скачивание paraphrase-multilingual-MiniLM-L12-v2")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    model.save("./model_cache")
    print("Модель сохранена в ./model_cache")
    print("python index.py <путь_к_папке>")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()