import sys

from codelens.config import EMBEDDING_MODEL_NAME, MODEL_CACHE_PATH


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        print("Выполните: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)
    print(f"Скачивание {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model.save(MODEL_CACHE_PATH)
    print(f"Модель сохранена в {MODEL_CACHE_PATH}")


if __name__ == "__main__":
    main()
