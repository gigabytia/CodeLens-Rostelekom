import sys
import argparse

from codelens.indexer import index_directory
from codelens.scorer import generate_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Индексация Python-кода для семантического поиска"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="./gymhero/gymhero",
        help="Путь к папке с Python-файлами (внутренняя gymhero/gymhero)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Принудительная полная переиндексация (игнорировать кеш)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Сгенерировать results.json и вычислить Precision@5",
    )
    args = parser.parse_args()

    if args.eval:
        generate_results()
    else:
        index_directory(args.directory, full_reindex=args.full)
