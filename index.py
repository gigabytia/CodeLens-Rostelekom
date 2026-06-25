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
        default=".",
        help="Путь к папке с Python-файлами (по умолчанию текущая директория)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--full",
        action="store_true",
        help="Принудительная полная переиндексация (игнорировать кеш)",
    )
    group.add_argument(
        "--eval",
        action="store_true",
        help="Сгенерировать results.json и вычислить метрики",
    )
    args = parser.parse_args()

    if args.eval:
        generate_results()
    elif args.full:
        index_directory(args.directory, full_reindex=True)
    else:
        index_directory(args.directory)
