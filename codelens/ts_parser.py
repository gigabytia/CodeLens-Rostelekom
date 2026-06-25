import re
from pathlib import Path
from codelens.logger import get_logger
from codelens.models import Chunk

logger = get_logger(__name__)

HAS_TREE_SITTER = False
try:
    from tree_sitter import Parser

    HAS_TREE_SITTER = True
except ImportError:
    Parser = None  # type: ignore
    HAS_TREE_SITTER = False
    logger.warning("tree-sitter не установлен. Java/JS парсинг будет через regex.")

_java_parser = None
_js_parser = None


def _get_java_parser():
    global _java_parser
    if _java_parser is not None:
        return _java_parser
    if Parser is None:
        return None
    try:
        import tree_sitter_java

        p = Parser()
        p.set_language(tree_sitter_java.language())
        _java_parser = p
        return p
    except Exception as e:
        logger.warning("tree-sitter-java не загружен: %s", e)
        return None


def _get_js_parser():
    global _js_parser
    if _js_parser is not None:
        return _js_parser
    if Parser is None:
        return None
    try:
        import tree_sitter_javascript

        p = Parser()
        p.set_language(tree_sitter_javascript.language())
        _js_parser = p
        return p
    except Exception as e:
        logger.warning("tree-sitter-javascript не загружен: %s", e)
        return None


def _get_doc_comment(source_lines: list[str], line_no: int) -> str:
    parts = []
    i = line_no - 2
    while i >= 0:
        stripped = source_lines[i].strip()
        if not stripped:
            if parts:
                break
            i -= 1
            continue
        if stripped.startswith("/*") or stripped.startswith("*") or stripped.startswith("//"):
            cleaned = stripped.lstrip("/*").lstrip("*").lstrip("/").strip()
            if cleaned:
                parts.insert(0, cleaned)
        elif stripped.startswith("*/"):
            i -= 1
            continue
        else:
            break
        i -= 1
    return " ".join(parts)


def _walk_tree(node, depth=0) -> list:
    results = [(node, depth)]
    for child in node.children:
        results.extend(_walk_tree(child, depth + 1))
    return results


def _child_by_type(node, child_type: str):
    for child in node.children:
        if child.type == child_type:
            return child
    return None


def _source_text(node, source_lines: list[str]) -> str:
    sr, sc = node.start_point
    er, ec = node.end_point
    if sr == er:
        return source_lines[sr][sc:ec]
    lines = [source_lines[sr][sc:]]
    for r in range(sr + 1, er):
        lines.append(source_lines[r])
    lines.append(source_lines[er][:ec])
    return "\n".join(lines)


def _node_name(node, source_lines: list[str]) -> str:
    id_node = _child_by_type(node, "identifier") or _child_by_type(node, "property_identifier")
    if id_node:
        sr, sc = id_node.start_point
        er, ec = id_node.end_point
        return source_lines[sr][sc:ec]
    return "unknown"


def parse_java_with_ts(source: str, source_lines: list[str], rel: str) -> list[Chunk]:
    parser = _get_java_parser()
    if parser is None:
        return parse_java_regex(source, source_lines, rel)
    tree = parser.parse(bytes(source, "utf-8"))
    return _extract_java_top(tree.root_node, source_lines, rel)


def _extract_java_top(root, source_lines: list[str], rel: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for child in root.children:
        if child.type == "class_declaration":
            _extract_java_class(child, source_lines, rel, chunks)
        elif child.type == "interface_declaration":
            _extract_java_interface(child, source_lines, rel, chunks)
    return chunks


def _extract_java_class(node, source_lines: list[str], rel: str, chunks: list[Chunk]) -> None:
    name = _node_name(node, source_lines)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    content = _source_text(node, source_lines)
    docstring = _get_doc_comment(source_lines, start)
    chunk_id = f"{rel}:{name}:{start}"
    embedding_text = f"class {name}\n{docstring}\n{content}"
    chunks.append(
        Chunk(
            chunk_id=chunk_id, file_path="", chunk_type="class", name=name,
            start_line=start, end_line=end, docstring=docstring, content=content,
            imports="", embedding_text=embedding_text,
        )
    )
    body = _child_by_type(node, "class_body")
    if body is None:
        return
    for child in body.children:
        if child.type == "method_declaration":
            _extract_java_method(child, source_lines, rel, chunks, name)


def _extract_java_method(node, source_lines: list[str], rel: str, chunks: list[Chunk], class_name: str) -> None:
    name = _node_name(node, source_lines)
    full_name = f"{class_name}.{name}"
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    content = _source_text(node, source_lines)
    docstring = _get_doc_comment(source_lines, start)
    chunk_id = f"{rel}:{full_name}:{start}"
    embedding_text = f"method {full_name}\n{docstring}\n{content}"
    chunks.append(
        Chunk(
            chunk_id=chunk_id, file_path="", chunk_type="method", name=full_name,
            start_line=start, end_line=end, docstring=docstring, content=content,
            imports="", embedding_text=embedding_text,
        )
    )


def _extract_java_interface(node, source_lines: list[str], rel: str, chunks: list[Chunk]) -> None:
    name = _node_name(node, source_lines)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    content = _source_text(node, source_lines)
    docstring = _get_doc_comment(source_lines, start)
    chunk_id = f"{rel}:{name}:{start}"
    embedding_text = f"interface {name}\n{docstring}\n{content}"
    chunks.append(
        Chunk(
            chunk_id=chunk_id, file_path="", chunk_type="class", name=name,
            start_line=start, end_line=end, docstring=docstring, content=content,
            imports="", embedding_text=embedding_text,
        )
    )


def parse_javascript_with_ts(source: str, source_lines: list[str], rel: str) -> list[Chunk]:
    parser = _get_js_parser()
    if parser is None:
        return parse_js_regex(source, source_lines, rel)
    tree = parser.parse(bytes(source, "utf-8"))
    return _extract_js_top(tree.root_node, source_lines, rel)


def _extract_js_top(root, source_lines: list[str], rel: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for child in root.children:
        if child.type == "class_declaration":
            _extract_js_class(child, source_lines, rel, chunks)
        elif child.type == "function_declaration":
            _extract_js_function(child, source_lines, rel, chunks)
    return chunks


def _extract_js_class(node, source_lines: list[str], rel: str, chunks: list[Chunk]) -> None:
    name = _node_name(node, source_lines)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    content = _source_text(node, source_lines)
    docstring = _get_doc_comment(source_lines, start)
    chunk_id = f"{rel}:{name}:{start}"
    embedding_text = f"class {name}\n{docstring}\n{content}"
    chunks.append(
        Chunk(
            chunk_id=chunk_id, file_path="", chunk_type="class", name=name,
            start_line=start, end_line=end, docstring=docstring, content=content,
            imports="", embedding_text=embedding_text,
        )
    )
    body = _child_by_type(node, "class_body")
    if body is None:
        return
    for child in body.children:
        if child.type == "method_definition":
            _extract_js_method(child, source_lines, rel, chunks, class_name=name)


def _extract_js_function(node, source_lines: list[str], rel: str, chunks: list[Chunk], class_name: str = "") -> None:
    name = _node_name(node, source_lines)
    full_name = f"{class_name}.{name}" if class_name else name
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    content = _source_text(node, source_lines)
    docstring = _get_doc_comment(source_lines, start)
    chunk_id = f"{rel}:{full_name}:{start}"
    embedding_text = f"function {full_name}\n{docstring}\n{content}"
    chunks.append(
        Chunk(
            chunk_id=chunk_id, file_path="", chunk_type="function", name=full_name,
            start_line=start, end_line=end, docstring=docstring, content=content,
            imports="", embedding_text=embedding_text,
        )
    )


def _extract_js_method(node, source_lines: list[str], rel: str, chunks: list[Chunk], class_name: str = "") -> None:
    name = _node_name(node, source_lines)
    full_name = f"{class_name}.{name}" if class_name else name
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    content = _source_text(node, source_lines)
    docstring = _get_doc_comment(source_lines, start)
    chunk_id = f"{rel}:{full_name}:{start}"
    embedding_text = f"method {full_name}\n{docstring}\n{content}"
    chunks.append(
        Chunk(
            chunk_id=chunk_id, file_path="", chunk_type="method", name=full_name,
            start_line=start, end_line=end, docstring=docstring, content=content,
            imports="", embedding_text=embedding_text,
        )
    )


def parse_java_regex(source: str, source_lines: list[str], rel: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    keyword = r"(?:public|private|protected|static|abstract|final|synchronized)\s+"
    class_pat = re.compile(rf"(?:{keyword})*(?:class|interface|@interface)\s+(\w+)")
    method_pat = re.compile(rf"(?:{keyword})*(?:<\w+>)?\s*(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+(?:,\s*\w+)*)?\s*\{{")

    for i, line in enumerate(source_lines):
        stripped = line.strip()
        if stripped.startswith("*") or stripped.startswith("//"):
            continue
        cm = class_pat.search(stripped)
        if cm:
            name = cm.group(1)
            start = i + 1
            end = _find_block_end(source_lines, i)
            content = "\n".join(source_lines[start - 1 : end])
            docstring = _get_doc_comment(source_lines, start)
            chunk_id = f"{rel}:{name}:{start}"
            embedding_text = f"class {name}\n{docstring}\n{content}"
            chunks.append(Chunk(
                chunk_id=chunk_id, file_path="", chunk_type="class", name=name,
                start_line=start, end_line=end, docstring=docstring, content=content,
                imports="", embedding_text=embedding_text,
            ))
            continue
        mm = method_pat.search(stripped)
        if mm:
            candidate = mm.group(1)
            skip = {"if", "else", "for", "while", "switch", "try", "catch", "return", "new", "this", "super"}
            if candidate in skip:
                continue
            start = i + 1
            end = _find_block_end(source_lines, i)
            content = "\n".join(source_lines[start - 1 : end])
            docstring = _get_doc_comment(source_lines, start)
            for cls in chunks:
                if cls.start_line <= start <= cls.end_line:
                    name = f"{cls.name}.{candidate}"
                    chunk_id = f"{rel}:{name}:{start}"
                    embedding_text = f"method {name}\n{docstring}\n{content}"
                    chunks.append(Chunk(
                        chunk_id=chunk_id, file_path="", chunk_type="method", name=name,
                        start_line=start, end_line=end, docstring=docstring, content=content,
                        imports="", embedding_text=embedding_text,
                    ))
                    break
    return chunks


def parse_js_regex(source: str, source_lines: list[str], rel: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, line in enumerate(source_lines):
        stripped = line.strip()
        if stripped.startswith("*") or stripped.startswith("//"):
            continue
        cm = re.match(r"(?:export\s+)?(?:default\s+)?class\s+(\w+)", stripped)
        if cm:
            name = cm.group(1)
            start = i + 1
            end = _find_block_end(source_lines, i)
            content = "\n".join(source_lines[start - 1 : end])
            docstring = _get_doc_comment(source_lines, start)
            chunk_id = f"{rel}:{name}:{start}"
            embedding_text = f"class {name}\n{docstring}\n{content}"
            chunks.append(Chunk(
                chunk_id=chunk_id, file_path="", chunk_type="class", name=name,
                start_line=start, end_line=end, docstring=docstring, content=content,
                imports="", embedding_text=embedding_text,
            ))
            continue
        fm = re.match(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)", stripped)
        if fm:
            name = fm.group(1)
            start = i + 1
            end = _find_block_end(source_lines, i)
            content = "\n".join(source_lines[start - 1 : end])
            docstring = _get_doc_comment(source_lines, start)
            chunk_id = f"{rel}:{name}:{start}"
            embedding_text = f"function {name}\n{docstring}\n{content}"
            chunks.append(Chunk(
                chunk_id=chunk_id, file_path="", chunk_type="function", name=name,
                start_line=start, end_line=end, docstring=docstring, content=content,
                imports="", embedding_text=embedding_text,
            ))
    return chunks


def _find_block_end(lines: list[str], open_line: int) -> int:
    depth = 0
    started = False
    for i in range(open_line, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
        if started and depth == 0:
            return i + 1
    return len(lines)


LANG_PARSERS = {
    "java": ("java", parse_java_with_ts),
    "javascript": ("js", parse_javascript_with_ts),
    "typescript": ("js", parse_javascript_with_ts),
    "python": None,
}


def parse_with_lang(lang: str, source: str, source_lines: list[str], rel: str) -> list[Chunk]:
    entry = LANG_PARSERS.get(lang)
    if entry is None:
        return []
    _ext, parser_fn = entry
    return parser_fn(source, source_lines, rel)
