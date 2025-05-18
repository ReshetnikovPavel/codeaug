import ast
import io
import tokenize
from collections import defaultdict
import random


def _get_scope(program: str, line: int) -> list[tuple[int, int]]:
    tokens = tokenize.tokenize(io.BytesIO(program.encode("utf-8")).readline)
    scopes = defaultdict(list)
    start = 1
    scope = 0
    target_scope = None
    for token in tokens:
        if token.start[0] <= line <= token.end[0]:
            target_scope = scope

        if token.type == tokenize.INDENT:
            scopes[scope].append((start, token.start[0] - 1))
            scope += 1
            start = token.start[0]
        elif token.type == tokenize.DEDENT:
            if target_scope is not None and scope == target_scope:
                scopes[scope].append((start, token.start[0] - 1))
                break
            if scope in scopes:
                del scopes[scope]
            scope -= 1

            start = token.start[0]
        elif token.type == tokenize.ENDMARKER:
            scopes[scope].append((start, token.start[0]))

    return scopes[target_scope]


def _get_closest_stmt_from_line(root: ast.AST, line: int) -> ast.AST | None:
    visitor = Stmts()
    visitor.visit(root)
    stmts = visitor.stmts
    max_depth = 0

    res = None
    for node, depth in stmts:
        if node.lineno <= line <= node.end_lineno and depth > max_depth:
            max_depth = depth
            res = node
    return res


class Stmts(ast.NodeVisitor):
    def __init__(self):
        self.stmts = []

    def visit(self, node, depth=0):
        if isinstance(node, ast.stmt):
            self.stmts.append((node, depth))
        for child in ast.iter_child_nodes(node):
            self.visit(child, depth=depth + 1)


def get_insert_ranges(
    program: str, root: ast.AST, stmt: ast.stmt | None
) -> list[tuple[int, int]]:
    if stmt is None:
        return [(root.lineno, root.end_lineno)]

    visitor = ChildNames()
    visitor.visit(stmt)
    child_names = visitor.child_names

    visitor = NamesByIds()
    visitor.visit(root)
    nodes_by_ids = visitor.nodes

    next_uses = [
        other.lineno
        for child in child_names
        for other in nodes_by_ids[child.id]
        if other.lineno > stmt.end_lineno
    ]
    next_uses.append(len(program.splitlines()))
    can_insert_forward_till = min(next_uses) - 1

    prev_uses = [
        other.lineno
        for child in child_names
        for other in nodes_by_ids[child.id]
        if other.lineno < stmt.lineno
    ]
    prev_uses.append(-1)
    can_insert_backward_till = max(prev_uses) + 1

    res = []
    for start, end in _get_scope(program, stmt.lineno):
        if end < can_insert_backward_till:
            continue
        if can_insert_forward_till < start:
            break
        s = can_insert_backward_till if start < can_insert_backward_till else start
        e = can_insert_forward_till if can_insert_forward_till < end else end
        res.append((s, e))
    return res


class NamesByLines(ast.NodeVisitor):
    def __init__(self):
        self.nodes = defaultdict(list)

    def visit_Name(self, node):
        self.nodes[node.lineno].append(node)
        self.generic_visit(node)


class ChildNames(ast.NodeVisitor):
    def __init__(self):
        self.child_names = None

    def visit(self, node):
        if isinstance(node, ast.Name):
            return [node]

        names = []
        for child in ast.iter_child_nodes(node):
            names.extend(self.visit(child))

        self.child_names = names
        return names


class NamesByIds(ast.NodeVisitor):
    def __init__(self):
        self.nodes = defaultdict(list)

    def visit_Name(self, node):
        self.nodes[node.id].append(node)
        self.generic_visit(node)


def _is_immovable_stmt(stmt: ast.stmt | None) -> bool:
    return type(stmt) in [
        ast.Return,
        ast.Raise,
        ast.Assert,
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.Break,
        ast.Continue,
    ]


def _in_ranges(ranges, target):
    for start, end in ranges:
        if start <= target <= end:
            return True
    return False


def move_stmt(program: str, from_line: int, to_line: int):
    root = ast.parse(program)

    from_stmt = _get_closest_stmt_from_line(root, from_line)
    if _is_immovable_stmt(from_stmt):
        return None
    from_lineno = from_stmt.lineno - 1 if from_stmt else from_line - 1
    from_end_lineno = from_stmt.end_lineno if from_stmt else from_line

    to_stmt = _get_closest_stmt_from_line(root, to_line)
    to_lineno = to_stmt.lineno - 1 if to_stmt else to_line - 1

    if _in_ranges(get_insert_ranges(program, root, from_stmt), to_lineno):
        lines = program.splitlines(keepends=True)
        return "".join(
            [
                *lines[:from_lineno],
                *lines[from_end_lineno:to_lineno],
                *lines[from_lineno:from_end_lineno],
                *lines[to_lineno:],
            ]
        )
    return None


def switch_stmts(program: str, from_line: int, to_line: int):
    root = ast.parse(program)

    from_stmt = _get_closest_stmt_from_line(root, from_line)
    if _is_immovable_stmt(from_stmt):
        return None
    from_lineno = from_stmt.lineno - 1 if from_stmt else from_line - 1
    from_end_lineno = from_stmt.end_lineno if from_stmt else from_line

    to_stmt = _get_closest_stmt_from_line(root, to_line)
    if _is_immovable_stmt(to_stmt):
        return None
    to_lineno = to_stmt.lineno - 1 if to_stmt else to_line - 1
    to_end_lineno = to_stmt.end_lineno if to_stmt else to_line

    if _in_ranges(get_insert_ranges(program, root, from_stmt), to_line) and _in_ranges(
        get_insert_ranges(program, root, to_stmt), from_line
    ):
        lines = program.splitlines(keepends=True)
        return "".join(
            [
                *lines[:from_lineno],
                *lines[to_lineno:to_end_lineno],
                *lines[from_end_lineno:to_lineno],
                *lines[from_lineno:from_end_lineno],
                *lines[to_end_lineno:],
            ]
        )
    return None


def get_random_from_ranges(ranges: list[tuple[int, int]]) -> int | None:
    if not ranges:
        return None

    total_length = sum(end - start + 1 for start, end in ranges)
    if total_length <= 0:
        return None

    random_offset = random.randint(0, total_length - 1)

    for start, end in ranges:
        range_length = end - start + 1
        if random_offset < range_length:
            return start + random_offset
        random_offset -= range_length
    return None


def move_stmt_randomly(program: str, from_line: int) -> str | None:
    root = ast.parse(program)

    from_stmt = _get_closest_stmt_from_line(root, from_line)
    if _is_immovable_stmt(from_stmt):
        return None
    from_lineno = from_stmt.lineno - 1 if from_stmt else from_line - 1
    from_end_lineno = from_stmt.end_lineno if from_stmt else from_line

    ranges = get_insert_ranges(program, root, from_stmt)
    to_line = get_random_from_ranges(ranges)
    if to_line is None or to_line == from_line + 1 or to_line == from_line:
        return None

    to_stmt = _get_closest_stmt_from_line(root, to_line)
    to_lineno = to_stmt.lineno - 1 if to_stmt else to_line - 1

    if _in_ranges(get_insert_ranges(program, root, from_stmt), to_lineno):
        lines = program.splitlines(keepends=True)
        return "".join(
            [
                *lines[:from_lineno],
                *lines[from_end_lineno:to_lineno],
                *lines[from_lineno:from_end_lineno],
                *lines[to_lineno:],
            ]
        )
    return None


def try_move_random_stmt(program: str, tries: int = 3) -> str | None:
    tried = set()
    lines = program.splitlines()
    while True:
        try:
            line = random.randint(1, len(lines))
            if line in tried:
                if len(tried) == len(lines):
                    return None
                continue
            tried.add(line)
            res = move_stmt_randomly(program, line)
            if res is not None:
                return res
            tries -= 1
            if tries == 0:
                return None
        except:  # noqa: E722
            continue


def move_random_stmt(program: str, tries: int = 3) -> str:
    return try_move_random_stmt(program, tries) or program
