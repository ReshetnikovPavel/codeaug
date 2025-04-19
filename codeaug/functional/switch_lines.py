import ast
import tokenize
from collections import defaultdict
import io

# Надо заменять не просто строчки, а блоки кода (типа весь for целиком)
# И нельзя перемещать строчки из одной функции в другую, с классами тоже какая-то дичь

# 1) Пойти по токенам, если индент, то ок, продолжаем, ждем пока будет дедент. В минус уходить нельзя, надо прерывать
# это и будет нужный скоуп


def get_scopes(program: str) -> defaultdict[int, list[tuple[int, int]]]:
    tokens = tokenize.tokenize(io.BytesIO(program.encode('utf-8')).readline)
    scopes = defaultdict(list)
    start = 0
    scope = 0
    for token in tokens:
        if token.type == tokenize.INDENT:
            scopes[scope].append((start - 1, token.start[0]))
            scope += 1
            start = token.start[0]
        elif token.type == tokenize.DEDENT:
            scopes[scope].append((start, token.start[0]))
            scope -= 1
            start = token.start[0]
        elif token.type == tokenize.ENDMARKER:
            scopes[scope].append((start, token.start[0]))
    return scopes


def get_scope_by_line(scopes: dict[int, list[tuple[int, int]]], line: int) -> int:
    for scope, ranges in scopes.items():
        for start, end in ranges:
            if start <= line < end:
                return scope
    raise ValueError("Scopes does not contain the line")


def get_ranges_with_same_scope(program: str, line: int) -> list[tuple[int, int]]:
    scopes = get_scopes(program)
    scope = get_scope_by_line(scopes, line)
    return scopes[scope]


def can_switch_to(program: str, line: int) -> list[tuple[int, int]]:
    module = ast.parse(program)
    visitor = NamesByLines()
    visitor.visit(module)
    nodes_by_lines = visitor.nodes
    visitor = NamesByIds()
    visitor.visit(module)
    nodes_by_ids = visitor.nodes

    next_uses = [
        other.lineno
        for node in nodes_by_lines[line]
        for other in nodes_by_ids[node.id]
        if other.lineno > node.lineno
    ]
    next_uses.append(len(program.splitlines()))
    can_switch_till = min(next_uses) - 1

    res = []
    for start, end in get_ranges_with_same_scope(program, line):
        if end > can_switch_till:
            res.append((start, end))
            break
        res.append((start, end))
    print(res)
    return res


class NamesByLines(ast.NodeVisitor):
    def __init__(self):
        self.nodes = defaultdict(list)

    def visit_Name(self, node):
        self.nodes[node.lineno].append(node)
        self.generic_visit(node)


class NamesByIds(ast.NodeVisitor):
    def __init__(self):
        self.nodes = defaultdict(list)

    def visit_Name(self, node):
        self.nodes[node.id].append(node)
        self.generic_visit(node)


def switch_lines(program: str, i: int, j: int) -> str | None:
    for start, end in can_switch_to(program, i):
        if j < start or end <= j:
            return None

    lines = program.splitlines(keepends=True)
    i -= 1
    j -= 1
    lines[i], lines[j] = lines[j], lines[i]
    return "".join(lines)


program = """       # 1
a = "Hello"         # 2
a = "Goodbye"       # 3
b = "World"         # 4
# print(a, b)       # 5
if a:               # 6
    print("AAAA")   # 7
    print("a")      # 8
else:               # 9
    print("meee")   # 10
print("f")          # 11
"""
print(switch_lines(program, 4, 8))
