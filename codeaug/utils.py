import itertools
import random
import sys
from typing import Callable, Generator

from tree_sitter import Node, Tree, Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())


def randomly(probability: float) -> bool:
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1 inclusive")
    return random.random() < probability


def traverse_tree(tree: Tree) -> Generator[Node, None, None]:
    cursor = tree.walk()

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent():
            break


def get_indentaion_by_node(program: bytes, node: Node) -> int:
    start = program.rfind(b"\n", None, node.start_byte) + 1
    return bytes(
        itertools.takewhile(lambda x: x == ord(" "), program[start : node.start_byte])
    )


def generic_visit(
    node: Node,
    program: bytes,
    visit: Callable[[Node, bytes], bytes],
    should_apply: Callable[[Node], bool],
) -> bytes:
    new_program = []
    prev = node.start_byte
    for child in node.children:
        new_program.append(program[prev : child.start_byte])
        new_program.append(visit(child, program, should_apply))
        prev = child.end_byte
    new_program.append(program[prev : node.end_byte])

    return b"".join(new_program)


def python_program_visitor(
    program: str,
    visit: Callable[[Node, bytes, Callable[[Node], bool]], bytes],
    should_apply: Callable[[Node], bool],
) -> str:
    assert isinstance(program, str)

    parser = Parser(PY_LANGUAGE)
    program = bytes(program, "utf-8")
    tree = parser.parse(program)
    root = tree.root_node
    return b"".join(
        [
            program[: root.start_byte],
            visit(tree.root_node, program, should_apply),
            program[root.end_byte :],
        ]
    ).decode("utf-8")


def invert_bool_expr(expr: bytes) -> bytes:
    return b"not (" + expr + b")"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
