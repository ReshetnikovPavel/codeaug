import itertools
import sys
from typing import Generator

from tree_sitter import Node, Tree


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


def generic_visit(program: bytes, node: Node, visit):
    new_program = []
    prev = node.start_byte
    for child in node.children:
        new_program.append(program[prev : child.start_byte])
        new_program.append(visit(child))
        prev = child.end_byte
    new_program.append(program[prev : node.end_byte])

    return b"".join(new_program)


def invert_bool_expr(expr: bytes) -> bytes:
    return b"not (" + expr + b")"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
