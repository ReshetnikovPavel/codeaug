from typing import Callable

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

from codeaug.utils import traverse_tree

PY_LANGUAGE = Language(tspython.language())


def remove_comments(
    program: str, should_apply: Callable[[Node], bool] = lambda _: True
) -> str:
    program = program.encode("utf-8")
    changed_program = []
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(program)
    prev = 0
    for node in traverse_tree(tree):
        if node.type == "comment" and should_apply(node):
            newline_index = program.rfind(b"\n", prev, node.start_byte)
            if newline_index == -1:
                newline_index = prev

            before_comment = program[newline_index + 1 : node.start_byte]
            if before_comment.strip() == b"":
                changed_program.append(program[prev:newline_index])
                prev = newline_index + 1 + len(before_comment)

            changed_program.append(program[prev : node.start_byte])
            prev = node.end_byte

    changed_program.append(program[prev:])
    res = b"".join(changed_program)
    return res.decode("utf-8")
