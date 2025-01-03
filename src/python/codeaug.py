import sys
from typing import Generator

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser, Tree

PY_LANGUAGE = Language(tspython.language())


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


def remove_comments(program: bytes) -> bytes:
    changed_program = []
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(program)
    prev = 0
    for node in traverse_tree(tree):
        if node.type == "comment":
            newline_index = program.rfind(b"\n", prev, node.start_byte)
            if newline_index == -1:
                newline_index = prev

            before_comment = program[newline_index + 1: node.start_byte]
            if before_comment.strip() == b"":
                changed_program.append(program[prev:newline_index])
                prev = newline_index + 1 + len(before_comment)

            changed_program.append(program[prev: node.start_byte])
            prev = node.end_byte

    changed_program.append(program[prev:])
    return b"".join(changed_program)


def _invert_bool_expression(expr: bytes) -> bytes:
    return b"not (" + expr + b")"


def _indent(expr: bytes) -> bytes:
    return expr.replace('\n', '\n    ')


def invert_if_statements(program: bytes) -> bytes:
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(program)
    # print(tree.root_node)

    def visit(node: Node) -> bytes:
        if (
            node.type == "if_statement"
            and node.child_by_field_name("condition")
            and node.child_by_field_name("consequence")
            and node.child_by_field_name("alternative")
        ):
            condition = node.child_by_field_name("condition")
            consequence = node.child_by_field_name("consequence")
            alternative = node.child_by_field_name("alternative")

            new_condition = _invert_bool_expression(condition.text)
            alternative_body = alternative.child_by_field_name("body")
            # if alternative.type == "else_clause":
            #     alternative_body = alternative.child_by_field_name("body")
            # elif alternative.type == "elif_clause":
            new_consequence = visit(alternative_body)
            new_alternatibe_body = visit(consequence)
            return b"".join(
                [
                    program[node.start_byte: condition.start_byte],
                    new_condition,
                    program[condition.end_byte: consequence.start_byte],
                    new_consequence,
                    program[consequence.end_byte: alternative_body.start_byte],
                    new_alternatibe_body,
                    program[alternative_body.end_byte: node.end_byte],
                ]
            )
        else:
            res = []
            prev = node.start_byte
            for child in node.children:
                res.extend([program[prev: child.start_byte], visit(child)])
                prev = child.end_byte
            res.append(program[prev: node.end_byte])
            return b"".join(res)

    return visit(tree.root_node)


if __name__ == "__main__":
    p = bytes(sys.stdin.read(), "utf-8")
    p = remove_comments(p)
    # p = invert_if_statements(p)
    print(p.decode("utf-8"))
