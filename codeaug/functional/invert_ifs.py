from typing import Callable
from tree_sitter import Node

from codeaug.utils import generic_visit, invert_bool_expr, python_program_visitor


def __is_if_else(node: Node):
    condition = node.child_by_field_name("condition")
    consequence = node.child_by_field_name("consequence")
    alternatives = node.children_by_field_name("alternative")
    return (
        condition
        and consequence
        and len(alternatives) == 1
        and alternatives[0].type == "else_clause"
    )


def __visit(node: Node, program: bytes, should_apply: Callable[[Node], bool]) -> bytes:
    new_program = []
    if __is_if_else(node) and should_apply(node):
        condition = node.child_by_field_name("condition")
        consequence = node.child_by_field_name("consequence")
        alternative = node.child_by_field_name("alternative")
        alternative = alternative.child_by_field_name("body")

        new_program.extend(
            [
                program[node.start_byte: condition.start_byte],
                invert_bool_expr(
                    program[condition.start_byte: condition.end_byte]),
                program[condition.end_byte: consequence.start_byte],
                __visit(alternative, program, should_apply),
                program[consequence.end_byte: alternative.start_byte],
                __visit(consequence, program, should_apply),
                program[alternative.end_byte: node.end_byte],
            ]
        )
        return b"".join(new_program)
    return generic_visit(node, program, __visit, should_apply)


def invert_ifs(
    program: str, should_apply: Callable[[Node], bool] = lambda _: True
) -> str:
    return python_program_visitor(program, __visit, should_apply)
