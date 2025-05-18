from typing import Callable

from tree_sitter import Node

from codeaug.utils import generic_visit, python_program_visitor


conditionals = {"<", ">", "<=", ">=", "==", "!="}
inverts = {
    b"<": b">",
    b">": b"<",
    b"<=": b">=",
    b">=": b"<=",
    b"==": b"==",
    b"!=": b"!=",
}


def __is_conditional(node: Node) -> bool:
    if node.type == "comparison_operator":
        operators = [
            child for child in node.children if child.type in conditionals]
        if len(operators) == 1:
            return True
    return False


def __get_operator(node: Node) -> Node:
    return [child for child in node.children if child.type in conditionals][0]


def __get_operands(node: Node) -> (Node, Node):
    operands = [
        child for child in node.children if child.type not in conditionals]
    return (operands[0], operands[1])


def __visit(node: Node, program: bytes, should_apply: Callable[[Node], bool]) -> bytes:
    new_program = []
    if __is_conditional(node) and should_apply(node):
        operator = __get_operator(node)
        left, right = __get_operands(node)
        inverted_operator = inverts[operator.text]
        new_program.extend(
            [
                program[node.start_byte: left.start_byte],
                right.text,
                program[left.end_byte: operator.start_byte],
                inverted_operator,
                program[operator.end_byte: right.start_byte],
                left.text,
                program[right.end_byte: node.end_byte],
            ]
        )

        return b"".join(new_program)

    return generic_visit(node, program, __visit, should_apply)


def switch_conditionals(
    program: str, should_apply: Callable[[Node], bool] = lambda _: True
) -> str:
    return python_program_visitor(program, __visit, should_apply)
