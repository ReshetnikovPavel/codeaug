from typing import Callable

from tree_sitter import Node

from codeaug.utils import generic_visit, get_indentaion_by_node, python_program_visitor


def __is_for_identifier_in_range(node: Node) -> bool:
    right = node.child_by_field_name("right")
    return (
        node.type == "for_statement"
        and right.type == "call"
        and right.child_by_field_name("function").type == "identifier"
        and right.child_by_field_name("function").text == b"range"
    )


def __get_range_arguments(node: Node) -> (bytes, bytes, bytes):
    range_arguments = [
        arg.text
        for arg in node.child_by_field_name("right")
        .child_by_field_name("arguments")
        .children
        if arg.text not in [b",", b"(", b")"]
    ]

    if len(range_arguments) == 1:
        return (b"0", range_arguments[0], b"1")
    elif len(range_arguments) == 2:
        return (range_arguments[0], range_arguments[1], b"1")
    elif len(range_arguments) == 3:
        return tuple(range_arguments)
    raise Exception("Range has too many arguments")


def __visit(node: Node, program: bytes, should_apply: Callable[[Node], bool]) -> bytes:
    new_program = []
    if __is_for_identifier_in_range(node) and should_apply(node):
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        body = node.child_by_field_name("body")

        start, stop, step = __get_range_arguments(node)
        assign = b"".join([left.text, b" = ", start])
        increment = b"".join([left.text, b" += ", step])
        condition = b"".join([left.text, b" < ", stop])

        inside_loop_indentation = get_indentaion_by_node(program, body)
        before_while_indent = get_indentaion_by_node(program, node)

        new_program.extend(
            [
                assign,
                b"\n",
                before_while_indent,
                b"while ",
                condition,
                program[right.end_byte : body.start_byte],
                __visit(body, program, should_apply),
                b"\n",
                inside_loop_indentation,
                increment,
            ]
        )

        return b"".join(new_program)
    return generic_visit(node, program, __visit, should_apply)


def replace_for_with_while(
    program: str, should_apply: Callable[[Node], bool] = lambda _: True
) -> str:
    return python_program_visitor(program, __visit, should_apply)
