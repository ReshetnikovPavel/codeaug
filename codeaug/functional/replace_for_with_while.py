import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from codeaug.utils import generic_visit, get_indentaion_by_node

PY_LANGUAGE = Language(tspython.language())


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


def replace_for_with_while(program: bytes) -> bytes:
    def visit(node: Node):
        new_program = []
        if __is_for_identifier_in_range(node):
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            body = node.child_by_field_name("body")

            start, stop, step = __get_range_arguments(node)
            assign = b"".join([left.text, b" = ", start])
            increment = b"".join([left.text, b" += ", step])
            condition = b"".join([left.text, b" < ", stop])

            inside_loop_indentation = get_indentaion_by_node(program, body)

            new_program.extend(
                [
                    assign,
                    b"\n",
                    b"while ",
                    condition,
                    replace_for_with_while(program[right.end_byte : node.end_byte]),
                    b"\n",
                    inside_loop_indentation,
                    increment,
                ]
            )

            return b"".join(new_program)
        return generic_visit(program, node, visit)

    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(program)
    return visit(tree.root_node)
