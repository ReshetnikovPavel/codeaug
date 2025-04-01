import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from codeaug.utils import generic_visit, invert_bool_expr


PY_LANGUAGE = Language(tspython.language())


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


def invert_ifs(program: bytes) -> bytes:
    def visit(node: Node):
        new_program = []
        if __is_if_else(node):
            condition = node.child_by_field_name("condition")
            consequence = node.child_by_field_name("consequence")
            alternative = node.child_by_field_name("alternative")
            alternative = alternative.child_by_field_name("body")

            new_program.extend(
                [
                    program[node.start_byte : condition.start_byte],
                    invert_bool_expr(
                        program[condition.start_byte : condition.end_byte]
                    ),
                    program[condition.end_byte : consequence.start_byte],
                    visit(alternative),
                    program[consequence.end_byte : alternative.start_byte],
                    visit(consequence),
                    program[alternative.end_byte : node.end_byte],
                ]
            )
            return b"".join(new_program)
        return generic_visit(program, node, visit)

    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(program)
    return visit(tree.root_node)
