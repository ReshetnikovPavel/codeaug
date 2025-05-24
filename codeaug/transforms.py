from typing import Callable
import tree_sitter
import codeaug.functional as F
import codeaug.utils


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, program):
        for t in self.transforms:
            program = t(program)
        return program

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class RemoveComments:
    def __init__(
        self, should_apply: Callable[[tree_sitter.Node], bool] = lambda _: True
    ):
        self.should_apply = should_apply

    def __call__(self, program: str):
        try:
            return F.remove_comments(program, self.should_apply)
        except Exception as e:
            codeaug.utils.eprint(e)
            return program


class InvertIfs:
    def __init__(
        self, should_apply: Callable[[tree_sitter.Node], bool] = lambda _: True
    ):
        self.should_apply = should_apply

    def __call__(self, program: str):
        try:
            return F.invert_ifs(program, self.should_apply)
        except Exception as e:
            codeaug.utils.eprint(e)
            return program


class ReplaceForWithWhile:
    def __init__(
        self, should_apply: Callable[[tree_sitter.Node], bool] = lambda _: True
    ):
        self.should_apply = should_apply

    def __call__(self, program: str):
        try:
            return F.replace_for_with_while(program, self.should_apply)
        except Exception as e:
            codeaug.utils.eprint(e)
            return program


class SwitchConditionals:
    def __init__(
        self, should_apply: Callable[[tree_sitter.Node], bool] = lambda _: True
    ):
        self.should_apply = should_apply

    def __call__(self, program: str):
        try:
            return F.switch_conditionals(program, self.should_apply)
        except Exception as e:
            codeaug.utils.eprint(e)
            return program


class RenameVariables:
    def __init__(
        self,
        rename_func: Callable[[str, str], str],
        should_apply: Callable[[str], bool] = lambda _: True,
    ):
        self.rename_func = rename_func
        self.should_apply = should_apply

    def __call__(self, program: str):
        try:
            return F.rename_variables(program, self.rename_func, self.should_apply)
        except Exception as e:
            codeaug.utils.eprint(e)
            return program


class RenameFunctions:
    def __init__(
        self,
        rename_func: Callable[[str, str], str],
        should_apply: Callable[[str], bool] = lambda _: True,
    ):
        self.rename_func = rename_func
        self.should_apply = should_apply

    def __call__(self, program: str):
        try:
            return F.rename_functions(program, self.rename_func, self.should_apply)
        except Exception as e:
            codeaug.utils.eprint(e)
            return program


class RenameClasses:
    def __init__(
        self,
        rename_func: Callable[[str, str], str],
        should_apply: Callable[[str], bool] = lambda _: True,
    ):
        self.rename_func = rename_func
        self.should_apply = should_apply

    def __call__(self, program: str):
        try:
            return F.rename_classes(program, self.rename_func, self.should_apply)
        except Exception as e:
            codeaug.utils.eprint(e)
            return program


class MoveRandomStmt:
    def __init__(self, tries: int = 3):
        self.tries = tries

    def __call__(self, program: str):
        try:
            return F.move_random_stmt(program, self.tries)
        except Exception as e:
            codeaug.utils.eprint(e)
            return program
