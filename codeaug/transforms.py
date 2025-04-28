import codeaug.functional as F


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
    def __call__(self, program: str):
        return F.remove_comments(program)


class InvertIfs:
    def __call__(self, program: str):
        return F.invert_ifs(program)


class MoveRandomStmt:
    def __init__(self, tries: int = 3):
        self.tries = tries

    def __call__(self, program: str, ):
        return F.move_random_stmt(program, self.tries)
