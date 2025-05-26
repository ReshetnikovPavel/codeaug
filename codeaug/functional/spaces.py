import tokenize
from io import BytesIO
from typing import Any, Callable


def remove_meaningless_spaces(input_str: str, should_apply: Callable[[Any], bool] = lambda _: True) -> str:
    try:
        input_bytes = input_str.encode("utf-8")
    except UnicodeEncodeError:
        input_bytes = input_str.encode("utf-8", errors="replace")

    tokens = []
    try:
        for tok in tokenize.tokenize(BytesIO(input_bytes).readline):
            if tok.type == tokenize.ENCODING:
                continue
            tokens.append((tok.type, tok.string))
    except tokenize.TokenError:
        return input_str

    output = []
    previous_type = None
    for tok_type, tok_string in tokens:
        if previous_type is not None:
            if previous_type == tokenize.NAME and tok_type == tokenize.NAME:
                output.append(" ")
        output.append(tok_string)
        previous_type = tok_type

    return "".join(output)

p = """
import tokenize
from io import BytesIO
from typing import Any, Callable


def remove_meaningless_spaces(input_str: str, should_apply: Callable[[Any], bool]) -> str:
    try:
        input_bytes = input_str.encode("utf-8")
    except UnicodeEncodeError:
        input_bytes = input_str.encode("utf-8", errors="replace")

    tokens = []
    try:
        for tok in tokenize.tokenize(BytesIO(input_bytes).readline):
            if tok.type == tokenize.ENCODING:
                continue
            tokens.append((tok.type, tok.string))
    except tokenize.TokenError:
        return input_str

    output = []
    previous_type = None
    for tok_type, tok_string in tokens:
        if previous_type is not None:
            if previous_type == tokenize.NAME and tok_type == tokenize.NAME:
                output.append(" ")
        output.append(tok_string)
        previous_type = tok_type

    return "".join(output)
"""
print(remove_meaningless_spaces(text))
