from codeaug.functional import invert_ifs


def test_simple_if():
    program = b"""
if x > 0:
    print('Positive')
"""
    expected_output = b"""
if not (x > 0):
    print('Positive')
"""
    actual = invert_ifs(program)
    assert actual == expected_output, actual


def test_if_else():
    program = b"""
if x > 0:
    print('Positive')
else:
    print('Non-positive')
"""
    expected_output = b"""
if not (x > 0):
    print('Non-positive')
else:
    print('Positive')
"""
    actual = invert_ifs(program)
    assert actual == expected_output, actual


def test_if_elif_else():
    program = b"""
if x > 0:
    print('Positive')
elif x == 0:
    print('Zero')
else:
    print('Negative')
"""
    expected_output = b"""
if not (x > 0):
    print('Positive')
elif not (x == 0):
    print('Zero')
else:
    print('Negative')
"""
    actual = invert_ifs(program)
    assert actual == expected_output, actual


def test_no_if_statements():
    program = b"""
print('Hello World')
"""
    expected_output = b"""
print('Hello World')
"""
    actual = invert_ifs(program)
    assert actual == expected_output, actual


def test_if_in_string():
    program = b"""
def example_function():
    x = "if x > 0: do something"
    return x
"""
    expected_output = b"""
def example_function():
    x = "if x > 0: do something"
    return x
"""
    actual = invert_ifs(program)
    assert actual == expected_output, actual


def test_if_in_comment():
    program = b"""
def example_function():
    # if x > 0: this is a comment
    return 42
"""
    expected_output = b"""
def example_function():
    # if x > 0: this is a comment
    return 42
"""
    actual = invert_ifs(program)
    assert actual == expected_output, actual


def test_empty_program():
    program = b""
    expected_output = b""
    actual = invert_ifs(program)
    assert actual == expected_output, actual


def test_complex_program():
    program = b"""
if x > 0:
    if y > 0:
        print('Positive')
    else:
        print('Non-positive')
else:
    print('Negative')
"""
    expected_output = b"""
if not (x > 0):
    if not (y > 0):
        print('Positive')
    else:
        print('Non-positive')
else:
    print('Negative')
"""
    actual = invert_ifs(program)
    assert actual == expected_output, actual
