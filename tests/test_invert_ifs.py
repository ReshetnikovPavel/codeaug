from codeaug.functional import invert_ifs


def assert_is_equal(actual, expected):
    if actual != expected:
        print("EXPECTED:::", expected)
        print("ACTUAL:::", actual)
        raise AssertionError("Code is not the same")


def test_if_else():
    program = """
if x > 0:
    print('Positive')
else:
    print('Non-positive')
"""
    expected_output = """
if not (x > 0):
    print('Non-positive')
else:
    print('Positive')
"""
    actual = invert_ifs(program)
    assert_is_equal(actual, expected_output)


def test_no_if_statements():
    program = """
print('Hello World')
"""
    expected_output = """
print('Hello World')
"""
    actual = invert_ifs(program)
    assert_is_equal(actual, expected_output)


def test_if_in_string():
    program = """
def example_function():
    x = "if x > 0: do something"
    return x
"""
    expected_output = """
def example_function():
    x = "if x > 0: do something"
    return x
"""
    actual = invert_ifs(program)
    assert_is_equal(actual, expected_output)


def test_if_in_comment():
    program = """
def example_function():
    # if x > 0: this is a comment
    return 42
"""
    expected_output = """
def example_function():
    # if x > 0: this is a comment
    return 42
"""
    actual = invert_ifs(program)
    assert_is_equal(actual, expected_output)


def test_empty_program():
    program = ""
    expected_output = ""
    actual = invert_ifs(program)
    assert_is_equal(actual, expected_output)


def test_complex_program():
    program = """
if x > 0:
    if y > 0:
        print('Positive')
    else:
        print('Non-positive')
else:
    print('Negative')
"""
    expected_output = """
if not (x > 0):
    print('Negative')
else:
    if not (y > 0):
        print('Non-positive')
    else:
        print('Positive')
"""
    actual = invert_ifs(program)
    assert_is_equal(actual, expected_output)
