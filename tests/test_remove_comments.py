from codeaug.functional import remove_comments


def test_single_line_comment():
    program = b"""
def example_function():
    x = 10  # This is a comment
    return x
"""
    expected_output = b"""
def example_function():
    x = 10  
    return x
"""
    actual = remove_comments(program)
    assert actual == expected_output, actual


def test_multiline_comment():
    program = b'''
def example_function():
    """This is a 
    multiline comment"""
    return 42
'''
    expected_output = b"""
def example_function():
    
    return 42
"""
    actual = remove_comments(program)
    assert actual == expected_output, actual


def test_multiline_single_quotes_comment():
    program = b"""
def example_function():
    '''This is a
    multiline comment'''
    return 42
"""
    expected_output = b"""
def example_function():
    
    return 42
"""
    actual = remove_comments(program)
    assert actual == expected_output, actual


def test_inline_comment():
    program = b"""
def example_function():
    x = 10  # Assign 10 to x
    return x  # Return the value of x
"""
    expected_output = b"""
def example_function():
    x = 10  
    return x  
"""
    actual = remove_comments(program)
    assert actual == expected_output, actual


def test_no_comments():
    program = b"""
def example_function():
    x = 10
    return x
"""
    expected_output = b"""
def example_function():
    x = 10
    return x
"""
    actual = remove_comments(program)
    assert actual == expected_output, actual


def test_empty_program():
    program = b""
    expected_output = b""
    actual = remove_comments(program)
    assert actual == expected_output, actual


def test_comment_at_start():
    program = b"""
# This is a comment at the start
def example_function():
    x = 10
    return x
"""
    expected_output = b"""
def example_function():
    x = 10
    return x
"""
    actual = remove_comments(program)
    assert actual == expected_output, actual


def test_comment_in_string():
    program = b"""
def example_function():
    x = "This is a string with a # comment"
    return x
"""
    expected_output = b"""
def example_function():
    x = "This is a string with a # comment"
    return x
"""
    actual = remove_comments(program)
    assert actual == expected_output, actual
