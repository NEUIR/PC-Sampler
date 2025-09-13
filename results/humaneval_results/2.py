from typing import List


def separate_paren_groups_prompt(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups_prompt('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    groups = []
    current_group = []
    stack = []
    
    for char in paren_string:
        if char == '(':
            current_group.append(char)
            stack.append(char)
        elif char == ')':
            if stack and stack[-1] == '(':
                groups.append(''.join(current_group))
                current_group = []
            else:
                groups.append(''.join(current_group))
                current_group = []
        else:
            current_group.append(char)
    
    if current_group:
        groups.append(''.join(current_group))
    
    return groups


METADATA = {
    'author': 'jt',
    'dataset': 'test'
}


def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == [
        '(()())', '((()))', '()', '((())()())'
    ]
    assert candidate('() (()) ((())) (((())))') == [
        '()', '(())', '((()))', '(((())))'
    ]
    assert candidate('(()(())((())))') == [
        '(()(())((())))'
    ]
    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']

if __name__ == "__main__":
    check(separate_paren_groups)