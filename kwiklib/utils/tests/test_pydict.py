"""Handle user-specified and default dictionaries."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from kwiklib import (python_to_pydict, pydict_to_python, get_pydict)

def test_python_to_pydict():
    python = """
    MYVAR1 = 'myvalue1'
    MYVAR2 = .123
    MYVAR3 = ['myvalue3', .456]
    """.replace('    ', '').strip()
    
    pydict = python_to_pydict(python)
    assert pydict['myvar1'] == 'myvalue1'
    assert pydict['myvar2'] == .123
    assert pydict['myvar3'] == ['myvalue3', .456]
    
def test_get_pydict():
    assert get_pydict(pydict_default={'myvar1': 'value1'}
            ).get('myvar1', None) == 'value1'
    
def test_pydict_to_python():
    pydict = dict(
        MYVAR1 = 'myvalue1',
        MYVAR2 = .123,
        MYVAR3 = ['myvalue3', .456])
    
    python = pydict_to_python(pydict)
    assert python == """
    MYVAR1 = 'myvalue1'
    MYVAR2 = 0.123
    MYVAR3 = ['myvalue3', 0.456]
    """.replace('    ', '').strip()
    