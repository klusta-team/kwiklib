"""Handle user-specified and default parameters."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from kwiklib import (python_to_pydict, pydict_to_python, 
    load_default_params, get_params)

def test_python_to_params():
    python = """
    MYVAR1 = 'myvalue1'
    MYVAR2 = .123
    MYVAR3 = ['myvalue3', .456]
    """.replace('    ', '').strip()
    
    params = python_to_pydict(python)
    assert params['myvar1'] == 'myvalue1'
    assert params['myvar2'] == .123
    assert params['myvar3'] == ['myvalue3', .456]
    
def test_get_params():
    assert get_params(sample_rate=1).get('filter_butter_order', None) == 3
    assert get_params(sample_rate=1, filter_butter_order=1)['filter_butter_order'] == 1
    
def test_params_to_python():
    params = dict(
        MYVAR1 = 'myvalue1',
        MYVAR2 = .123,
        MYVAR3 = ['myvalue3', .456],
        SAMPLE_RATE = 1)
    
    python = pydict_to_python(params)
    assert python == """
    MYVAR1 = 'myvalue1'
    MYVAR2 = 0.123
    MYVAR3 = ['myvalue3', 0.456]
    SAMPLE_RATE = 1
    """.replace('    ', '').strip()
    
def test_default_params():
    params_default = load_default_params(dict(sample_rate=1))
    assert params_default['filter_butter_order'] == 3
    

