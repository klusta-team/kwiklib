"""Utility I/O tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import tempfile

from kwiklib.dataio.utils import *


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# KWIK file creation tests
# -----------------------------------------------------------------------------
def test_json():
    dirpath = tempfile.mkdtemp()
    path = os.path.join(dirpath, 'test.json')
    
    d = {'a': [2, {'b': [{'c': 0}, {'c': [1, 2]}], 'd': 3}]}
    save_json(path, d)
    d_bis = load_json(path)
    assert d == d_bis
    
    os.remove(path)
    
def test_convert_dtype():
    x = np.random.rand(10)
    for dtype in (np.int16, np.float32):
        x_bis = convert_dtype(x, dtype)
        assert x_bis.dtype == dtype
        assert x.mean() != 0
    
    x = np.random.randint(size=10, low=1, high=10)
    for dtype in (np.int16, np.float32):
        x_bis = convert_dtype(x, dtype)
        assert x_bis.dtype == dtype
        assert x.mean() != 0
    
def test_ensure_vector():
    assert np.array_equal(ensure_vector(1), [1])
    assert np.array_equal(ensure_vector(1.), [1.])
    assert np.array_equal(ensure_vector([1]), [1])
    assert np.array_equal(ensure_vector([1,2]), [1, 2])
    