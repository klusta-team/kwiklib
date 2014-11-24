"""Utility I/O functions."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json

import numpy as np


# -----------------------------------------------------------------------------
# Data type conversions
# -----------------------------------------------------------------------------
_maxint16 = 2.**12
_maxint16inv = 1./_maxint16
_maxint8 = 127
_maxint8inv = 1./255
_dtype_factors = {
    (np.int16, np.float32): _maxint16inv,
    (np.int16, np.float64): _maxint16inv,
    (np.int32, np.float32): _maxint16inv,
    (np.int32, np.float64): _maxint16inv,
    (np.int64, np.float32): _maxint16inv,
    (np.int64, np.float64): _maxint16inv,
    (np.float32, np.int16): _maxint16,
    (np.float64, np.int16): _maxint16,
    (np.int8, np.float32): _maxint8inv,
    (np.float32, np.int8): _maxint8,
}

def _get_dtype(dtype):
    if isinstance(dtype, np.dtype):
        return dtype.type
    else:
        return dtype

def convert_dtype(data, dtype=None, factor=None):
    if not dtype:
        return data
    if data.shape[0] == 0:
        return data.astype(dtype)
    dtype_old = data.dtype
    if dtype_old == dtype:
        return data
    key = (_get_dtype(dtype_old), _get_dtype(dtype))
    factor = factor or _dtype_factors.get(key, 1)
    if dtype_old in (np.float32, np.float64):
        factor = factor/np.abs(data).max()
    # We avoid unnecessary array copy when factor == 1
    if factor != 1:
        return (data * factor).astype(dtype)
    else:
        return data.astype(dtype) 

def ensure_vector(data, size=None):
    """Ensure data is a 1D vector."""
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        data = np.array([data])
    assert data.ndim == 1
    # If size is specified, tile the vector by this size.
    if len(data) == 1 and size is not None:
        data = np.tile(data, size)
    return data
    

# -----------------------------------------------------------------------------
# JSON functions
# -----------------------------------------------------------------------------
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def save_json(path, d):
    with open(path, 'w') as f:
        json.dump(d, f, indent=4)

