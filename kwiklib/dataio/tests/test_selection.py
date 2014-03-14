"""Unit tests for selection module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter
import tempfile
import os

import numpy as np
import pandas as pd
import tables as tb

from kwiklib.dataio.selection import select, slice_to_indices


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def generate_clusters(indices, nspikes=100):
    """Generate all spikes in cluster 0, except some in cluster 1."""
    # 2 different clusters, with 3 spikes in cluster 1
    clusters = np.zeros(nspikes, dtype=np.int32)
    clusters[indices] = 1
    return clusters
    
def generate_earray(a):
    dirpath = tempfile.mkdtemp()
    path = os.path.join(dirpath, 'test.h5')
    f = tb.openFile(path, 'w')
    arr = f.createEArray('/', 'arr', tb.Float32Atom(), (0, a.shape[1]))
    arr.append(a)
    f.flush()
    f.close()
    return path
    
def generate_data2D(nspikes=100, ncols=5):
    data = np.random.randn(nspikes, ncols)
    return data

def assert_indexing(indices, n=10):
    values = range(n)
    stop = indices.stop or n
    assert np.array_equal(slice_to_indices(indices, stop=stop), 
                          values[indices])
    

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_slice_to_indices():
    assert_indexing(slice(0, 10, 1))
    
    assert_indexing(slice(None, None, None))
    
    assert_indexing(slice(0, None, None))
    assert_indexing(slice(None, 10, None))
    assert_indexing(slice(None, None, 1))
    
    assert_indexing(slice(None, 10, 1))
    assert_indexing(slice(0, None, 1))
    assert_indexing(slice(0, 10, None))
    
    assert_indexing(slice(1, None, None))
    assert_indexing(slice(1, None, 2))
    
    assert_indexing(slice(1, 9, None))
    assert_indexing(slice(1, 9, 2))
    
    assert_indexing(slice(1, 8, None))
    assert_indexing(slice(1, 8, 2))
    
    assert_indexing(slice(1, 8, None))
    assert_indexing(slice(1, 8, 2))
    assert_indexing(slice(1, 8, 3))
    assert_indexing(slice(1, 8, 4))
    assert_indexing(slice(1, 8, 5))
    assert_indexing(slice(1, 8, 6))
    
def test_select_numpy():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    assert np.array_equal(select(clusters, [9, 11]), [0, 0])
    assert np.array_equal(select(clusters, [10, 99]), [1, 0])
    assert np.array_equal(select(clusters, [20, 25, 25]), [1, 1, 1])

def test_select_pandas():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    
    # test selection of Series (1D)
    clusters = pd.Series(clusters)
    assert np.array_equal(select(clusters, [9, 11]), [0, 0])
    assert np.array_equal(select(clusters, [10, 99]), [1, 0])
    assert np.array_equal(select(clusters, [20, 25, 25]), [1, 1, 1])
    
    # test selection of Series (3D)
    clusters = pd.DataFrame(clusters)
    assert np.array_equal(np.array(select(clusters, [9, 11])).ravel(), [0, 0])
    assert np.array_equal(np.array(select(clusters, [10, 99])).ravel(), [1, 0])
    assert np.array_equal(np.array(select(clusters, [20, 25, 25])).ravel(), [1, 1, 1])
    
    # test selection of Panel (4D)
    clusters = pd.Panel(np.expand_dims(clusters, 3))
    assert np.array_equal(np.array(select(clusters, [9, 11])).ravel(), [0, 0])
    assert np.array_equal(np.array(select(clusters, [10, 99])).ravel(), [1, 0])
    
def test_select_pytables():
    a = np.random.randn(100, 10)
    
    path = generate_earray(a)
    f = tb.openFile(path, 'r')
    arr = f.root.arr
    
    s = select(arr, slice(0, 10, 2))
    assert isinstance(s, pd.DataFrame)
    assert s.shape == (5, 10)
    
    assert np.array_equal(select(arr, 10), arr[[10], :])
    assert np.array_equal(select(arr, [10]), arr[[10], :])
    assert np.array_equal(select(arr, [10, 20]), arr[[10, 20], :])
    assert np.array_equal(select(arr, slice(10, 20)), arr[10:20, :])
    
    f.close()
    os.remove(path)
    
def test_select_single():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    assert select(clusters, 10) == 1
