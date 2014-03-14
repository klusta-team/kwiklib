"""Chunking tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from kwiklib.dataio import chunk_bounds, Chunk, excerpts


# -----------------------------------------------------------------------------
# Chunking Tests
# -----------------------------------------------------------------------------
def test_chunk_bounds():
    chunks = chunk_bounds(200, 100, overlap=20)
    
    assert next(chunks) == (0, 100, 0, 90)
    assert next(chunks) == (80, 180, 90, 170)
    assert next(chunks) == (160, 200, 170, 200)
    
def test_chunk():
    data = np.random.randn(200, 4)
    chunks = chunk_bounds(data.shape[0], 100, overlap=20)
    
    # Chunk 1.
    ch = Chunk(data=data, bounds=next(chunks))
    
    assert ch.window_full == (0, 100)
    assert ch.window_keep == (0, 90)
    
    assert np.array_equal(ch.data_chunk_full, data[0:100])
    assert np.array_equal(ch.data_chunk_keep, data[0:90])
    
    # Chunk 2.
    ch = Chunk(data=data, bounds=next(chunks))
    
    assert ch.window_full == (80, 180)
    assert ch.window_keep == (90, 170)
    
    assert np.array_equal(ch.data_chunk_full, data[80:180])
    assert np.array_equal(ch.data_chunk_keep, data[90:170])
    
def test_excerpts_1():
    bounds = [(start, end) for (start, end) in excerpts(100, 
                                                        nexcerpts=3,
                                                        excerpt_size=10)]
    assert bounds == [(0, 10), (45, 55), (90, 100)]
    
def test_excerpts_2():
    bounds = [(start, end) for (start, end) in excerpts(10, 
                                                        nexcerpts=3,
                                                        excerpt_size=10)]
    assert bounds == [(0, 10),]
    
    