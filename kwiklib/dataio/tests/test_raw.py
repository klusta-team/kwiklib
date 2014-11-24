"""Raw data reader tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import tempfile
import shutil

import numpy as np
from nose import with_setup
import tables as tb

from kwiklib.dataio import (NumPyRawDataReader, DatRawDataReader,
    create_kwd, convert_dat_to_kwd, add_recording_in_kwd, read_raw)
from kwiklib.utils import create_trace


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
DIRPATH = tempfile.mkdtemp()
FILENAME = 'mydatfile.dat'
FILENAME2 = 'mydatfile2.dat'
FILENAME_KWD = "test.kwd"
PATH = os.path.join(DIRPATH, FILENAME)
PATH2 = os.path.join(DIRPATH, FILENAME2)
NSAMPLES = 2000
NCHANNELS = 4

def dat_setup_1():
    trace = create_trace(NSAMPLES, NCHANNELS)
    trace.tofile(PATH)
    
def dat_teardown_1():
    os.remove(PATH)
    
def dat_setup_2():
    trace = create_trace(NSAMPLES, NCHANNELS)
    trace.tofile(PATH)
    trace2 = create_trace(2 * NSAMPLES, NCHANNELS)
    trace2.tofile(PATH2)
    
def dat_teardown_2():
    os.remove(PATH)
    os.remove(PATH2)
    
def kwd_setup():
    # Create an empty KWD file.
    create_kwd(os.path.join(DIRPATH, FILENAME_KWD))
    data = create_trace(NSAMPLES, NCHANNELS)
    add_recording_in_kwd(os.path.join(DIRPATH, FILENAME_KWD), data=data)
    
def kwd_teardown():
    os.remove(os.path.join(DIRPATH, FILENAME_KWD))
    
# -----------------------------------------------------------------------------
# Chunking Tests
# -----------------------------------------------------------------------------
def test_raw_data_1():
    data = np.random.randn(200, 4)
    rd = NumPyRawDataReader(data)
    chunks = rd.chunks(chunk_size=100, chunk_overlap=20)
    
    ch = next(chunks)
    assert ch.window_full == (0, 100)
    assert ch.window_keep == (0, 90)
    assert np.array_equal(ch.data_chunk_full, data[0:100])
    assert np.array_equal(ch.data_chunk_keep, data[0:90])
    
    ch = next(chunks)
    assert ch.window_full == (80, 180)
    assert ch.window_keep == (90, 170)
    assert np.array_equal(ch.data_chunk_full, data[80:180])
    assert np.array_equal(ch.data_chunk_keep, data[90:170])
    
    assert str(ch)
    
    
# -----------------------------------------------------------------------------
# DAT Reader Tests
# -----------------------------------------------------------------------------
@with_setup(dat_setup_1, dat_teardown_1)
def test_raw_dat_1():
    """Test 1 file with 20k samples."""
    with DatRawDataReader(PATH, dtype=np.int16, shape=(-1, NCHANNELS)) as reader:
        assert len([chunk for chunk in reader.chunks(NSAMPLES, 0)]) == 1
    
@with_setup(dat_setup_2, dat_teardown_2)
def test_raw_dat_2():
    """Test the concatenation of 1 file with 20k samples and 1 other with 40k."""
    with DatRawDataReader([PATH, PATH2], dtype=np.int16, shape=(-1, NCHANNELS)) as reader:
        for chunk, rec in zip(reader.chunks(NSAMPLES, 0), [0, 1, 1]):
            assert chunk.recording == rec
    
    
# -----------------------------------------------------------------------------
# KWD Reader Tests
# -----------------------------------------------------------------------------
@with_setup(kwd_setup, kwd_teardown)
def test_raw_kwd_1():
    with read_raw(os.path.join(DIRPATH, FILENAME_KWD)) as reader:
        data = np.vstack([chunk.data_chunk_full 
                          for chunk in reader.chunks(20000)])
        assert data.shape == (NSAMPLES, NCHANNELS)
        assert data.dtype == np.int16
    
def test_raw_data_iterator():
    data = np.random.randn(200, 4)
    rd = NumPyRawDataReader(data)
    assert len([ch for ch in rd.chunks(chunk_size=100, chunk_overlap=20)]) == 3
    
    
# -----------------------------------------------------------------------------
# DAT ==> KWD conversion Tests
# -----------------------------------------------------------------------------
@with_setup(dat_setup_2, dat_teardown_2)
def test_convert():
    """Test conversion from dat to kwd."""
    # Create the DAT reader with two DAT files.
    dat_reader = DatRawDataReader([PATH, PATH2], 
        dtype=np.int16, dtype_to=np.int16, shape=(-1, NCHANNELS))
    
    # Create an empty KWD file.
    kwd_file = "test.kwd"
    create_kwd(kwd_file)
    
    # Convert from DAT to the newly created KWD file.
    convert_dat_to_kwd(dat_reader, kwd_file)
    
    # Now, check that the conversion worked. Read the data in DAT and KWD
    # and check the arrays are the same.
    f = tb.openFile(kwd_file, 'r')
    # Load full KWD data.
    data_kwd = np.vstack([f.root.recordings._f_getChild('0').data,
                          f.root.recordings._f_getChild('1').data])
    # Load full DAT data.
    data_dat = np.vstack([chunk.data_chunk_full 
                          for chunk in dat_reader.chunks(20000)])
    # Assert the two are equal.
    assert np.array_equal(data_kwd, data_dat)
    f.close()
    
    os.remove(kwd_file)