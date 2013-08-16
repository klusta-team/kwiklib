"""Unit tests for tracetools module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd
import tables as tb

from kwiklib.dataio import (save_binary, close_kwd, raw_to_kwd, read_dat, )
from kwiklib.dataio.tests import (create_trace, duration, freq, nchannels,
    TEST_FOLDER)

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------
nsamples = int(duration * freq)

def setup():
    # Create mock directory if needed.
    dir = TEST_FOLDER
    if not os.path.exists(dir):
        os.mkdir(dir)
        
    dat = create_trace(nsamples, nchannels)
    
    # Create mock DAT file.
    save_binary(os.path.join(dir, 'test.dat'), dat)
    

# -----------------------------------------------------------------------------
# KWD tests
# -----------------------------------------------------------------------------
def test_kwd_1():
    # Open the mock data.
    dir = TEST_FOLDER
    filename_dat = os.path.join(dir, 'test.dat')
    filename_kwd = os.path.join(dir, 'test.kwd')
    
    # Convert the DAT file in KWD.
    raw_to_kwd(filename_dat, filename_kwd, nchannels)
    
    # Load DAT file (memmap).
    dat = read_dat(filename_dat, nchannels)
    assert dat.shape == (nsamples, nchannels)
    
    # Load KWD file.
    file_kwd = tb.openFile(filename_kwd)
    kwd = file_kwd.root.data[:]
    assert kwd.shape == (nsamples, nchannels)

    # Check they are identical.
    assert np.array_equal(dat, kwd)
    
    # Close the KWD file.
    close_kwd(file_kwd)
    
    
