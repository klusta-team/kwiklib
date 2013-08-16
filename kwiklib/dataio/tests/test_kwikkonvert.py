"""Unit tests for kwikkonvert tool."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import pprint

import numpy as np
import numpy.random as rnd
import pandas as pd
import tables as tb

from kwiklib.dataio import save_binary, read_dat, close_kwd, paramspy_to_json
from kwiklib.dataio.kwikkonvert import kwikkonvert
from kwiklib.dataio.tests import (create_trace, duration, freq, nchannels,
    TEST_FOLDER, nsamples, fetdim)


# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------
nsamples_wave = nsamples
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
# KwikKonvert tests
# -----------------------------------------------------------------------------
def test_kwikkonvert_1():
    # Open the mock data.
    dir = TEST_FOLDER
    filename_dat = os.path.join(dir, 'test.dat')
    filename_kwd = os.path.join(dir, 'test.kwd')
    filename_prm = os.path.join(dir, 'params.prm')
    filename_prb = os.path.join(dir, 'myprobe.prb')
    
    # Write the PRM file.
    params_py = """
    NCHANNELS = {nchannels}
    SAMPLING_FREQUENCY = {freq}
    WAVEFORMS_NSAMPLES = {nsamples_wave}
    FETDIM = {fetdim}
    PRB_FILE = 'myprobe.prb'
    IGNORED_CHANNELS = []
    RAW_DATA_FILES = ['test.dat']
    """.replace('    ', '').format(
        nchannels=nchannels,
        freq=freq,
        nsamples_wave=nsamples_wave,
        fetdim=fetdim,
    )
    with open(filename_prm, 'w') as f:
        f.write(params_py)
    
    # Write the PRB file.
    probe = """
    {
        "shanks": 
            [
                {
                    "shank_index": 1,
                    "channels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                    "graph": []
                }
            ]
    }"""
    with open(filename_prb, 'w') as f:
        f.write(probe)
    
    # Convert the DAT file in KWD.
    kwikkonvert(filename_prm, overwrite=True)
    
    # Load DAT file (memmap).
    dat = read_dat(filename_dat, nchannels)
    assert dat.shape == (nsamples, nchannels)
    
    # Load KWD file.
    file_kwd = tb.openFile(filename_kwd)
    kwd = file_kwd.root.data[:]
    assert kwd.shape == (nsamples, nchannels)

    # Check they are identical.
    assert np.array_equal(dat, kwd)
    
    # Check PRM_JSON.
    params_json = paramspy_to_json(params_py)
    params_json2 = file_kwd.getNodeAttr('/metadata', 'PRM_JSON')
    assert params_json == params_json2
    
    # Check PRB_JSON.
    probe_json = probe
    probe_json2 = file_kwd.getNodeAttr('/metadata', 'PRB_JSON')
    assert probe_json == probe_json2
    
    # Close the KWD file.
    close_kwd(file_kwd)
    
    