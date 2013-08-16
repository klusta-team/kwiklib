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
        
    # Create 2 DAT files with nsamples and 2*nsamples samples, respectively.
    # The generated KWD file must have 3*nsamples samples.
    dat1 = create_trace(nsamples, nchannels)
    dat2 = create_trace(nsamples * 2, nchannels)
    
    # Create mock DAT file.
    save_binary(os.path.join(dir, 'test1.dat'), dat1)
    save_binary(os.path.join(dir, 'test2.dat'), dat2)
    
def write_params(filename_prm, files):
    # Write the PRM file.
    params_py = """
    NCHANNELS = {nchannels}
    SAMPLING_FREQUENCY = {freq}
    WAVEFORMS_NSAMPLES = {nsamples_wave}
    FETDIM = {fetdim}
    PRB_FILE = 'myprobe.prb'
    IGNORED_CHANNELS = []
    RAW_DATA_FILES = {files}
    """.replace('    ', '').format(
        nchannels=nchannels,
        freq=freq,
        nsamples_wave=nsamples_wave,
        fetdim=fetdim,
        files=str(files),
    )
    with open(filename_prm, 'w') as f:
        f.write(params_py)
    return params_py

def write_probe(filename_prb):
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
    return probe

    
# -----------------------------------------------------------------------------
# KwikKonvert tests
# -----------------------------------------------------------------------------
def test_kwikkonvert_1():
    # Open the mock data.
    dir = TEST_FOLDER
    filename_dat = os.path.join(dir, 'test1.dat')
    filename_kwd = os.path.join(dir, 'test1.raw.kwd')
    filename_prm = os.path.join(dir, 'params.prm')
    filename_prb = os.path.join(dir, 'myprobe.prb')
    
    # Write PRM file.
    params_py = write_params(filename_prm, ['test1.dat'])
    
    # Write PRB file.
    probe = write_probe(filename_prb)

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
    
def test_kwikkonvert_2():
    # Open the mock data.
    dir = TEST_FOLDER
    filename_dat1 = os.path.join(dir, 'test1.dat')
    filename_dat2 = os.path.join(dir, 'test2.dat')
    filename_kwd = os.path.join(dir, 'test1.raw.kwd')
    filename_prm = os.path.join(dir, 'params.prm')
    filename_prb = os.path.join(dir, 'myprobe.prb')
    
    # Write PRM file.
    params_py = write_params(filename_prm, ['test1.dat', 'test2.dat'])
    
    # Write PRB file.
    probe = write_probe(filename_prb)

    # Convert the DAT file in KWD.
    kwikkonvert(filename_prm, overwrite=True)
    
    # Load DAT file (memmap).
    dat1 = read_dat(filename_dat1, nchannels)
    dat2 = read_dat(filename_dat2, nchannels)
    dat = np.vstack((dat1, dat2))
    
    assert dat1.shape == (nsamples, nchannels)
    assert dat2.shape == (nsamples * 2, nchannels)
    
    
    # Load KWD file.
    file_kwd = tb.openFile(filename_kwd)
    kwd = file_kwd.root.data[:]
    assert kwd.shape == (nsamples * 3, nchannels)

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
    