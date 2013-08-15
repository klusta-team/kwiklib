"""Unit tests for params module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import pprint

import numpy as np
import numpy.random as rnd
import pandas as pd

from kwiklib.dataio import (paramsxml_to_json, paramspy_to_json, 
    load_params_json, params_to_json)


# -----------------------------------------------------------------------------
# Probe tests
# -----------------------------------------------------------------------------
def test_params_py():
    params_py = """
    VAR1 = 0
    VAR2 = 1.23
    VAR3 = 'this is a string'
    VAR4 = {'key1': 'value1', 'key2': 4.56}
    VAR5 = [1, 2, 3]
    """.replace('    ', '')
    params_json = paramspy_to_json(params_py)
    params = json.loads(params_json)
    
    assert params['VAR1'] == 0
    assert params['VAR2'] == 1.23
    assert params['VAR3'] == 'this is a string'
    assert params['VAR4'] == {'key1': 'value1', 'key2': 4.56}
    assert params['VAR5'] == [1, 2, 3]
    
def assert_params(params):
    assert params['freq'] == 20000.
    assert params['nsamples'] == 20
    assert params['fetdim'] == 3
    assert params['dead_channels'] == []
    assert params['probe_file'] == 'myprobe.prb'
    assert params['raw_data_files'] == ['myfile1.ns5', 'myfile2.ns5']

def test_params_json():
    params_py = """
    SAMPLING_FREQUENCY = 20000.
    WAVEFORMS_NSAMPLES = 20
    FETDIM = 3
    PRB_FILE = 'myprobe.prb'
    DEAD_CHANNELS = []
    RAW_DATA_FILES = ['myfile1.ns5', 'myfile2.ns5']
    """.replace('    ', '')
    params_json = paramspy_to_json(params_py)
    params = load_params_json(params_json)
    params_json2 = params_to_json(params)
    assert params_json == params_json2
    
def test_params_json_kwik():
    params_py = """
    SAMPLING_FREQUENCY = 20000.
    WAVEFORMS_NSAMPLES = 20
    FETDIM = 3
    PRB_FILE = 'myprobe.prb'
    RAW_DATA_FILES = ['myfile1.ns5', 'myfile2.ns5']
    """.replace('    ', '')
    params_json = paramspy_to_json(params_py)
    params = load_params_json(params_json)
    assert_params(params)
    
def test_params_json_spikedetekt():
    params_py = """
    SAMPLERATE = 20000.
    T_BEFORE = .0005 # time before peak in extracted spike
    T_AFTER = .0005 # time after peak in extracted spike
    FPC = 3
    PROBE_FILE = 'myprobe.prb'
    RAW_DATA_FILES = ['myfile1.ns5', 'myfile2.ns5']
    """.replace('    ', '')
    params_json = paramspy_to_json(params_py)
    params = load_params_json(params_json)
    assert_params(params)
    
