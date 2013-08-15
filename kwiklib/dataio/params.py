"""This module provides functions used to load params files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import tables
import time

import numpy as np
import matplotlib.pyplot as plt

from klustersloader import (find_filenames, find_index, read_xml,
    filename_to_triplet, triplet_to_filename, find_indices,
    find_hdf5_filenames,
    read_clusters, read_cluster_info, read_group_info,)
from tools import MemMappedText, MemMappedBinary


# -----------------------------------------------------------------------------
# Params file functions
# -----------------------------------------------------------------------------
def paramsxml_to_json(metadata_xml):
    """Convert PARAMS from XML to JSON."""
    shanks = metadata_xml['shanks']
    params = dict(
        SAMPLING_FREQUENCY=metadata_xml['freq'],
        FETDIM={shank: metadata_xml[shank]['fetdim'] 
            for shank in shanks},
        WAVEFORMS_NSAMPLES={shank: metadata_xml[shank]['nsamples'] 
            for shank in shanks},
    )
    return json.dumps(params, indent=4)

def paramspy_to_json(metadata_py):
    """metadata_py is a string containing Python code where each line is
    VARNAME = VALUE"""
    metadata = {}
    exec metadata_py in {}, metadata
    return json.dumps(metadata)


# -----------------------------------------------------------------------------
# Params parse functions
# -----------------------------------------------------------------------------
def get_freq(params):
    # Kwik format.
    if 'SAMPLING_FREQUENCY' in params:
        return float(params['SAMPLING_FREQUENCY'])
    # Or SpikeDetekt format.
    elif 'SAMPLERATE' in params:
        return float(params['SAMPLERATE'])

def get_nsamples(params, freq):
    # Kwik format.
    if 'WAVEFORMS_NSAMPLES' in params:
        # First case: it's a dict channel: nsamples.
        if isinstance(params['WAVEFORMS_NSAMPLES'], dict):
            return {int(key): value 
                for key, value in params['WAVEFORMS_NSAMPLES'].iteritems()}
        # Second case: it's a single value, the same nsamples for all channels.
        else:
            return int(params['WAVEFORMS_NSAMPLES'])
    # or SpikeDetekt format.
    elif 'T_BEFORE' in params and 'T_AFTER' in params:
        return int(freq * (float(params['T_BEFORE']) + float(params['T_AFTER'])))

def get_fetdim(params):
    # Kwik format.
    if 'FETDIM' in params:
        if isinstance(params['FETDIM'], dict):
            return {int(key): value 
                for key, value in params['FETDIM'].iteritems()}
        else:
            return int(params['FETDIM'])
    # or SpikeDetekt format.
    elif 'FPC' in params:
        return params['FPC']

def get_raw_data_files(params):
    files = params.get('RAW_DATA_FILES', [])
    if isinstance(files, basestring):
        return [files]
    else:
        return files

def get_probe_file(params):
    if 'PRB_FILE' in params:
        return params['PRB_FILE']
    elif 'PROBE_FILE' in params:
        return params['PROBE_FILE']

def get_dead_channels(params):
    return params.get('DEAD_CHANNELS', [])
    
def load_params_json(params_json):
    if not params_json:
        return None
    params_dict = json.loads(params_json)
    
    params = {}
    params['freq'] = get_freq(params_dict)
    params['nsamples'] = get_nsamples(params_dict, params['freq'])
    params['fetdim'] = get_fetdim(params_dict)
    params['raw_data_files'] = get_raw_data_files(params_dict)
    params['probe_file'] = get_probe_file(params_dict)
    params['dead_channels'] = get_dead_channels(params_dict)
    
    return params

def params_to_json(params):
    
    params_ns = {}
    params_ns['SAMPLING_FREQUENCY'] = params['freq']
    params_ns['WAVEFORMS_NSAMPLES'] = params['nsamples']
    params_ns['FETDIM'] = params['fetdim']
    params_ns['RAW_DATA_FILES'] = params['raw_data_files']
    params_ns['PRB_FILE'] = params['probe_file']
    params_ns['DEAD_CHANNELS'] = params['dead_channels']
    
    return json.dumps(params_ns)

def load_prm(prm_filename):
    with open(prm_filename, 'r') as f:
        params_text = f.read()
    # First possibility: the PRM is in Python.
    try:
        params_json = paramspy_to_json(params_text)
    # Otherwise, it is already in JSON.
    except:
        params_json = params_text
    # Parse the JSON parameters file.
    return load_params_json(params_json)
    

