# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
import os
import sys
import re

import numpy as np

from kwiklib.dataio import (paramspy_to_json, load_params_json, load_prm, 
    params_to_json, load_prb, raw_to_kwd)


# -----------------------------------------------------------------------------
# Conversion functions
# -----------------------------------------------------------------------------
def get_abs_path(file, dir):
    """Ensure a file path is absolute. If it's relative, it's relative to
    the folder where the PRM is stored."""
    if os.path.isabs(file):
        return os.path.abspath(file)
    else:
        return os.path.abspath(os.path.join(dir, file))

def convert_raw_file(filename_raw, nchannels, params_json='', probe_json='',
        overwrite=False):
    base, ext = os.path.splitext(filename_raw)
    # if ext == '.dat':
    # Remove the leading dot ('.').
    ext = ext[1:]
    filename_kwd = base + '.kwd'
    # Raise an error if the KWD file already exists, unless overwrite is 
    # True.
    if not overwrite and os.path.exists(filename_kwd):
        raise IOError("The KWD file '{0:s}' already exists.".format(filename_kwd))
    raw_to_kwd(filename_raw, filename_kwd, nchannels, ext=ext,
        params_json=params_json, probe_json=probe_json)
    
    
# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def kwikkonvert(prm_filename, overwrite=False):

    dir = os.path.dirname(prm_filename)
    if not os.path.exists(prm_filename):
        raise IOError("The PRM file '{0:s}' does not exist.".format(prm_filename))
    
    # Parse the PRM file.
    params = load_prm(prm_filename)
    params_json = params_to_json(params)
    nchannels = params['nchannels']
    
    # Get the probe file.
    prb_filename = get_abs_path(params['probe_file'], dir)
    if not params['probe_file']:
        raise IOError("You need to specify in the PRM file the path to the PRB file.")
    elif not os.path.exists(prb_filename):
        raise IOError("The PRB file '{0:s}' does not exist.".format(prb_filename))
    with open(prb_filename, 'r') as f:
        probe_json = f.read()
    
    # Get the raw data files.
    files = params['raw_data_files']
    files = [get_abs_path(file, dir) for file in files]
    
    for file in files:
        convert_raw_file(file, nchannels, params_json=params_json, 
            probe_json=probe_json, overwrite=overwrite)

    
