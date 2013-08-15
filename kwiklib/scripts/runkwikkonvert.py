# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
import os
import sys
import re

import numpy as np

from kwiklib.dataio import (paramspy_to_json, load_params_json, load_prm, 
    params_to_json, load_prb, dat_to_kwd)


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
DESCRIPTION = """KwikKonvert

Convert files in the legacy file format (.fet, .spk, .clu, and so on) into the
Kwik format.

"""


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

def convert_raw_file(filename_raw, nchannels, params_json='', probe_json=''):
    base, ext = os.path.splitext(filename_raw)
    if ext == '.dat':
        filename_kwd = base + '.kwd'
        dat_to_kwd(filename_raw, filename_kwd, nchannels, 
            params_json=params_json, probe_json=probe_json)
    
        
    
# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    # Parse the arguments.
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('prm', metavar='PRM filename', type=str, nargs=1,
                       help='the PRM file, containing all parameters')
    args = parser.parse_args()
    
    # Get the PRM filename.
    assert len(args.prm) == 1
    prm_filename = os.path.abspath(args.prm[0])
    dir = os.path.dirname(prm_filename)
    if not os.path.exists(prm_filename):
        raise IOError("The PRM file '{0:s}' does not exist.".format(prm_filename))
    
    # Parse the PRM file.
    params = load_prm(prm_filename)
    params_json = params_to_json(params)
    nchannels = params['nchannels']
    print nchannels
    
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
            probe_json=probe_json)

if __name__ == '__main__':
    main()


