# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
import os
import sys
import re

import numpy as np

from kwiklib.dataio import paramspy_to_json, load_params_json


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
    if not os.path.exists(prm_filename):
        raise IOError("The PRM file '{0:s}' does not exist.".format(prm_filename))
    
    # Parse the PRM file.
    params = load_prm(prm_filename)
    
    # Get the raw data files.
    files = params['raw_data_files']
    
    dat_to_kwd(filename_dat, filename_kwd, nchannels, nsamples=None,
        metadata=None):
    
    

if __name__ == '__main__':
    main()


