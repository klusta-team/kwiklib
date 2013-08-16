# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
import os
import sys
import re

import numpy as np

from kwiklib.dataio import (paramspy_to_json, load_params_json, load_prm, 
    params_to_json, load_prb)
from kwiklib.dataio.kwikkonvert import kwikkonvert
import kwiklib.utils.logger as log


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
DESCRIPTION = """KwikKonvert

Convert files in the legacy file format (.fet, .spk, .clu, and so on) into the
Kwik format.

"""


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    # Parse the arguments.
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('prm', metavar='PRM_filename', type=str, nargs=1,
                       help=("The (absolute or relative) path to the "
                             "PRM file, containing all parameters"))
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite the KWD file if it already exists')
    parser.add_argument('--verbose', action='store_true',
                       help='Display information during the conversion')
    args = parser.parse_args()
    
    # Get the PRM filename.
    assert len(args.prm) == 1
    prm_filename = os.path.abspath(args.prm[0])
    
    kwikkonvert(prm_filename, overwrite=args.overwrite, verbose=True)

if __name__ == '__main__':
    main()


