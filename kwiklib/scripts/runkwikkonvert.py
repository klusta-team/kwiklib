# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
import os
import sys
import re

import numpy as np


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
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('prm', metavar='PRM filename', type=str, nargs=1,
                       help='the PRM file, containing all parameters')

    args = parser.parse_args()
    
    assert len(args.prm) == 1
    
    prm_filename = os.path.abspath(args.prm[0])
    if not os.path.exists(prm_filename):
        raise IOError("The PRM file '{0:s}' does not exist.".format(prm_filename))
    

if __name__ == '__main__':
    main()