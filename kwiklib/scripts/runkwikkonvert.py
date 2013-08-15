# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
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
# Main function
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('prm', metavar='PRM filename', type=str, nargs=1,
                       help='the PRM file, containing all parameters')

    args = parser.parse_args()
    print(args.prm)

if __name__ == '__main__':
    main()