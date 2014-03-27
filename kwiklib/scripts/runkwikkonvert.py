# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
import os
import sys
import re

from kwiklib.dataio import klusters_to_kwik


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
    parser.add_argument('xmlpath', type=str, #nargs=1,
                       help=("The path to the XML file"))
    args = parser.parse_args()
    filename = args.xmlpath
    
    assert filename, ArgumentError("Please provide the path to the XML file.")
    
    dir, filename = os.path.split(filename)
    base, ext = os.path.splitext(filename)
    
    assert ext == '.xml', ArgumentError("The file needs to be a valid XML file.")
    
    klusters_to_kwik(filename=filename, dir=dir, progress_report=None)
    
if __name__ == '__main__':
    main()

