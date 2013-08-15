"""This module provides functions used to read and write KWD files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import tables
import numpy as np

from tools import MemMappedArray


# -----------------------------------------------------------------------------
# Conversion functions
# -----------------------------------------------------------------------------
def create_kwd(filename_kwd):
    file_kwd = tables.openFile(filename_kwd, mode='w')
    file_kwd.createGroup('/', 'metadata')
    file_kwd.setNodeAttr('/', 'VERSION', 1)
    return file_kwd

def write_metadata(file_kwd, metadata):
    # TODO
    pass
    
def write_raw_data(file_kwd, filename_dat, nchannels, 
        nsamples=None):
    # Create the EArray.
    data = file_kwd.createEArray('/', 'data', tables.Int16Atom(), 
        (0, nchannels), expectedrows=nsamples)
    
    # Open the DAT file.
    dat = MemMappedArray(filename_dat, np.int16)
    chunk_nrows = 1000
    chunk_pos = 0
    while True:
        # Read the chunk from the DAT file.
        i0, i1 = (chunk_pos * nchannels), (chunk_pos + chunk_nrows) * nchannels
        chunk = dat[i0:i1]
        chunk = chunk.reshape((-1, nchannels))
        if chunk.size == 0:
            break
        # Write the chunk in the EArray.
        data.append(chunk)
        chunk_pos += chunk.shape[0]
    return data

def close_kwd(file_kwd):
    file_kwd.flush()
    file_kwd.close()
    
def dat_to_kwd(filename_dat, filename_kwd, nchannels, nsamples=None,
        metadata=None):
    if os.path.exists(filename_kwd):
        return
    file_kwd = create_kwd(filename_kwd)
    # TODO
    write_metadata(file_kwd, metadata)
    write_raw_data(file_kwd, filename_dat, nchannels, nsamples=None)
    
    close_kwd(file_kwd)
    
    