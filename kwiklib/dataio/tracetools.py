"""This module provides functions used to read and write KWD files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import tables
import numpy as np

from hdf5tools import write_metadata
from tools import MemMappedArray


# -----------------------------------------------------------------------------
# Conversion functions
# -----------------------------------------------------------------------------
def create_kwd(filename_kwd):
    file_kwd = tables.openFile(filename_kwd, mode='w')
    file_kwd.createGroup('/', 'metadata')
    file_kwd.setNodeAttr('/', 'VERSION', 1)
    return file_kwd

def get_header_size(filename_raw, ext):
    if ext == 'dat':
        return 0
    elif ext == 'ns5':
        # TODO
        return 0
    
def write_raw_data(file_kwd, filename_raw, ext=None, nchannels=None, 
        datatype=None):
    if datatype is None:
        datatype = np.int16
    # Wa can infer the total number of samples from the file size and the
    # data type size.
    nsamples = (os.path.getsize(filename_raw) // 
        (nchannels * np.dtype(datatype).itemsize))
    
    # Get the header.
    header_size = get_header_size(filename_raw, ext)
    
    # Create the EArray.
    data = file_kwd.createEArray('/', 'data', tables.Int16Atom(), 
        (0, nchannels), expectedrows=nsamples)
    
    # Open the raw data file.
    raw = MemMappedArray(filename_raw, np.int16, header_size=header_size)
    chunk_nrows = 1000
    chunk_pos = 0
    while True:
        # Read the chunk from the RAW file.
        i0, i1 = (chunk_pos * nchannels), (chunk_pos + chunk_nrows) * nchannels
        chunk = raw[i0:i1]
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
    
def raw_to_kwd(filename_raw, filename_kwd, nchannels, ext=None, params_json='',
        probe_json=''):
    # if os.path.exists(filename_kwd):
        # raise IOError("The KWD file '{0:s}' already exists.".format(filename_kwd))
    file_kwd = create_kwd(filename_kwd)
    write_metadata(file_kwd, params_json, probe_json)
    write_raw_data(file_kwd, filename_raw, nchannels=nchannels, ext=ext)
    close_kwd(file_kwd)
    
    