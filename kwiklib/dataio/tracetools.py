"""This module provides functions used to read and write KWD files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import struct

import tables
import numpy as np

from hdf5tools import write_metadata
from tools import MemMappedArray
from kwiklib.utils import logger as log


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

        f = open(filename_raw, 'rb')
        file_total_size = os.path.getsize(filename_raw)
        
        sample_width = 2  # int16 samples
    
        # Read File_Type_ID and check compatibility
        # If v2.2 is used, this value will be 'NEURALCD', which uses a slightly
        # more complex header. Currently unsupported.
        File_Type_ID = [chr(ord(c)) \
            for c in f.read(8)]
        if "".join(File_Type_ID) != 'NEURALSG':
            log.info( "Incompatible ns5 file format. Only v2.1 is supported.\nThis will probably not work.")

        # Skip the next field.
        f.read(16)

        # Read Period.
        period, = struct.unpack('<I', f.read(4))
        freq = period * 30000.0

        # Read Channel_Count and Channel_ID
        Channel_Count, = struct.unpack('<I', f.read(4))
        
        Channel_ID = [struct.unpack('<I', f.read(4))[0]
            for n in xrange(Channel_Count)]
            
        # Compute total header length
        Header = 8 + 16 + 4 + 4 + \
            4*Channel_Count # in bytes

        # determine length of file
        n_samples = (file_total_size - Header) // (Channel_Count * sample_width)
        # Length = np.float64(n_samples) / Channel_Count
        file_total_size2 = sample_width * Channel_Count * n_samples + Header
    
        # Sanity check.
        if file_total_size != file_total_size2:
            fields = ["{0:s}={1:s}".format(key, str(locals()[key])) 
                for key in ('period', 'freq', 'Channel_Count', 'Channel_ID',
                    'n_samples')]
            raise ValueError("The file seems corrupted: " + ", ".join(fields))
    
        return Header
    
def get_or_create_table(file_kwd, nchannels=None, nsamples=None):
    if not '/data' in file_kwd:
        # Create the EArray.
        data = file_kwd.createEArray('/', 'data', tables.Int16Atom(), 
            (0, nchannels), expectedrows=nsamples)
    data = file_kwd.getNode('/data')
    return data
    
def write_raw_data(file_kwd, filename_raw, nchannels=None, 
        datatype=None):
    if datatype is None:
        datatype = np.int16
        
    # Get the RAW data file extension.
    base, ext = os.path.splitext(filename_raw)
    # Remove the leading dot ('.').
    ext = ext[1:]
        
    # Get the header.
    header_size = get_header_size(filename_raw, ext)
    
    # We can infer the total number of samples from the file size and the
    # data type size.
    nsamples = (os.path.getsize(filename_raw) // 
        (nchannels * np.dtype(datatype).itemsize))
    
    # Get or create the EArray.
    data = get_or_create_table(file_kwd, nchannels=nchannels, nsamples=nsamples)
    
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
    
def raw_to_kwd(filename_raw, filename_kwd, nchannels, params_json='',
        probe_json=''):
    # if os.path.exists(filename_kwd):
        # raise IOError("The KWD file '{0:s}' already exists.".format(filename_kwd))
    file_kwd = create_kwd(filename_kwd)
    write_metadata(file_kwd, params_json, probe_json)
    write_raw_data(file_kwd, filename_raw, nchannels=nchannels)
    close_kwd(file_kwd)
    
    