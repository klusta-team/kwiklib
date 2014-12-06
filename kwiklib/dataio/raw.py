"""Reading raw data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import tables as tb
from nose import with_setup

from kwik import open_file, add_recording_in_kwd
from experiment import Experiment
from chunks import chunk_bounds, Chunk, Excerpt, excerpts
from kwiklib.utils.six import Iterator, string_types


# -----------------------------------------------------------------------------
# Raw data readers
# -----------------------------------------------------------------------------
class BaseRawDataReader(object):
    def __init__(self, dtype_to=None):
        self.dtype_to = dtype_to
        self.nrecordings = 1
    
    def next_recording(self):
        for self.recording in range(self.nrecordings):
            yield self.recording, self.get_recording_data(self.recording)
    
    def get_recording_data(self, recording):
        # TO BE OVERRIDEN
        # return data
        raise NotImplementedError()
    
    def chunks(self, chunk_size=None, chunk_overlap=0):
        for recording, data in self.next_recording():
            assert chunk_size is not None, "You need to specify a chunk size."""
            for bounds in chunk_bounds(data.shape[0], 
                                       chunk_size=chunk_size, 
                                       overlap=chunk_overlap):
                yield Chunk(data, bounds=bounds, dtype=self.dtype_to, 
                            recording=recording, nrecordings=self.nrecordings)
        
    def excerpts(self, nexcerpts=None, excerpt_size=None):
        for recording, data in self.next_recording():
            for bounds in excerpts(data.shape[0],
                                   nexcerpts=nexcerpts, 
                                   excerpt_size=excerpt_size):
                yield Excerpt(data, bounds=bounds, dtype=self.dtype_to, 
                              recording=recording, nrecordings=self.nrecordings)
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
        
class NumPyRawDataReader(BaseRawDataReader):
    def __init__(self, data, dtype_to=None):
        self._data = data
        self.nsamples, self.nchannels = data.shape
        super(NumPyRawDataReader, self).__init__(dtype_to=dtype_to)

    def get_recording_data(self, recording):
        return self._data
        
    def __repr__(self):
        return "<NumPyRawReader {0:d}x{1:d} {2:s} array>".format(
            self.nsamples, self.nchannels, str(self._data.dtype)
        )
        
class ExperimentRawDataReader(BaseRawDataReader):
    def __init__(self, experiment, dtype_to=None):
        self.experiment = experiment
        super(ExperimentRawDataReader, self).__init__(dtype_to=dtype_to)
        self.nrecordings = len(experiment.recordings)

    def get_recording_data(self, recording):
        data = self.experiment.recordings[recording].raw
        return data

    def __repr__(self):
        return "<ExperimentRawReader {0:s}>".format(
            self.experiment)
        
class KwdRawDataReader(BaseRawDataReader):
    def __init__(self, kwd, dtype_to=None):
        
        if isinstance(kwd, string_types):
            kwd = open_file(kwd, 'r')
            self.to_close = True
        else:
            self.to_close = False

        self._kwd = kwd
        nrecordings = self._kwd.root.recordings._v_nchildren
        super(KwdRawDataReader, self).__init__(dtype_to=dtype_to)
        
    def get_recording_data(self, recording):
        data = self._kwd.root.recordings._f_getChild(str(recording)).data
        return data
    
    def __repr__(self):
        return "<KwdRawReader {0:s}>".format(
            self._kwd.filename)
        
    def __exit__(self, *args):
        if self.to_close:
            self._kwd.close()
    
class DatRawDataReader(BaseRawDataReader):
    """Read a DAT file by chunks."""
    def __init__(self, filenames, dtype=None, dtype_to=None, 
                 shape=None, datlen=None):
        """
        
        Arguments:
        * dtype: the dtype of the DAT file.
        * dtype_to: the dtype of the chunks to read.
        * datlen: length of the array. By default, None = full length of the 
          data.
        """
        if not isinstance(filenames, list):
            filenames = [filenames]
        self.filenames = filenames
        self.nrecordings = len(self.filenames)
        
        if dtype is None:
            dtype = np.int16
        self.dtype = np.dtype(dtype)
        
        if dtype_to is None:
            dtype_to = np.int16
        self.dtype_to = np.dtype(dtype_to)
        
        self.datlen = datlen
        _, self.nchannels = shape
        
    def get_recording_data(self, recording):
        filename = self.filenames[recording]
        # Find file size.
        size = os.stat(filename).st_size
        row_size = self.nchannels * self.dtype.itemsize
        if size % row_size != 0:
            raise ValueError(("Shape error: the file {f} has S={s} bytes, "
                "but there are C={c} channels. C should be a divisor of S."
                "").format(f=filename, s=size, c=self.nchannels))
        self.nsamples = size // row_size
        shape = (self.nsamples, self.nchannels)
        data = np.memmap(filename, dtype=self.dtype,
                           mode='r',
                           offset=0,
                           shape=shape)
        # Return full recording...
        if self.datlen is None:
            return data
        # ... or restrict the length of the recording.
        else:
            return data[:self.datlen,:]
        
    def __repr__(self):
        return "<DatRawReader {0:s}>".format(
            ', '.join(self.filenames))
        

# -----------------------------------------------------------------------------
# Main raw data reading function
# -----------------------------------------------------------------------------
def read_raw(raw, nchannels=None):
    if isinstance(raw, np.ndarray):
        return NumPyRawDataReader(raw)
    elif isinstance(raw, Experiment):
        return ExperimentRawDataReader(raw)
    elif isinstance(raw, string_types):
        if raw.endswith('.dat'):
            assert nchannels > 0, ("The number of channels must be specified "
                "in order to read from a .dat file.")
            return DatRawDataReader(raw, dtype=np.int16, shape=(0, nchannels))
        elif raw.endswith('.kwd'):
            return KwdRawDataReader(raw)
        else:
            raise ArgumentError("Unknown file extension for the raw data.")
    elif isinstance(raw, list):
        # List of .dat files.
        for _ in raw:
            assert _.endswith('.dat')
        assert nchannels > 0, ("The number of channels must be specified "
            "in order to read from a .dat file.")
        return DatRawDataReader(raw, dtype=np.int16, shape=(0, nchannels))
            
def convert_dat_to_kwd(dat_reader, kwd_file, chunk_size=20000):
    with open_file(kwd_file, 'a') as kwd:
        for chunk in dat_reader.chunks(chunk_size):
            data = chunk.data_chunk_full
            rec = chunk.recording
            try:
                # Add the data to the KWD file, in /recordings/[X]/data.
                kwd.root.recordings._f_getChild(str(rec)).data.append(data)
            except tb.NoSuchNodeError:
                # If /recordings/[X] does not exist, add this recording
                # to the KWD file and the data as well.
                add_recording_in_kwd(kwd, recording_id=rec,
                                     nchannels=chunk.nchannels,
                                     nsamples=chunk.nsamples,
                                     data=data)
                