"""Chunking routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from kwiklib.dataio.utils import convert_dtype


# -----------------------------------------------------------------------------
# Chunking functions
# -----------------------------------------------------------------------------
def chunk_bounds(nsamples, chunk_size, overlap=0):
    """Returns chunks of the form:
    [ overlap/2 | chunk_size-overlap | overlap/2 ]
    s_start   keep_start           keep_end     s_end
    Except for the first and last chunks which do not have a left/right overlap
    
    """
    s_start = 0
    s_end = chunk_size
    keep_start = s_start
    keep_end = s_end - overlap // 2
    yield s_start, s_end, keep_start, keep_end
    
    while s_end - overlap + chunk_size < nsamples:
        s_start = s_end - overlap
        s_end = s_start + chunk_size
        keep_start = keep_end
        keep_end = s_end - overlap // 2
        if s_start < s_end:
            yield s_start, s_end, keep_start, keep_end
        
    s_start = s_end - overlap
    s_end = nsamples
    keep_start = keep_end
    keep_end = s_end
    if s_start < s_end:
        yield s_start, s_end, keep_start, keep_end

def excerpt_step(nsamples, nexcerpts=None, excerpt_size=None):
    step = max((nsamples - excerpt_size) // (nexcerpts - 1),
               excerpt_size)
    return step
    
def excerpts(nsamples, nexcerpts=None, excerpt_size=None):
    """Yield (start, end) where start is included and end is excluded."""
    step = excerpt_step(nsamples, 
                        nexcerpts=nexcerpts,
                        excerpt_size=excerpt_size)
    for i in range(nexcerpts):
        start = i * step
        if start >= nsamples:
            break
        end = min(start + excerpt_size, nsamples)
        yield start, end
    
    
# -----------------------------------------------------------------------------
# Chunk class
# -----------------------------------------------------------------------------
class Chunk(object):
    def __init__(self, data=None, nsamples=None, nchannels=None,
                 bounds=None, dtype=None, recording=0, nrecordings=1):
        self._data = data
        if nsamples is None and nchannels is None:
            nsamples, nchannels = data.shape
        self.nsamples = nsamples
        self.nchannels = nchannels
        self.dtype = dtype
        self.recording = recording
        self.nrecordings = nrecordings
        self.s_start, self.s_end, self.keep_start, self.keep_end = bounds
        self.window_full = self.s_start, self.s_end
        self.window_keep = self.keep_start, self.keep_end
        
    @property
    def data_chunk_full(self):
        chunk = self._data[self.s_start:self.s_end,:]
        return convert_dtype(chunk, self.dtype)
    
    @property
    def data_chunk_keep(self):
        chunk =  self._data[self.keep_start:self.keep_end,:]
        return convert_dtype(chunk, self.dtype)
    
    def __repr__(self):
        return "<Chunk ({0:d}/{1:d}), recording {recording}>".format(
            self.s_start,
            self._data.shape[0], recording=self.recording,
        )
        
        
# -----------------------------------------------------------------------------
# Excerpt class
# -----------------------------------------------------------------------------
class Excerpt(object):
    def __init__(self, data=None, nsamples=None, nchannels=None,
                 bounds=None, dtype=None, recording=0, nrecordings=1):
        self._data = data
        if nsamples is None and nchannels is None:
            nsamples, nchannels = data.shape
        self.nsamples = nsamples
        self.nchannels = nchannels
        self.dtype = dtype
        self.recording = recording
        self.start, self.end = bounds
        self.size = self.end - self.start
        self.window = bounds
        
    @property
    def data(self):
        excerpt = self._data[self.start:self.end,:]
        return convert_dtype(excerpt, self.dtype)
    
    def __repr__(self):
        return "<Excerpt [{0:d}:{1:d}], maxlen={2:d}>".format(
            self.start, self.end, self._data.shape[0]
        )
        
        