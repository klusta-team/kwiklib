"""Generate a DAT file with random raw data."""
from kwiklib.dataio import save_binary
from kwiklib.dataio.tests.mock_data import create_trace

nsamples = 20000. * 60
nchannels = 32
filename = "myexperiment.dat"
dat = create_trace(nsamples, nchannels)
save_binary(filename, dat)
