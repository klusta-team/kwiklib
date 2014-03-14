"""Testing utility functions."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def create_trace(nsamples, nchannels):
    noise = np.array(np.random.randint(size=(nsamples, nchannels),
        low=-1000, high=1000), dtype=np.int16)
    t = np.linspace(0., 100., nsamples)
    low = np.array(10000 * np.cos(t), dtype=np.int16)
    return noise + low[:, np.newaxis]
    