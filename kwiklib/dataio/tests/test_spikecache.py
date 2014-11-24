"""Experiment tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import tables as tb
from nose import with_setup

from kwiklib.dataio.spikecache import SpikeCache

def test_spikecache_1():
    nspikes = 100000
    nclusters = 100
    nchannels = 8
    spike_clusters = np.random.randint(size=nspikes, low=0, high=nclusters)
    
    sc = SpikeCache(spike_clusters=spike_clusters,
                    cache_fraction=.1,
                    features_masks=np.zeros((nspikes, 3*nchannels, 2)),
                    waveforms_raw=np.zeros((nspikes, 20, nchannels)),
                    waveforms_filtered=np.zeros((nspikes, 20, nchannels)))
                    
    ind, fm = sc.load_features_masks(.1)
    
    assert len(ind) == nspikes // 100
    assert fm.shape[0] == nspikes // 100
    
    ind, fm = sc.load_features_masks(clusters=[10, 20])
    
    assert len(ind) == fm.shape[0]
    assert np.allclose(ind, np.nonzero(np.in1d(spike_clusters, (10, 20)))[0])
    
    ind, fm = sc.load_features_masks(clusters=[1000])
    assert len(ind) == 0
    assert len(fm) == 0
    
def test_spikecache_2():
    nspikes = 100000
    nclusters = 100
    nchannels = 8
    spike_clusters = np.random.randint(size=nspikes, low=0, high=nclusters)
    
    sc = SpikeCache(spike_clusters=spike_clusters,
                    cache_fraction=.1,
                    features_masks=np.zeros((nspikes, 3*nchannels, 2)),
                    waveforms_raw=np.zeros((nspikes, 20, nchannels)),
                    waveforms_filtered=np.zeros((nspikes, 20, nchannels)))
           
    ind, waveforms = sc.load_waveforms(clusters=[10], count=10)
    assert len(ind) == waveforms.shape[0]
    assert len(ind) >= 10
    
    ind, waveforms = sc.load_waveforms(clusters=[10, 20], count=10)
    assert len(ind) == waveforms.shape[0]
    assert len(ind) >= 20
    
    ind, waveforms = sc.load_waveforms(clusters=[1000], count=10)
    assert len(ind) == 0
   
   
   