"""HDF5 tools tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import tempfile

import numpy as np
import tables as tb
from nose import with_setup

from kwiklib.utils.six import itervalues
from kwiklib.dataio.kwik import *


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
DIRPATH = tempfile.mkdtemp()

def setup_create(create_default=False):
    prm = {'nfeatures': 3, 'waveforms_nsamples': 20, 'has_masks': False,
           'nchannels': 3}
    prb = {0:
        {
            'channels': [4, 6, 8],
            'graph': [[4, 6], [8, 4]],
            'geometry': {4: [0.4, 0.6], 6: [0.6, 0.8], 8: [0.8, 0.0]},
        }
    }
    
    create_files('myexperiment', dir=DIRPATH, prm=prm, prb=prb,
                 create_default_info=create_default)

def setup_create_default():
    setup_create(True)
                 
def teardown_create():
    files = get_filenames('myexperiment', dir=DIRPATH)
    [os.remove(path) for path in itervalues(files)]

    
# -----------------------------------------------------------------------------
# Filename tests
# -----------------------------------------------------------------------------
def test_get_filenames():
    filenames = get_filenames('myexperiment')
    assert os.path.basename(filenames['kwik']) == 'myexperiment.kwik'
    assert os.path.basename(filenames['kwx']) == 'myexperiment.kwx'
    assert os.path.basename(filenames['raw.kwd']) == 'myexperiment.raw.kwd'
    assert os.path.basename(filenames['low.kwd']) == 'myexperiment.low.kwd'
    assert os.path.basename(filenames['high.kwd']) == 'myexperiment.high.kwd'
    
def test_basename_1():
    bn = 'myexperiment'
    filenames = get_filenames(bn)
    kwik = filenames['kwik']
    kwx = filenames['kwx']
    kwdraw = filenames['raw.kwd']
    
    assert get_basename(kwik) == bn
    assert get_basename(kwx) == bn
    assert get_basename(kwdraw) == bn
    
def test_basename_2():
    kwik = '/my/path/experiment.kwik'
    kwx = '/my/path/experiment.kwx'
    kwdhigh = '/my/path/experiment.high.kwd'
    
    assert get_basename(kwik) == 'experiment'
    assert get_basename(kwx) == 'experiment'
    assert get_basename(kwdhigh) == 'experiment'
    
    
# -----------------------------------------------------------------------------
# HDF5 creation functions tests
# -----------------------------------------------------------------------------
def test_create_kwik():
    path = os.path.join(DIRPATH, 'myexperiment.kwik')
    
    prm = {
        'waveforms_nsamples': 20,
        'nfeatures': 3*32,
    }
    prb = {0:
        {
            'channels': [4, 6, 8],
            'graph': [[4, 6], [8, 4]],
            'geometry': {4: [0.4, 0.6], 6: [0.6, 0.8], 8: [0.8, 0.0]},
        }
    }
    
    create_kwik(path, prm=prm, prb=prb)
    
    f = tb.openFile(path, 'r')
    channel = f.root.channel_groups.__getattr__('0').channels.__getattr__('4')
    assert channel._v_attrs.name == 'channel_4'
    
    f.close()
    os.remove(path)
    
def test_create_kwx():
    path = os.path.join(DIRPATH, 'myexperiment.kwx')
    
    # Create the KWX file.
    waveforms_nsamples = 20
    nchannels = 32
    nchannels2 = 24
    nfeatures = 3*nchannels
    prm = {
        'waveforms_nsamples': waveforms_nsamples,
        'nfeatures': 3*nchannels,
    }
    prb = {0:
        {
            'channels': np.arange(nchannels),
        },
        1: {
            'channels': nchannels + np.arange(nchannels2),
            'nfeatures': 3*nchannels2
        },
        2: {
            'channels': nchannels + nchannels2 + np.arange(nchannels),
            'nfeatures': 2*nchannels
        },
    }
    
    create_kwx(path, prb=prb, prm=prm)
    
    # Open the KWX file.
    f = tb.openFile(path, 'r')
    
    # Group 1
    fm1 = f.root.channel_groups.__getattr__('1').features_masks
    wr1 = f.root.channel_groups.__getattr__('1').waveforms_raw
    wf1 = f.root.channel_groups.__getattr__('1').waveforms_filtered
    assert fm1.shape[1:] == (3*nchannels2, 2)
    assert wr1.shape[1:] == (waveforms_nsamples, nchannels2)
    assert wf1.shape[1:] == (waveforms_nsamples, nchannels2)

    # Group 2
    fm2 = f.root.channel_groups.__getattr__('2').features_masks
    wr2 = f.root.channel_groups.__getattr__('2').waveforms_raw
    wf2 = f.root.channel_groups.__getattr__('2').waveforms_filtered
    assert fm2.shape[1:] == (2*nchannels, 2)
    assert wr2.shape[1:] == (waveforms_nsamples, nchannels)
    assert wf2.shape[1:] == (waveforms_nsamples, nchannels)
    
    f.close()
    
    # Delete the file.
    os.remove(path)
    
def test_create_kwd():
    path = os.path.join(DIRPATH, 'myexperiment.raw.kwd')
    
    # Create the KWD file.
    nchannels_tot = 32*3
    prm = {'nchannels': nchannels_tot}
    
    create_kwd(path, type='raw', prm=prm,)
    
    # Open the KWX file.
    f = tb.openFile(path, 'r')
    
    assert f.root.recordings
    
    f.close()
    
    # Delete the file.
    os.remove(path)
    
def test_create_empty():
    files = create_files('myexperiment', dir=DIRPATH)
    [os.remove(path) for path in itervalues(files)]
    
@with_setup(setup_create_default, teardown_create)
def test_create_default():
    path = os.path.join(DIRPATH, 'myexperiment.kwik')
    
    prm = {
        'waveforms_nsamples': 20,
        'nfeatures': 3*32,
    }
    prb = {0:
        {
            'channels': [4, 6, 8],
            'graph': [[4, 6], [8, 4]],
            'geometry': {4: [0.4, 0.6], 6: [0.6, 0.8], 8: [0.8, 0.0]},
        }
    }
    
    files = open_files('myexperiment', dir=DIRPATH)
    f = files['kwik']
    assert f.root.channel_groups.__getattr__('0').cluster_groups.main.__getattr__('0')._f_getAttr('name') == 'Noise'
    assert hasattr(f.root.channel_groups.__getattr__('0').clusters.main, '0')
    
    close_files(files)
    
    
# -----------------------------------------------------------------------------
# Item creation functions tests
# -----------------------------------------------------------------------------
@with_setup(setup_create, teardown_create)
def test_add_recording():
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    
    sample_rate = 20000.
    start_time = 10.
    start_sample = 200000.
    bit_depth = 16
    band_high = 100.
    band_low = 500.
    nchannels = 32
    nsamples = 0
    
    add_recording(files, 
                  sample_rate=sample_rate,
                  start_time=start_time, 
                  start_sample=start_sample,
                  bit_depth=bit_depth,
                  band_high=band_high,
                  band_low=band_low,
                  nchannels=nchannels,
                  nsamples=nsamples,
                  )
    
    rec = files['kwik'].root.recordings.__getattr__('0')
    assert rec._v_attrs.sample_rate == sample_rate
    assert rec._v_attrs.start_time == start_time
    assert rec._v_attrs.start_sample == start_sample
    assert rec._v_attrs.bit_depth == bit_depth
    assert rec._v_attrs.band_high == band_high
    assert rec._v_attrs.band_low == band_low
    
    close_files(files)
    
@with_setup(setup_create, teardown_create)
def test_add_event_type():
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    add_event_type(files, 'myevents')
    events = files['kwik'].root.event_types.myevents.events
    
    assert isinstance(events.time_samples, tb.EArray)
    assert isinstance(events.recording, tb.EArray)
    events.user_data
    
    close_files(files)

@with_setup(setup_create, teardown_create)
def test_add_cluster_group():
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    add_cluster_group(files, channel_group_id='0', id='0', name='Noise')
    noise = files['kwik'].root.channel_groups.__getattr__('0').cluster_groups.main.__getattr__('0')
    
    assert noise._v_attrs.name == 'Noise'
    noise.application_data.klustaviewa._v_attrs.color
    noise.user_data
    
    remove_cluster_group(files, channel_group_id='0', id='0')
    assert not hasattr(
        files['kwik'].root.channel_groups.__getattr__('0').cluster_groups.main,
        '0')
        
    close_files(files)

@with_setup(setup_create, teardown_create)
def test_add_cluster():
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    add_cluster(files, channel_group_id='0',)
    cluster = files['kwik'].root.channel_groups.__getattr__('0').clusters.main.__getattr__('0')
    
    cluster._v_attrs.cluster_group
    cluster._v_attrs.mean_waveform_raw
    cluster._v_attrs.mean_waveform_filtered
    
    cluster.quality_measures
    cluster.application_data.klustaviewa._v_attrs.color
    cluster.user_data
    
    remove_cluster(files, channel_group_id='0', id='0')
    assert not hasattr(
        files['kwik'].root.channel_groups.__getattr__('0').clusters.main,
        '0')
    
    close_files(files)

@with_setup(setup_create, teardown_create)
def test_add_clustering():
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    nspikes = 100
    
    add_spikes(files, channel_group_id='0', 
               time_samples=np.arange(nspikes),
               features=np.random.randn(nspikes, 3),
               fill_empty=False,
               )
               
    spike_clusters = np.random.randint(size=nspikes, low=3, high=20)
    add_clustering(files, name='myclustering', spike_clusters=spike_clusters)
    clusters = files['kwik'].root.channel_groups.__getattr__('0').spikes.clusters.myclustering[:]
    
    assert np.allclose(spike_clusters, clusters)
    
    clustering = files['kwik'].root.channel_groups.__getattr__('0').clusters.myclustering
    assert not hasattr(clustering, '0')
    for i in np.unique(spike_clusters):
        assert clustering.__getattr__(str(i)).application_data. \
                klustaviewa._f_getAttr('color') > 0
    
    close_files(files)
    
@with_setup(setup_create, teardown_create)
def test_add_clustering_overwrite():
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    nspikes = 100
    
    add_spikes(files, channel_group_id='0', 
               time_samples=np.arange(nspikes),
               features=np.random.randn(nspikes, 3),
               fill_empty=False,
               )
               
    spike_clusters = np.random.randint(size=nspikes, low=3, high=20)
    add_clustering(files, name='main', spike_clusters=spike_clusters,
                   overwrite=True)
    clusters = files['kwik'].root.channel_groups.__getattr__('0').spikes.clusters.main[:]
    
    assert np.allclose(spike_clusters, clusters)
    
    clustering = files['kwik'].root.channel_groups.__getattr__('0').clusters.main
    assert not hasattr(clustering, '0')
    for i in np.unique(spike_clusters):
        assert clustering.__getattr__(str(i)).application_data. \
                klustaviewa._f_getAttr('color') > 0
    
    close_files(files)

@with_setup(setup_create, teardown_create)
def test_add_spikes():
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    nspikes = 7
    
    add_spikes(files, channel_group_id='0', 
               time_samples=1,
               )
    add_spikes(files, channel_group_id='0', 
               time_samples=np.arange(1),
               )
    add_spikes(files, channel_group_id='0', 
               time_samples=np.arange(2),
               masks=np.random.randn(2, 3),
               )
    add_spikes(files, channel_group_id='0', 
               time_samples=np.arange(2),
               waveforms_raw=np.random.randn(2, 20, 3),
               )
    add_spikes(files, channel_group_id='0', 
               time_samples=4,
               waveforms_raw=np.random.randn(20, 3),
               waveforms_filtered=np.random.randn(20, 3),
               )
               
    spikes = files['kwx'].root.channel_groups.__getattr__('0')
    assert spikes.waveforms_raw.shape == (nspikes, 20, 3)
    assert spikes.waveforms_filtered.shape == (nspikes, 20, 3)
    close_files(files)
    
@with_setup(setup_create, teardown_create)
def test_add_spikes_fm():
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    nspikes = 7
    
    add_spikes(files, channel_group_id='0', 
               time_samples=np.arange(nspikes),
               features=np.random.randn(nspikes, 3),
               fill_empty=False,
               )
               
    spikes = files['kwx'].root.channel_groups.__getattr__('0')
    
    assert spikes.waveforms_raw.shape == (0, 20, 3)
    assert spikes.waveforms_filtered.shape == (0, 20, 3)
    assert spikes.features_masks.shape == (nspikes, 3)
    
    close_files(files)

@with_setup(setup_create, teardown_create)
def test_to_contiguous():
    """Convert an EArray to contiguous Array."""
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    
    n = 100000
    fm = files['kwx'].root.channel_groups.__getattr__('0').features_masks
    s = fm.shape[1:]
    a = fm.atom
    
    X = np.random.rand(n, *s)
    fm.append(X)
    
    assert isinstance(fm, tb.EArray)
    assert fm.shape[0] == n
    assert fm.shape[1:] == s
    assert fm.atom == a
    
    to_contiguous(fm, nspikes=n)
    
    fm = files['kwx'].root.channel_groups.__getattr__('0').features_masks
    assert isinstance(fm, tb.Array) and not isinstance(fm, tb.EArray)
    assert fm.shape[0] == n
    assert fm.shape[1:] == s
    assert fm.atom == a
    
    Y = fm[...]
    
    assert np.allclose(X, Y)
    
    close_files(files)

