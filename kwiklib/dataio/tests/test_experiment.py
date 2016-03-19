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

from kwiklib.dataio.kwik import (add_recording, create_files, open_files,
    close_files, add_event_type, add_cluster_group, get_filenames,
    add_cluster)
from kwiklib.dataio.experiment import (Experiment, _resolve_hdf5_path,
    ArrayProxy, DictVectorizer)
from kwiklib.utils.six import itervalues
from kwiklib.utils.logger import info


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
DIRPATH = tempfile.mkdtemp()

def _setup(_name, has_masks=True):
    # Create files.
    prm = {'nfeatures': 3, 'waveforms_nsamples': 10, 'nchannels': 3,
           'sample_rate': 20000.,
           'nfeatures_per_channel': 1,
           'has_masks': has_masks
           }
    prb = {0:
        {
            'channels': [4, 6, 8],
            'graph': [[4, 6], [8, 4]],
            'geometry': {4: [0.4, 0.6], 6: [0.6, 0.8], 8: [0.8, 0.0]},
        }
    }
    create_files(_name, dir=DIRPATH, prm=prm, prb=prb)

    # Open the files.
    files = open_files(_name, dir=DIRPATH, mode='a')

    # Add data.
    add_recording(files,
                  sample_rate=20000.,
                  start_time=10.,
                  start_sample=200000.,
                  bit_depth=16,
                  band_high=100.,
                  band_low=500.,
                  nchannels=3,)
    add_event_type(files, 'myevents')
    add_cluster_group(files, channel_group_id='0', id='0', name='Noise')
    add_cluster(files, channel_group_id='0', cluster_group=0)

    # Close the files
    close_files(files)

def _teardown(_name):
    files = get_filenames(_name, dir=DIRPATH)
    [os.remove(path) for path in itervalues(files)]

def setup(): _setup('myexperiment')
def teardown(): _teardown('myexperiment')
def setup2(): _setup('myexperiment2')
def teardown2(): _teardown('myexperiment2')
def setup_nomasks(): _setup('myexperiment_nomasks', has_masks=False)
def teardown_nomasks(): _teardown('myexperiment_nomasks')


# -----------------------------------------------------------------------------
# Experiment creation tests
# -----------------------------------------------------------------------------
def test_resolve_hdf5_path():
    path = "{kwx}/channel_groups/0"

    files = open_files('myexperiment', dir=DIRPATH)
    assert _resolve_hdf5_path(files, path)

    close_files(files)

def test_experiment_channels():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        assert exp.name == 'myexperiment'
        assert exp.application_data
        # assert exp.user_data
        assert exp.application_data.spikedetekt.nchannels == 3
        assert exp.application_data.spikedetekt.waveforms_nsamples == 10
        assert exp.application_data.spikedetekt.nfeatures == 3

        # Channel group.
        chgrp = exp.channel_groups[0]
        assert chgrp.name == 'channel_group_0'
        assert np.array_equal(chgrp.adjacency_graph, [[4, 6], [8, 4]])
        assert chgrp.application_data
        # assert chgrp.user_data

        # Channels.
        channels = chgrp.channels
        assert list(sorted(channels.keys())) == [4, 6, 8]

        # Channel.
        ch = channels[4]
        assert ch.name == 'channel_4'
        ch.kwd_index
        ch.ignored

        assert np.allclose(ch.position, [.4, .6])
        ch.voltage_gain
        ch.display_threshold
        assert ch.application_data
        # assert ch.user_data

def test_experiment_spikes():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        spikes = chgrp.spikes
        assert isinstance(spikes.time_samples, tb.EArray)
        assert spikes.time_samples.dtype == np.uint64
        assert spikes.time_samples.ndim == 1

        t = spikes.concatenated_time_samples
        # assert isinstance(t, np.ndarray)
        assert t.dtype == np.uint64
        assert t.ndim == 1

        assert isinstance(spikes.time_fractional, tb.EArray)
        assert spikes.time_fractional.dtype == np.uint8
        assert spikes.time_fractional.ndim == 1

        assert isinstance(spikes.recording, tb.EArray)
        assert spikes.recording.dtype == np.uint16
        assert spikes.recording.ndim == 1

        assert isinstance(spikes.clusters.main, tb.EArray)
        assert spikes.clusters.main.dtype == np.uint32
        assert spikes.clusters.main.ndim == 1

        assert isinstance(spikes.clusters.original, tb.EArray)
        assert spikes.clusters.original.dtype == np.uint32
        assert spikes.clusters.original.ndim == 1

        assert isinstance(spikes.features_masks, tb.EArray)
        assert spikes.features_masks.dtype == np.float32
        assert spikes.features_masks.ndim == 3

        assert isinstance(spikes.waveforms_raw, tb.EArray)
        assert spikes.waveforms_raw.dtype == np.int16
        assert spikes.waveforms_raw.ndim == 3

        assert isinstance(spikes.waveforms_filtered, tb.EArray)
        assert spikes.waveforms_filtered.dtype == np.int16
        assert spikes.waveforms_filtered.ndim == 3

def test_experiment_features():
    """Test the wrapper around features implementing a custom cache."""
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        spikes = chgrp.spikes

        assert isinstance(spikes.features_masks, tb.EArray)
        assert spikes.features_masks.dtype == np.float32
        assert spikes.features_masks.ndim == 3

        assert isinstance(spikes.waveforms_raw, tb.EArray)
        assert spikes.waveforms_raw.dtype == np.int16
        assert spikes.waveforms_raw.ndim == 3

        assert isinstance(spikes.waveforms_filtered, tb.EArray)
        assert spikes.waveforms_filtered.dtype == np.int16
        assert spikes.waveforms_filtered.ndim == 3

def test_experiment_setattr():
    with Experiment('myexperiment', dir=DIRPATH, mode='a') as exp:
        chgrp = exp.channel_groups[0]
        color0 = chgrp.clusters.main[0].application_data.klustaviewa.color
        # By default, the cluster's color is 1.
        assert color0 == 1
        # Set it to 0.
        chgrp.clusters.main[0].application_data.klustaviewa.color = 0
        # We check that the color has changed in the file.
        assert chgrp.clusters.main[0].application_data.klustaviewa._f_getAttr('color') == 0

    # Close and open the file.
    with Experiment('myexperiment', dir=DIRPATH, mode='a') as exp:
        chgrp = exp.channel_groups[0]
        # Check that the change has been saved on disk.
        assert chgrp.clusters.main[0].application_data.klustaviewa.color == 0
        # Set back the field to its original value.
        chgrp.clusters.main[0].application_data.klustaviewa.color = color0

def test_experiment_vectorizer():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        clustering = exp.channel_groups[0].clusters.main
        dv = DictVectorizer(clustering, 'application_data.klustaviewa.color')
        assert np.array_equal(dv[0], 1)
        assert np.array_equal(dv[:], [1])

@with_setup(setup2, teardown2)  # Create brand new files.
def test_experiment_add_spikes():
    with Experiment('myexperiment2', dir=DIRPATH, mode='a') as exp:
        chgrp = exp.channel_groups[0]
        spikes = chgrp.spikes

        assert spikes.features_masks.shape == (0, 3, 2)
        assert isinstance(spikes.features, ArrayProxy)
        assert spikes.features.shape == (0, 3)

        spikes.add(time_samples=1000)
        spikes.add(time_samples=2000)

        assert len(spikes) == 2
        assert spikes.features_masks.shape == (2, 3, 2)

        assert isinstance(spikes.features, ArrayProxy)
        assert spikes.features.shape == (2, 3)

@with_setup(setup_nomasks, teardown_nomasks)  # Create brand new files.
def test_experiment_add_spikes_nomasks():
    with Experiment('myexperiment_nomasks', dir=DIRPATH, mode='a') as exp:
        chgrp = exp.channel_groups[0]
        spikes = chgrp.spikes

        assert spikes.features_masks.shape == (0, 3)
        assert isinstance(spikes.features, tb.Array)
        assert spikes.features.shape == (0, 3)

        spikes.add(time_samples=1000)
        spikes.add(time_samples=2000)

        assert len(spikes) == 2
        assert spikes.features_masks.shape == (2, 3)

        assert isinstance(spikes.features, tb.Array)
        assert spikes.features.shape == (2, 3)

@with_setup(setup2, teardown2)  # Create brand new files.
def test_experiment_add_cluster():
    with Experiment('myexperiment2', dir=DIRPATH, mode='a') as exp:
        chgrp = exp.channel_groups[0]
        chgrp.clusters.main.add_cluster(id=27, color=34)
        assert 27 in chgrp.clusters.main.keys()
        assert chgrp.clusters.main[27].application_data.klustaviewa.color == 34
        assert np.allclose(chgrp.clusters.main.color[:], [1, 34])

        chgrp.clusters.main.remove_cluster(id=27)
        assert 27 not in chgrp.clusters.main.keys()

@with_setup(setup2, teardown2)  # Create brand new files.
def test_experiment_add_cluster_group():
    with Experiment('myexperiment2', dir=DIRPATH, mode='a') as exp:
        chgrp = exp.channel_groups[0]
        chgrp.cluster_groups.main.add_group(id=27, name='boo', color=34)
        assert 27 in chgrp.cluster_groups.main.keys()
        assert chgrp.cluster_groups.main[27].name == 'boo'
        assert chgrp.cluster_groups.main[27].application_data.klustaviewa.color == 34

        chgrp.cluster_groups.main.remove_group(id=27)
        assert 27 not in chgrp.cluster_groups.main.keys()

def test_experiment_clusters():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        cluster = chgrp.clusters.main[0]

@with_setup(setup2, teardown2)  # Create brand new files.
def test_experiment_copy_clusters():
    with Experiment('myexperiment', dir=DIRPATH, mode='a') as exp:
        clusters = exp.channel_groups[0].spikes.clusters

        # Adding spikes.
        for i in range(10):
            exp.channel_groups[0].spikes.add(time_samples=i*1000, cluster=10+i)

        main = clusters.main[:]
        original = clusters.original[:]

        assert len(main) == 10
        assert len(original) == 10
        assert np.allclose(main, np.arange(10, 20))
        assert np.allclose(original, np.zeros(10))

        # Change original clusters on disk.
        clusters.original[1:10:2] = 123
        assert np.all(clusters.main[1:10:2] != 123)

        # Copy clusters from original to main.
        clusters.copy('original', 'main')
        assert np.all(clusters.main[1:10:2] == 123)

@with_setup(setup,)
def test_experiment_cluster_groups():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        cluster_group = chgrp.cluster_groups.main[0]
        assert cluster_group.name == 'Noise'

        assert cluster_group.application_data
        # assert cluster_group.user_data

        assert np.array_equal(chgrp.cluster_groups.main.color[:], [1])
        assert np.array_equal(chgrp.clusters.main.group[:], [0])

def test_experiment_recordings():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        rec = exp.recordings[0]
        assert rec.name == 'recording_0'
        assert rec.sample_rate == 20000.
        assert rec.start_time == 10.
        assert rec.start_sample == 200000.
        assert rec.bit_depth == 16
        assert rec.band_high == 100.
        assert rec.band_low == 500.

        rd = rec.raw
        assert isinstance(rd, tb.EArray)
        assert rd.shape == (0, 3)
        assert rd.dtype == np.int16

def test_experiment_events():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        evtp = exp.event_types['myevents']
        evtp.application_data
        # evtp.user_data

        samples = evtp.events.time_samples
        assert isinstance(samples, tb.EArray)
        assert samples.dtype == np.uint64

        recordings = evtp.events.recording
        assert isinstance(recordings, tb.EArray)
        assert recordings.dtype == np.uint16

        # evtp.events.user_data

def test_experiment_repr():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        s = str(exp)

def test_experiment_repr_nokwd():
    kwd = os.path.join(DIRPATH, 'myexperiment.raw.kwd')
    kwd2 = os.path.join(DIRPATH, 'myexperiment2.raw.kwd')

    # Move a KWD file and test if Experiment works without KWD.
    os.rename(kwd, kwd2)

    info("The following error message is expected (part of the unit test)")
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        s = str(exp)

    os.rename(kwd2, kwd)

