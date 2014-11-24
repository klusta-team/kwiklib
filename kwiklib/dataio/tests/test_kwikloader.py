"""Unit tests for loader module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
from collections import Counter

import numpy as np
import numpy.random as rnd
import pandas as pd
import shutil
from nose.tools import with_setup

from mock_data import setup as setup_klusters
from mock_data import (teardown, TEST_FOLDER, nspikes, nclusters, nsamples, 
    nchannels, fetdim)
from kwiklib.dataio import (KwikLoader, Experiment, klusters_to_kwik,
    check_dtype, check_shape, get_array, select, get_indices)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
def setup():
    setup_klusters()
    klusters_to_kwik(filename='test', dir=TEST_FOLDER)
    
    
# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_kwik_loader_1():
    
    # Open the mock data.
    dir = TEST_FOLDER
    xmlfile = os.path.join(dir, 'test.xml')
    l = KwikLoader(filename=xmlfile)
    
    # Get full data sets.
    features = l.get_features()
    # features_some = l.get_some_features()
    masks = l.get_masks()
    waveforms = l.get_waveforms()
    clusters = l.get_clusters()
    spiketimes = l.get_spiketimes()
    nclusters = len(Counter(clusters))
    
    # probe = l.get_probe()
    cluster_colors = l.get_cluster_colors()
    cluster_groups = l.get_cluster_groups()
    group_colors = l.get_group_colors()
    group_names = l.get_group_names()
    cluster_sizes = l.get_cluster_sizes()
    
    # Check the shape of the data sets.
    # ---------------------------------
    assert check_shape(features, (nspikes, nchannels * fetdim + 1))
    # assert features_some.shape[1] == nchannels * fetdim + 1
    assert check_shape(masks, (nspikes, nchannels * fetdim + 1))
    assert check_shape(waveforms, (nspikes, nsamples, nchannels))
    assert check_shape(clusters, (nspikes,))
    assert check_shape(spiketimes, (nspikes,))
    
    # assert check_shape(probe, (nchannels, 2))
    assert check_shape(cluster_colors, (nclusters,))
    assert check_shape(cluster_groups, (nclusters,))
    assert check_shape(group_colors, (4,))
    assert check_shape(group_names, (4,))
    assert check_shape(cluster_sizes, (nclusters,))
    
    
    # Check the data type of the data sets.
    # -------------------------------------
    assert check_dtype(features, np.float32)
    assert check_dtype(masks, np.float32)
    # HACK: Panel has no dtype(s) attribute
    # assert check_dtype(waveforms, np.float32)
    assert check_dtype(clusters, np.int32)
    assert check_dtype(spiketimes, np.float64)
    
    # assert check_dtype(probe, np.float32)
    assert check_dtype(cluster_colors, np.int32)
    assert check_dtype(cluster_groups, np.int32)
    assert check_dtype(group_colors, np.int32)
    assert check_dtype(group_names, object)
    assert check_dtype(cluster_sizes, np.int32)
    
    l.close()
        
def test_kwik_loader_control():
    # Open the mock data.
    dir = TEST_FOLDER
    xmlfile = os.path.join(dir, 'test.xml')
    l = KwikLoader(filename=xmlfile)
    
    # Take all spikes in cluster 3.
    spikes = get_indices(l.get_clusters(clusters=3))
    
    # Put them in cluster 4.
    l.set_cluster(spikes, 4)
    spikes_new = get_indices(l.get_clusters(clusters=4))
    
    # Ensure all spikes in old cluster 3 are now in cluster 4.
    assert np.all(np.in1d(spikes, spikes_new))
    
    # Change cluster groups.
    clusters = [2, 3, 4]
    group = 0
    l.set_cluster_groups(clusters, group)
    groups = l.get_cluster_groups(clusters)
    assert np.all(groups == group)
    
    # Change cluster colors.
    clusters = [2, 3, 4]
    color = 12
    l.set_cluster_colors(clusters, color)
    colors = l.get_cluster_colors(clusters)
    assert np.all(colors == color)
    
    # Change group name.
    group = 0
    name = l.get_group_names(group)
    name_new = 'Noise new'
    assert name == 'Noise'
    l.set_group_names(group, name_new)
    assert l.get_group_names(group) == name_new
    
    # Change group color.
    groups = [1, 2]
    colors = l.get_group_colors(groups)
    color_new = 10
    l.set_group_colors(groups, color_new)
    assert np.all(l.get_group_colors(groups) == color_new)
    
    # Add cluster and group.
    spikes = get_indices(l.get_clusters(clusters=3))[:10]
    # Create new group 100.
    l.add_group(100, 'New group', 10)
    # Create new cluster 10000 and put it in group 100.
    l.add_cluster(10000, 100, 10)
    # Put some spikes in the new cluster.
    l.set_cluster(spikes, 10000)
    clusters = l.get_clusters(spikes=spikes)
    assert np.all(clusters == 10000)
    groups = l.get_cluster_groups(10000)
    assert groups == 100
    l.set_cluster(spikes, 2)
    
    # Remove the new cluster and group.
    l.remove_cluster(10000)
    l.remove_group(100)
    assert np.all(~np.in1d(10000, l.get_clusters()))
    assert np.all(~np.in1d(100, l.get_cluster_groups()))
    
    l.close()
    
@with_setup(setup)
def test_kwik_save():
    """WARNING: this test should occur at the end of the module since it
    changes the mock data sets."""
    # Open the mock data.
    dir = TEST_FOLDER
    xmlfile = os.path.join(dir, 'test.xml')
    l = KwikLoader(filename=xmlfile)
    
    clusters = l.get_clusters()
    cluster_colors = l.get_cluster_colors()
    cluster_groups = l.get_cluster_groups()
    group_colors = l.get_group_colors()
    group_names = l.get_group_names()
    
    # Set clusters.
    indices = get_indices(clusters)
    l.set_cluster(indices[::2], 2)
    l.set_cluster(indices[1::2], 3)
    
    # Set cluster info.
    cluster_indices = l.get_clusters_unique()
    l.set_cluster_colors(cluster_indices[::2], 10)
    l.set_cluster_colors(cluster_indices[1::2], 20)
    l.set_cluster_groups(cluster_indices[::2], 1)
    l.set_cluster_groups(cluster_indices[1::2], 0)
    
    # Save.
    l.remove_empty_clusters()
    l.save()
    
    clusters = l.get_clusters()
    cluster_colors = l.get_cluster_colors()
    cluster_groups = l.get_cluster_groups()
    group_colors = l.get_group_colors()
    group_names = l.get_group_names()
    
    assert np.all(clusters[::2] == 2)
    assert np.all(clusters[1::2] == 3)
    
    assert np.all(cluster_colors[::2] == 10)
    assert np.all(cluster_colors[1::2] == 20)
    
    print cluster_groups
    
    assert np.all(cluster_groups[::2] == 1)
    assert np.all(cluster_groups[1::2] == 0)

    l.close()
    
    
    