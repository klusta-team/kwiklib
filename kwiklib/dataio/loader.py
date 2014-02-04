"""This module provides utility classes and functions to load spike sorting
data sets."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import os.path
import re
from collections import Counter

import numpy as np
import pandas as pd

from tools import (load_text, normalize,
    load_binary, load_pickle, save_text, get_array, 
    first_row, load_binary_memmap)
from selection import (select, select_pairs, get_spikes_in_clusters,
    get_some_spikes_in_clusters, get_some_spikes, get_indices)
from kwiklib.utils.logger import (debug, info, warn, exception, FileLogger,
    register, unregister)
from kwiklib.utils.colors import COLORS_COUNT, generate_colors


# -----------------------------------------------------------------------------
# Default cluster/group info
# -----------------------------------------------------------------------------
def default_cluster_info(clusters_unique):
    n = len(clusters_unique)
    cluster_info = pd.DataFrame({
        'color': generate_colors(n),
        'group': 3 * np.ones(n)},
        dtype=np.int32,
        index=clusters_unique)
    # Put cluster 0 in group 0 (=noise), cluster 1 in group 1 (=MUA)
    if 0 in clusters_unique:
        cluster_info['group'][0] = 0
    if 1 in clusters_unique:
        cluster_info['group'][1] = 1
    return cluster_info

def default_group_info():
    group_info = np.zeros((4, 3), dtype=object)
    group_info[:, 0] = np.arange(4)
    group_info[:, 1] = generate_colors(group_info.shape[0])
    group_info[:, 2] = np.array(['Noise', 'MUA', 'Good', 'Unsorted'],
        dtype=object)
    group_info = pd.DataFrame(
        {'color': group_info[:, 1].astype(np.int32),
         'name': group_info[:, 2]},
         index=group_info[:, 0].astype(np.int32))
    return group_info


# -----------------------------------------------------------------------------
# Cluster renumbering
# -----------------------------------------------------------------------------
def reorder(x, order):
    x_reordered = np.zeros_like(x)
    for i, o in enumerate(order):
        x_reordered[x == o] = i
    return x_reordered

def renumber_clusters(clusters, cluster_info):
    clusters_unique = get_array(get_indices(cluster_info))
    nclusters = len(clusters_unique)
    assert np.array_equal(clusters_unique, np.unique(clusters))
    clusters_array = get_array(clusters)
    groups = get_array(cluster_info['group'])
    colors = get_array(cluster_info['color'])
    groups_unique = np.unique(groups)
    # Reorder clusters according to the group.
    clusters_unique_reordered = np.hstack(
        [sorted(clusters_unique[groups == group]) for group in groups_unique])
    # WARNING: there's a +2 offset to avoid conflicts with the old convention
    # cluster 0 = noise, cluster 1 = MUA.
    clusters_renumbered = reorder(clusters_array, clusters_unique_reordered) + 2
    cluster_permutation = reorder(clusters_unique_reordered, clusters_unique)
    # Reorder cluster info.
    groups_reordered = groups[cluster_permutation]
    colors_reordered = colors[cluster_permutation]
    # Recreate new cluster info.
    cluster_info_reordered = pd.DataFrame({'color': colors_reordered, 
        'group': groups_reordered}, dtype=np.int32, 
        index=(np.arange(nclusters) + 2))
    return clusters_renumbered, cluster_info_reordered

