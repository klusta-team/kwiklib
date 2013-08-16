"""This module provides functions used to load KWA (kwailiary) files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import tables
import time

import numpy as np
import matplotlib.pyplot as plt

from selection import get_indices


# -----------------------------------------------------------------------------
# Probe file functions
# -----------------------------------------------------------------------------
def kwa_to_json(kwa_dict):
    """Convert a KWA dictionary to JSON.
    cluster_colors and group_colors are pandas.Series objects."""
    kwa_full = {}
    kwa_full['shanks'] = []
    for shank, kwa in kwa_dict['shanks'].iteritems():
        cluster_colors = kwa['cluster_colors']
        group_colors = kwa['group_colors']
        clusters = get_indices(cluster_colors)
        groups = get_indices(group_colors)
        kwa_shank = dict(
            clusters=[{'cluster': str(cluster), 'color': str(cluster_colors[cluster])}
                for cluster in clusters],
            groups_of_clusters=[{'group': str(group), 'color': str(group_colors[group])}
                for group in groups],
            shank_index=shank
        )
        kwa_full['shanks'].append(kwa_shank)
    return json.dumps(kwa_full, indent=4)

def load_kwa_json(kwa_json):
    """Convert from KWA JSON into two NumPy arrays with the cluster colors and group colors."""
    if not kwa_json:
        return None
    kwa_dict = json.loads(kwa_json)
    auxdata = {}
    auxdata['shanks'] = {}
    auxdata['channels'] = {}
    auxdata['groups_of_channels'] = {}
    
    # load list of cluster and group colors for each shank
    if kwa_dict.get('shanks', {}):
        for kwa in kwa_dict['shanks']:
            shank = kwa['shank_index']
            cluster_colors = [int(o['color']) for o in kwa['clusters']]
            group_colors = [int(o['color']) for o in kwa['groups_of_clusters']]
            auxdata['shanks'][int(kwa['shank_index'])] = dict(cluster_colors=cluster_colors, 
                group_colors=group_colors)
    
    # load list of cluster and group colors for each shank
    if kwa_dict.get('channels', {}):
        for kwa in kwa_dict['channels']:
            channel = kwa.get('channel', {})
            channel_group = kwa.get('group', {})
            channel_color = kwa.get('color', {})
            channel_name = kwa.get('name', {})
            channel_visible = kwa.get('visible', {})
            auxdata['channels'][int(channel)] = dict(color=channel_color, 
                name=channel_name, group=channel_group, visible=channel_visible)

    if kwa_dict.get('groups_of_channels', {}):
        for kwa in kwa_dict['groups_of_channels']:
            channel_group = kwa.get('group', {})
            channel_group_color = kwa.get('color', {})
            channel_group_name = kwa.get('name', {})
            channel_group_visible = kwa.get('visible', {})
            auxdata['groups_of_channels'][int(channel_group)] = dict(color=channel_group_color, 
                name=channel_group_name, visible=channel_group_visible)

    return auxdata
    
def write_kwa(filename_kwa, kwa):
    kwa_json = kwa_to_json(kwa)
    with open(filename_kwa, 'w') as f:
        f.write(kwa_json)
    

