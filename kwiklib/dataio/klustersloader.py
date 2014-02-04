"""This module provides utility classes and functions to load spike sorting
data sets."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import os.path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from loader import (default_group_info, reorder, renumber_clusters,
    default_cluster_info)
from tools import (load_text, normalize,
    load_binary, load_pickle, save_text, get_array,
    first_row, load_binary_memmap)
from selection import (select, select_pairs, get_spikes_in_clusters,
    get_some_spikes_in_clusters, get_some_spikes, get_indices)
from kwiklib.utils.logger import (register, unregister, FileLogger, 
    debug, info, warn)
from kwiklib.utils.colors import COLORS_COUNT, generate_colors


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def find_index(filename):
    """Search the file index of the filename, if any, or return None."""
    r = re.search(r"([^\n]+)\.([^\.]+)\.([0-9]+)$", filename)
    if r:
        return int(r.group(3))
    # If the filename has no index in it, and if the file does not actually
    # exist, return the index of an existing filename.
    # if not os.path.exists(filename):
    return find_index(find_filename(filename, 'fet'))

def find_indices(filename, dir='', files=[]):
    """Return the list of all indices for the given filename, present
    in the filename's directory."""
    # get the extension-free filename, extension, and file index
    # template: FILENAME.xxx.0  => FILENAME (can contain points), 0 (index)
    # try different patterns
    patterns = [r"([^\n]+)\.([^\.]+)\.([0-9]+)$",
                r"([^\n]+)\.([^\.]+)$"]
    for pattern in patterns:
        r = re.search(pattern, filename)
        if r:
            filename = r.group(1)
            # extension = r.group(2)
            break
    
    # get the full path
    if not dir:
        dir = os.path.dirname(filename)
    filename = os.path.basename(filename)
    # try obtaining the list of all files in the directory
    if not files:
        try:
            files = os.listdir(dir)
        except (WindowsError, OSError, IOError):
            raise IOError("Error when accessing '{0:s}'.".format(dir))
    
    # If the requested filename does not have a file index, then get the 
    # smallest available fileindex in the files list.
    fileindex_set = set()
    for file in files:
        r = re.search(r"([^\n]+)\.([^\.]+)\.([0-9]+)$", file)
        if r:
            if r.group(1) == filename:
                fileindex_set.add(int(r.group(3)))
        
    return sorted(fileindex_set)
          
def find_filename(filename, extension_requested, dir='', files=[]):
    """Search the most plausible existing filename corresponding to the
    requested approximate filename, which has the required file index and
    extension.
    
    Arguments:
    
      * filename: the full filename of an existing file in a given dataset
      * extension_requested: the extension of the file that is requested
    
    """
    
    # get the extension-free filename, extension, and file index
    # template: FILENAME.xxx.0  => FILENAME (can contain points), 0 (index)
    # try different patterns
    patterns = [r"([^\n]+)\.([^\.]+)\.([0-9]+)$",
                r"([^\n]+)\.([^\.]+)$"]
    fileindex = None
    for pattern in patterns:
        r = re.search(pattern, filename)
        if r:
            filename = r.group(1)
            extension = r.group(2)
            if len(r.groups()) >= 3:
                fileindex = int(r.group(3))
            # else:
                # fileindex = None
            break
    
    # get the full path
    if not dir:
        dir = os.path.dirname(filename)
    filename = os.path.basename(filename)
    # try obtaining the list of all files in the directory
    if not files:
        try:
            files = os.listdir(dir)
        except (WindowsError, OSError, IOError):
            raise IOError("Error when accessing '{0:s}'.".format(dir))
    
    # If the requested filename does not have a file index, then get the 
    # smallest available fileindex in the files list.
    if fileindex is None:
        fileindex_set = set()
        for file in files:
            r = re.search(r"([^\n]+)\.([^\.]+)\.([0-9]+)$", file)
            if r:
                fileindex_set.add(int(r.group(3)))
        if fileindex_set:
            fileindex = sorted(fileindex_set)[0]
    
    # try different suffixes
    if fileindex is not None:
        suffixes = [
                    '.{0:s}.{1:d}'.format(extension_requested, fileindex),
                    '.{0:s}'.format(extension_requested),
                    ]
    else:
        suffixes = [
                    # '.{0:s}.{1:d}'.format(extension_requested, fileindex),
                    '.{0:s}'.format(extension_requested),
                    ]
    
    # find the real filename with the longest path that fits the requested
    # filename
    for suffix in suffixes:
        filtered = []
        prefix = filename
        while prefix and not filtered:
            filtered = filter(lambda file: (file.startswith(prefix) and 
                file.endswith(suffix)), files)
            prefix = prefix[:-1]
        # order by increasing length and return the shortest
        filtered = sorted(filtered, cmp=lambda k, v: len(k) - len(v))
        if filtered:
            return os.path.join(dir, filtered[0])
    
    return None

def find_any_filename(filename, extension_requested, dir='', files=[]):
    # get the full path
    if not dir:
        dir = os.path.dirname(filename)
    
    # try obtaining the list of all files in the directory
    if not files:
        try:
            files = os.listdir(dir)
        except (WindowsError, OSError, IOError):
            raise IOError("Error when accessing '{0:s}'.".format(dir))
    
    filtered = filter(lambda f: f.endswith('.' + extension_requested), files)
    if filtered:
        return os.path.join(dir, filtered[0])
    
def find_filename_or_new(filename, extension_requested,
        have_file_index=True, dir='', files=[]):
    """Find an existing filename with a requested extension, or create
    a new filename based on an existing file."""
    # Find the filename with the requested extension.
    filename_found = find_filename(filename, extension_requested, dir=dir, files=files)
    # If it does not exist, find a file that exists, and replace the extension 
    # with the requested one.
    if not filename_found:
        if have_file_index:
            file, fileindex = os.path.splitext(filename)
            try:
                fileindex = int(fileindex[1:])
            except:
                # We request a filename with a file index but none exists.
                fileindex = 1
            file = '.'.join(file.split('.')[:-1])
            filename_new = "{0:s}.{1:s}.{2:d}".format(file, 
                extension_requested, int(fileindex))
        else:
            dots = filename.split('.')
            # Trailing file index?
            try:
                if int(dots[-1]) >= 0:
                    file = '.'.join(dots[:-2])
            except:
                file = '.'.join(dots[:-1])
            filename_new = "{0:s}.{1:s}".format(file, extension_requested)
        return filename_new
    else:
        return filename_found
    
def find_filenames(filename):
    """Find the filenames of the different files for the current
    dataset."""
    filenames = {}
    for ext in ['xml', 'fet', 'spk', 'uspk', 'res', 'dat',]:
        filenames[ext] = find_filename(filename, ext) or ''
    for ext in ['clu', 'aclu', 'cluinfo', 'acluinfo', 'groupinfo', 'kvwlg']:
        filenames[ext] = find_filename_or_new(filename, ext)
    filenames['probe'] = (find_filename(filename, 'probe') or
                          find_any_filename(filename, 'probe'))
    filenames['mask'] = (find_filename(filename, 'fmask') or
                         find_filename(filename, 'mask'))
    # HDF5 file format
    filenames.update(find_hdf5_filenames(filename))
    return filenames

def filename_to_triplet(filename):
    patterns = [r"([^\n]+)\.([^\.]+)\.([0-9]+)$",
                r"([^\n]+)\.([^\.]+)$"]
    fileindex = None
    for pattern in patterns:
        r = re.search(pattern, filename)
        if r:
            filename = r.group(1)
            extension = r.group(2)
            if len(r.groups()) >= 3:
                fileindex = int(r.group(3))
            return (filename, extension, fileindex)
    return (filename, )
    
def triplet_to_filename(triplet):
    return '.'.join(map(str, triplet))
    
def find_hdf5_filenames(filename):
    filenames = {}
    # Find KWIK and KWA files.
    for key in ['kwik', 'kwa']:
        filenames['hdf5_' + key] = os.path.abspath(
            find_filename_or_new(filename, key, have_file_index=False))
    # Find KWD files.
    for key in ['raw', 'low', 'high']:
        filenames['hdf5_' + key] = os.path.abspath(
            find_filename_or_new(filename, key + '.kwd', have_file_index=False))
    return filenames


# -----------------------------------------------------------------------------
# File reading functions
# -----------------------------------------------------------------------------
def read_xml(filename_xml, fileindex=1):
    """Read the XML file associated to the current dataset,
    and return a metadata dictionary."""
    
    tree = ET.parse(filename_xml)
    root = tree.getroot()
    
    d = {}

    ac = root.find('acquisitionSystem')
    if ac is not None:
        nc = ac.find('nChannels')
        if nc is not None:
            d['total_channels'] = int(nc.text)
        sr = ac.find('samplingRate')
        if sr is not None:
            d['rate'] = float(sr.text)

    sd = root.find('spikeDetection')
    if sd is not None:
        cg = sd.find('channelGroups')
        if cg is not None:
            # find the group corresponding to the fileindex
            g = cg.findall('group')[fileindex - 1]
            if g is not None:
                ns = g.find('nSamples')
                if ns is not None:
                    d['nsamples'] = int(ns.text)
                nf = g.find('nFeatures')
                if nf is not None:
                    d['fetdim'] = int(nf.text)
                c = g.find('channels')
                if c is not None:
                    d['nchannels'] = len(c.findall('channel'))
    
    if 'nchannels' not in d:
        d['nchannels'] = d['total_channels']
    
    # klusters tests
    metadata = dict(
        nchannels=d['nchannels'],
        nsamples=d['nsamples'],
        fetdim=d['fetdim'],
        freq=d['rate'])
    
    return metadata

# Features.
def process_features(features, fetdim, nchannels, freq, nfet=None):
    features = np.array(features, dtype=np.float32)
    nspikes, ncol = features.shape
    if nfet is not None:
        nextrafet = nfet - fetdim * nchannels
    else:
        nextrafet = ncol - fetdim * nchannels
            
    # get the spiketimes
    spiketimes = features[:,-1].copy()
    spiketimes *= (1. / freq)
    
    # normalize normal features while keeping symmetry
    features_normal = normalize(features[:,:fetdim * nchannels],
                                        symmetric=True)
    # TODO: put the following line in FeatureView: it is necessary for correct
    # normalization of the times.
    features_time = spiketimes.reshape((-1, 1)) * 1. / spiketimes[-1] * 2 - 1
    # features_time = spiketimes.reshape((-1, 1)) * 1. / spiketimes[-1]# * 2 - 1
    # normalize extra features without keeping symmetry
    if nextrafet > 1:
        features_extra = normalize(features[:,-nextrafet:-1],
                                            symmetric=False)
        features = np.hstack((features_normal, features_extra, features_time))
    else:
        features = np.hstack((features_normal, features_time))
    return features, spiketimes
    
def read_features(filename_fet, nchannels, fetdim, freq, do_process=True):
    """Read a .fet file and return the normalize features array,
    as well as the spiketimes."""
    try:
        features = load_text(filename_fet, np.int64, skiprows=1, delimiter=' ')
    except ValueError:
        features = load_text(filename_fet, np.float32, skiprows=1, delimiter='\t')
    if do_process:
        return process_features(features, fetdim, nchannels, freq, 
            nfet=first_row(filename_fet))
    else:
        return features
    
# Clusters.
def process_clusters(clusters):
    return clusters[1:]

def read_clusters(filename_clu):
    clusters = load_text(filename_clu, np.int32)
    return process_clusters(clusters)

# RES file.
def process_res(spiketimes, freq=None):
    if freq is None:
        return spiketimes
    else:
        return spiketimes * 1. / freq

def read_res(filename_res, freq=None):
    res = load_text(filename_res, np.int32)
    return process_res(res, freq)

# Cluster info.
def process_cluster_info(cluster_info):
    cluster_info = pd.DataFrame({'color': cluster_info[:, 1], 
        'group': cluster_info[:, 2]}, dtype=np.int32, index=cluster_info[:, 0])
    return cluster_info
    
def read_cluster_info(filename_acluinfo):
    # For each cluster (absolute indexing): cluster index, color index, 
    # and group index
    cluster_info = load_text(filename_acluinfo, np.int32)
    return process_cluster_info(cluster_info)
    
# Group info.
def process_group_info(group_info):
    group_info = pd.DataFrame(
        {'color': group_info[:, 1].astype(np.int32),
         'name': group_info[:, 2]}, index=group_info[:, 0].astype(np.int32))
    return group_info

def read_group_info(filename_groupinfo):
    # For each group (absolute indexing): color index, and name
    group_info = load_text(filename_groupinfo, str, delimiter='\t')
    return process_group_info(group_info)
    
# Masks.
def process_masks(masks_full, fetdim):
    masks = masks_full[:,:-1:fetdim]
    return masks, masks_full

def read_masks(filename_mask, fetdim):
    masks_full = load_text(filename_mask, np.float32, skiprows=1)
    return process_masks(masks_full, fetdim)
    
# Waveforms.
def process_waveforms(waveforms, nsamples, nchannels):
    waveforms = np.array(waveforms, dtype=np.float32)
    waveforms = normalize(waveforms, symmetric=True)
    waveforms = waveforms.reshape((-1, nsamples, nchannels))
    return waveforms

def read_waveforms(filename_spk, nsamples, nchannels):
    waveforms = np.array(load_binary(filename_spk), dtype=np.float32)
    n = waveforms.size
    if n % nsamples != 0 or n % nchannels != 0:
        waveforms = load_text(filename_spk, np.float32)
    return process_waveforms(waveforms, nsamples, nchannels)
    
# DAT.
def read_dat(filename_dat, nchannels, dtype=np.int16):
    nsamples = (os.path.getsize(filename_dat) // 
        (nchannels * np.dtype(dtype).itemsize))
    return load_binary_memmap(filename_dat, dtype=dtype,
                             shape=(nsamples, nchannels))

# Probe.
def process_probe(probe):
    return normalize(probe)

def read_probe(filename_probe, fileindex):
    """fileindex is the shank index."""
    if not filename_probe:
        return
    if os.path.exists(filename_probe):
        # Try the text-flavored probe file.
        try:
            probe = load_text(filename_probe, np.float32)
        except:
            # Or try the Python-flavored probe file (SpikeDetekt, with an
            # extra field 'geometry').
            try:
                ns = {}
                execfile(filename_probe, ns)
                probe = ns['geometry'][fileindex]
                probe = np.array([probe[i] for i in sorted(probe.keys())],
                                    dtype=np.float32)
            except:
                return None
        return process_probe(probe)


# -----------------------------------------------------------------------------
# File saving functions
# -----------------------------------------------------------------------------
def save_cluster_info(filename_cluinfo, cluster_info):
    cluster_info_array = np.hstack((cluster_info.index.reshape((-1, 1)), 
        cluster_info.values))
    save_text(filename_cluinfo, cluster_info_array)
    
def save_group_info(filename_groupinfo, group_info):
    group_info_array = np.hstack((group_info.index.reshape((-1, 1)), 
        group_info.values))
    save_text(filename_groupinfo, group_info_array, fmt='%s', delimiter='\t')
    
def save_clusters(filename_clu, clusters):
    save_text(filename_clu, clusters, header=len(np.unique(clusters)))

def convert_to_clu(clusters, cluster_groups):
    # cluster_groups = cluster_info['group']
    clusters_new = np.array(clusters, dtype=np.int32)
    for i in (0, 1):
        clusters_new[cluster_groups.ix[clusters] == i] = i
    # clusters_unique = np.unique(set(clusters_new).union(set([0, 1])))
    # clusters_renumbered = reorder(clusters_new, clusters_unique)
    # return clusters_renumbered
    return clusters_new
