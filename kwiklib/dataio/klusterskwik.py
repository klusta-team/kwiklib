"""This module provides functions used to write HDF5 files in the new file
format."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# import json
import os
import tables
import time
import shutil
from collections import OrderedDict

import numpy as np
# import matplotlib.pyplot as plt

from probe import old_to_new
# from params import paramsxml_to_json
from klustersloader import (find_filenames, find_index, read_xml,
    filename_to_triplet, triplet_to_filename, find_indices,
    find_hdf5_filenames, find_filename, find_any_filename, 
    find_filename_or_new,
    read_clusters, read_cluster_info, read_group_info,)
from loader import (default_cluster_info, default_group_info)
# from auxtools import kwa_to_json, write_kwa
from tools import MemMappedText, MemMappedBinary
from utils import convert_dtype
from kwik import (create_files, open_files, close_files, add_spikes, 
    to_contiguous, add_cluster, add_cluster_group)
from probe import generate_probe


# -----------------------------------------------------------------------------
# Klusters files readers
# -----------------------------------------------------------------------------
def open_klusters_oneshank(filename):
    filenames = find_filenames(filename)
    fileindex = find_index(filename)
    
    # Open small Klusters files.
    data = {}
    metadata = read_xml(filenames['xml'], fileindex)
    data['clu'] = read_clusters(filenames['clu'])
    
    # Read .aclu data.
    if 'aclu' in filenames and os.path.exists(filenames['aclu']):
        data['aclu'] = read_clusters(filenames['aclu'])
    else:
        data['aclu'] = data['clu']
        
    # Read .acluinfo data.
    if 'acluinfo' in filenames and os.path.exists(filenames['acluinfo']):
        data['acluinfo'] = read_cluster_info(filenames['acluinfo'])
    # If the ACLUINFO does not exist, try CLUINFO (older file extension)
    elif 'cluinfo' in filenames and os.path.exists(filenames['cluinfo']):
        data['acluinfo'] = read_cluster_info(filenames['cluinfo'])
    else:
        data['acluinfo'] = default_cluster_info(np.unique(data['aclu']))
        
    # Read group info.
    if 'groupinfo' in filenames and os.path.exists(filenames['groupinfo']):
        data['groupinfo'] = read_group_info(filenames['groupinfo'])
    else:
        data['groupinfo'] = default_group_info()
    
    # Find out the number of columns in the .fet file.
    with open(filenames['fet'], 'r') as f:
        f.readline()
        # Get the number of non-empty columns in the .fet file.
        data['fetcol'] = len([col for col in f.readline().split(' ') if col.strip() != ''])
    
    metadata['nspikes'] = len(data['clu'])
    data['fileindex'] = fileindex

    # Open big Klusters files.
    data['fet'] = MemMappedText(filenames['fet'], np.int64, skiprows=1)
    if 'spk' in filenames and os.path.exists(filenames['spk'] or ''):
        data['spk'] = MemMappedBinary(filenames['spk'], np.int16, 
            rowsize=metadata['nchannels'] * metadata['nsamples'])
    if 'uspk' in filenames and os.path.exists(filenames['uspk'] or ''):
        data['uspk'] = MemMappedBinary(filenames['uspk'], np.int16, 
            rowsize=metadata['nchannels'] * metadata['nsamples'])
    if 'mask' in filenames and os.path.exists(filenames['mask'] or ''):
        data['mask'] = MemMappedText(filenames['mask'], np.float32, skiprows=1)

    # data['metadata'] = metadata
    data.update(metadata)
    
    return data
    
def open_klusters(filename):
    indices = find_indices(filename)
    triplet = filename_to_triplet(filename)
    filenames_shanks = {}
    for index in indices:
        filenames_shanks[index] = triplet_to_filename(triplet[:2] + (index,))
    klusters_data = {index: open_klusters_oneshank(filename) 
        for index, filename in filenames_shanks.iteritems()}
    shanks = filenames_shanks.keys()
           
    # Find the dataset filenames and load the metadata.
    filenames = find_filenames(filename)
    # Metadata common to all shanks.
    metadata = read_xml(filenames['xml'], 1)
    # Metadata specific to each shank.
    metadata.update({shank: read_xml(filenames['xml'], shank)
        for shank in shanks})
    metadata['shanks'] = sorted(shanks)
    metadata['has_masks'] = (('mask' in filenames 
                                    and filenames['mask'] is not None) or (
                                  'fmask' in filenames 
                                    and filenames['fmask'] is not None
                                  ))
    
    klusters_data['name'] = triplet[0]
    klusters_data['metadata'] = metadata
    klusters_data['shanks'] = shanks
    klusters_data['filenames'] = filenames
    
    # Load probe file.
    filename_probe = filenames['probe']
    # It no probe file exists, create a default, linear probe with the right
    # number of channels per shank.
    if not filename_probe:
        # Generate a probe filename.
        filename_probe = find_filename_or_new(filename, 'default.probe',
            have_file_index=False)
        shanks = {shank: klusters_data[shank]['nchannels']
            for shank in filenames_shanks.keys()}
        probe_python = generate_probe(shanks, 'complete')
        # with open(filename_probe, 'w') as f:
            # f.write(probe_python)
        # save_probe(filename_probe, probe_python)
        klusters_data['prb'] = probe_python
    else:
        probe_ns = {}
        execfile(filename_probe, {}, probe_ns)
        klusters_data['probe'] = probe_ns
    
    return klusters_data

def probe_to_prb(probe):
    return old_to_new(probe)
    
def metadata_to_prm(metadata):
    return dict(
        nchannels=metadata['nchannels'],
        sample_rate=metadata['freq'],
        waveforms_nsamples=metadata['nsamples'],
        nfeatures_per_channel=metadata['fetdim'],
        has_masks=metadata['has_masks'],
    )
   

# -----------------------------------------------------------------------------
# HDF5 writer
# ----------------------------------------------------------------------------- 
def klusters_to_kwik(filename=None, dir='.', progress_report=None):
    with KwikWriter(filename, dir=dir) as f:
        # Callback function for progress report.
        if progress_report is not None:
            f.progress_report(progress_report)
        f.convert()

class KwikWriter(object):
    def __init__(self, filename=None, dir=None):
        self._progress_callback = None
        self.filename = filename
        self.dir = dir
        if dir:
            self.filename = os.path.join(dir, filename)
        
    def __enter__(self):
        if self.filename:
            self.open(self.filename)
        return self
            
    def open(self, filename=None):
        if filename is not None:
            self.filename = filename
        self.klusters_data = open_klusters(self.filename)
        self.filenames = self.klusters_data['filenames']
        self.name = self.klusters_data['name']
        
        # Backup the original CLU file.
        filename_clu_original = find_filename_or_new(self.filename, 'clu_original')
        shutil.copyfile(self.filenames['clu'], filename_clu_original)
        
        if 'probe' in self.klusters_data:
            prb = probe_to_prb(self.klusters_data['probe'])
        else:
            prb = self.klusters_data['prb']
        prm = metadata_to_prm(self.klusters_data['metadata'])
        
        for chgrp in prb.keys():
            prb[chgrp]['nfeatures'] = self.klusters_data[chgrp]['fetcol']
        
        self.filenames_kwik = create_files(self.name, prm=prm, prb=prb)
        self.files = open_files(self.name, mode='a')
        
        self.shanks = sorted([key for key in self.klusters_data.keys() 
            if isinstance(key, (int, long))])
        self.shank = self.shanks[0]
        self.spike = 0
        
    def read_next_spike(self):
        if self.spike >= self.klusters_data[self.shank]['nspikes']:
            return {}
        data = self.klusters_data[self.shank]
        read = {}
        read['cluster'] = data['aclu'][self.spike]
        fet = data['fet'].next()
        read['time'] = fet[-1]
        read['fet'] = convert_dtype(fet, np.float32)
        if 'spk' in data:
            read['spk'] = data['spk'].next()
        if 'uspk' in data:
            read['uspk'] = data['uspk'].next()
        if 'mask' in data:
            read['mask'] = data['mask'].next()
        # else:
            # read['mask'] = np.ones_like(read['fet'])
        self.spike += 1
        return read
        
    def write_spike(self, read):
        wr = read.get('uspk', None)
        wf = read.get('spk', None)
        
        if wr is not None:
            wr = wr.reshape((1, -1, self.nchannels))
        if wf is not None:
            wf = wf.reshape((1, -1, self.nchannels))
        
        add_spikes(self.files, channel_group_id=str(self.shank),
            time_samples=read['time'],
            cluster=read['cluster'], 
            cluster_original=read['cluster'],
            features=read['fet'], 
            masks=read.get('mask', None),
            waveforms_raw=wr, 
            waveforms_filtered=wf,
            fill_empty=False)

    def report_progress(self):
        if self._progress_callback:
            self._progress_callback(
                self.spike, 
                self.klusters_data[self.shank]['nspikes'], 
                self.shanks.index(self.shank),
                len(self.shanks))
        
    def convert(self):
        """Convert the old file format to the new HDF5-based format."""
        # Convert in HDF5 by going through all spikes.
        for self.shank in self.shanks:
            self.nchannels = self.klusters_data[self.shank]['nchannels']
            
            # Write cluster info.
            acluinfo = self.klusters_data[self.shank]['acluinfo']
            groupinfo = self.klusters_data[self.shank]['groupinfo']
            
            for clustering in ('main', 'original'):
                for clu, info in acluinfo.iterrows():
                    add_cluster(self.files,
                        channel_group_id=str(self.shank), 
                        id=str(clu),
                        cluster_group=info['group'],
                        color=info['color'],
                        clustering=clustering)
                
                # Write cluster group info.
                for clugrp, info in groupinfo.iterrows():
                    add_cluster_group(self.files,
                        channel_group_id=str(self.shank), 
                        id=str(clugrp), 
                        name=info['name'], 
                        color=info['color'],
                        clustering=clustering)
            
            # Write spike data.
            self.spike = 0
            read = self.read_next_spike()
            self.report_progress()
            while read:
                self.write_spike(read)
                read = self.read_next_spike()
                self.report_progress()
                
            # Convert features masks array from chunked to contiguous
            node = self.files['kwx'].root.channel_groups.__getattr__(str(self.shank)).features_masks
            to_contiguous(node, self.spike)
        
    def progress_report(self, fun):
        self._progress_callback = fun
        return fun
        
    def close(self):
        """Close all files."""
        
        # Close the memory-mapped Klusters files.
        if hasattr(self, 'shanks'):
            for shank in self.shanks:
                for data in self.klusters_data[shank]:
                    if isinstance(data, (MemMappedBinary, MemMappedText)):
                        data.close()
        
        close_files(self.files)
        
    def __del__(self):
        self.close()
        
    def __exit__(self, exception_type, exception_val, trace):
        self.close()


