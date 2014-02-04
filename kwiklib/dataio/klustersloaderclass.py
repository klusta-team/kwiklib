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

from loader import (Loader, default_group_info, reorder, renumber_clusters,
    default_cluster_info)
from tools import (load_text, normalize,
    load_binary, load_pickle, save_text, get_array,
    first_row, load_binary_memmap)
from selection import (select, select_pairs, get_spikes_in_clusters,
    get_some_spikes_in_clusters, get_some_spikes, get_indices)
from kwiklib.utils.logger import (register, unregister, FileLogger, 
    debug, info, warn)
from kwiklib.utils.colors import COLORS_COUNT, generate_colors
from klustersloaderclass import *

    
# -----------------------------------------------------------------------------
# Klusters Loader
# -----------------------------------------------------------------------------
class KlustersLoader(Loader):
    def open(self, filename):
        """Open a file."""
        self.filename = filename
        # Find the file index associated to the filename, or 1 by default.
        self.fileindex = find_index(filename) or 1
        self.find_filenames()
        self.save_original_clufile()
        self.read()
        
    def find_filenames(self):
        # """Find the filenames of the different files for the current
        # dataset."""
        for ext, filename in find_filenames(self.filename).iteritems():
            setattr(self, 'filename_' + ext, filename)
        
    def save_original_clufile(self):
        filename_clu_original = find_filename(self.filename, 'clu_original')
        if filename_clu_original is None:
            if os.path.exists(self.filename_clu):
                # Save the original clu file if it does not exist yet.
                with open(self.filename_clu, 'r') as f:
                    clu = f.read()
                with open(self.filename_clu.replace('.clu.', 
                    '.clu_original.'), 'w') as f:
                    f.write(clu)
            if os.path.exists(self.filename_aclu):
                # Save the original clu file if it does not exist yet.
                with open(self.filename_aclu, 'r') as f:
                    clu = f.read()
                with open(self.filename_aclu.replace('.aclu.', 
                    '.aclu_original.'), 'w') as f:
                    f.write(clu)
    
    
    # Internal read methods.
    # ----------------------
    def read_metadata(self):
        try:
            self.metadata = read_xml(self.filename_xml, self.fileindex)
        except:
            # Die if no XML file is available for this dataset, as it contains
            # critical metadata.
            raise IOError("The XML file is missing.")
            
        self.nsamples = self.metadata.get('nsamples')
        self.nchannels = self.metadata.get('nchannels')
        self.fetdim = self.metadata.get('fetdim')
        self.freq = self.metadata.get('freq')
        
    def read_probe(self):
        if self.filename_probe is None:
            info("No probe file has been found.")
            self.probe = None
        else:
            try:
                self.probe = read_probe(self.filename_probe, self.fileindex)
                info("Successfully loaded {0:s}".format(self.filename_probe))
            except Exception as e:
                info(("There was an error while loading the probe file "
                          "'{0:s}' : {1:s}").format(self.filename_probe,
                            e.message))
                self.probe = None
    
    def read_features(self):
        try:
            self.features, self.spiketimes = read_features(self.filename_fet,
                self.nchannels, self.fetdim, self.freq)
            info("Successfully loaded {0:s}".format(self.filename_fet))
        except IOError:
            raise IOError("The FET file is missing.")
        # Convert to Pandas.
        self.features = pd.DataFrame(self.features, dtype=np.float32)
        self.duration = self.spiketimes[-1]
        self.spiketimes = pd.Series(self.spiketimes, dtype=np.float32)
        
        # Count the number of spikes and save it in the metadata.
        self.nspikes = self.features.shape[0]
        self.metadata['nspikes'] = self.nspikes
        self.nextrafet = self.features.shape[1] - self.nchannels * self.fetdim
    
    def read_res(self):
        try:
            self.spiketimes_res = read_res(self.filename_res, self.freq)
            self.spiketimes_res = pd.Series(self.spiketimes_res, dtype=np.float32)
        except IOError:
            warn("The RES file is missing.")
    
    def read_clusters(self):
        try:
            # Try reading the ACLU file, or fallback on the CLU file.
            if os.path.exists(self.filename_aclu):
                self.clusters = read_clusters(self.filename_aclu)
                info("Successfully loaded {0:s}".format(self.filename_aclu))
            else:
                self.clusters = read_clusters(self.filename_clu)
                info("Successfully loaded {0:s}".format(self.filename_clu))
        except IOError:
            warn("The CLU file is missing.")
            # Default clusters if the CLU file is not available.
            self.clusters = np.zeros(self.nspikes, dtype=np.int32)
        # Convert to Pandas.
        self.clusters = pd.Series(self.clusters, dtype=np.int32)
        
        # Count clusters.
        self._update_data()
    
    def read_cluster_info(self):
        try:
            self.cluster_info = read_cluster_info(self.filename_acluinfo)
            info("Successfully loaded {0:s}".format(self.filename_acluinfo))
        except IOError:
            info("The CLUINFO file is missing, generating a default one.")
            self.cluster_info = default_cluster_info(self.clusters_unique)
                
        if not np.array_equal(self.cluster_info.index, self.clusters_unique):
            info("The CLUINFO file does not correspond to the loaded CLU file.")
            self.cluster_info = default_cluster_info(self.clusters_unique)
            
        self.cluster_colors = self.cluster_info['color'].astype(np.int32)
        self.cluster_groups = self.cluster_info['group'].astype(np.int32)
        
    def read_group_info(self):
        try:
            self.group_info = read_group_info(self.filename_groupinfo)
            info("Successfully loaded {0:s}".format(self.filename_groupinfo))
        except IOError:
            info("The GROUPINFO file is missing, generating a default one.")
            self.group_info = default_group_info()
        
        # Convert to Pandas.
        self.group_colors = self.group_info['color'].astype(np.int32)
        self.group_names = self.group_info['name']
        
    def read_masks(self):
        try:
            self.masks, self.masks_full = read_masks(self.filename_mask,
                                                     self.fetdim)
            info("Successfully loaded {0:s}".format(self.filename_mask))
        except IOError:
            warn("The MASKS/FMASKS file is missing.")
            # Default masks if the MASK/FMASK file is not available.
            self.masks = np.ones((self.nspikes, self.nchannels))
            self.masks_full = np.ones(self.features.shape)
        self.masks = pd.DataFrame(self.masks)
        self.masks_full = pd.DataFrame(self.masks_full)
    
    def read_waveforms(self):
        try:
            self.waveforms = read_waveforms(self.filename_spk, self.nsamples,
                                            self.nchannels)
            info("Successfully loaded {0:s}".format(self.filename_spk))
        except IOError:
            warn("The SPK file is missing.")
            self.waveforms = np.zeros((self.nspikes, self.nsamples, 
                self.nchannels))
        # Convert to Pandas.
        self.waveforms = pd.Panel(self.waveforms, dtype=np.float32)
    
    def read_dat(self):
        try:
            self.dat = read_dat(self.filename_dat, self.nchannels)
        except IOError:
            warn("The DAT file is missing.")
    
    def read_fil(self):
        try:
            self.fil = read_dat(self.filename_fil, self.nchannels)
        except IOError:
            warn("The FIL file is missing.")
    
    # def read_stats(self):
        # self.ncorrbins = 100 #SETTINGS.get('correlograms.ncorrbins', 100)
        # self.corrbin = .001  #SETTINGS.get('correlograms.corrbin', .001)

        
    # Log file.
    # ---------
    def initialize_logfile(self):
        # filename = self.filename_fet.replace('.fet.', '.kvwlg.')
        self.logfile = FileLogger(self.filename_kvwlg, name='datafile', 
            level=self.userpref['loglevel_file'])
        # Register log file.
        register(self.logfile)
        
    
    # Public methods.
    # ---------------
    def read(self):
        self.initialize_logfile()
        # Load the similarity measure chosen by the user in the preferences
        # file: 'gaussian' or 'kl'.
        # Refresh the preferences file when a new file is opened.
        # USERPREF.refresh()
        self.similarity_measure = self.userpref['similarity_measure'] or 'gaussian'
        debug("Similarity measure: {0:s}.".format(self.similarity_measure))
        info("Opening {0:s}.".format(self.filename))
        self.report_progress(0, 5)
        self.read_metadata()
        self.read_probe()
        self.report_progress(1, 5)
        self.read_features()
        self.report_progress(2, 5)
        self.read_res()
        self.read_clusters()
        self.report_progress(3, 5)
        self.read_cluster_info()
        self.read_group_info()
        self.read_masks()
        self.report_progress(4, 5)
        self.read_waveforms()
        self.report_progress(5, 5)
        # self.read_stats()
    
    def save(self, renumber=False):
        self.update_cluster_info()
        self.update_group_info()
        
        if renumber:
            self.renumber()
            clusters = get_array(self.clusters_renumbered)
            cluster_info = self.cluster_info_renumbered
        else:
            clusters = get_array(self.clusters)
            cluster_info = self.cluster_info
        
        # Save both ACLU and CLU files.
        save_clusters(self.filename_aclu, clusters)
        save_clusters(self.filename_clu, 
            convert_to_clu(clusters, cluster_info['group']))
        
        # Save CLUINFO and GROUPINFO files.
        save_cluster_info(self.filename_acluinfo, cluster_info)
        save_group_info(self.filename_groupinfo, self.group_info)
    
    def close(self):
        if hasattr(self, 'logfile'):
            unregister(self.logfile)
            
    def __del__(self):
        self.close()
        
    
# -----------------------------------------------------------------------------
# Memory Loader
# -----------------------------------------------------------------------------
class MemoryLoader(Loader):
    def __init__(self, parent=None, **kwargs):
        super(MemoryLoader, self).__init__(parent)
        self.read(**kwargs)
    
    
    # Internal read methods.
    # ----------------------
    def read_metadata(self, nsamples=None, nchannels=None, fetdim=None,
        freq=None):
        self.nsamples = nsamples
        self.nchannels = nchannels
        self.fetdim = fetdim
        self.freq = freq
        
    def read_probe(self, probe):
        try:
            self.probe = process_probe(probe)
        except Exception as e:
            info(("There was an error while loading the probe: "
                      "'{0:s}'").format(e.message))
            self.probe = None
    
    def read_features(self, features):
        self.features, self.spiketimes = process_features(features,
            self.nchannels, self.fetdim, self.freq)
        # Convert to Pandas.
        self.features = pd.DataFrame(self.features, dtype=np.float32)
        self.duration = self.spiketimes[-1]
        self.spiketimes = pd.Series(self.spiketimes, dtype=np.float32)
        
        # Count the number of spikes and save it in the metadata.
        self.nspikes = self.features.shape[0]
        self.nextrafet = self.features.shape[1] - self.nchannels * self.fetdim
        
    def read_clusters(self, clusters):
        self.clusters = process_clusters(clusters)
        # Convert to Pandas.
        self.clusters = pd.Series(self.clusters, dtype=np.int32)
        # Count clusters.
        self._update_data()
    
    def read_cluster_info(self, cluster_info):
        self.cluster_info = process_cluster_info(cluster_info)
                
        assert np.array_equal(self.cluster_info.index, self.clusters_unique), \
            "The CLUINFO file does not correspond to the loaded CLU file."
            
        self.cluster_colors = self.cluster_info['color'].astype(np.int32)
        self.cluster_groups = self.cluster_info['group'].astype(np.int32)
        
    def read_group_info(self, group_info):
        self.group_info = process_group_info(group_info)
        # Convert to Pandas.
        self.group_colors = self.group_info['color'].astype(np.int32)
        self.group_names = self.group_info['name']
        
    def read_masks(self, masks):
        self.masks, self.masks_full = process_masks(masks, self.fetdim)
        self.masks = pd.DataFrame(self.masks)
        self.masks_full = pd.DataFrame(self.masks_full)
    
    def read_waveforms(self, waveforms):
        self.waveforms = process_waveforms(waveforms, self.nsamples,
                                        self.nchannels)
        # Convert to Pandas.
        self.waveforms = pd.Panel(self.waveforms, dtype=np.float32)
    
    # def read_stats(self):
        # self.ncorrbins = 100 #SETTINGS.get('correlograms.ncorrbins', 100)
        # self.corrbin = .001 #SETTINGS.get('correlograms.corrbin', .001)
    
    
    # Public methods.
    # ---------------
    def read(self, nsamples=None, nchannels=None, fetdim=None,
            freq=None, probe=None, features=None, clusters=None,
            cluster_info=None, group_info=None, channel_info=None,
            channel_group_info=None, masks=None, waveforms=None):
        self.read_metadata(nsamples=nsamples, nchannels=nchannels,
            fetdim=fetdim, freq=freq)
        self.read_probe(probe)
        self.read_features(features)
        self.read_clusters(clusters)
        self.read_cluster_info(cluster_info)
        self.read_group_info(group_info)
        self.read_masks(masks)
        self.read_waveforms(waveforms)
        # self.read_stats()
    
    
    