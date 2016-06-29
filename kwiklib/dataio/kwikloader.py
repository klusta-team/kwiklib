"""This module provides utility classes and functions to load spike sorting
data sets."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import os.path
import shutil
import re
from collections import Counter

import numpy as np
import pandas as pd
import tables as tb

from loader import (Loader, default_group_info, reorder, renumber_clusters,
    default_cluster_info)
from klustersloader import (find_filenames, save_clusters, convert_to_clu,
    find_filename, find_filename_or_new)
from tools import (load_text, normalize,
    load_binary, load_pickle, save_text, get_array,
    first_row, load_binary_memmap)
from selection import (select, select_pairs, get_spikes_in_clusters,
    get_some_spikes_in_clusters, get_some_spikes, get_indices, pandaize)
from kwiklib.utils.logger import (debug, info, warn, exception, FileLogger,
    register, unregister)
from kwiklib.utils.colors import COLORS_COUNT, generate_colors
from kwiklib.dataio.kwik import add_cluster
from kwiklib.dataio.klusterskwik import klusters_to_kwik
from .experiment import Experiment


def add_missing_clusters(exp):

    shanks = sorted(exp.channel_groups.keys())

    for shank in shanks:
        cg = exp.channel_groups[shank]
        clusters = cg.clusters.main.keys()
        clusters_unique = np.unique(cg.spikes.clusters.main[:])
        # Find missing clusters in the kwik file.
        missing = sorted(set(clusters_unique)-set(clusters))

        # Add all missing clusters with a default color and "Unsorted" cluster group (group #3).
        for idx in missing:
            info("Adding missing cluster %d in shank %d." % (idx, shank))
            add_cluster(exp._files, channel_group_id='%d' % shank,
                        id=str(idx),
                        clustering='main',
                        cluster_group=3)


# -----------------------------------------------------------------------------
# HDF5 Loader
# -----------------------------------------------------------------------------
class KwikLoader(Loader):
    # TODO: change the clustering ('main' by default)

    def __init__(self, parent=None, filename=None, userpref=None):
        self.experiment = None
        super(KwikLoader, self).__init__(parent=parent, filename=filename, userpref=userpref)

    # Read functions.
    # ---------------
    def _report_progress_open(self, spike, nspikes, shank, nshanks):
        i = shank * 100 + float(spike)/nspikes*100
        n = nshanks * 100
        self.report_progress(i, n)

    def _consistency_check(self):
        exp = self.experiment
        chgrp = self.shank

        cg = exp.channel_groups[chgrp]
        clusters = cg.clusters.main.keys()
        clusters_unique = np.unique(cg.spikes.clusters.main[:])

        # Find missing clusters in the kwik file.
        missing = sorted(set(clusters_unique)-set(clusters))

        # Add all missing clusters with a default color and "Unsorted" cluster group (group #3).
        for idx in missing:
            warn("Consistency check: adding cluster %d in the kwik file" % idx)
            add_cluster(exp._files, channel_group_id='%d' % chgrp,
                        id=idx,
                        clustering='main',
                        cluster_group=3)

    def open(self, filename=None, shank=None):
        """Open everything."""
        if filename is None:
            filename = self.filename
        else:
            self.filename = filename
        dir, basename = os.path.split(filename)

        # Converting to kwik if needed
        # kwik = find_filename(basename, 'kwik', dir=dir)
        # xml = find_filename(basename, 'xml', dir=dir)
        # self.filename_clu = find_filename(basename, 'clu', dir=dir)
        self._filenames = find_filenames(filename)
        kwik = find_filename(basename, 'kwik', dir=dir)
        xml = self._filenames['xml']
        clu = self._filenames['clu']

        self.log_filename = find_filename_or_new(filename, 'kvlog', dir=dir)


        # Backup the .clu file.
        clu_original = find_filename_or_new(filename, 'clu_original')
        if os.path.exists(clu) and not os.path.exists(clu_original):
            shutil.copyfile(clu, clu_original)

        if not kwik:
            assert xml, ValueError("I need a valid .kwik file")
            return

        self.experiment = Experiment(basename, dir=dir, mode='a')

        # CONSISTENCY CHECK
        # add missing clusters
        add_missing_clusters(self.experiment)

        # TODO
        # self.initialize_logfile()
        # Load the similarity measure chosen by the user in the preferences
        # file: 'gaussian' or 'kl'.
        # Refresh the preferences file when a new file is opened.
        # USERPREF.refresh()
        self.similarity_measure = self.userpref['similarity_measure'] or 'gaussian'
        debug("Similarity measure: {0:s}.".format(self.similarity_measure))
        info("Opening {0:s}.".format(self.experiment.name))
        self.shanks = sorted(self.experiment.channel_groups.keys())

        self.freq = self.experiment.application_data.spikedetekt.sample_rate

        self.fetdim = self.experiment.application_data.spikedetekt.n_features_per_channel
        self.nsamples = self.experiment.application_data.spikedetekt.extract_s_before + self.experiment.application_data.spikedetekt.extract_s_after

        self.set_shank(shank or self.shanks[0])

    # Shank functions.
    # ----------------
    def get_shanks(self):
        """Return the list of shanks available in the file."""
        return self.shanks

    def set_shank(self, shank):
        """Change the current shank and read the corresponding tables."""
        if not shank in self.shanks:
            warn("Shank {0:d} is not in the list of shanks: {1:s}".format(
                shank, str(self.shanks)))
            return
        self.shank = shank

        # CONSISTENCY CHECK
        # self._consistency_check()

        self.nchannels = len(self.experiment.channel_groups[self.shank].channels)

        clusters = self.experiment.channel_groups[self.shank].spikes.clusters.main[:]
        self.clusters = pd.Series(clusters, dtype=np.int32)
        self.nspikes = len(self.clusters)

        self.features = self.experiment.channel_groups[self.shank].spikes.features
        self.masks = self.experiment.channel_groups[self.shank].spikes.masks
        self.waveforms = self.experiment.channel_groups[self.shank].spikes.waveforms_filtered

        if self.features is not None:
            nfet = self.features.shape[1]
            self.nextrafet = (nfet - self.nchannels * self.fetdim)
        else:
            self.nextrafet = 0

        # Load concatenated time samples: those are the time samples +
        # the start time of the corresponding recordings.
        spiketimes = self.experiment.channel_groups[self.shank].spikes.concatenated_time_samples[:] * (1. / self.freq)
        self.spiketimes = pd.Series(spiketimes, dtype=np.float64)
        self.duration = spiketimes[-1]

        self._update_data()

        self.read_clusters()

    def copy_clustering(self, clustering_from='original',
                        clustering_to='main'):
        clusters = self.experiment.channel_groups[self.shank].spikes.clusters
        clusters.copy(clustering_from, clustering_to)


    # Read contents.
    # ---------------------
    def get_probe_geometry(self):
        return np.array([c.position
            for c in self.experiment.channel_groups[self.shank].channels])

    def read_clusters(self):
        # Read the cluster info.
        clusters = self.experiment.channel_groups[self.shank].clusters.main.keys()
        cluster_groups = [c.cluster_group or 0 for c in self.experiment.channel_groups[self.shank].clusters.main.values()]

        # cluster_colors = [c.application_data.klustaviewa.color
        #     if c.application_data.klustaviewa.color is not None
        #     else 1
        #     for c in self.experiment.channel_groups[self.shank].clusters.main.values()]

        groups = self.experiment.channel_groups[self.shank].cluster_groups.main.keys()
        group_names = [g.name or 'Group' for g in self.experiment.channel_groups[self.shank].cluster_groups.main.values()]
        # group_colors = [g.application_data.klustaviewa.color or 1 for g in self.experiment.channel_groups[self.shank].cluster_groups.main.values()]

        # Create the cluster_info DataFrame.
        self.cluster_info = pd.DataFrame(dict(
            # color=cluster_colors,
            group=cluster_groups,
            ), index=clusters)
        # self.cluster_colors = self.cluster_info['color'].astype(np.int32)
        self.cluster_groups = self.cluster_info['group'].astype(np.int32)

        # Create the group_info DataFrame.
        self.group_info = pd.DataFrame(dict(
            # color=group_colors,
            name=group_names,
            ), index=groups)
        # self.group_colors = self.group_info['color'].astype(np.int32)
        self.group_names = self.group_info['name']


    # Writing capabilities.
    # ---------------------
    def set_cluster(self, spikes, cluster):
        if not hasattr(spikes, '__len__'):
            spikes = [spikes]

        self.experiment.channel_groups[self.shank].spikes.clusters.main[spikes] = cluster
        clusters = self.experiment.channel_groups[self.shank].spikes.clusters.main[:]
        self.clusters = pd.Series(clusters, dtype=np.int32)

        self._update_data()

    def set_cluster_groups(self, clusters, group):
        # self.cluster_groups.ix[clusters] = group
        if not hasattr(clusters, '__len__'):
            clusters = [clusters]

        clusters_gr = self.experiment.channel_groups[self.shank].clusters.main
        for cl in clusters:
            clusters_gr[cl].cluster_group = group

        self.read_clusters()

    def set_cluster_colors(self, clusters, color):
        # self.cluster_colors.ix[clusters] = color
        if not hasattr(clusters, '__len__'):
            clusters = [clusters]
        clusters_gr = self.experiment.channel_groups[self.shank].clusters.main
        for cl in clusters:
            clusters_gr[cl].application_data.klustaviewa.color = color

        self.read_clusters()

    def set_group_names(self, groups, name):
        # self.group_names.ix[groups] = name
        if not hasattr(groups, '__len__'):
            groups = [groups]
        groups_gr = self.experiment.channel_groups[self.shank].cluster_groups.main
        for gr in groups:
            groups_gr[gr].name = name

        self.read_clusters()

    def set_group_colors(self, groups, color):
        # self.group_colors.ix[groups] = color
        if not hasattr(groups, '__len__'):
            groups = [groups]

        groups_gr = self.experiment.channel_groups[self.shank].cluster_groups.main
        # for gr in groups:
        #     groups_gr[gr].application_data.klustaviewa.color = color

        self.read_clusters()


    # Add.
    def add_cluster(self, cluster, group, color):
        # if cluster not in self.cluster_groups.index:
            # self.cluster_groups = self.cluster_groups.append(
                # pd.Series([group], index=[cluster])).sort_index()
        # if cluster not in self.cluster_colors.index:
            # self.cluster_colors = self.cluster_colors.append(
                # pd.Series([color], index=[cluster])).sort_index()

        self.experiment.channel_groups[self.shank].clusters.main.add_cluster(
            id=cluster,
            # color=color,
            cluster_group=group)

        self.read_clusters()

    def add_clusters(self, clusters, groups):
        # if cluster not in self.cluster_groups.index:
            # self.cluster_groups = self.cluster_groups.append(
                # pd.Series([group], index=[cluster])).sort_index()
        # if cluster not in self.cluster_colors.index:
            # self.cluster_colors = self.cluster_colors.append(
                # pd.Series([color], index=[cluster])).sort_index()
        for cluster, group in zip(clusters, groups):
            self.experiment.channel_groups[self.shank].clusters.main.add_cluster(
                id=cluster, cluster_group=group)
        self.read_clusters()

    def add_group(self, group, name):
        # if group not in self.group_colors.index:
            # self.group_colors = self.group_colors.append(
                # pd.Series([color], index=[group])).sort_index()
        # if group not in self.group_names.index:
            # self.group_names = self.group_names.append(
                # pd.Series([name], index=[group])).sort_index()

        groups = self.experiment.channel_groups[self.shank].cluster_groups.main
        groups.add_group(id=group, name=name,)

        self.read_clusters()

    # Remove.
    def remove_cluster(self, cluster):
        if np.any(np.in1d(cluster, self.clusters)):
            raise ValueError(("Cluster {0:d} is not empty and cannot "
            "be removed.").format(cluster))

        self.experiment.channel_groups[self.shank].clusters.main.remove_cluster(
            id=cluster,)

        self.read_clusters()

    def remove_group(self, group):
        if np.any(np.in1d(group, self.cluster_groups)):
            raise ValueError(("Group {0:d} is not empty and cannot "
            "be removed.").format(group))

        self.experiment.channel_groups[self.shank].cluster_groups.main.remove_group(
            id=group,)

        self.read_clusters()

    # Access to the data: spikes
    # --------------------------
    def select(self, spikes=None, clusters=None):
        if clusters is not None:
            if not hasattr(clusters, '__len__'):
                clusters = [clusters]
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        self.spikes_selected = spikes
        self.clusters_selected = clusters

    # Log file.
    # ---------
    def initialize_logfile(self):
        self.logfile = FileLogger(self.filename_log, name='datafile',
            level=self.userpref['loglevel_file'])
        # Register log file.
        register(self.logfile)

    # Save.
    # -----
    def save(self, renumber=False):
        self.report_progress_save(1, 4)

        if renumber:
            self.renumber()
            self.clusters = self.clusters_renumbered
            self.cluster_info = self.cluster_info_renumbered
            self._update_data()

        # Save the clusters in the .clu file.
        clu = self._filenames['clu']
        clu_split = clu.split('.')
        clu_split[-1] = str(self.shank)
        clu = '.'.join(clu_split)
        save_clusters(clu,
            convert_to_clu(self.clusters, self.cluster_info['group']))

        self.report_progress_save(2, 4)

        # self.close()
        self.report_progress_save(3, 4)

        # self.open()
        self.report_progress_save(4, 4)


    # Close functions.
    # ----------------
    def close(self):
        """Close the kwik HDF5 file."""
        # if hasattr(self, 'kwik') and self.kwik.isopen:
            # self.kwik.flush()
            # self.kwik.close()
        if self.experiment is not None:
            self.experiment.close()
            self.experiment = None
        if hasattr(self, 'logfile'):
            unregister(self.logfile)


