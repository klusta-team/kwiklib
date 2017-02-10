"""Object-oriented interface to an experiment's data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import os.path as op
import re
from itertools import chain
from collections import OrderedDict

import numpy as np
import pandas as pd
import tables as tb

from klusta.traces.waveform import WaveformLoader, SpikeLoader
from klusta.kwik.model import (_concatenate_virtual_arrays,
                               _dat_to_traces,
                               )
from klusta.traces.filter import apply_filter, bandpass_filter

from selection import select, slice_to_indices
from kwiklib.dataio.kwik import (get_filenames, open_files, close_files,
    add_spikes, add_cluster, add_cluster_group, remove_cluster,
    remove_cluster_group)
from kwiklib.dataio.utils import convert_dtype
from kwiklib.dataio.spikecache import SpikeCache
from kwiklib.utils.six import (iteritems, string_types, iterkeys,
    itervalues, next)
from kwiklib.utils.wrap import wrap
from kwiklib.utils.logger import warn, debug, info


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def _resolve_hdf5_path(files, path):
    """Resolve a HDF5 external link. Return the referred node (group or
    dataset), or None if it does not exist.

    Arguments:
      * files: a dict {type: file_handle}.
      * path: a string like "{type}/path/to/node" where `type` is one of
        `kwx`, `raw.kwd`, etc.

    """
    nodes = path.split('/')
    path_ext = '/' + '/'.join(nodes[1:])
    type = nodes[0]
    pattern = r'\{([a-zA-Z\._]+)\}'
    assert re.match(pattern, type)
    r = re.search(pattern, type)
    assert r
    type = r.group(1)
    # Resolve the link.
    file = files.get(type, None)
    if file:
        return file.getNode(path_ext)
    else:
        return None

def _get_child_id(child):
    id = child._v_name
    if id.isdigit():
        return int(id)
    else:
        return id

def _print_instance(obj, depth=0, name=''):
    # Handle the first element of the list/dict.
    r = []
    if isinstance(obj, (list, dict)):
        if not obj:
            r = []
            return r
        if isinstance(obj, list):
            sobj = obj[0]
            key = '0'
        elif isinstance(obj, dict):
            key, sobj = next(iteritems(obj))
        if isinstance(sobj, (list, dict, int, long, string_types, np.ndarray,
                      float)):
            r = []
        else:
            r = [(depth+1, str(key))] + _print_instance(sobj, depth+1)
    # Arrays do not have children.
    elif isinstance(obj, (np.ndarray, tb.EArray)):
        r = []
    # Handle class instances.
    elif hasattr(obj, '__dict__'):
        vs = vars(obj)
        if hasattr(obj, '__dir__'):
            vs.update({name: getattr(obj, name)
                        for name in dir(obj)
                            if name not in ('CLASS', 'TITLE', 'VERSION')})
        fields = {k: v
            for k, v in iteritems(vs)
                if not k.startswith('_')}
        r = list(chain(*[_print_instance(fields[n], depth=depth+1, name=str(n))
                for n in sorted(iterkeys(fields))]))
    else:
        r = []
    # Add the current object's display string.
    if name:
        if isinstance(obj, tb.EArray):
            s = name + ' [{dtype} {shape}]'.format(dtype=obj.dtype,
                shape=obj.shape)
        elif isinstance(obj, (string_types, int, long, float, tuple)) or obj is None:
            s = name + ' = ' + str(obj)
        else:
            s = name
        r = [(depth, s)] + r
    return r

class ArrayProxy(object):
    """Proxy to a view of an array."""
    def __init__(self, arr, col=None):
        self._arr = arr
        self._col = col
        self.dtype = arr.dtype

    @property
    def shape(self):
        return self._arr.shape[:-1]

    def __getitem__(self, item):
        if self._col is None:
            return self._arr[item]
        else:
            if isinstance(item, tuple):
                item += (self._col,)
                return self._arr[item]
            else:
                return self._arr[item, ..., self._col]


# -----------------------------------------------------------------------------
# Node wrappers
# -----------------------------------------------------------------------------
class Node(object):
    _files = None
    _kwik = None
    _node = None
    _root = None

    def __init__(self, files, node=None, root=None):
        self._files = files
        self._kwik = self._files.get('kwik', None)
        assert self._kwik is not None
        if node is None:
            node = self._kwik.root
        self._node = node
        self._root = root

    def _gen_children(self, container_name=None, child_class=None):
        """Return a dictionary {child_id: child_instance}."""
        # The container with the children is either the current node, or
        # a child of this node.
        if container_name is None:
            container = self._node
        else:
            container = self._node._f_getChild(container_name)
        l = [
            (_get_child_id(child), child_class(self._files, child, root=self._root))
                for child in container
            ]
        l = sorted(l, key=lambda (x,y): x)
        return OrderedDict(l)

    def _get_child(self, child_name):
        """Return the child specified by its name.
        If this child has a `hdf5_path` special, then the path is resolved,
        and the referred child in another file is returned.
        """
        child = self._node._f_getChild(child_name)
        try:
            # There's a link that needs to be resolved: return it.
            path = child._f_getAttr('hdf5_path')
            return _resolve_hdf5_path(self._files, path)
        except AttributeError:
            # No HDF5 external link: just return the normal child.
            return child

    def __getattr__(self, key):
        try:
            return self.__dict__[key]
        except:
            try:
                return self._node._f_getAttr(key)
            except AttributeError:
                warn(("{key} needs to be an attribute of "
                     "{node}").format(key=key, node=self._node._v_name))
                return None

    def __setattr__(self, key, value):
        try:
            self._node._f_getAttr(key)
            self._node._f_setAttr(key, value)
        except AttributeError:
            super(Node, self).__setattr__(key, value)

class NodeWrapper(object):
    """Like a PyTables node, but supports in addition: `node.attr`."""
    def __init__(self, node):
        self._node = node

    def __getitem__(self, key):
        return self._node[key]

    def __getattr__(self, key):
        # Do not override if key is an attribute of this class.
        if key.startswith('_'):
            try:
                return self.__dict__[key]
            # Accept nodewrapper._method if _method is a method of the PyTables
            # Node object.
            except KeyError:
                return getattr(self._node, key)
        try:
            # Return the wrapped node if the child is a group.
            attr = getattr(self._node, key)
            if isinstance(attr, tb.Group):
                return NodeWrapper(attr)
            else:
                return attr
        # Return the attribute.
        except:
            try:
                return self._node._f_getAttr(key)
            except AttributeError:
                # NOTE: old format
                if key == 'n_features_per_channel':
                    return self._node._f_getAttr('nfeatures_per_channel')
                warn(("{key} needs to be an attribute of "
                     "{node}").format(key=key, node=self._node._v_name))
                return None

    def __setattr__(self, key, value):
        if key.startswith('_'):
            self.__dict__[key] = value
            return
        # Ensure the key is an existing attribute to the current node.
        try:
            self._node._f_getAttr(key)
        except AttributeError:
            raise "{key} needs to be an attribute of {node}".format(
                key=key, node=self._node._v_name)
        # Set the attribute.
        self._node._f_setAttr(key, value)

    def __dir__(self):
        return sorted(dir(self._node) + self._node._v_attrs._v_attrnames)

    def __repr__(self):
        return self._node.__repr__()

class DictVectorizer(object):
    """This object serves as a vectorized proxy for a dictionary of objects
    that have individual fields of interest. For example: d={k: obj.attr1}.
    The object dv = DictVectorizer(d, 'attr1.subattr') can be used as:

        dv[3]
        dv[[1,2,5]]
        dv[2:4]

    """
    def __init__(self, dict, path):
        self._dict = dict
        self._path = path.split('.')

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def _get_path(self, key):
        """Resolve the path recursively for a given key of the dictionary."""
        val = self._dict[key]
        for p in self._path:
            val = getattr(val, p)
        return val

    def _set_path(self, key, value):
        """Resolve the path recursively for a given key of the dictionary,
        and set a value."""
        val = self._dict[key]
        for p in self._path[:-1]:
            val = getattr(val, p)
        setattr(val, key, value)

    def __getitem__(self, item):
        if isinstance(item, slice):
            item = slice_to_indices(item, lenindices=len(self._dict),
                                    keys=sorted(self._dict.keys()))
        if hasattr(item, '__len__'):
            return np.array([self._get_path(k) for k in item])
        else:
            return self._get_path(item)

    def __setitem__(self, item, value):
        if key.startswith('_'):
            self.__dict__[key] = value
            return
        if isinstance(item, slice):
            item = slice_to_indices(item, lenindices=len(self._dict))
        if hasattr(item, '__len__'):
            if not hasattr(value, '__len__'):
                value = [value] * len(item)
            for k, val in zip(item, value):
                self._set_path(k, value)
        else:
            return self._set_path(item, value)


def _read_traces(files, dtype=None, n_channels=None):
    kwd_path = None
    dat_path = None
    kwik = files['kwik']

    recordings = kwik.root.recordings
    traces = []
    # opened_files = []

    # HACK when there is no recordings: find a .dat file with the same
    # base name in the current directory.
    if not recordings:
        name = op.splitext(op.basename(kwik.filename))[0]
        p = op.join(op.dirname(op.realpath(kwik.filename)), name + '.dat')
        if op.exists(p):
            dat = _dat_to_traces(p, dtype=dtype or 'int16',
                                 n_channels=n_channels)
            traces.append(dat)

    for recording in recordings:
        # Is there a path specified to a .raw.kwd file which exists in
        # [KWIK]/recordings/[X]/raw? If so, open it.
        raw = recording.raw
        if 'hdf5_path' in raw._v_attrs:
            kwd_path = raw._v_attrs.hdf5_path[:-8]
            kwd = files['raw.kwd']
            if kwd is None:
                debug("%s not found, trying same basename in KWIK dir" %
                      kwd_path)
            else:
                debug("Loading traces: %s" % kwd_path)
                traces.append(kwd.root.recordings._f_getChild(str(recording._v_name)).data)
                # opened_files.append(kwd)
                continue
        # Is there a path specified to a .dat file which exists?
        if 'dat_path' in raw._v_attrs:
            dtype = kwik.root.application_data.spikedetekt._v_attrs.dtype[0]
            if dtype:
                dtype = np.dtype(dtype)

            n_channels = kwik.root.application_data.spikedetekt._v_attrs. \
                n_channels
            if n_channels:
                n_channels = int(n_channels)

            assert dtype is not None
            assert n_channels
            dat_path = raw._v_attrs.dat_path
            if not op.exists(dat_path):
                debug("%s not found, trying same basename in KWIK dir" %
                      dat_path)
                name = op.splitext(op.basename(kwik.filename))[0]
                dat_path = op.join(op.dirname(op.realpath(kwik.filename)), name + '.dat')
            if op.exists(dat_path):
                debug("Loading traces: %s" % dat_path)
                dat = _dat_to_traces(dat_path, dtype=dtype,
                                     n_channels=n_channels)
                traces.append(dat)
                # opened_files.append(dat)
                continue

    if not traces:
        warn("No traces found: the waveforms won't be available.")
    return _concatenate_virtual_arrays(traces)


# -----------------------------------------------------------------------------
# Experiment class and sub-classes.
# -----------------------------------------------------------------------------
class Experiment(Node):
    """An Experiment instance holds all information related to an
    experiment. One can access any information using a logical structure
    that is somewhat independent from the physical representation on disk.
    """
    def __init__(self, name=None, dir=None, files=None, mode='r', prm={}):
        """`name` must correspond to the basename of the files."""
        self.name = name
        self._dir = dir
        self.dir = dir
        self._mode = mode
        self._files = files
        self._prm = prm
        if self._files is None:
            self._files = open_files(self.name, dir=self._dir, mode=self._mode)
        def _get_filename(file):
            if file is None:
                return None
            else:
                return os.path.realpath(file.filename)
        self._filenames = {type: _get_filename(file)
            for type, file in iteritems(self._files)}
        super(Experiment, self).__init__(self._files)
        self._root = self._node

        # Ensure the version of the kwik format is exactly 2.
        assert self._root._f_getAttr('kwik_version') == 2

        self.application_data = NodeWrapper(self._root.application_data)
        # self.user_data = NodeWrapper(self._root.user_data)

        self.channel_groups = self._gen_children('channel_groups', ChannelGroup)
        self.recordings = self._gen_children('recordings', Recording)
        # self.event_types = self._gen_children('event_types', EventType)

        # Initialize the spike cache of all channel groups.
        for grp in self.channel_groups.itervalues():
            grp.spikes.init_cache()

    def gen_filename(self, extension):
        if extension.startswith('.'):
            extension = extension[1:]
        return os.path.splitext(self._filenames['kwik'])[0] + '.' + extension

    def __enter__(self):
        return self

    def close(self):
        if self._files is not None:
            close_files(self._files)

    def __repr__(self):
        n = "<Experiment '{name}'>".format(name=self.name)
        l = _print_instance(self, name=n)
        # print l
        return '\n'.join('    '*d + s for d, s in l)

    def __exit__ (self, type, value, tb):
        self.close()

class ChannelGroup(Node):
    def __init__(self, files, node=None, root=None):
        super(ChannelGroup, self).__init__(files, node, root=root)

        # self.application_data = NodeWrapper(self._node.application_data)
        # self.user_data = NodeWrapper(self._node.user_data)

        self.channels = self._gen_children('channels', Channel)
        self.clusters = ClustersNode(self._files, self._node.clusters, root=self._root)
        self.cluster_groups = ClusterGroupsNode(self._files, self._node.cluster_groups, root=self._root)

        self.spikes = Spikes(self._files, self._node.spikes, root=self._root)

class Spikes(Node):
    def __init__(self, files, node=None, root=None):
        super(Spikes, self).__init__(files, node, root=root)

        self.time_samples = self._node.time_samples
        self.time_fractional = self._node.time_fractional
        self.recording = self._node.recording
        self.clusters = Clusters(self._files, self._node.clusters, root=self._root)

        # Add concatenated time samples
        self.concatenated_time_samples = self._compute_concatenated_time_samples()

        self.channel_group_id = self._node._v_parent._v_name

        # Get large datasets, that may be in external files.
        # self.features_masks = self._get_child('features_masks')
        # self.waveforms_raw = self._get_child('waveforms_raw')
        # self.waveforms_filtered = self._get_child('waveforms_filtered')

        # Load features masks directly from KWX.
        g = self.channel_group_id
        path = '/channel_groups/{}/features_masks'.format(g)
        if files['kwx']:
            self.features_masks = files['kwx'].getNode(path)
        else:
            self.features_masks = None

        # Load raw data directly from raw data.
        traces = _read_traces(files)

        b = self._root.application_data.spikedetekt._f_getAttr('extract_s_before')
        a = self._root.application_data.spikedetekt._f_getAttr('extract_s_after')

        order = self._root.application_data.spikedetekt._f_getAttr('filter_butter_order')
        rate = self._root.application_data.spikedetekt._f_getAttr('sample_rate')
        low = self._root.application_data.spikedetekt._f_getAttr('filter_low')
        if 'filter_high_factor' in self._root.application_data.spikedetekt._v_attrs:
            high = self._root.application_data.spikedetekt._f_getAttr('filter_high_factor') * rate
        else:
            # NOTE: old format
            high = self._root.application_data.spikedetekt._f_getAttr('filter_high')
        b_filter = bandpass_filter(rate=rate,
                                   low=low,
                                   high=high,
                                   order=order)

        debug("Enable waveform filter.")

        def the_filter(x, axis=0):
            return apply_filter(x, b_filter, axis=axis)

        filter_margin = order * 3

        channels = self._root.channel_groups._f_getChild(self.channel_group_id)._f_getAttr('channel_order')
        _waveform_loader = WaveformLoader(n_samples=(b, a),
                                          traces=traces,
                                          filter=the_filter,
                                          filter_margin=filter_margin,
                                          scale_factor=.01,
                                          channels=channels,
                                          )
        self.waveforms_raw = SpikeLoader(_waveform_loader,
                                         self.concatenated_time_samples)
        self.waveforms_filtered = self.waveforms_raw

        nspikes = len(self.time_samples)

        if self.waveforms_raw is not None:
            self.nsamples, self.nchannels = self.waveforms_raw.shape[1:]

        if self.features_masks is None:
            self.features_masks = np.zeros((nspikes, 1, 1), dtype=np.float32)

        if len(self.features_masks.shape) == 3:
            self.features = ArrayProxy(self.features_masks, col=0)
            self.masks = ArrayProxy(self.features_masks, col=1)
        elif len(self.features_masks.shape) == 2:
            self.features = self.features_masks
            self.masks = None  #np.ones_like(self.features)
        self.nfeatures = self.features.shape[1]

    def _compute_concatenated_time_samples(self):
        t_rel = self.time_samples[:]
        recordings = self.recording[:]
        if len(recordings) == 0 and len(t_rel) > 0:
            recordings = np.zeros_like(t_rel)
        # Get list of recordings.
        recs = self._root.recordings
        recs = sorted([int(_._v_name) for _ in recs._f_listNodes()])
        # Get their start times.
        if not recs:
            return t_rel
        start_times = np.zeros(max(recs)+1, dtype=np.uint64)
        for r in recs:
            recgrp = getattr(self._root.recordings, str(r))
            sample_rate = recgrp._f_getAttr('sample_rate')
            start_time = recgrp._f_getAttr('start_time') or 0.
            start_times[r] = int(start_time * sample_rate)
        return t_rel + start_times[recordings]

    def add(self, **kwargs):
        """Add a spike. Only `time_samples` is mandatory."""
        add_spikes(self._files, channel_group_id=self.channel_group_id, **kwargs)

    def init_cache(self):
        """Initialize the cache for the features & masks."""
        self._spikecache = SpikeCache(
            # TODO: handle multiple clusterings in the spike cache here
            spike_clusters=self.clusters.main,
            features_masks=self.features_masks,
            waveforms_raw=self.waveforms_raw,
            waveforms_filtered=self.waveforms_filtered,
            # TODO: put this value in the parameters
            cache_fraction=1.,)

    def load_features_masks_bg(self, *args, **kwargs):
        return self._spikecache.load_features_masks_bg(*args, **kwargs)

    def load_features_masks(self, *args, **kwargs):
        return self._spikecache.load_features_masks(*args, **kwargs)

    def load_waveforms(self, *args, **kwargs):
        return self._spikecache.load_waveforms(*args, **kwargs)

    def __getitem__(self, item):
        raise NotImplementedError("""It is not possible to select entire spikes
            yet.""")

    def __len__(self):
        return self.time_samples.shape[0]

class Clusters(Node):
    """The parent of main, original, etc. Contains multiple clusterings."""
    def __init__(self, files, node=None, root=None):
        super(Clusters, self).__init__(files, node, root=root)
        # Each child of the Clusters group is assigned here.
        for node in self._node._f_iterNodes():
            setattr(self, node._v_name, node)

    def copy(self, clustering_from, clustering_to):
        spike_clusters_from = self._node._f_getChild(clustering_from)[:]
        clusters_to = self._node._f_getChild(clustering_to)
        clusters_to[:] = spike_clusters_from

        group_from = self._node._v_parent._v_parent.clusters._f_getChild(clustering_from)
        group_to = self._node._v_parent._v_parent.clusters._f_getChild(clustering_to)

        group_from._f_copy(newname=clustering_to, overwrite=True, recursive=True)

class Clustering(Node):
    """An actual clustering, with the cluster numbers for all spikes."""
    def __init__(self, files, node=None, root=None, child_class=None):
        super(Clustering, self).__init__(files, node, root=root)
        self._child_class = child_class
        self._update()

    def _update(self):
        self._dict = self._gen_children(child_class=self._child_class)
        self.color = DictVectorizer(self._dict, 'application_data.klustaviewa.color')

    def __getitem__(self, item):
        return self._dict[item]

    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return len(self._dict)

    def __contains__(self, v):
        return v in self._dict

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def iteritems(self):
        return self._dict.iteritems()

class ClustersClustering(Clustering):
    """An actual clustering, with color and group."""
    # def __init__(self, *args, **kwargs):
        # super(ClustersClustering, self).__init__(*args, **kwargs)
        # self.group = DictVectorizer(self._dict, 'cluster_group')

    def _update(self):
        self._dict = self._gen_children(child_class=self._child_class)
        self.color = DictVectorizer(self._dict, 'application_data.klustaviewa.color')
        self.group = DictVectorizer(self._dict, 'cluster_group')

    def add_cluster(self, id=None, color=None, **kwargs):
        channel_group_id = self._node._v_parent._v_parent._v_name
        clustering = self._node._v_name
        add_cluster(self._files, channel_group_id=channel_group_id,
                    color=color,
                    id=str(id), clustering=clustering, **kwargs)
        self._update()

    def remove_cluster(self, id=None,):
        channel_group_id = self._node._v_parent._v_parent._v_name
        clustering = self._node._v_name
        remove_cluster(self._files, channel_group_id=channel_group_id,
                       id=str(id), clustering=clustering)
        self._update()

class ClusterGroupsClustering(Clustering):
    def _update(self):
        self._dict = self._gen_children(child_class=self._child_class)
        self.color = DictVectorizer(self._dict, 'application_data.klustaviewa.color')
        self.name = DictVectorizer(self._dict, 'name')

    def add_group(self, id=None, color=None, name=None):
        channel_group_id = self._node._v_parent._v_parent._v_name
        clustering = self._node._v_name
        add_cluster_group(self._files, channel_group_id=channel_group_id,
                    color=color, name=name,
                    id=str(id), clustering=clustering, )
        self._update()

    def remove_group(self, id=None,):
        channel_group_id = self._node._v_parent._v_parent._v_name
        clustering = self._node._v_name
        remove_cluster_group(self._files, channel_group_id=channel_group_id,
                       id=str(id), clustering=clustering)
        self._update()

class ClustersNode(Node):
    """The parent of clustering types: main, original..."""
    def __init__(self, files, node=None, root=None):
        super(ClustersNode, self).__init__(files, node, root=root)
        # Each child of the group is assigned here.
        for node in self._node._f_iterNodes():
            setattr(self, node._v_name, ClustersClustering(self._files, node,
                child_class=Cluster, root=self._root))

class ClusterGroupsNode(Node):
    def __init__(self, files, node=None, root=None):
        super(ClusterGroupsNode, self).__init__(files, node, root=root)
        # Each child of the group is assigned here.
        for node in self._node._f_iterNodes():
            setattr(self, node._v_name, ClusterGroupsClustering(self._files, node, child_class=ClusterGroup))

class Channel(Node):
    def __init__(self, files, node=None, root=None):
        super(Channel, self).__init__(files, node, root=root)

        # self.application_data = NodeWrapper(self._node.application_data)
        # self.user_data = NodeWrapper(self._node.user_data)

class Cluster(Node):
    def __init__(self, files, node=None, root=None):
        super(Cluster, self).__init__(files, node, root=root)

        # self.cluster_group = self._node._v_attrs.cluster_group
        # self.mean_waveform_raw = self._node._v_attrs.mean_waveform_raw
        # self.mean_waveform_filtered = self._node._v_attrs.mean_waveform_filtered

        self.application_data = NodeWrapper(self._node.application_data)
        # self.color = self.application_data.klustaviewa.color
        # self.user_data = NodeWrapper(self._node.user_data)
        # self.quality_measures = NodeWrapper(self._node.quality_measures)

    def __getattr__(self, name):
        if name == 'cluster_group':
            def _process(cg):
                if hasattr(cg, '__len__'):
                    if len(cg) > 0:
                        return cg[0]
                    else:
                        return 0
                return cg
            return _process(self._node._v_attrs.cluster_group)
        return super(Cluster, self).__getattr__(name)

class ClusterGroup(Node):
    def __init__(self, files, node=None, root=None):
        super(ClusterGroup, self).__init__(files, node, root=root)

        # self.application_data = NodeWrapper(self._node.application_data)
        # self.user_data = NodeWrapper(self._node.user_data)

class Recording(Node):
    def __init__(self, files, node=None, root=None):
        super(Recording, self).__init__(files, node, root=root)

        # self.name = self._node._v_attrs.name
        # self.start_time = self._node._v_attrs.start_time
        # self.start_sample = self._node._v_attrs.start_sample
        # self.sample_rate = self._node._v_attrs.sample_rate
        # self.bit_depth = self._node._v_attrs.bit_depth
        # self.band_high = self._node._v_attrs.band_high
        # self.band_low = self._node._v_attrs.band_low

        # self.raw = self._get_child('raw')
        # self.high = self._get_child('high')
        # self.low = self._get_child('low')

        # self.user_data = NodeWrapper(self._node.user_data)

class EventType(Node):
    def __init__(self, files, node=None, root=None):
        super(EventType, self).__init__(files, node, root=root)

        self.events = Events(self._files, self._node.events)

        # self.application_data = NodeWrapper(self._node.application_data)
        # self.user_data = NodeWrapper(self._node.user_data)

class Events(Node):
    def __init__(self, files, node=None, root=None):
        super(Events, self).__init__(files, node, root=root)

        self.time_samples = self._node.time_samples
        self.recording = self._node.recording

        # self.user_data = NodeWrapper(self._node.user_data)
