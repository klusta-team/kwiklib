"""This module provides functions used to generate and load probe files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import pprint
import itertools

from tools import MemMappedText, MemMappedBinary


# -----------------------------------------------------------------------------
# Probe file functions
# -----------------------------------------------------------------------------
def flatten(l):
    return sorted(set([item for sublist in l for item in sublist]))

def generate_probe(channel_groups, topology='linear'):
    """channel_groups is a dict {channel_group: nchannels}."""
    
    if not isinstance(channel_groups, dict):
        channel_groups = {0: channel_groups}
    
    groups = sorted(channel_groups.keys())
    r = {}
    curchannel = 0
    for i in range(len(groups)):
        id = groups[i]  # channel group index
        n = channel_groups[id]  # number of channels
        
        channels = range(curchannel, curchannel + n)
        
        if topology == 'linear':
            graph = [[ch, ch + 1] for ch in channels[:-1]]
        elif topology == 'complete':
            graph = map(list, list(itertools.product(channels, repeat=2)))
        
        geometry = {channels[_]: [float(i), float(_)]
                    for _ in range(n)}
        
        d = {'channels': channels,
             'graph': graph,
             'geometry': geometry,
            }
        r[id] = d
        curchannel += n
    return r     
    
def load_probe(filename):
    prb = {}
    execfile(filename, {}, prb)
    return prb['channel_groups']
    
def save_probe(filename, prb):
    with open(filename, 'w') as f:
        f.write('channel_groups = ' + str(prb))
    
    