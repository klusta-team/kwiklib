"""Probe tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from kwiklib.utils import Probe, python_to_pydict


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
prb = {0:
            {
                "channels": [0, 1, 2, 3],
                "graph": [[0, 1], [2, 3]],
                "geometry": {0: [0.1, 0.2], 1: [0.3, 0.4]}
            }, 1:
            {
                "channels": [4, 5, 6, 7],
                "graph": [[4, 5], [6, 7]],
                "geometry": {4: [0.1, 0.2], 5: [0.3, 0.4]}
            }
        }

def test_probe():
    probe = Probe(prb, name="My test probe")
    
    assert len(probe.channel_groups) == 2
    
    assert probe.channel_groups[0].channels == list(range(4))
    assert probe.channel_groups[0].graph == [(0, 1), (2, 3)]
    assert np.array_equal(probe.channel_groups[0].geometry_arr,
                          np.array([[.1, .2], [.3, .4]]))
    
    assert probe.channel_groups[1].channels == list(range(4, 8))
    assert probe.channel_groups[1].graph == [(4, 5), (6, 7)]
    assert np.array_equal(probe.channel_groups[1].geometry_arr,
                          np.array([(.1, .2), (.3, .4)]))
    
    for i in range(4):
        assert probe.channel_to_group[i] == 0
    for i in range(4, 8):
        assert probe.channel_to_group[i] == 1
    
    assert list(probe.adjacency_list[0]) == [1]
    assert list(probe.adjacency_list[1]) == [0]
    assert list(probe.adjacency_list[2]) == [3]
    assert list(probe.adjacency_list[3]) == [2]
    assert list(probe.adjacency_list[4]) == [5]
    assert list(probe.adjacency_list[5]) == [4]
    assert list(probe.adjacency_list[6]) == [7]
    assert list(probe.adjacency_list[7]) == [6]
    