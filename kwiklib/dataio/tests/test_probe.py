"""Test probe functions."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import tempfile

from kwiklib.dataio.probe import generate_probe, load_probe, save_probe


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_linear_probe_0():
    probe = generate_probe({})
    assert probe == {}

def test_linear_probe_1():
    probe = generate_probe({0: 8})
    assert len(probe) == 1
    assert probe[0]['channels'] == range(8)
    assert probe[0]['graph']
    assert probe[0]['geometry']

def test_linear_probe_2():
    probe = generate_probe({0: 8, 1: 6, 3: 4})
    assert len(probe) == 3
    
    assert probe[0]['channels'] == range(8)
    
    assert probe[1]['channels'] == range(8, 14)
    
    assert probe[3]['channels'] == range(14, 18)

def test_linear_probe_3():
    probe = generate_probe({3: 8})
    assert len(probe) == 1
    assert probe[3]['channels'] == range(8)

def test_complete_probe_1():
    probe = generate_probe({3: 8}, 'complete')
    assert len(probe) == 1
    assert probe[3]['channels'] == range(8)
    for i in range(8):
        for j in range(i+1, 8):
            assert [i, j] in probe[3]['graph']

def test_load_probe_1():
    dir = tempfile.gettempdir()
    filename = os.path.join(dir, 'probe.py')
    
    probe = generate_probe(8)
    save_probe(filename, probe)
    
    probe_bis = load_probe(filename)
    
    assert probe == probe_bis
    