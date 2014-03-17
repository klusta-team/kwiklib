import tables as tb
from mock_data import *
from kwiklib.dataio import Experiment
from kwiklib.dataio.klusterskwik import klusters_to_kwik

def test_conversion_1():
    
    klusters_to_kwik(filename='test', dir=TEST_FOLDER)
    
    with Experiment('test', dir=TEST_FOLDER, mode='r') as exp:
        # Ensure features masks is contiguous
        assert isinstance(exp.channel_groups[1].spikes.features_masks, tb.Array)
        assert not isinstance(exp.channel_groups[1].spikes.features_masks, tb.EArray)
        
        