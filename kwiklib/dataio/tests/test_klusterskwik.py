import tables as tb
from mock_data import *
from kwiklib.dataio import Experiment
from kwiklib.dataio.klusterskwik import klusters_to_kwik

def test_conversion_1():
    
    # Convert klusters data to kwik.
    klusters_to_kwik(filename='test', dir=TEST_FOLDER)
    
    with Experiment('test', dir=TEST_FOLDER, mode='r') as exp:
        
        # Check cluster / cluster group metadata.
        assert np.allclose(sorted(exp.channel_groups[1].clusters.main.keys()),
            range(2, 22))
        assert np.allclose(exp.channel_groups[1].clusters.main.color[:],
            range(1, 21))
        assert np.all(exp.channel_groups[1].clusters.main.group[:] == 3)
        
        # Ensure features masks is contiguous.
        assert isinstance(exp.channel_groups[1].spikes.features_masks, tb.Array)
        assert not isinstance(exp.channel_groups[1].spikes.features_masks, tb.EArray)
        
        nspikes = len(exp.channel_groups[1].spikes.clusters.main[:])
        assert exp.channel_groups[1].spikes.features_masks.shape[0] == nspikes
        assert exp.channel_groups[1].spikes.waveforms_raw.shape[0] == nspikes
        assert exp.channel_groups[1].spikes.waveforms_filtered.shape[0] == nspikes
        
        