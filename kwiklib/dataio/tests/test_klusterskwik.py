import os
import tables as tb
from mock_data import *
from kwiklib.dataio import Experiment, read_features, read_clusters
from kwiklib.dataio.klusterskwik import klusters_to_kwik

def test_conversion_1():
    
    # Convert klusters data to kwik.
    klusters_to_kwik(filename='test', dir=TEST_FOLDER)
    
    fet = read_features(os.path.join(TEST_FOLDER, 'test.fet.1'), 
                        nchannels, fetdim, freq, do_process=False)
    
    clu = read_clusters(os.path.join(TEST_FOLDER, 'test.clu.1'))
    
    with Experiment('test', dir=TEST_FOLDER, mode='r') as exp:
        
        # Check cluster / cluster group metadata.
        assert np.allclose(sorted(exp.channel_groups[1].clusters.main.keys()),
            range(2, 22))
        assert np.allclose(exp.channel_groups[1].clusters.main.color[:],
            range(1, 21))
        assert np.all(exp.channel_groups[1].clusters.main.group[:] == 3)
        
        # Check original == main.
        assert np.allclose(
            sorted(exp.channel_groups[1].clusters.main.keys()),
            sorted(exp.channel_groups[1].clusters.original.keys()),
            )
        assert np.allclose(
            exp.channel_groups[1].clusters.main.color[:],
            exp.channel_groups[1].clusters.original.color[:],
            )
        assert np.allclose(
            exp.channel_groups[1].clusters.main.group[:],
            exp.channel_groups[1].clusters.original.group[:],
        )
        
        # Test spike clusters.
        assert np.allclose(
            exp.channel_groups[1].spikes.clusters.main[:],
            clu)
        assert np.allclose(
            exp.channel_groups[1].spikes.clusters.main[:],
            exp.channel_groups[1].spikes.clusters.original[:])
        
        # Ensure features masks is contiguous.
        assert isinstance(exp.channel_groups[1].spikes.features_masks, tb.Array)
        assert not isinstance(exp.channel_groups[1].spikes.features_masks, tb.EArray)
        
        # Check features and waveforms.
        nspikes = len(exp.channel_groups[1].spikes.clusters.main[:])
        assert exp.channel_groups[1].spikes.features_masks.shape[0] == nspikes
        # No uspk file ==> no waveforms_raw
        assert exp.channel_groups[1].spikes.waveforms_raw.shape[0] == 0
        assert exp.channel_groups[1].spikes.waveforms_filtered.shape[0] == nspikes
        
        assert exp.channel_groups[1].spikes.time_samples[:].sum() > 0
        assert exp.channel_groups[1].spikes.features_masks[:].sum() > 0
        assert exp.channel_groups[1].spikes.waveforms_filtered[:].sum() > 0
        
        fet_kwik = exp.channel_groups[1].spikes.features[:]
        
        # Check equality between original and kwik (normalized) features array.
        fet = fet.ravel()
        fet_kwik = fet_kwik.ravel()
        ind = fet !=0
        d = (fet[ind] / fet_kwik[ind])
        assert d.max() - d.min() <= .1
        