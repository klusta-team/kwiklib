from mock_data import *
from kwiklib.dataio import Experiment
from kwiklib.dataio.klusterskwik import klusters_to_kwik

def test_conversion_1():
    
    klusters_to_kwik(filename='test', dir=TEST_FOLDER)
    
    with Experiment('test', dir=TEST_FOLDER, mode='r') as exp:
        print exp