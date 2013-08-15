"""Unit tests for auxtools module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import pprint

import numpy as np
import numpy.random as rnd
import pandas as pd

from kwiklib.dataio import kwa_to_json, load_kwa_json

# -----------------------------------------------------------------------------
# KWA tests
# -----------------------------------------------------------------------------
def test_kwa_1():
    kwa = {}
    kwa['shanks'] = {1: {'cluster_colors': [1, 2, 5],
               'group_colors': [4],},
           2: {'cluster_colors': [6],
               'group_colors': [1, 3],}
               }
    kwa_json = kwa_to_json(kwa)
    kwa2 = load_kwa_json(kwa_json)
    
    for shank in (1, 2):
        for what in ('cluster_colors', 'cluster_colors'):
            assert kwa2['shanks'][shank][what] == kwa['shanks'][shank][what]
    
