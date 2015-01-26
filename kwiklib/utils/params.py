"""Handle user-specified and default parameters."""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import pprint

from kwiklib.utils import python_to_pydict, to_lower, get_pydict
from six import string_types, iteritems


# -----------------------------------------------------------------------------
# Python script <==> dictionaries conversion
# -----------------------------------------------------------------------------
def get_params(filename=None, **kwargs):
    """Return all the parameters, retrieved following this order of priority:
    
    * parameters specified as keyword arguments in this function,
    * parameters specified in the .PRM file given in `filename`,
    * default parameters.
    
    """
    # Extract sample_rate before loading the default parameters.
    # This is because some default parameters are expressed as a function
    # of the sample rate.
    sample_rate = get_pydict(filename).get('sample_rate', None) or kwargs['sample_rate']
    if 'sample_rate' not in kwargs:
        kwargs['sample_rate'] = sample_rate
    default = load_default_params(kwargs)
    params = get_pydict(filename=filename, 
                      pydict_default=default,
                      **kwargs)
    # Set waveforms_nsamples, which is defined as extract_s_before + extract_s_after
    params['waveforms_nsamples'] = params['extract_s_before'] + params['extract_s_after']
    return params


# -----------------------------------------------------------------------------
# Default parameters
# -----------------------------------------------------------------------------
def load_default_params(namespace=None):
    """Load default parameters, in a given namespace (empty by default)."""
    if namespace is None:
        namespace = {}
    # The default parameters are read in a namespace that must contain
    # sample_rate.
    assert namespace['sample_rate'] > 0
    folder = os.path.dirname(os.path.realpath(__file__))
    params_default_path = os.path.join(folder, 'params_default.py')
    with open(params_default_path, 'r') as f:
        params_default_python = f.read()
    params_default = python_to_pydict(params_default_python, namespace.copy())
    return to_lower(params_default)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def display_params(prm):
    return pprint.pformat(prm)
