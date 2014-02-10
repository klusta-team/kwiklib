import numpy as np
import tables as tb

class ReSelect(object):
    def __init__(self, index):
        self.index = index
        
class Loop(object):
    def __init__(self, index):
        self.index = index
        
def determine_selection(arr, index):
    # DEBUG
    # if not hasattr(arr, chunkshape):
        # return arr[index]
    cs = arr.chunkshape
    # Make sure index is a tuple...
    if not isinstance(index, tuple):
        return select(arr, (index,))
    assert isinstance(index, tuple)
    assert len(index) <= arr.ndim
    # ... with as many elements as arr has dimensions.
    if len(index) < arr.ndim:
        index = index + (slice(None),) * (arr.ndim - len(index))
    index = list(index)
    # Determine, for each axis, how to select.
    for i in range(len(index)):
        if isinstance(index[i], (np.ndarray, list)):
            if cs[i] > 2:
                index[i] = ReSelect(index[i])
            else:
                index[i] = Loop(index[i])
    return tuple(index)
        
def select(arr, index):
    index = determine_selection(index)
    if arr.ndim == 1:
        if isinstance(index, ReSelect):
            return arr[:][index.index]
        elif isinstance(index, Loop):
            return np.concatenate([arr[i] for i in index.index], axis=0)
        else:
            return arr[index]
    elif arr.ndim == 2:
        if isinstance(index[0], ReSelect) or isinstance(index[1], ReSelect):
            if isinstance(index[0], ReSelect):
                return arr[:,index[1]][index[0],...]
            else:
                return arr[index[0],:][...,index[1]]
        elif isinstance(index[0], Loop) or isinstance(index[1], Loop):
            if isinstance(index[0], Loop):
                return np.concatenate([arr[i,index[1]] for i in index[0].index], axis=0)
            else:
                return np.concatenate([arr[index[0],i] for i in index[1].index], axis=1)
        else:
            return arr[index]
    elif arr.ndim == 3:
        if isinstance(index[0], ReSelect) or isinstance(index[1], ReSelect) or isinstance(index[2], ReSelect):
            if isinstance(index[0], ReSelect):
                return arr[:,index[1],index[2]][index[0],...]
            elif isinstance(index[1], ReSelect):
                return arr[index[0],:,index[2]][:,index[1],:]
            else:
                return arr[index[0],index[1],:][:,:,index[2]]
        elif isinstance(index[0], Loop) or isinstance(index[1], Loop) or isinstance(index[2], Loop):
            if isinstance(index[0], Loop):
                return np.concatenate([arr[i,index[1],index[2]] for i in index[0].index], axis=0)
            elif isinstance(index[1], Loop):
                return np.concatenate([arr[index[0],i,index[2]] for i in index[1].index], axis=1)
            else:
                return np.concatenate([arr[index[0],index[1],i] for i in index[2].index], axis=2)
        else:
            return arr[index]
        
class ArrayWrapper(object):
    def __init__(self, arr):
        self._arr = arr
        self.ndim = arr.ndim
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.chunkshape = arr.chunkshape
    
    def __getitem__(self, index):
        return select(self._arr, index)
    
    def __setitem__(self, index, value):
        self._arr[index] = value

if __name__ == '__main__':
    pass
    
        