import os
import time
import h5py
import numpy as np

def create_contiguous(filename):
    with h5py.File(filename, "w") as f:
        a = f.create_dataset('/test', dtype=np.float32, shape=(n,k))
        n_ = n//10
        for i in range(10):
            print i,
            a[i*n_:(i+1)*n_,...] = np.random.rand(n_, k)
            
n, k = 5000000, 100
filename_contiguous = 'features_contiguous.h5'
if not os.path.exists(filename_contiguous):
    create_contiguous(filename_contiguous)

ind = np.random.randint(size=50000, low=0, high=n)
ind = np.unique(ind)

with h5py.File(filename_contiguous, "r") as f:
    a = f['/test']
    out = np.empty((len(ind),) +  a.shape[1:], dtype=a.dtype)
    t0 = time.clock()
    for j, i in enumerate(ind):
        out[j:j+1,...] = a[i:i+1,...]
    print
    print "%.1f s" % (time.clock() - t0)
