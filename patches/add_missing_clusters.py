"""
This script fixes corrupted data files that have missing clusters in the kwik file.

What this script does is to add new cluster groups in the HDF5 file for every missing 
cluster. These missing clusters are added to the Unsorted group with a random color.

Call this script like this to fix a data file:

    python add_missing_clusters.py myfile.kwik

"""

import sys
from kwiklib import *

print "Loading..."
exp = Experiment(sys.argv[1], mode='a')

shanks = sorted(exp.channel_groups.keys())

for shank in shanks:
    cg = exp.channel_groups[shank]
    clusters = cg.clusters.main.keys()
    clusters_unique = np.unique(cg.spikes.clusters.main[:])
    # Find missing clusters in the kwik file.
    missing = sorted(set(clusters_unique)-set(clusters))

    # Add all missing clusters with a default color and "Unsorted" cluster group (group #3).
    for idx in missing:
        print "Adding missing cluster %d in shank %d." % (idx, shank)
        add_cluster(exp._files, channel_group_id='%d' % shank,
                    id=str(idx),
                    clustering='main',
                    cluster_group=3)

exp.close()
print "Done!"
