"""
This script fixes data files that were recorded with a version of spikedetekt2
that contained a minor bug. This bug prevented the program to save the start time
of each recording. As a result, KlustaViewa was not able to display correlograms
because the spike times were not sorted (the spike times were in local time coordinate
rather than absolute time coordinate, starting again at 0 at each new recording).

What this script does is to add these start_time fields in the kwik file by
looking at the spike times. It adds a dead time of 10 seconds before two consecutive
recordings.

Call this script like this to fix a data file:

    python add_start_time.py myfile.kwik

"""
import sys
import numpy as np
from kwiklib import Experiment

exp = Experiment(sys.argv[1], mode='a')

shanks = sorted(exp.channel_groups.keys())
recordings = sorted(exp.recordings.keys())

print "Loading..."
changes = {}
for chgroup in shanks:
    spkrec = exp.channel_groups[chgroup].spikes.recording[:]
    i = np.nonzero(np.abs(np.diff(spkrec)) > 0)[0]
    changes[chgroup] = (i, spkrec[i+0])
# for each shank, list of boundaries for spike times, and corresponding recording

last_t = 0
for recidx in recordings:
    print "Processing recording %d" % recidx

    rec = exp.recordings[recidx]
    sr = rec.sample_rate
    
    rec.start_time = last_t / float(sr)
    
    t = 0
    for chgroup in shanks:
        mychanges, spkrec = changes[chgroup]
        i = np.nonzero(spkrec == recidx)[0]
        if i.size == 0:
            continue
        spkidx = mychanges[i][0]
        t = max(t, exp.channel_groups[chgroup].spikes.time_samples[spkidx])
    # t is now the largest spike time in that recording (local frame)
    
    last_t += int(t + (10 * sr))

exp.close()
print "Done!"