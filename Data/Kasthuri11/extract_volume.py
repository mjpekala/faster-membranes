# quick and dirty script to pull a volume out of an hdf5 file downloaded
# from the openconnectome project

import sys, os
import numpy as np
import h5py


inFile = sys.argv[1]
if not os.path.exists(inFile):
	raise RuntimeError('could not find input file %s' % inFile)

f = h5py.File(inFile, 'r')
group = f[f.keys()[0]]

volume = group[u'CUTOUT']

outFile = inFile.replace('.hdf5', '')
np.save(outFile, volume)
