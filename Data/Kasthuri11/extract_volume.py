# quick and dirty script to pull a volume out of an hdf5 file downloaded
# from the openconnectome project

import sys, os
import numpy as np
import h5py
from scipy.io import savemat


if __name__ == "__main__": 
    inFile = sys.argv[1] 
    if not os.path.exists(inFile): 
        raise RuntimeError('could not find input file %s' % inFile)

    f = h5py.File(inFile, 'r')
    group = f[f.keys()[0]]

    volume = group[u'CUTOUT']

    outFile = inFile.replace('.hdf5', '')
    np.save(outFile, volume.value)

    # Can also save a matlab version, if desired (is optional).
    outFile = inFile.replace('.hdf5', '.mat')
    savemat(outFile, {'X' : np.transpose(volume.value, (1,2,0))})

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
