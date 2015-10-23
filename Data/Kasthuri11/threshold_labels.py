# Use this to convert membrane probabilities into class labels.

import sys, os
import numpy as np
import h5py


THRESH_UPPER = .6
THRESH_LOWER = .1

inFile = sys.argv[1]
if not os.path.exists(inFile):
	raise RuntimeError('could not find input file %s' % inFile)

P = np.load(inFile)

Y = -1 * np.ones(P.shape, dtype=np.uint8)
Y[P > THRESH_UPPER] = 1     # high probability -> use as positive examples
Y[P < THRESH_LOWER] = 0     # low probability -> use as negative examples

outFile = inFile.replace('.npy', '') + '-thresh'
np.save(outFile, Y)
