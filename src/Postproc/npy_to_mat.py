"""
Manually converts a .npy output from pycaffe wrappers into a .mat
file.  Only needed if there is some problem with the .mat file
produced by the pycaffe wrapper scripts.

Example:
   python  npy_to_mat.py  YhatDeploy.npy
"""

import sys
import numpy as np
import scipy.io


if __name__ == "__main__":
    inFile = sys.argv[1]

    print('[info]: loading file %s ...' % inFile)
    Yhat = np.load(inFile)
    print('[info]: volume has shape: %s' % str(Yhat.shape))
    

    # Grab the estimates associated with class 1 (assumed to be the
    # biological structure of interest in a binary classification
    # problem, e.g. the non-membrane vs membrane problem).
    Yhat = Yhat[1,:,:,:]

    # transpose to put dimensions in matlab canonical order.
    Yhat = np.transpose(Yhat, [1, 2, 0])
    
    print('[info]: new shape: %s' % str(Yhat.shape))

    print('[info]: saving to matlab format (this may take a few minutes for large volumes)...')
    scipy.io.savemat(inFile + ".mat", {'Yhat' : Yhat})
    print('[info]: done!')
    
    
