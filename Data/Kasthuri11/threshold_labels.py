# THRESHOLD_LABELS  Use this to convert membrane probabilities into class labels.
#
#   The way we are setting this problem up is a little unusual.
#   Instead of gold standard ground truth class labels, we are using
#   the outputs of another classifier that we trust (but do not have
#   access to) as a source of ground truth.
#
#   The trusted classifier generates outputs that live in [0 1].  We
#   could try to match these outputs by setting up a regression
#   problem.  Alternately, we can map these classifier outputs to
#   binary class labels by thresholding.
#
#   This script takes the latter approach.  Rather than pick a single
#   score threshold, we pick two thresholds.  Scores above the upper
#   threshold are used as positive class labels.  Those below the
#   lower score threshold are negative class labels.  Those in between
#   are not used for training.
#


import sys, os
import numpy as np
from scipy.stats.mstats import mquantiles as quantile
import h5py


# Different ways of thresholding the scores in [0 1]
THRESH_QUANTILES = [.2, .9]
THRESH_MANUAL = [.1, .6]


if __name__ == "__main__":
    inFile = sys.argv[1]
    if not os.path.exists(inFile):
        raise RuntimeError('could not find input file %s' % inFile)

    P = np.load(inFile)

    if True: 
        thresh = THRESH_MANUAL
    else: 
        thresh = quantile(np.reshape(P, (P.size,)), THRESH_QUANTILES)
        print('[info]: Thresholding data using quantiles: %s' % (THRESH_QUANTILES)) 
    print('[info]: Using thresholds:                  %s' % (thresh))

    Y = -1 * np.ones(P.shape, dtype=np.uint8)
    Y[P >= thresh[1]] = 1     # high probability -> use as positive examples
    Y[P <= thresh[0]] = 0     # low probability -> use as negative examples
 
    outFile = inFile.replace('.npy', '') + '-thresh'
    np.save(outFile, Y)

    pct = 100.0 * np.sum(Y>=0) / Y.size
    print('[info]: Training labels use %0.2f%% of input volume.' % pct)


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
