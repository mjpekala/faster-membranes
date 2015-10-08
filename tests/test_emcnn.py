"""Unit test for emcnn.py

To run (from pwd):
    PYTHONPATH=../src python test_emcnn.py
"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"


import unittest
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as smetrics

import emcnn



class TestEmcnn(unittest.TestCase):
    def test_omit_labels(self):
        Y = np.zeros((3,3,3))
        Y[0,0:3] = range(3)

        toOmit, pctOmitted = emcnn._omit_labels(Y)
        self.assertTrue(len(toOmit) == 0)

        Y[0,1] = -1;
        toOmit, pctOmitted = emcnn._omit_labels(Y)
        self.assertTrue(len(toOmit) == 1)
        
        toOmit, pctOmitted = emcnn._omit_labels(Y, [1])
        self.assertTrue(len(toOmit) == 2)

        
    def test_xform_minibatch(self):
        X = np.random.randint(0,10, size=(2,100,3,3))

        # if prob of flip is 0 -> no data augmentation
        for ii in range(30):
            Xprime = emcnn._xform_minibatch(X,prob=0.0)
            self.assertTrue(np.all(X == Xprime))
        
        # if prob of flip is high -> some data augmentation
        numDiff = 0
        for ii in range(30):
            Xprime = emcnn._xform_minibatch(X,prob=0.9)
            self.assertTrue(np.any(Xprime > 0))  # make sure didn't nuke data
            if np.any(X != Xprime): numDiff += 1
        self.assertTrue(numDiff>0)

        
    def test_load_data(self):
        # TODO: make this search the PYTHONPATH for files...
        xfile = os.path.join('Data', 'ISBI2012', 'train-volume.tif')
        yfile = os.path.join('Data', 'ISBI2012', 'train-labels.tif')
        if not os.path.exists(xfile): 
            xfile = os.path.join('..', xfile)
            yfile = os.path.join('..', yfile)

        # test loading raw ISBI data (no preprocessing)
        whichSlices = [0,1]
        X, Y = emcnn._load_data(xfile, yfile, 30, whichSlices)
        self.assertTrue(X.shape[0] == len(whichSlices))

        yAll = np.unique(Y);  yAll.sort()
        self.assertTrue(np.all(yAll == [0, 1]))
        
        self.assertTrue(np.max(X) <= 1.0)
        self.assertTrue(np.min(X) >= 0.0)

        

        
if __name__ == "__main__":
    unittest.main()


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
