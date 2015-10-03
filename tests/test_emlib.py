"""Unit test for emlib.py

To run:
    PYTHONPATH=../src python test_metrics.py
"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"


import unittest
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as smetrics

import emlib



class TestEmlib(unittest.TestCase):
    def test_metrics(self):
        Y = np.random.randint(0,2,size=(2,5,5))
        Yhat = np.random.randint(0,2,size=(2,5,5))

        C,acc,prec,recall,f1 = emlib.metrics(Y, Yhat, display=False)
        prec2, recall2, f12, supp = smetrics(np.reshape(Y, (Y.size,)), 
                np.reshape(Yhat, (Yhat.size,)))

        self.assertAlmostEqual(prec, prec2[1])
        self.assertAlmostEqual(recall, recall2[1])
        self.assertAlmostEqual(f1, f12[1])
        




if __name__ == "__main__":
    unittest.main()


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
