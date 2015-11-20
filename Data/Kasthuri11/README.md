## The Kasthuri data set

The Kasthuri 11 data set consists of a training volume, a test volume, and a set of training labels corresponding to probability estimates created from a CNN.  To make a local copy of this data set run (from the command line):
```
    ./getdata.sh
```

This will download the data and also run the necessary preprocessing.  Necessary preprocessing steps include:

-   Creating .npy files containing only the raw data (no metadata).  This is required by our particular wrapper around Caffe.

-  Thresholding the training probability estimates to produce binary class labels.  Note that one could also consider formulating the training as a regression problem (vs binary classification).  The thresholds being used as of this writing (October 2015) were chosen arbitrarily and are possibly suboptimal.  These can be adjusted within the file *threshold_labels.py*.  If  you'd like to visualize the impact of different thresholds on the training labels, in matlab do the following (assuming you have already run the *getdata.sh* script):
```
    load train-labels.mat
    view_train_labels(X)
```
