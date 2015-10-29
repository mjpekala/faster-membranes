## The Kasthuri data set

The Kasthuri 11 data set consists of a training volume, a test volume, and a set of training labels corresponding to probability estimates created from a CNN.  To make a local copy of this data set run the script
```
    getdata.sh
```

This will download the data and also run the necessary preprocessing.  Necessary preprocessing steps include
  o  Creating .npy files containing only the raw data (no metadata).  This is required by our particular wrapper around Caffe.
  o  Thresholding the training probability estimates to produce binary class labels.  Note that one could also consider formulating the training as a regression problem (vs binary classification), which would arguably make better use of the training data.  Note also that the thresholds being used as of this writing (October 2015) were chosen arbitrarily and are possibly suboptimal.
