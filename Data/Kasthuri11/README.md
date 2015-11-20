## The Kasthuri data set

For our experiments, we use two subsets of the Kasthuri 11 data set as our train and test volumes.   The subset "AC4" is used for training and "AC3" for test.  These volumes were not ground-truthed manually; instead, our "ground truth" is in the form of probability estimates produced by a trusted convolutional neural network (CNN).  So we are using the outputs of one CNN to train another (a bit unorthodox - somewhat reminiscant of Hinton's distallation idea, except we don't have gold standard binary class labels to accompany the outputs of the CNN). 

To make a local copy of this data set run (from the command line):
```
    ./getdata.sh
```

This will download the data and also run the necessary preprocessing.  Necessary preprocessing steps include:

-   Creating .npy files containing only the raw data (no metadata).  This is required by our particular wrapper around Caffe.

-  Thresholding the training probability estimates to produce binary class labels.  One could also consider formulating the training as a regression problem (vs binary classification).  The thresholds being used as of this writing (October 2015) were chosen arbitrarily and are possibly suboptimal.  These can be adjusted within the file *threshold_labels.py*.  If  you'd like to visualize the impact of different thresholds on the training labels, in matlab do the following (assuming you have already run the *getdata.sh* script):
```
    load train-labels.mat
    view_train_labels(X)
```
