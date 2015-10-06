# faster-membranes
This repository hosts an experiment for segmenting EM images, with the goal of working more quickly than brute-force application of CNNs on a per-pixel basis.

## Quick start

-  Install Caffe (including the Python interface) and Caffe con Troll (CcT).
-  Edit paths towards the top of the Makefile as needed for your system.
-  Preprocess ISBI2012 by calling 

    make data

-  Do either a timing experiment or extract probability estimates.  See
   the Makefile for details.  A quick example is:

    make CNN=lenet-py GPU=1 pycaffe-train
    make CNN=lenet-py GPU=1 pycaffe-predict
