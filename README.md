# faster-membranes
This repository hosts an experiment for segmenting EM images, with the
goal of working more quickly than brute-force application of CNNs on a
per-pixel basis.

This code is in an experimental state and subject to change.


## Quick start

-  Install Caffe (including the Python interface) and Caffe con Troll (CcT).
-  Edit make.config as needed for your system (and experiment of interest).
-  For caffe (command line) or CcT, preprocess the ISBI 2012 data set via:
```
    make lmdb
```
-  Do either a timing experiment or extract probability estimates.  See
   the Makefile for details.  A quick example is:
```
    make CNN=n3_py GPU=1 isbi2012-train
    make CNN=n3_py GPU=1 EVAL_PCT=.1 isbi2012-deploy
```
