# faster-membranes
This repository hosts an experiment for segmenting EM images, with the
goal of higher throughput relative to brute-force application of sliding window CNNs to all pixels in a volume.

This code is in an experimental state and subject to change.


## Quick start

-  Install Caffe (including the Python interface) and Caffe con Troll (CcT).
-  Edit make.config as needed for your system (and experiment of interest).
-  To extract probability estimates (here, for ISBI 2012):
```
    make CNN=n3_py GPU=1 isbi2012-train
    make CNN=n3_py GPU=1 EVAL_PCT=.1 isbi2012-deploy
```
  Outputs will be placed in the "Experiments" subdirectory.

-  To run timing estimates for CcT vs Caffe:
```
    make lmdb
    make GPU=1 caffe-train
    make caffe-time-cpu
    make cct-train
```
