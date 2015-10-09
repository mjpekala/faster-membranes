# faster-membranes
This repository hosts an experiment for segmenting EM images, with the
goal of working more quickly than brute-force application of CNNs on a
per-pixel basis.

This code is in an experimental state and subject to change.


## Quick start

-  Install Caffe (including the Python interface) and Caffe con Troll (CcT).
-  Edit make.config as needed for your system (and experiment of interest).
-  To extract probability estimates (here, for ISBI 2012):
```
    make CNN=n3_py GPU=1 isbi2012-train
    make CNN=n3_py GPU=1 EVAL_PCT=.1 isbi2012-deploy
```

-  To run timing estimates for CcT vs Caffe:
```
    make lmdb
    make GPU=1 caffe-train
    make caffe-time-cpu
    make cct-train
```
