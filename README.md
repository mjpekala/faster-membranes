# faster-membranes
This repository hosts an experiment for segmenting EM images, with the
goal of higher throughput relative to exhaustive application of sliding window CNNs.

This code is in an experimental state and subject to change.


## Quick start

-  Install Caffe (including the Python interface) and Caffe con Troll (CcT).
-  Edit make.config as needed for your system (and experiment of interest).
-  To extract probability estimates (here, for ISBI 2012):
```
    make CNN=n3_py GPU=1 isbi2012-train
    make CNN=n3_py GPU=1 EVAL_PCT=.1 isbi2012-deploy
```
  Outputs will be placed in the "Experiments" subdirectory.  Code for filling in unevaluated pixels can be found in src/Postproc.

-  To run timing estimates for CcT vs Caffe:
```
    make lmdb
    make GPU=1 caffe-train
    make caffe-time-cpu
    make cct-train
```


## Caffe Notes
As of Oct 12, 2015 the following modifications to Caffe source code are required in order to run this software:

- Fix the memory leak in the memory data layer: https://github.com/BVLC/caffe/issues/2334
- Add support for MCMC sampling in test mode:  https://github.com/yaringal/DropoutUncertaintyCaffeModels
