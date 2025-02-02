# faster-membranes
**UPDATE (March 2015)**: This code is being replaced by a Theano/Keras implementation (which is a bit easier to set up and run on some systems). A prototype/experimental version is [here](https://github.com/mjpekala/bio-segmentation); the "official" version will be incorporated into [this](https://github.com/neurodata/ndparse) much larger neuroscience toolkit.

This repository hosts an experiment for segmenting EM images, with the
goal of higher throughput relative to exhaustive application of sliding window CNNs.

Note: this code is in an experimental state and subject to change.
Furthermore, no attempt has (yet) been made to optimize these CNN models.



## Quick start

### Configuring Caffe
This software requires Caffe (tested with version 1.0, release candidate 2) and optionally Caffe con Troll (alpha release).

- https://github.com/BVLC/caffe/releases
- https://github.com/HazyResearch/CaffeConTroll

Note that the following modifications to Caffe are required:

- Fix the memory leak in the memory data layer (see https://github.com/BVLC/caffe/issues/2334).  Also initialize labels_ and data_ in the MemoryDataLayer (include/caffe/data_layers.hpp) if you want the Caffe unit tests to pass.
```
   Dtype* data_ = NULL;
   Dtype* labels_ = NULL;
```
- Optional - for uncertainty quantification, apply the patch described here: https://github.com/yaringal/DropoutUncertaintyCaffeModels.  Note that I called the new parameter "do_mc" instead of "sample_weights_test".


### Running Experiments

-  Install Caffe (including the Python interface) and Caffe con Troll (CcT) and modify as described above.
-  Edit make.config as needed for your system (and experiment of interest).
-  To extract probability estimates (here, for ISBI 2012):
```
    make CNN=n3_py GPU=1 isbi2012-train
    make CNN=n3_py GPU=1 EVAL_PCT=.1 isbi2012-deploy
```
  Outputs will be placed in the "Experiments" subdirectory.  Code for filling in unevaluated pixels can be found in src/Postproc.

-  To run timing estimates for CcT vs Caffe:
```
    make CNN=lenet_lmdb lmdb
    make CNN=lenet_lmdb GPU=1 caffe-train
    make CNN=lenet_lmdb caffe-time-cpu
    make CNN=lenet_lmdb cct-train
```

- For Kasthuri data set
```
   make EXPERIMENT=KAST CNN=n3_py GPU=1 kast-train
   make EXPERIMENT=KAST CNN=n3_py GPU=2 EVAL_PCT=0.1 CAFFE_MODEL=iter_450000.caffemodel kast-deploy-train
   make EXPERIMENT=KAST CNN=n3_py GPU=3 EVAL_PCT=1.0 CAFFE_MODEL=iter_450000.caffemodel kast-deploy-train
   make EXPERIMENT=KAST CNN=n3_py GPU=4 EVAL_PCT=0.1 CAFFE_MODEL=iter_450000.caffemodel kast-deploy-test
   make EXPERIMENT=KAST CNN=n3_py GPU=5 EVAL_PCT=1.0 CAFFE_MODEL=iter_450000.caffemodel kast-deploy-test
```
