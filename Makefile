#-------------------------------------------------------------------------------
# This makefile sets up a few classification problems related to the 
# ISBI 2012 challenge data set.  The goal is to make it relatively easy
# to run timing experiments with Caffe and Caffe con Troll (CcT) on this
# data.  Also, we provide ways of extracting probability maps.
# 
# 1. Preprocess ISBI data set (e.g. create LMDB databases).
#    You should only need to do this once:
#       make lmdb
#
#    If you want to re-generate at some point, you must first
#    delete the existing LMDB database. Do this via:
#       make lmdb-clean
#
# 2. Extract probability maps.  This requires you (a) train a model
#    using pycaffe and then (b) deploy the model on the validation set.
#    This can take a long time; hence we nohup these tasks implicitly.
#    
#    Example: training and deploying the "lenet" model:
#       make CNN=lenet-py GPU=1 pycaffe-train
#       make CNN=lenet-py GPU=1 pycaffe-predict
#                    - or - 
#       make CNN=lenet-py GPU=2 EVAL_PCT=.1 pycaffe-predict
#
#    Example: training and deploying the "N3" model:
#       (do exactly the same as above but with CNN=n3)
#
# 3. To generate timing estimates for Caffe:
#       make CNN=lenet GPU=3 caffe-train
#       make CNN=lenet GPU=4 caffe-time-gpu
#       make CNN=lenet caffe-time-cpu
# 
# 4. To generate timing estimates for Caffe con Troll (CcT):
#       make cct-train
#       make cct-fwd-time > fwdtime.txt
#
#
# NOTES:
# o For CcT compatability, we create LMDB databases that contain many 
#   pre-computed tiles; in the past we created these tiles "lazily" to
#   avoid creating large data sets with lots of redundancy.
#
# o A future possible direction is some combination of CcT and 
#   dense classification techniques (e.g. the semantic segmentation
#   approach of Long et. al.)
#
#-------------------------------------------------------------------------------


# you can ignore these...needed to define paths properly later below
MAKE_NAME := $(abspath $(lastword $(MAKEFILE_LIST)))
BASE_DIR := $(patsubst %/,%,$(dir $(MAKE_NAME)))
PROJ_NAME := $(notdir $(BASE_DIR))


#-------------------------------------------------------------------------------
include make.config



#-------------------------------------------------------------------------------
# MACROS that are system-independent (you can probably ignore these)
# 
#-------------------------------------------------------------------------------

SRC=$(BASE_DIR)/src
DATA_DIR=$(BASE_DIR)/Data/ISBI2012
LMDB_DIR=$(BASE_DIR)/Data/ISBI2012/$(EXPERIMENT)
MODEL_DIR=$(BASE_DIR)/Models/$(CNN)
OUT_DIR=$(MODEL_DIR)/$(EXPERIMENT)

# Different ways to run python.
# (we always need PyCaffe and emlib.py in the PYTHONPATH)
PY=PYTHONPATH=$(PYCAFFE):$(SRC) python
IPY=PYTHONPATH=$(PYCAFFE):$(SRC) ipython -i --
PYNOHUP=PYTHONPATH=$(PYCAFFE):$(SRC) nohup python
PYPROF=PYTHONPATH=$(PYCAFFE):$(SRC) python -m cProfile -s cumtime

# Number of iterations to use in timing experiments
NITERS=100

TAR=$(PROJ_NAME).tar


#-------------------------------------------------------------------------------
# "administrative" targets
#-------------------------------------------------------------------------------
default:
	@echo ""
	@echo "Experiment:  $(EXPERIMENT)"
	@echo "Base dir:    $(BASE_DIR)"
	@echo "Data dir:    $(DATA_DIR)"
	@echo "Output dir:  $(OUT_DIR)"
	@echo "Using caffe: $(PYCAFFE)"
	@echo ""
	@echo "Please explicitly choose a target"


tar :
	\rm -f $(BASE_DIR)/$(TAR)
	pushd $(BASE_DIR)/.. && tar cvf $(TAR) `find ./$(PROJ_NAME) -name \*.py -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./$(PROJ_NAME) -name \*.m -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./$(PROJ_NAME) -name \*.md -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./$(PROJ_NAME) -name \*.txt -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./$(PROJ_NAME) -name \*.tif -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./$(PROJ_NAME) -name Makefile -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./$(PROJ_NAME) -name \*.prototxt -print`



#-------------------------------------------------------------------------------
# Data preprocessing
# (includes creating LMDB databases)
# You have to do this (just once) before running any other targets.
#-------------------------------------------------------------------------------

lmdb:
	@$(PY) $(SRC)/preprocess.py \
		-X $(BASE_DIR)/Data/ISBI2012/train-volume.tif \
		-Y $(BASE_DIR)/Data/ISBI2012/train-labels.tif \
		--train-slices $(SLICE_TRAIN) \
		--valid-slices $(SLICE_VALID) \
		--test-slices $(SLICE_TEST) \
		--brightness-quantile 1.0 \
		--out-dir $(LMDB_DIR)

	@$(PY) $(SRC)/make_lmdb.py \
		-X $(LMDB_DIR)/Xtrain.npy \
		-Y $(LMDB_DIR)/Ytrain.npy \
		--num-examples $(N_TILES) \
		-o $(LMDB_DIR)/train.lmdb

	@$(PY) $(SRC)/make_lmdb.py \
		-X $(LMDB_DIR)/Xvalid.npy \
		-Y $(LMDB_DIR)/Yvalid.npy \
		--num-examples $(N_TILES) \
		-o $(LMDB_DIR)/valid.lmdb


# Deletes data preprocessing
lmdb-clean:
	\rm -rf $(LMDB_DIR)/{train,valid}.lmdb 
	\rm -f $(LMDB_DIR)/{X,Y}*npy
	\rm -f $(LMDB_DIR)/{X,Y}*mat


#-------------------------------------------------------------------------------
# Working with Caffe
#-------------------------------------------------------------------------------

#--------------------------------------------------
# Train a model using either:
# 
#   a) Command-line caffe and pre-computed tiles
#      stored in an LMDB database
#      (this enables timing comparisons with CcT)
#      
#   b) PyCaffe and "lazily" created tiles
#      (this uses more of the available data and is
#       recommended if you care about classification
#       performance)
#      
#--------------------------------------------------
caffe-train:
	@mkdir -p $(OUT_DIR)
	nohup $(CAFFE) train \
		-solver $(MODEL_DIR)/$(CNN)-solver.prototxt \
	       	-gpu $(GPU) \
		> $(OUT_DIR)/caffe.$(CNN).train.out &


pycaffe-train:
	@mkdir -p $(OUT_DIR)
	$(PYNOHUP) $(SRC)/emcnn.py \
		--x-train $(DATA_DIR)/train-volume.tif \
		--y-train $(DATA_DIR)/train-labels.tif \
		--train-slices $(SLICE_TRAIN) \
		--x-valid $(DATA_DIR)/train-volume.tif \
		--y-valid $(DATA_DIR)/train-labels.tif \
		--valid-slices $(SLICE_VALID) \
		--solver $(MODEL_DIR)/$(CNN)-solver.prototxt \
		--rotate-data $(ROTATE) \
		--gpu $(GPU) \
		--out-dir $(OUT_DIR) \
		> $(OUT_DIR)/pycaffe.$(CNN).train.out &


pycaffe-predict:
	$(PYNOHUP) $(SRC)/emcnn.py \
		--network $(MODEL_DIR)/$(CNN)-net.prototxt \
		--model $(OUT_DIR)/$(CAFFE_MODEL) \
		--x-deploy $(DATA_DIR)/train-volume.tif \
		--gpu $(GPU) \
		--out-dir $(OUT_DIR) \
		--eval-pct $(EVAL_PCT) \
		> $(OUT_DIR)/pycaffe.$(CNN).predict.out &


pycaffe-predict-test:
	$(PYNOHUP) $(SRC)/emcnn.py \
		--network $(MODEL_DIR)/$(CNN)-net.prototxt \
		--model $(OUT_DIR)/$(CAFFE_MODEL) \
		--x-deploy $(DATA_DIR)/test-volume.tif \
		--gpu $(GPU) \
		--out-dir $(OUT_DIR) \
		--eval-pct $(EVAL_PCT) \
		> $(OUT_DIR)/pycaffe.$(CNN).predict.test.out &


#--------------------------------------------------
# Produce timing estimates (using model created above)
# using caffe command line for either:
# 
#   a) GPU cards
#   b) CPU only (very slow!)
#  
#--------------------------------------------------
caffe-time-gpu:
	$(CAFFE) time \
		-model $(MODEL_DIR)/$(CNN)-net.prototxt \
		-weights $(OUT_DIR)/$(CAFFE_MODEL) \
		-iterations $(NITERS) -gpu $(GPU)


caffe-time-cpu:
	$(CAFFE) time \
		-model $(MODEL_DIR)/$(CNN)-net.prototxt \
		-weights $(OUT_DIR)/$(CAFFE_MODEL) \
		-iterations $(NITERS)


#-------------------------------------------------------------------------------
# Working with CcT
#-------------------------------------------------------------------------------
cct-train:
	nohup $(CCT) train $(MODEL_DIR)/$(CNN)-solver.prototxt > cct.train.out &

cct-fwd-time:
	@grep "Forward Pass" cct.train.out | awk '{print $$6}'


