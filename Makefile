#-------------------------------------------------------------------------------
# This makefile sets up a few classification problems related to the 
# ISBI 2012 challenge data set.  The goal is to make it relatively easy
# to run timing experiments with Caffe and Caffe con Troll (CcT) on this
# data.  Also, we provide ways of extracting probability maps.
# 
# 1. Preprocess ISBI data set (e.g. create LMDB databases).
#    You should only need to do this once:
#       make data
#
#    If you want to re-generate at some point, you must first
#    delete the existing LMDB database. Do this via:
#       make data-clean
#
# 2. Train models using Caffe (command line and/or pycaffe).
#    This can take a long time; hence we nohup these implicitly:
#       make CNN=lenet-py GPU=1 pycaffe-train
#       make CNN=n3-py GPU=2 pycaffe-train
#       make CNN=lenet GPU=3 caffe-train
#       make CNN=n3 GPU=4 caffe-train
#       make CNN=lenet cct-train
#
# 3. Extract predictions from the Caffe model:
#      make CNN=lenet-py GPU=1 pycaffe-predict
#      make CNN=lenet-py GPU=2 EVAL_PCT=.1 pycaffe-predict
#
# 
# 4. To generate timing estimates for Caffe:
#      make caffe-time-gpu
#      make caffe-time-cpu
# 
# 5. To generate timing estimates for Caffe con Troll (CcT):
#      make cct-time-cpu
#
#
# 6. Extract predictions for CcT:
#      TODO: this is going to require some new code.
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
# MACROS you *need* to configure for your particular system.
#
#-------------------------------------------------------------------------------

# I assume caffe and caffe-ct are in your path; if not
# update these macros accordingly
#
# Update PYCAFFE as appropriate for your caffe
PYCAFFE=/home/pekalmj1/Apps/caffe/python
CAFFE=caffe
CCT=caffe-ct


#-------------------------------------------------------------------------------
# MACROS you may want to change to control the experimental setup
#
#-------------------------------------------------------------------------------

# Experiment parameters related to the data set.
# You can put different train/test splits into different "experiments".
#
# Note: if you change the EXPERIMENT, you'll need to manually hack
#       the caffe *.net files (at least for the command-line version)
#
EXPERIMENT=ISBI_Train20
S_TRAIN="range(0,20)"
S_VALID="range(20,30)"
S_TEST="[]"
N_TILES=200000


# Specify which CNN and model to use.
#    CNN \in {n3, n3-py, lenet, lenet-py}
# The "-py" models are for pycaffe targets.
CNN=lenet
CAFFE_MODEL=iter_080000.caffemodel
CCT_MODEL=trained_model.bin.25-09-2015-04-46-54

# Extra synthetic data augmentation during training?
ROTATE=0

# How much of the volume to evaluate in deploy mode \in [0,1]
EVAL_PCT=1.0

# You may want to override this from the command line.
# (see examples above).
# On our cluster, you should avoid using gpu 0.
GPU=1


#-------------------------------------------------------------------------------
# MACROS you can probably ignore...
# 
#-------------------------------------------------------------------------------

SRC=$(BASE_DIR)/src
DATA_DIR=$(BASE_DIR)/Data/ISBI2012/$(EXPERIMENT)
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
	@echo $(BASE_DIR)
	@echo $(DATA_DIR)
	@echo $(OUT_DIR)
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

data:
	@$(PY) $(SRC)/preprocess.py \
		-X $(BASE_DIR)/Data/ISBI2012/train-volume.tif \
		-Y $(BASE_DIR)/Data/ISBI2012/train-labels.tif \
		--train-slices $(S_TRAIN) \
		--valid-slices $(S_VALID) \
		--test-slices $(S_TEST) \
		--brightness-quantile 1.0 \
		--out-dir $(DATA_DIR)

	@$(PY) $(SRC)/make_lmdb.py \
		-X $(DATA_DIR)/Xtrain.npy \
		-Y $(DATA_DIR)/Ytrain.npy \
		--num-examples $(N_TILES) \
		-o $(DATA_DIR)/train.lmdb

	@$(PY) $(SRC)/make_lmdb.py \
		-X $(DATA_DIR)/Xvalid.npy \
		-Y $(DATA_DIR)/Yvalid.npy \
		--num-examples $(N_TILES) \
		-o $(DATA_DIR)/valid.lmdb


# Deletes data preprocessing
data-clean:
	\rm -rf $(DATA_DIR)/{train,valid}.lmdb 
	\rm -f $(DATA_DIR)/{X,Y}*npy
	\rm -f $(DATA_DIR)/{X,Y}*mat


#-------------------------------------------------------------------------------
# Working with Caffe
#-------------------------------------------------------------------------------

#--------------------------------------------------
# Train a model using either:
# 
#   a) Command-line caffe and pre-computed tiles
#      stored in an LMDB database
#      
#   b) PyCaffe and dynamically "lazily" created tiles
#      (this is a much larger data set)
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
		--x-train $(DATA_DIR)/Xtrain.npy \
		--y-train $(DATA_DIR)/Ytrain.npy \
		--x-valid $(DATA_DIR)/Xvalid.npy \
		--y-valid $(DATA_DIR)/Yvalid.npy \
		--solver $(MODEL_DIR)/$(CNN)-solver.prototxt \
		--rotate-data $(ROTATE) \
		--gpu $(GPU) \
		--out-dir $(OUT_DIR) \
		> $(OUT_DIR)/pycaffe.$(CNN).train.out &


pycaffe-predict:
	$(PYNOHUP) $(SRC)/emcnn.py \
		--network $(MODEL_DIR)/$(CNN)-net.prototxt \
		--model $(OUT_DIR)/$(CAFFE_MODEL) \
		--x-deploy $(DATA_DIR)/Xvalid.npy \
		--gpu $(GPU) \
		--out-dir $(OUT_DIR) \
		--eval-pct $(EVAL_PCT) \
		> $(OUT_DIR)/pycaffe.$(CNN).predict.out &


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


