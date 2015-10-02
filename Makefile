#-------------------------------------------------------------------------------
# This makefile sets up a few classification problems related to the 
# ISBI 2012 challenge data set.  The goal is to make it relatively easy
# to run timing experiments with Caffe and Caffe con Troll (CcT) on this
# data.  Also, we provide ways of extracting probability maps.
# 
# 1. To create the LMDB database from raw ISBI data:
#      make lmdb-train
#      make lmdb-valid
#
# 2. Train models using Caffe:
#      make CNN=lenet GPU=1 caffe-train
#      make CNN=lenet GPU=2 pycaffe-train
#      make CNN=n3 GPU=3 caffe-train
#      make CNN=n3 GPU=4 pycaffe-train
# 
# 3. To generate timing estimates for Caffe:
#      make caffe-time-gpu
#      make caffe-time-cpu
# 
# 4. To generate timing estimates for Caffe con Troll (CcT):
#      make cct-time-cpu
#
# 5. Extract predictions from the Caffe model:
#      make CNN=lenet GPU=3 caffe-predict
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

# Specify which CNN to use.
CNN=lenet
SOLVER=./Models/$(CNN)/$(CNN)-solver.prototxt
NET=./Models/$(CNN)/$(CNN)-net.prototxt
DEPLOY=./Models/$(CNN)/$(CNN)-deploy.prototxt

CAFFE_MODEL=$(BASE_DIR)/Models/$(CNN)/iter_15000.caffemodel
CCT_MODEL=trained_model.bin.25-09-2015-04-46-54


# Experiment parameters related to the data set.
# You can put different train/test splits into different "experiments".
EXPERIMENT=$(BASE_DIR)/Data/ISBI2012/ISBI_Train20
S_TRAIN="range(0,20)"
S_VALID="range(20,30)"
S_TEST="[]"
N_TILES=200000

# You may want to override this from the command line.
# (see examples above).
# On our cluster, you should avoid using gpu 0.
GPU=1


#-------------------------------------------------------------------------------
# MACROS you can probably ignore...
# 
#-------------------------------------------------------------------------------

SRC=$(BASE_DIR)/src

# Different ways to run python.
# (we always need PyCaffe and emlib.py in the PYTHONPATH)
PY=PYTHONPATH=$(PYCAFFE):$(SRC) python
PYNOHUP=PYTHONPATH=$(PYCAFFE):$(SRC) nohup python
PYPROF=PYTHONPATH=$(PYCAFFE):$(SRC) python -m cProfile -s cumtime

# Number of iterations to use in timing experiments
NITERS=100

TAR=$(PROJ_NAME).tar


#-------------------------------------------------------------------------------
# "administrative" targets
#-------------------------------------------------------------------------------
default:
	@echo $(BASE_DIR)
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
		--out-dir $(EXPERIMENT)

	@$(PY) $(SRC)/make_lmdb.py \
		-X $(EXPERIMENT)/Xtrain.npy \
		-Y $(EXPERIMENT)/Ytrain.npy \
		--num-examples $(N_TILES) \
		-o $(EXPERIMENT)/train.lmdb

	@$(PY) $(SRC)/make_lmdb.py \
		-X $(EXPERIMENT)/Xvalid.npy \
		-Y $(EXPERIMENT)/Yvalid.npy \
		--num-examples $(N_TILES) \
		-o $(EXPERIMENT)/valid.lmdb


# Deletes data preprocessing
data-clean:
	\rm -rf $(EXPERIMENT)/train.lmdb $(EXPERIMENT)/valid.lmdb 
	\rm -f $(EXPERIMENT)/{X,Y}*npy


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
	nohup $(CAFFE) train \
		-solver $(CNN)/$(CNN)-solver.prototxt \
	       	-gpu $(GPU) \
		> $(CNN)/caffe.$(CNN).train.out &


pycaffe-train:
	$(PYNOHUP) emcnn.py \
		--x-train $(EXPERIMENT)/Xtrain.npy \
		--y-train $(EXPERIMENT)/Ytrain.npy \
		--x-valid $(EXPERIMENT)/Xvalid.npy \
		--y-valid $(EXPERIMENT)/Yvalid.npy \
		--solver $(CNN)/$(CNN)-solver-py.prototxt \
		--gpu $(GPU) \
		--out-dir $(EXPERIMENT) \
		> $(EXPERIMENT)/pycaffe.$(CNN).train.out &


#--------------------------------------------------
# Produce timing estimates (using model created above)
# using caffe command line for either:
# 
#   a) GPU cards
#   b) CPU only (very slow!)
#  
#--------------------------------------------------
caffe-time-gpu:
	$(CAFFE) time -model $(NET) -weights $(CAFFE_MODEL) -iterations $(NITERS) -gpu $(GPU)


caffe-time-cpu:
	nohup $(CAFFE) time -model $(NET) -weights $(CAFFE_MODEL) -iterations $(NITERS) > caffe.time.cpu.out &


#--------------------------------------------------
# Generate pixel-level predictions using Caffe.
#
# Note this works with either model developed during training
# (just point the --model argument at the correct file)
#--------------------------------------------------
caffe-predict:
	$(PY) deploy2.py -n $(DEPLOY) -m $(CAFFE_MODEL) -X ISBI2012/train-volume.tif  --gpu $(GPU) --max-brightness $(MAX_BRIGHT) --eval-pct .45



#-------------------------------------------------------------------------------
# Working with CcT
#-------------------------------------------------------------------------------
cct-train:
	nohup $(CCT) train $(SOLVER) > cct.train.out &


cct-time-cpu:
	nohup $(CCT) test $(SOLVER) -i $(CCT_MODEL) > cct.time.cpu.out &


cct-clean :
	\rm -f train_preprocessed.bin val_preprocessed.bin
