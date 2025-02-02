#-------------------------------------------------------------------------------
# This makefile sets up a few classification problems related to 
# electron microscopy data sets.  The goal is to make it relatively easy
# to run timing experiments with Caffe and Caffe con Troll (CcT).
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


include make.config


#-------------------------------------------------------------------------------
# "administrative" targets
#-------------------------------------------------------------------------------


tar:
	\rm -f $(BASE_DIR)/$(TAR)
	pushd $(BASE_DIR)/.. && tar cvf $(TAR) `find ./$(PROJ_NAME) -name \*.py -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./$(PROJ_NAME) -name \*.m -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./$(PROJ_NAME) -name \*.md -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./$(PROJ_NAME) -name \*.txt -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./$(PROJ_NAME) -name \*.tif -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./$(PROJ_NAME) -name Makefile -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./$(PROJ_NAME) -name \*.prototxt -print`


unittest:
	$(PY) tests/test_emlib.py
	$(PY) tests/test_emcnn.py

#### **** TODO: move everything from here down into separate makefiles ****

#-------------------------------------------------------------------------------
#  II. Caffe vs CcT timing estimates.
#      At the moment, this requires creating a separate LMDB database.
#-------------------------------------------------------------------------------

lmdb:
	@$(PY) $(SRC)/preprocess.py \
		-X $(BASE_DIR)/Data/ISBI2012/train-volume.tif \
		-Y $(BASE_DIR)/Data/ISBI2012/train-labels.tif \
		--train-slices "range(0,20)" \
		--valid-slices "range(20,30)" \
		--brightness-quantile 1.0 \
		--out-dir $(OUT_DIR)

	@$(PY) $(SRC)/make_lmdb.py \
		-X $(OUT_DIR)/Xtrain.npy \
		-Y $(OUT_DIR)/Ytrain.npy \
		--num-examples 200000 \
		-o $(OUT_DIR)/train.lmdb

	@$(PY) $(SRC)/make_lmdb.py \
		-X $(OUT_DIR)/Xvalid.npy \
		-Y $(OUT_DIR)/Yvalid.npy \
		--num-examples 200000 \
		-o $(OUT_DIR)/valid.lmdb


caffe-train:
	@mkdir -p $(OUT_DIR)
	nohup $(CAFFE) train \
		-solver $(MODEL_DIR)/$(CNN)_solver.prototxt \
	       	-gpu $(GPU) \
		> $(OUT_DIR)/caffe.$(CNN).train.out &


caffe-time-gpu:
	$(CAFFE) time \
		-model $(MODEL_DIR)/$(CNN)_net.prototxt \
		-weights $(OUT_DIR)/$(CAFFE_MODEL) \
		-iterations $(NITERS) -gpu $(GPU)


caffe-time-cpu:
	$(CAFFE) time \
		-model $(MODEL_DIR)/$(CNN)_net.prototxt \
		-weights $(OUT_DIR)/$(CAFFE_MODEL) \
		-iterations $(NITERS)


cct-train:
	nohup $(CCT) train $(MODEL_DIR)/$(CNN)_solver.prototxt > cct.train.out &

cct-fwd-time:
	@grep "Forward Pass" cct.train.out | awk '{print $$6}'




#-------------------------------------------------------------------------------
#  III. Construct maps for the Kasthuri data set
#-------------------------------------------------------------------------------

# Train on the first 80 slices, reserving the rest for validation.
# This creates the model needed for the deploy targets.
kast-train:
	@mkdir -p $(OUT_DIR)
	$(PYNOHUP) $(SRC)/emcnn.py \
		--x-train $(BASE_DIR)/Data/Kasthuri11/train-volume.npy \
		--y-train $(BASE_DIR)/Data/Kasthuri11/train-labels-thresh.npy \
		--train-slices "range(0,80)" \
		--x-valid $(BASE_DIR)/Data/Kasthuri11/train-volume.npy \
		--y-valid $(BASE_DIR)/Data/Kasthuri11/train-labels-thresh.npy \
		--valid-slices "range(90,100)" \
		--solver $(MODEL_DIR)/$(CNN)_solver.prototxt \
		--rotate-data $(ROTATE) \
		--gpu $(GPU) \
		--out-dir $(OUT_DIR) \
		> $(OUT_DIR)/pycaffe.$(CNN).kast.train.out &


# Deploy classifier on training data set.
# This is only useful for diagnostic purposes.
kast-deploy-train:
	@mkdir -p $(OUT_DIR)
	$(PYNOHUP) $(SRC)/emcnn.py \
		--x-deploy $(BASE_DIR)/Data/Kasthuri11/train-volume.npy \
		--network $(MODEL_DIR)/$(CNN)_net.prototxt \
		--model $(OUT_DIR)/$(CAFFE_MODEL) \
		--gpu $(GPU) \
		--eval-pct $(EVAL_PCT) \
		--out-dir $(OUT_DIR) \
		> $(OUT_DIR)/pycaffe.$(CNN).kast.deploy.train.$(NOW).out &


# Deploy classifier on the test data set.
kast-deploy-test:
	@mkdir -p $(OUT_DIR)
	$(PYNOHUP) $(SRC)/emcnn.py \
		--x-deploy $(BASE_DIR)/Data/Kasthuri11/test-volume.npy \
		--network $(MODEL_DIR)/$(CNN)_net.prototxt \
		--model $(OUT_DIR)/$(CAFFE_MODEL) \
		--gpu $(GPU) \
		--eval-pct $(EVAL_PCT) \
		--out-dir $(OUT_DIR) \
		> $(OUT_DIR)/pycaffe.$(CNN).kast.deploy.test.$(NOW).out &

