""" EMCNN: PyCaffe for electron microscopy (EM) classification problems.

This code uses Caffe's python API to solve per-pixel classification
problems on EM data sets.  

The specific command line parameters used dictate whether this is used
to train a new CNN or deploy and existing one.

For a list of supported arguments, please see _get_args().

== ASSUMPTIONS ==

o Your convolutional neural network (CNN) must have the following 
  layers and tops/outputs:

    top name |  Layer type       |  Reason
    ---------+-------------------+----------------------------------
      loss   |  SoftmaxWithLoss  |  Reporting training performance
      acc    |  Accuracy         |  Reporting training performance
      prob   |  Softmax          |  Extracting estimates

o The input data volume (denoted X in this code) is 
  assumed to take values in [0 255]
  (this is the case for ISBI 2012)

o The labels volume (denoted Y) are integer valued.  
  Any pixels with negative "labels" are ignored during processing.
  This means that the Y volume serves a dual purpose (class labels
  and a processing mask).  If you don't have class labels but
  still want to mask out pixels (e.g. in deploy mode) then just
  set Y=0 for pixels you want to process and Y=-1 for those you don't.

"""


__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"


import sys, os, argparse, time, datetime
from pprint import pprint
from random import shuffle, choice
import pdb

import numpy as np
import scipy

from sobol_lib import i4_sobol_generate as sobol
import emlib
from pycaffe2 import SGDSolverMemoryData




def _train_mode_args():
    """Parameters for training a CNN.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--solver', dest='solver', 
		    type=str, required=True, 
		    help='Caffe prototxt file (train mode)')

    parser.add_argument('--x-train', dest='emTrainFile', 
		    type=str, required=True,
		    help='Filename of the training volume (train mode)')
    parser.add_argument('--y-train', dest='labelsTrainFile', 
		    type=str, required=True,
		    help='Filename of the training labels (train mode)')

    parser.add_argument('--x-valid', dest='emValidFile', 
		    type=str, required=True,
		    help='Filename of the validation volume (train mode)')
    parser.add_argument('--y-valid', dest='labelsValidFile', 
		    type=str, required=True,
		    help='Filename of the validation labels (train mode)')

    parser.add_argument('--rotate-data', dest='rotateData', 
		    type=int, default=0, 
		    help='(optional) 1 := apply arbitrary rotations')

    parser.add_argument('--omit-labels', dest='omitLabels', 
		    type=str, default='[]', 
		    help='(optional) list of additional labels to omit')
    
    parser.add_argument('--train-slices', dest='trainSlices', 
		    type=str, default='', 
		    help='(optional) limit to a subset of X/Y train')
    
    parser.add_argument('--valid-slices', dest='validSlices', 
		    type=str, default='', 
		    help='(optional) limit to a subset of X/Y validation')

    return parser


def _deploy_mode_args():
    """Parameters for deploying a CNN.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--network', dest='network', 
		    type=str, required=True,
		    help='Caffe network file (train mode)')

    parser.add_argument('--model', dest='model', 
		    type=str, required=True,
		    help='Caffe model file to use for weights (deploy mode)')

    parser.add_argument('--x-deploy', dest='emDeployFile', 
		    type=str, required=True,
		    help='Filename of the "deploy" volume (deploy mode)')

    parser.add_argument('--eval-pct', dest='evalPct', 
		    type=float, default=1.0,
		    help='what percentage of pixels to evaluate in each slice')
    
    parser.add_argument('--deploy-slices', dest='deploySlices', 
		    type=str, default='', 
		    help='(optional) limit to a subset of X/Y deploy volume')

    # Ref: Gal, Ghahramani "Dropout as a Bayesian Approximation:
    #      Representing Model Uncertainty in Deep Learning," arXiv, 2015.
    parser.add_argument('--n-monte-carlo', dest='nMC', 
		    type=int, default=0, 
		    help='(optional) if you have a model with dropout, the number of monte carlo samples to use for estimating uncertainty. This is highly experimental...')

    return parser



def _get_args():
    """Command line parameters.

    Note: use either the training parameters or the deploy parameters;
          not both.
    """

    if '--solver' in sys.argv:
        print('[emcnn]  Will run in TRAIN mode\n')
        parser = _train_mode_args()
        mode = 'train'
    else:
        print('[emcnn]  Will run in DEPLOY mode\n')
        parser = _deploy_mode_args()
        mode = 'deploy'
    
    #----------------------------------------
    # Parameters that apply to all modes
    #----------------------------------------
    parser.add_argument('--gpu', dest='gpu', 
		    type=int, default=-1, 
		    help='GPU ID to use')

    parser.add_argument('--out-dir', dest='outDir', 
		    type=str, default='', 
		    help='(optional) overrides the snapshot directory')

    args = parser.parse_args()
    args.mode = mode


    # map strings to python objects; XXX: a better way than eval()
    if (args.mode == 'train') and args.omitLabels: 
        args.omitLabels = eval(args.omitLabels)

    if (args.mode == 'train') and args.trainSlices: 
        args.trainSlices = eval(args.trainSlices)

    if (args.mode == 'train') and args.validSlices: 
        args.validSlices = eval(args.validSlices)
        
    if (args.mode == 'deploy') and args.deploySlices: 
        args.deploySlices = eval(args.deploySlices)

    return args



#-------------------------------------------------------------------------------
# some helper functions
#-------------------------------------------------------------------------------
numel = lambda X: X.size # np.prod(X.shape)


border_size = lambda batchDim: int(batchDim[2]/2)
prune_border_3d = lambda X, bs: X[:, bs:(-bs), bs:(-bs)]
prune_border_4d = lambda X, bs: X[:, :, bs:(-bs), bs:(-bs)]


            
def _omit_labels(Y, extraOmits=[]):
    """Determines which labels to omit.
    
    This is the union of the set of all negative labels in Y 
    with the set extraOmits.
    """
    # we always omit labels < 0
    omitLabels = np.unique(Y)
    omitLabels = list(omitLabels[omitLabels<0])

    # optionally exclude additional pixels
    if extraOmits: 
        omitLabels = omitLabels + extraOmits

    # determine the total pct omitted
    numOmitted = 0
    for yi in omitLabels:
        numOmitted += np.sum(Y==yi)
    
    return omitLabels, 100.0*numOmitted / Y.size



def _load_data(xName, yName, tileRadius, onlySlices, omitLabels=None):
    """Loads data sets and does basic preprocessing.
    """
    X = emlib.load_cube(xName, np.float32)

    # usually we expect fewer slices in Z than pixels in X or Y.
    # Make sure the dimensions look ok before proceeding.
    assert(X.shape[0] < X.shape[1])
    assert(X.shape[0] < X.shape[2])

    if onlySlices: 
        X = X[onlySlices,:,:] 
    print('[emCNN]:    data shape: %s' % str(X.shape))

    X = emlib.mirror_edges(X, tileRadius)

    # Scale data to live in [0 1].
    # *** ASSUMPTION *** original data is in [0 255]
    if np.max(X) > 1:
        X = X / 255.
    print('[emCNN]:    data min/max: %0.2f / %0.2f' % (np.min(X), np.max(X)))

    # Also obtain labels file (if provided - e.g. in deploy mode
    # we may not have labels...)
    if yName: 
        Y = emlib.load_cube(yName, np.float32)

        if onlySlices: 
            Y = Y[onlySlices,:,:] 
        print('[emCNN]:    labels shape: %s' % str(Y.shape))

        # ** ASSUMPTION **: Special case code for membrane detection / ISBI volume
        yAll = np.unique(Y)
        yAll.sort()
        if (len(yAll) == 2) and (yAll[0] == 0) and (yAll[1] == 255):
            print('[emCNN]:    ISBI-style labels detected.  converting 0->1, 255->0')
            Y[Y==0] = 1;      #  membrane
            Y[Y==255] = 0;    #  non-membrane

        # Labels must be natural numbers (contiguous integers starting at 0)
        # because they are mapped to indices at the output of the network.
        # This next bit of code remaps the native y values to these indices.
        omitLabels, pctOmitted = _omit_labels(Y, omitLabels)
        Y = emlib.fix_class_labels(Y, omitLabels).astype(np.int32)

        print('[emCNN]:    yAll is %s' % str(np.unique(Y)))
        print('[emCNN]:    will use %0.2f%% of volume' % (100.0 - pctOmitted))

        Y = emlib.mirror_edges(Y, tileRadius)

        return X, Y
    else:
        return X



#-------------------------------------------------------------------------------
# Functions for training a CNN
#-------------------------------------------------------------------------------


def _xform_minibatch(X, arbitraryRotation=False):
    """Synthetic data augmentation for one mini-batch.

    The default set of data augmentation operations correspond to
    the symmetries of the square (a non abelian group).  The
    elements of this group are:

      o four rotations (0, pi/2, pi, 3*pi/4)
        Denote these by: R0 R1 R2 R3

      o two mirror images (about y-axis or x-axis)
        Denote these by: M1 M2

      o two diagonal flips (about y=-x or y=x)
        Denote these by: D1 D2

    This page has a nice visual depiction:
      http://www.cs.umb.edu/~eb/d4/


    Parameters: 
       X := Mini-batch data (# slices, # channels, rows, colums) 
       
       arbitraryRotation := a boolean; when true, will rotate the mini-batch X
                 by some angle in [0, 2*pi)
    """

    def R0(X):
        return X  # this is the identity map

    def M1(X):
        return X[:,:,::-1,:]

    def M2(X): 
        return X[:,:,:,::-1]

    def D1(X):
        return np.transpose(X, [0, 1, 3, 2])

    def R1(X):
        return D1(M2(X))   # = rot90 on the last two dimensions

    def R2(X):
        return M2(M1(X))

    def R3(X): 
        return D2(M2(X))

    def D2(X):
        return R1(M1(X))


    if arbitraryRotation: 
        # rotation by an arbitrary angle 
        # Note: this is very slow!!
        # Note: this should probably be implemented at a higher level
        #       (than the individual mini-batch) so we can incorporate
        #       context rather than filling in pixels with a background color.
        angle = np.random.rand() * 360.0 
        fillColor = np.max(X) 
        X2 = scipy.ndimage.rotate(X, angle, axes=(2,3), reshape=False, cval=fillColor)
    else:
        symmetries = [R0, R1, R2, R3, M1, M2, D1, D2]
        op = choice(symmetries) 
        
        # For some reason, the implementation of row and column reversals, 
        #     e.g.      X[:,:,::-1,:]
        # break PyCaffe.  Numpy must be doing something under the hood 
        # (e.g. changing from C order to Fortran order) to implement this 
        # efficiently which is incompatible w/ PyCaffe.  
        # Hence the explicit construction of X2 with order 'C' below.
        X2 = np.zeros(X.shape, dtype=np.float32, order='C') 
        X2[...] = op(X)

    return X2



def train_one_epoch(solverMD, X, Y, 
        batchDim,
        outDir='./', 
        omitLabels=[], 
        data_augment=None):
    """ Trains a CNN for a single epoch.

    PARAMETERS:
      solverMD  : an SGDSolverMemoryData object
      X         : a data volume/tensor with dimensions (#slices, height, width)
      Y         : a labels tensor with same size as X
      batchDim  : the tuple (#classes, minibatchSize, height, width)
      outDir    : output directory (e.g. for model snapshots)
      omitLabels : class labels to skip during training (or [] for none)
      data_agument : synthetic data augmentation function

    """

    # Pre-allocate some variables & storage.
    #
    tileRadius = int(batchDim[2]/2)
    Xi = np.zeros(batchDim, dtype=np.float32)
    yi = np.zeros((batchDim[0],), dtype=np.float32)
    yMax = np.max(Y).astype(np.int32)
    
    lastChatter = -2
    startTime = time.time()

    it = emlib.stratified_interior_pixel_generator(Y, tileRadius, batchDim[0], omitLabels=omitLabels) 

    for Idx, epochPct in it: 
        # Map the indices Idx -> tiles Xi and labels yi 
        # 
        # Note: if Idx.shape[0] < batchDim[0] (last iteration of an epoch) 
        # a few examples from the previous minibatch will be "recycled" here. 
        # This is intentional (to keep batch sizes consistent even if data 
        # set size is not a multiple of the minibatch size). 
        # 
        for jj in range(Idx.shape[0]): 
            a = Idx[jj,1] - tileRadius 
            b = Idx[jj,1] + tileRadius + 1 
            c = Idx[jj,2] - tileRadius 
            d = Idx[jj,2] + tileRadius + 1 
            Xi[jj, 0, :, :] = X[ Idx[jj,0], a:b, c:d ]
            yi[jj] = Y[ Idx[jj,0], Idx[jj,1], Idx[jj,2] ] 

        # label-preserving data transformation (synthetic data generation)
        if data_augment is not None:
            Xi = data_augment(Xi)

        assert(not np.any(np.isnan(Xi)))
        assert(not np.any(np.isnan(yi)))

        #----------------------------------------
        # one forward/backward pass and update weights
        #----------------------------------------
        out = solverMD.step(Xi, yi)

        #----------------------------------------
        # Some events occur on regular intervals.
        # Address these here.
        #----------------------------------------
        if solverMD.is_time_for_snapshot():
            fn = os.path.join(outDir, 'iter_%06d.caffemodel' % solverMD._iter)
            solverMD._solver.net.save(str(fn))
            print "[emCNN]: Saved snapshot."

        if solverMD.is_training_complete():
            break  # we hit max_iter on a non-epoch boundary...all done.


        elapsed = (time.time() - startTime) / 60.0
        if (lastChatter+2) < elapsed:  # notify progress every 2 min
            lastChatter = elapsed
            print('[emCNN]: we are %0.2f%% complete with this epoch' % (100.*epochPct))
            sys.stdout.flush()
                
    # all finished with this epoch
    print "[emCNN]:    epoch complete."
    sys.stdout.flush()



def _train_network(args):
    """ Main CNN training loop.

    Creates PyCaffe objects and calls train_one_epoch until done.
    """
    #----------------------------------------
    # parse information from the prototxt files
    #----------------------------------------
    solverParam = caffe_pb2.SolverParameter()
    text_format.Merge(open(args.solver).read(), solverParam)

    netFn = solverParam.net
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFn).read(), netParam)

    batchDim = emlib.infer_data_dimensions(netFn)
    assert(batchDim[2] == batchDim[3])  # tiles must be square
    print('[emCNN]: batch shape: %s' % str(batchDim))
   
    if args.outDir:
        outDir = args.outDir # overrides snapshot prefix
    else: 
        outDir = str(solverParam.snapshot_prefix)   # unicode -> str
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    # choose a synthetic data generating function
    if args.rotateData:
        syn_func = lambda V: _xform_minibatch(V, arbitraryRotation=True)
        print('[emCNN]:   WARNING: applying arbitrary rotations to data.  This may degrade performance in some cases...\n')
    else:
        syn_func = lambda V: _xform_minibatch(V, arbitraryRotation=False)


    #----------------------------------------
    # Create the Caffe solver
    # Note this assumes a relatively recent PyCaffe
    #----------------------------------------
    solver = caffe.SGDSolver(args.solver)
    solverMD = SGDSolverMemoryData(solver, solverParam)
    solverMD.print_network()

    #----------------------------------------
    # Load data
    #----------------------------------------
    bs = border_size(batchDim)
    print "[emCNN]: tile radius is: %d" % bs
    
    print "[emCNN]: loading training data..."
    Xtrain, Ytrain = _load_data(args.emTrainFile,
            args.labelsTrainFile,
            tileRadius=bs,
            onlySlices=args.trainSlices,
            omitLabels=args.omitLabels)
    
    print "[emCNN]: loading validation data..."
    Xvalid, Yvalid = _load_data(args.emValidFile,
            args.labelsValidFile,
            tileRadius=bs,
            onlySlices=args.validSlices,
            omitLabels=args.omitLabels)

    #----------------------------------------
    # Do training; save results
    #----------------------------------------
    omitLabels = set(args.omitLabels).union([-1,])   # always omit -1
    currEpoch = 1
    sys.stdout.flush()

    while not solverMD.is_training_complete():
        print "[emCNN]: Starting epoch %d" % currEpoch

        train_one_epoch(solverMD, Xtrain, Ytrain, 
            batchDim, outDir, 
            omitLabels=omitLabels,
            data_augment=syn_func)

        currEpoch += 1

        print "[emCNN]: Making predictions on validation data..."
        Mask = np.ones(Xvalid.shape, dtype=np.bool)
        Mask[Yvalid<0] = False
        Prob = predict(solver.net, Xvalid, Mask, batchDim)

        # discard mirrored edges and form class estimates
        Yhat = np.argmax(Prob, 0) 
        Yhat[Mask==False] = -1;
        Prob = prune_border_4d(Prob, bs)
        Yhat = prune_border_3d(Yhat, bs)

        # compute some metrics
        print('[emCNN]: Validation set performance:')
        emlib.metrics(prune_border_3d(Yvalid, bs), Yhat, display=True)

 
    solver.net.save(str(os.path.join(outDir, 'final.caffemodel')))
    np.save(os.path.join(outDir, 'Yhat.npz'), Prob)
    scipy.io.savemat(os.path.join(outDir, 'Yhat.mat'), {'Yhat' : Prob})
    print('[emCNN]: training complete.')



#-------------------------------------------------------------------------------
# Functions for "deploying" a CNN (i.e. forward pass only)
#-------------------------------------------------------------------------------


def predict(net, X, Mask, batchDim, nMC=0):
    """Generates predictions for a data volume.

    PARAMETERS:
      X        : a data volume/tensor with dimensions (#slices, height, width)
      Mask     : a boolean tensor with the same size as X.  Only positive
                 elements will be classified.  The prediction for all 
                 negative elements will be -1.  Use this to run predictions
                 on a subset of the volume.
      batchDim : a tuple of the form (#classes, minibatchSize, height, width)

    """    
    # *** This code assumes a layer called "prob"
    if 'prob' not in net.blobs: 
        raise RuntimeError("Can't find a layer called 'prob'")

    print "[emCNN]: Evaluating %0.2f%% of cube" % (100.0*np.sum(Mask)/numel(Mask)) 

    # Pre-allocate some variables & storage.
    #
    tileRadius = int(batchDim[2]/2)
    Xi = np.zeros(batchDim, dtype=np.float32)
    yi = np.zeros((batchDim[0],), dtype=np.float32)
    nClasses = net.blobs['prob'].data.shape[1]

    # if we don't evaluate all pixels, the 
    # ones not evaluated will have label -1
    if nMC <= 0: 
        Prob = -1*np.ones((nClasses, X.shape[0], X.shape[1], X.shape[2]))
    else:
        Prob = -1*np.ones((nMC, X.shape[0], X.shape[1], X.shape[2]))
        print "[emCNN]: Generating %d MC samples for class 0" % nMC
        if nClasses > 2: 
            print "[emCNN]: !!!WARNING!!! nClasses > 2 but we are only extracting MC samples for class 0 at this time..."


    # do it
    startTime = time.time()
    cnnTime = 0
    lastChatter = -2
    it = emlib.interior_pixel_generator(X, tileRadius, batchDim[0], mask=Mask)

    for Idx, epochPct in it: 
        # Extract subtiles from validation data set 
        for jj in range(Idx.shape[0]): 
            a = Idx[jj,1] - tileRadius 
            b = Idx[jj,1] + tileRadius + 1 
            c = Idx[jj,2] - tileRadius 
            d = Idx[jj,2] + tileRadius + 1 
            Xi[jj, 0, :, :] = X[ Idx[jj,0], a:b, c:d ]
            yi[jj] = 0  # this is just a dummy value

        #---------------------------------------- 
        # forward pass only (i.e. no backward pass)
        #----------------------------------------
        if nMC <= 0: 
            # this is the typical case - just one forward pass
            _tmp = time.time() 
            net.set_input_arrays(Xi, yi)
            out = net.forward() 
            cnnTime += time.time() - _tmp 
            
            # On some version of Caffe, Prob is (batchSize, nClasses, 1, 1) 
            # On newer versions, it is natively (batchSize, nClasses) 
            # The squeeze here is to accommodate older versions 
            ProbBatch = np.squeeze(out['prob']) 
            
            # store the per-class probability estimates.  
            # 
            # * On the final iteration, the size of Prob  may not match 
            #   the remaining space in Yhat (unless we get lucky and the 
            #   data cube size is a multiple of the mini-batch size).  
            #   This is why we slice yijHat before assigning to Yhat. 
            for jj in range(nClasses): 
                pj = ProbBatch[:,jj]      # get probabilities for class j 
                assert(len(pj.shape)==1)  # should be a vector (vs tensor) 
                Prob[jj, Idx[:,0], Idx[:,1], Idx[:,2]] = pj[:Idx.shape[0]]   # (*)
        else:
            # Generate MC-based uncertainty estimates
            # (instead of just a single point estimate)
            _tmp = time.time() 
            net.set_input_arrays(Xi, yi)

            # do nMC forward passes and save the probability estimate
            # for class 0.
            for ii in range(nMC): 
                out = net.forward() 
                ProbBatch = np.squeeze(out['prob']) 
                p0 = ProbBatch[:,0]      # get probabilities for class 0
                assert(len(p0.shape)==1)  # should be a vector (vs tensor) 
                Prob[ii, Idx[:,0], Idx[:,1], Idx[:,2]] = p0[:Idx.shape[0]]   # (*)
            cnnTime += time.time() - _tmp 
            

        elapsed = (time.time() - startTime) / 60.0

        if (lastChatter+2) < elapsed:  # notify progress every 2 min
            lastChatter = elapsed
            print('[emCNN]: elapsed=%0.2f min; %0.2f%% complete' % (elapsed, 100.*epochPct))
            sys.stdout.flush()

    # done
    print('[emCNN]: Total time to evaluate cube: %0.2f min (%0.2f CNN min)' % (elapsed, cnnTime/60.))
    return Prob




def _deploy_network(args):
    """ Runs Caffe in deploy mode (where there is no solver).
    """

    #----------------------------------------
    # parse information from the prototxt files
    #----------------------------------------
    netFn = str(args.network)  # unicode->str to avoid caffe API problems
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFn).read(), netParam)

    batchDim = emlib.infer_data_dimensions(netFn)
    assert(batchDim[2] == batchDim[3])  # tiles must be square
    print('[emCNN]: batch shape: %s' % str(batchDim))

    if args.outDir:
        outDir = args.outDir  # overrides default
    else: 
        # there is no snapshot dir in a network file, so default is
        # to just use the location of the network file.
        outDir = os.path.dirname(args.network)

    # Add a timestamped subdirectory.
    ts = datetime.datetime.now()
    subdir = "Deploy_%s_%02d:%02d" % (ts.date(), ts.hour, ts.minute)
    outDir = os.path.join(outDir, subdir)

    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    print('[emCNN]: writing results to: %s' % outDir)

    # save the parameters we're using
    with open(os.path.join(outDir, 'params.txt'), 'w') as f:
        pprint(args, stream=f)

    #----------------------------------------
    # Create the Caffe network
    # Note this assumes a relatively recent PyCaffe
    #----------------------------------------
    phaseTest = 1  # 1 := test mode
    net = caffe.Net(netFn, args.model, phaseTest)

    #----------------------------------------
    # Load data
    #----------------------------------------
    bs = border_size(batchDim)
    print "[emCNN]: loading deploy data..."
    Xdeploy = _load_data(args.emDeployFile,
                         None,
                         tileRadius=bs,
                         onlySlices=args.deploySlices)
    print "[emCNN]: tile radius is: %d" % bs

    # Create a mask volume (vs list of labels to omit) due to API of emlib
    if args.evalPct < 1: 
        Mask = np.zeros(Xdeploy.shape, dtype=np.bool)
        m = Xdeploy.shape[-2]
        n = Xdeploy.shape[-1]
        nToEval = np.round(args.evalPct*m*n).astype(np.int32)
        idx = sobol(2, nToEval ,0)
        idx[0] = np.floor(m*idx[0])
        idx[1] = np.floor(n*idx[1])
        idx = idx.astype(np.int32)
        Mask[:,idx[0], idx[1]] = True
        pct = 100.*np.sum(Mask) / Mask.size
        print("[emCNN]: subsampling volume...%0.2f%% remains" % pct)
    else:
        Mask = np.ones(Xdeploy.shape, dtype=np.bool)

    #----------------------------------------
    # Do deployment & save results
    #----------------------------------------
    sys.stdout.flush()

    if args.nMC < 0: 
        Prob = predict(net, Xdeploy, Mask, batchDim)
    else:
        Prob = predict(net, Xdeploy, Mask, batchDim, nMC=args.nMC)

    # discard mirrored edges 
    Prob = prune_border_4d(Prob, bs)

    net.save(str(os.path.join(outDir, 'final.caffemodel')))
    np.save(os.path.join(outDir, 'YhatDeploy'), Prob)
    scipy.io.savemat(os.path.join(outDir, 'YhatDeploy.mat'), {'Yhat' : Prob})

    print('[emCNN]: deployment complete.')



#-------------------------------------------------------------------------------
if __name__ == "__main__":
    args = _get_args()

    # these are imported here to facilitate unit testing on machines that
    # don't have caffe
    import caffe
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format

    # specify CPU or GPU
    if args.gpu >= 0:
	caffe.set_mode_gpu()
	caffe.set_device(args.gpu)
    else:
	caffe.set_mode_cpu()

    print(args)

    # train or deploy
    if args.mode == 'train':
        _train_network(args)
    else:
        _deploy_network(args)


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
