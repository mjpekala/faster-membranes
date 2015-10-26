""" PYCAFFE2 : Code to facilitate interacting with Caffe
               via Python.
"""


__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"


import copy, time, sys
import numpy as np



class SGDSolverMemoryData:
    """
    This class is a *partial* re-implementation of the Caffe code
    that resides in src/caffe/solver.cpp.  Only a subset of Caffe's
    capabilities are supported here.

    That this object exists is an artifact of:

      o An issue/incompatability between PyCaffe and the memory
        data layer object (which has issues of its own) and

      o A need for us to use the memory data layer so that we
        can extract sliding windows on-the-fly (vs needing
        to pre-generate all possible tiles, which would 
        consume an infeasible amount of storage).

    I'm not particularly happy about this; once the python 
    caffe interface matures (or we move away from sliding
    windows to some more elegant way of computing dense estimates)
    this code can be retired.
    """

    def __init__(self, solver, param, verbose=True):
        if param.lr_policy not in [u'step', u'inv']:
            raise ValueError('sorry - I only support "step" or "inv" policies at this time')

        if param.solver_type != param.SolverType.Value('SGD'):
            raise ValueError('sorry - I only support SGD at this time')

        self._param = copy.deepcopy(param) # not sure if caffe mucks with these uder the hood as training progresses - hopefully not, but just in case...
        self._solver = solver
	self._verbose = verbose

        self._cnnTime = 0.0   # := track time spent doing CNN ops
	self._bornTime = time.time()
        self._iter = 0        # := current training iteration
        self._V = {}          # := previous SGD updates (for momentum)

        self._lastLoss = np.NaN
        self._lastAcc = np.NaN

        self._nHist = 30
        self._lastLossN = []


    def step(self, X, y):
        out = self._step(X, y, self._iter)
        self._iter += 1

	if self._verbose and (self._iter % self._param.display) == 1:
		self.print_state()

        return out


    def is_training_complete(self):
        return (self._iter >= self._param.max_iter)


    def is_time_for_snapshot(self):
        return ((self._iter % self._param.snapshot) == 0)


    def print_state(self):
        """Displays info. about learning progress to stdout.
        """

        print "[SGD2]: completed iteration %d (of %d)" % (self._iter, self._param.max_iter)

        totalTime = (time.time() - self._bornTime) / 60.0
        print "[SGD2]: %0.2f min elapsed (%0.2f CNN min)" % (totalTime, self._cnnTime/60.)
        print "[SGD2]:   learning rate:    %0.2e" % self._get_learning_rate(self._iter)
        if len(self._lastLossN): 
            print "[SGD2]:   last loss:        %0.2e" % self._lastLossN[-1]
            print "[SGD2]:   avg loss:         %0.2e" % np.mean(self._lastLossN)
        if self._lastAcc: 
            print "[SGD2]:   last acc (train): %0.2f" % self._lastAcc 

        sys.stdout.flush()



    def print_network(self): 
        """Displays network information to stdout.
        """ 
        # XXX: difference between params and layers?
        for name, blobs in self._solver.net.params.iteritems(): 
            for bIdx, b in enumerate(blobs): 
                print("  %s[%d] : %s" % (name, bIdx, b.data.shape))

        for ii,layer in enumerate(self._solver.net.layers):
            print("  layer %d: %s" % (ii, layer.type))
            for jj,blob in enumerate(layer.blobs):
                print("    blob %d has size %s" % (jj, str(blob.data.shape)))


    def _get_learning_rate(self, currIter):
        """
        Returns the learning rate as a function of the current iteration.

        Based on SGDSolver::GetLearningRate in caffe.cpp
        \begin{quote}
          The currently implemented learning rate policies are:
           - fixed: always return base_lr.
           - step: return base_lr * gamma ^ (floor(iter / step))
           - exp: return base_lr * gamma ^ iter
           - inv: return base_lr * (1 + gamma * iter) ^ (- power)
           - multistep: similar to step but it allows non uniform steps defined by
           stepvalue
           - poly: the effective learning rate follows a polynomial decay, to be
           zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
           - sigmoid: the effective learning rate follows a sigmod decay
           return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
           
           where base_lr, max_iter, gamma, step, stepvalue and power are defined
           in the solver parameter protocol buffer, and iter is the current iter
         \end{quote}
        """
        if self._param.lr_policy == u'step':
            _tmp = self._param.gamma ** (np.floor(currIter/self._param.stepsize))
            return self._param.base_lr * _tmp
        elif self._param.lr_policy == u'inv':
            _tmp = (1.0 + self._param.gamma * currIter) ** (-1.0*self._param.power)
            return self._param.base_lr * _tmp;
        else:
            raise RuntimeError('currently unsupported learning rate policy')



    def _step(self, X, y, currIter):
        """
        Executes a single forward/backward pass of SGD.

          X : The inputs/features for this minibatch; dimensions:
               (#examples, #channels, height, width)
          y : A 1d vector of features for this minibatch; dimensions:
               (#examples,)
          currIter : The current training iteration (scalar integer)

        Note that you can pass this function any training iteration
        you like.  Usually this will be self._iter.

        See SGDSolver::ComputeUpdateValue() in caffe.cpp
        """

        # convert vector of labels into an appropriately-sized tensor.
        if y.ndim == 1: 
            yTensor = np.ascontiguousarray(y[:, np.newaxis, np.newaxis, np.newaxis]) 
        else:
            yTensor = y
        assert(yTensor.ndim) == 4

        # caffe wants data to be float32
        yTensor = yTensor.astype(np.float32)
        X = X.astype(np.float32)

        # the learning rate for this iteration
        alpha = self._get_learning_rate(currIter)

        tic = time.time() 
        self._solver.net.set_input_arrays(X, yTensor) 
        out = self._solver.net.forward() 
        self._solver.net.backward() 

        for lIdx, layer in enumerate(self._solver.net.layers): 
            for bIdx, blob in enumerate(layer.blobs): 
                if np.any(np.isnan(blob.diff)): 
                    raise RuntimeError("NaN detected in gradient of layer %d" % lIdx) 
                # Some networks scale the learning rate for weights 
                # and biases differently; handle that here:
                # TODO: figure out where the layer-specific parameters live.
                #    In c++, they are here:
                #      this->net_->params_lr()
                #      this->net_->params_weight_decay() 
                # Unclear if/where these are available via python...
                alphaLocal = alpha * 1.0 
                decayLocal = self._param.weight_decay * 1.0

                # SGD with momentum
                key = (lIdx, bIdx) 
                V = self._V.get(key, 0.0) 
                Vnext = self._param.momentum * V - alphaLocal * blob.diff
                blob.data[...] += Vnext
                self._V[key] = Vnext 
                       
                # also, weight decay (optional)
                # XXX: make sure it is correct to apply in this
                #      manner (i.e. apart from momentum)
                #
                #   w_i <- w_i - alpha \nabla grad - alpha * lambda * w_i
                #
                if decayLocal > 0:
                    blob.data[...] *= (1.0 - alphaLocal * decayLocal)

        self._cnnTime += time.time() - tic

        # try to extract some performance information.
        # makes assumptions about naming conventions of CNN layers.
        # Use try/catch in case these assumptions do not hold.
        try: 
            loss = np.squeeze(out.get('loss', None))
            if loss:
                self._lastLossN.append(float(loss))
            while len(self._lastLossN) > self._nHist:
                self._lastLossN.pop(0)
        except: 
            pass

        try:
            self._lastAcc = np.squeeze(out.get('acc', None))
        except:
            pass

        return out


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
