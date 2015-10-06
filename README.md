# faster-membranes
This repository hosts an experiment for segmenting EM images, with the goal of working more quickly than brute-force application of CNNs on a per-pixel basis.

## Quick start

o  Install Caffe (including the Python interface) and Caffe con Troll (CcT).
o  Edit paths towards the top of the Makefile as needed for your system.
o  Preprocess ISBI2012 by calling "make data"
o  Do either a timing experiment or extract probability estimates.  See
   the Makefile for examples of how to do both of these things.
