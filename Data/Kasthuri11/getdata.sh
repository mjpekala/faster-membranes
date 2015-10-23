#!/bin/bash

curl http://openconnecto.me/ocp/ca/ac3ac4/ac4membraneIDSIA/hdf5/1/4400,5424/5440,6464/1100,1200/ > train-labels.hdf5

curl http://openconnecto.me/ocp/ca/kasthuri11cc/image/hdf5/1/4400,5424/5440,6464/1100,1200/ > train-volume.hdf5

curl http://openconnecto.me/ocp/ca/kasthuri11cc/image/hdf5/1/1/5472,6496/8712,9736/1000,1256/ > test-volume.hdf5


