#!/bin/bash

# set up conda env
ENVNAME=rl-augment
ENVDIR=${ENVNAME}_env
mkdir $ENVDIR
tar -xzf ${ENVNAME}.tar.gz -C $ENVDIR
rm ${ENVNAME}.tar.gz # remove env tarball
source $ENVDIR/bin/activate

git clone git@github.com:NicholasCorrado/Gymnasium-Robotics.git -b no-dict

pip install -e .
pip install -e augment/my-gym
pip install -e augment/Gymnasium-Robotics