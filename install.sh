#!/bin/bash

# set up conda env
ENVNAME=rl-augment
ENVDIR=${ENVNAME}_env
cp /staging/ncorrado/${ENVNAME}.tar.gz .
mkdir $ENVDIR
tar -xzf ${ENVNAME}.tar.gz -C $ENVDIR
rm ${ENVNAME}.tar.gz # remove env tarball
source $ENVDIR/bin/activate


pip install -e .

cd augment
git clone https://github.com/NicholasCorrado/Gymnasium-Robotics.git -b sb3
git clone https://github.com/NicholasCorrado/panda-gym.git -b sb3
pip install -e Gymnasium-Robotics
pip install -e panda-gym
pip install -e my-gym
cd ..
