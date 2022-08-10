#!/bin/bash

# set up conda env
ENVNAME=rl-augment
ENVDIR=${ENVNAME}_env
mkdir $ENVDIR
tar -xzf ${ENVNAME}.tar.gz -C $ENVDIR
rm ${ENVNAME}.tar.gz # remove env tarball
source $ENVDIR/bin/activate

# set up  mujoco
#./mujoco_get_dependencies.sh
source ./mujoco_setup.sh

git clone https://github.com/DLR-RM/stable-baselines3.git
pip install -e stable-baselines3
pip install -e my-gym

pip install -e .