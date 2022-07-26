#!/bin/bash

# set up conda env
mkdir augment_env
tar -xzf augment.tar.gz -C augment_env
rm augment.tar.gz # remove env tarball
source augment_env/bin/activate

# set up  mujoco
source ./mujoco_setup.sh
