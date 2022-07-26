#!/bin/bash

# set up conda env
mkdir augment_env
tar -xzf augment.tar.gz -C augment_env
rm augment.tar.gz # remove env tarball
source augment_env/bin/activate

# set up  mujoco
./mujoco_get_dependencies.sh
source ./mujoco_setup.sh

git clone -b latent https://github.com/DLR-RM/stable-baselines3.git
pip install -e stable-baselines3

pip install -e .