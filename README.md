# RL-Augment

Install MuJoCo for Apple M1 Mac (see https://github.com/openai/mujoco-py/issues/662)
```commandline
pip install mujoco_py
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210/


mkdir -p $HOME/.mujoco/mujoco210         # Remove existing installation if any
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/Headers/ $HOME/.mujoco/mujoco210/include
mkdir -p $HOME/.mujoco/mujoco210/bin
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.*.dylib $HOME/.mujoco/mujoco210/bin/libmujoco210.dylib
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.*.dylib /usr/local/lib/

# For M1 (arm64) mac users:
# The released binary doesn't ship glfw3, so need to install on your own
brew install glfw
ln -sf /opt/homebrew/lib/libglfw.3.dylib $HOME/.mujoco/mujoco210/bin

# Please make sure /opt/homebrew/bin/gcc-11  exists: install gcc if you haven't already
# brew install gcc
export CC=/opt/homebrew/bin/gcc-11         # see https://github.com/openai/mujoco-py/issues/605

pip install mujoco-py && python -c 'import mujoco_py'
```
