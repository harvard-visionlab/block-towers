# block-towers
Code for generating and rendering block towers using MuJoCo physics simulator via dm_control

# environment

Create a mujoco environment, install dependencies, and add install kernel (jupyterlab/jupyternotebook)
```
conda create --name mujoco python=3.10.12
source activate mujoco
python3 -m pip install -r requirements.txt
ipython kernel install --user --name=mujoco
python3 -m pip install -e .
python3 env_test.py
```