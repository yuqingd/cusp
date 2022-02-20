# README

Codebase for It Takes Four to Tango: Multiagent Self Play for Automatic Curriculum Generation. Built off of the Pytorch SAC implementation at https://github.com/denisyarats/pytorch_sac. 

## Installation Instructions
### For CUDA:
1. ``sudo apt-get install -y libglew-dev``
2. Add to .bash_rc ``export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so``
3. install cuda https://pytorch.org/get-started/locally/ 

### Setup repository
1. Install conda for creating the virtual env: https://docs.conda.io/en/latest/miniconda.html 
2. Install Mujoco: https://www.roboti.us/index.html
3. Install any necessary packages for DMC: https://github.com/deepmind/dm_control 
    1.  for Mujoco-py: ``sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3``, ``sudo apt-get install patchelf. ``
4. Install dmc2gym: https://github.com/denisyarats/dmc2gym 
5. Clone the repository and create the conda env using 
``conda env create -f environment.yml`` and activate the env
6. Install local version of dm_control for modified envs: ``pip install -e dm_control/``


## Training
To train CuSP, run

`` python train.py env_name=point_mass exp_name=test num_steps=6e3 goal_algo=cusp seed=0 num_steps_alice=100 num_steps_bob=100 symmetrize=True before_update_stale_regrets=50 stale_regret_coeff=.9 ``

| Command      | Description |
| ----------- | ----------- |
| ``env_name``      | Specify env to run --     ``point_mass, point_mass_maze0, manipulator_reach, manipulator_toss, walker``      |
| ``num_steps``      | Total number of training rounds       |
| ``goal_algo``      | Goal generation algorithm -- ``cusp, asp, goalgan, dr``       |
| ``num_steps_alice``   | Max Alice trajectory length        |
| ``num_steps_bob``   | Max Bob trajectory length        |
| ``symmetrize``    | If true, ymmetrize training setup to have two goal generators        |
| ``before_update_stale_regrets``    |  Training episode at which we begin stale regret updates      |
| ``stale_regret_coeff``    |  Weighing (beta) of regret updates     |

For detailed configs, see ``config/train.yaml``.  Results will be logged in ``logdir/``.
