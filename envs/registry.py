from envs.SelfPlay import *
from gym.envs.robotics import FetchPushEnv
from gym.wrappers import TimeLimit
import dmc2gym
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
import torch
import numpy as np

def make_vec_envs(env_name,
                  seed,
                  log_dir,
                  num_processes,
                  max_episode_steps=None,
                  num_subgoals=1,
                  additional_dims=0):
    envs = [
        make_env(env_name, seed, log_dir, max_episode_steps, allow_early_resets=True,  num_subgoals=num_subgoals, additional_dims=additional_dims)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    return envs

def make_env(env_name, seed, log_dir, max_episode_steps=None, allow_early_resets=True, num_subgoals=1, additional_dims=0):
    def _thunk():
        if (env_name == 'fetch_push'):
            env = FetchPushEnv()
            env.seed(seed)
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
            env = SelfPlayEnvFetchPush(additional_dims, env)

        elif (env_name == 'point_mass'):
            env = dmc2gym.make(domain_name='point_mass', task_name='easy', seed=seed, episode_length=max_episode_steps)
            env = SelfPlayEnvPointMass(None, additional_dims, env)

        elif ('point_mass_maze' in env_name):
            maze_num = env_name[-1]
            if maze_num.isdigit():
                task_name = 'maze' + maze_num
            else:
                task_name = 'maze'
            env = dmc2gym.make(domain_name='point_mass', task_name=task_name, seed=seed, episode_length=max_episode_steps)
            env = SelfPlayEnvPointMass(maze_num, additional_dims, env)

        elif ('manipulator' in env_name):
            env = dmc2gym.make(domain_name='manipulator', task_name='bring_ball', seed=seed, episode_length=max_episode_steps)
            if 'toss' in env_name:
                if 'multi' in env_name:
                    env = SelfPlayEnvManipulatorMultiGoal('toss', num_subgoals, env)
                else:
                    env = SelfPlayEnvManipulator('toss', additional_dims, env)
            elif 'reach' in env_name:
                if 'multi' in env_name:
                    env = SelfPlayEnvManipulatorMultiGoal('reach', num_subgoals, env)
                else:
                    env = SelfPlayEnvManipulator('reach', additional_dims, env)
            else:
                raise NotImplementedError
            
        elif ('walker' in env_name):
            env = dmc2gym.make(domain_name='walker', task_name='walk', seed=seed, episode_length=max_episode_steps)
            env = SelfPlayEnvWalker('small' in env_name, additional_dims, env)

        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir),
                allow_early_resets=allow_early_resets)
        return env

    return _thunk