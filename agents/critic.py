import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig

import utils.utils as utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dim, hidden_depth, goal_generator_cfg=None, device=None,
                 goal_space=None, goal_observation_space=None):
        super().__init__()
        self.goal_gen = None
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        if isinstance(goal_generator_cfg, DictConfig):
            obs_dim =  goal_observation_space['full_state'].shape[-1] + goal_observation_space['latent'].shape[-1]
        elif goal_dim != 0:
            self.obs_embed = utils.mlp(obs_dim, hidden_dim//2, hidden_dim//2,
                               hidden_depth)
            self.goal_embed = utils.mlp(goal_dim, hidden_dim//2, hidden_dim//2,
                               hidden_depth)
            
            obs_dim = hidden_dim

            

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        if isinstance(obs, dict):
            # obs = self.goal_gen.input_forward(obs)
            concat_obs = []
            for k, v in obs.items():
                concat_obs.append(v)
            obs =  torch.cat(concat_obs, axis=-1)
        elif self.goal_dim!=0:
            assert self.obs_dim + self.goal_dim == obs.shape[-1]
            obs_embed = self.obs_embed(obs[..., :self.obs_dim])
            goal_embed = self.goal_embed(obs[..., -self.goal_dim:])
            obs = torch.cat([obs_embed, goal_embed], axis=-1)
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)
