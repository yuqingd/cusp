import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

import utils.utils as utils
import hydra
from omegaconf.dictconfig import DictConfig



class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds, goal_generator_cfg=None, device=None,
                 goal_space=None, goal_observation_space=None):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.goal_gen = None
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        if isinstance(goal_generator_cfg, DictConfig):
            obs_dim =  goal_observation_space['full_state'].shape[-1] + goal_observation_space['latent'].shape[-1]
            self.goal_space = goal_space
            self.device = device
        elif goal_dim != 0:
            self.obs_embed = utils.mlp(obs_dim, hidden_dim//2, hidden_dim//2,
                               hidden_depth)
            self.goal_embed = utils.mlp(goal_dim, hidden_dim//2, hidden_dim//2,
                               hidden_depth)
            
            obs_dim = hidden_dim

        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.goal_space = goal_space
        self.apply(utils.weight_init)

    def forward(self, obs):
        
        if isinstance(obs, dict):
            # obs = self.goal_gen.input_forward(obs)
            concat_obs = []
            for k, v in obs.items():
                concat_obs.append(v)
            obs =  torch.cat(concat_obs, axis=-1)
        elif self.goal_dim != 0:
            assert self.obs_dim + self.goal_dim == obs.shape[-1]
            obs_embed = self.obs_embed(obs[..., :self.obs_dim])
            goal_embed = self.goal_embed(obs[..., -self.goal_dim:])
            obs = torch.cat([obs_embed, goal_embed], axis=-1) 

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                        1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std
        
        if self.goal_space is not None:
            if (self.goal_space.high == float("inf")).any() or (self.goal_space.low == float("-inf")).any():
                dist = pyd.Normal(mu, std)
            else: 
                dist = SquashedNormal(mu, std)
        else:
            dist = SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)