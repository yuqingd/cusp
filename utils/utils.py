
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import time
from gym.spaces import Dict, Box, Discrete, Tuple, MultiBinary, MultiDiscrete
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import imageio
from torch import distributions as pyd
import random
import math

def flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

def to_tensor(obs, device):
    if isinstance(obs, dict):
        tensor_obs_dict = {}
        for k, v in obs.items():
            tensor_obs_dict[k] = v.to(device) if torch.is_tensor(v) else torch.FloatTensor(v).to(device)
        return tensor_obs_dict
    else:
        return torch.FloatTensor(obs).to(device)
        
class VideoRecorder(object):
    def __init__(self, dir_name, height=420, width=420, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            try:
                frame = env.render(
                    mode='rgb_array',
                    height=self.height,
                    width=self.width,
                    camera_id=self.camera_id
                )
            except TypeError:
                frame = env.render(
                    mode='rgb_array',
                )
            if frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)


def unflatten(space, x):
    if isinstance(space, MultiBinary):
        return np.asarray(x).reshape(space.shape)
    elif isinstance(space, MultiDiscrete):
        return np.asarray(x).reshape(space.shape)
    else:
        raise NotImplementedError


def flatten(space, x):
    """Flatten a data point from a space.

    This is useful when e.g. points from spaces must be passed to a neural
    network, which only understands flat arrays of floats.

    Accepts a space and a point from that space. Always returns a 1D array.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    """
    if isinstance(space, Box):
        return np.asarray(x, dtype=space.dtype).flatten()
    elif isinstance(space, Discrete):
        onehot = np.zeros(space.n, dtype=space.dtype)
        onehot[x] = 1
        return onehot
    elif isinstance(space, Tuple):
        return np.concatenate(
            [flatten(s, x_part) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, dict):
        flattened = []
        for val in x.values():
            flattened.extend(val.squeeze() if isinstance(val[0], np.ndarray) else val)
        return flattened
    elif isinstance(space, MultiBinary):
        return np.asarray(x, dtype=space.dtype).flatten()
    elif isinstance(space, MultiDiscrete):
        return np.asarray(x, dtype=space.dtype).flatten()
    else:
        raise NotImplementedError


def flatten_space(space):
    """Flatten a space into a single ``Box``.
    This is equivalent to ``flatten()``, but operates on the space itself. The
    result always is a `Box` with flat boundaries. The box has exactly
    ``flatdim(space)`` dimensions. Flattening a sample of the original space
    has the same effect as taking a sample of the flattenend space.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    """

    if isinstance(space, Box):
        return Box(space.low.flatten(), space.high.flatten(), dtype=space.dtype)
    if isinstance(space, Discrete):
        return Box(low=0, high=1, shape=(space.n, ), dtype=space.dtype)
    if isinstance(space, Tuple):
        space = [flatten_space(s) for s in space.spaces]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
            dtype=np.result_type(*[s.dtype for s in space])
        )
    if isinstance(space, Dict):
        space = [flatten_space(s) for s in space.spaces.values()]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
            dtype=np.result_type(*[s.dtype for s in space])
        )
    if isinstance(space, MultiBinary):
        return Box(low=0,
                   high=1,
                   shape=(space.n, ),
                   dtype=space.dtype
                   )
    if isinstance(space, MultiDiscrete):
        return Discrete(sum(space.nvec))

    raise NotImplementedError

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])



class RolloutStorage(object):
    def __init__(self, num_steps, obs_space, action_space,
                 recurrent_hidden_state_size, goal_shape=None, goal_components=None, goal_buffer_size=None, num_processes=1):
        if isinstance(obs_space, Dict):
            self.obs = {}
            for k, v in obs_space.spaces.items():
                self.obs[k] = torch.zeros(num_steps + 1, num_processes, *v.shape)
        else:
            obs_shape = obs_space.shape
            self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)

        if isinstance(recurrent_hidden_state_size, tuple):
            ac_size, v_size = recurrent_hidden_state_size
            self.recurrent_hidden_states_v = torch.zeros(
                num_steps + 1, num_processes, v_size)
            self.recurrent_hidden_states_ac =  torch.zeros(
                num_steps + 1, num_processes, ac_size)
        else:
            self.recurrent_hidden_states = torch.zeros(
                num_steps + 1, num_processes, recurrent_hidden_state_size)

        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps

        if goal_shape is not None:
            self.goal_buffer = torch.zeros(goal_buffer_size, num_processes, goal_shape * goal_components)
        else:
            self.goal_buffer = None

        self.goal_shape = goal_shape
        self.goal_components = goal_components
        self.goal_buffer_size = goal_buffer_size
        self.step = 0

    def to(self, device):
        if isinstance(self.obs, dict):
            for v in self.obs.values():
                v.to(device)
        else:
            self.obs = self.obs.to(device)

        if hasattr(self, 'recurrent_hidden_states_ac'):
            self.recurrent_hidden_states_ac = self.recurrent_hidden_states_ac.to(device)
            self.recurrent_hidden_states_v = self.recurrent_hidden_states_v.to(device)
        else:
            self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        if self.goal_buffer_size is not None:
            self.goal_buffer = self.goal_buffer.to(device)

    def insert_act(self, actions, action_log_probs, value_preds, recurrent_hidden_states):
        self.actions[self.step].copy_(actions.squeeze())
        self.action_log_probs[self.step].copy_(action_log_probs)#.unsqueeze(-1))
        self.value_preds[self.step].copy_(value_preds)
        if isinstance(recurrent_hidden_states, tuple):
            ac_rnn_hxs, v_rnn_hxs = recurrent_hidden_states
            self.recurrent_hidden_states_ac[self.step + 1].copy_(ac_rnn_hxs.squeeze(0))
            self.recurrent_hidden_states_v[self.step + 1].copy_(v_rnn_hxs.squeeze(0))
        else:
            self.recurrent_hidden_states[self.step +
                                         1].copy_(recurrent_hidden_states)

    def insert_after_act(self, obs, rewards, masks, bad_masks):
        if isinstance(obs, dict):
            for k, v in obs.items():
                self.obs[k][self.step + 1].copy_(obs[k])
        else:
            self.obs[self.step + 1].copy_(obs)

        self.rewards[self.step].copy_(rewards.permute(1,0))
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def insert_init_obs(self, obs):
        assert self.step == 0

        if isinstance(obs, dict):
            for k, v in obs.items():
                self.obs[k][self.step].copy_(obs[k])
        else:
            self.obs[self.step].copy_(obs)
    
    def insert_reward(self, step, reward):
        self.rewards[step].copy_(reward.permute(1,0))


    def insert_goal(self, goal):
        assert self.goal_buffer_size is not None
        old_buff = self.goal_buffer[:-1].clone().detach()
        self.goal_buffer[1:].copy_(old_buff)
        self.goal_buffer[0].copy_(goal)

    def after_update(self):
        if isinstance(self.obs, dict):
            for k, v in self.obs.items():
                self.obs[k][0].copy_(self.obs[k][-1])
        else:
            self.obs[0].copy_(self.obs[-1])

        if hasattr(self, 'recurrent_hidden_states_ac'):
            self.recurrent_hidden_states_ac[0].copy_(self.recurrent_hidden_states_ac[-1])
            self.recurrent_hidden_states_v[0].copy_(self.recurrent_hidden_states_v[-1])
        else:
            self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])

        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def get_utility(self, gamma):
        utility = 0
        for step in range(self.rewards.size(0)):
            utility += self.rewards[step] * gamma ** step * self.masks[step]
        return utility
    

    def get_last_obs(self):
        last_obs = {}
        for k, v in self.obs.items():
            last_obs[k] = v[-1]
        return last_obs

    def get_last_recurrent(self):
        if hasattr(self, 'recurrent_hidden_states_ac'):
            return (self.recurrent_hidden_states_ac[-1], self.recurrent_hidden_states_v[-1])
        else:
            return self.recurrent_hidden_states[-1]


    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):

            if isinstance(self.obs, dict):
                obs_batch = {k : [] for k in self.obs.keys()}
            else:
                obs_batch = []
            recurrent_hidden_states_batch_ac = []
            recurrent_hidden_states_batch_v = []

            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            goals_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                if isinstance(obs_batch, dict):
                    for k, v in self.obs.items():
                        obs_batch[k].append(v[1:, ind])
                else:
                    obs_batch.append(self.obs[1:, ind])
                recurrent_hidden_states_batch_ac.append(
                    self.recurrent_hidden_states_ac[0:1, ind])

                recurrent_hidden_states_batch_v.append(
                    self.recurrent_hidden_states_v[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[1:, ind])
                return_batch.append(self.returns[1:, ind])
                masks_batch.append(self.masks[1:, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                if advantages is not None:
                    adv_targ.append(advantages[:, ind])

                if self.goal_buffer is not None:
                    goals_batch.append(self.goal_buffer[:, ind])


            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            if isinstance(obs_batch, dict):
                for k, v in self.obs.items():
                    obs_batch[k] = torch.stack(obs_batch[k], 1)
            else:
                obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            if advantages is not None:
                adv_targ = torch.stack(adv_targ, 1)
            if self.goal_buffer is not None:
                goals_batch = torch.stack(goals_batch, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch_ac = torch.stack(
                recurrent_hidden_states_batch_ac, 1).view(N, -1)
            recurrent_hidden_states_batch_v = torch.stack(
                recurrent_hidden_states_batch_v, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            if isinstance(obs_batch, dict):
                for k, v in self.obs.items():
                    obs_batch[k] = _flatten_helper(T, N, obs_batch[k])
            else:
                obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            if advantages is not None:
                adv_targ = _flatten_helper(T, N, adv_targ)
            if self.goal_buffer is not None:
                goals_batch = _flatten_helper(self.goal_buffer_size, N, goals_batch)

            yield obs_batch, (recurrent_hidden_states_batch_ac, recurrent_hidden_states_batch_v), actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, goals_batch, T, N


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
