import numpy as np
import torch
from gym.spaces import Dict
import os

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, goal_shape=None, reward_shape=1):
        self.capacity = capacity
        self.device = device

        if isinstance(obs_shape, Dict):
            self.obses = {}
            self.next_obses = {}
            for k, v in obs_shape.spaces.items():
                obs_dtype = np.float32 if len(v.shape) == 1 else np.uint8
                self.obses[k] = np.empty((capacity, *v.shape), dtype=obs_dtype)
                self.next_obses[k] = np.empty((capacity, *v.shape), dtype=obs_dtype)
        else:
             # the proprioceptive obs is stored as float32, pixels obs as uint8
            self.obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
            self.obses = np.empty((capacity, *obs_shape), dtype=self.obs_dtype)
            self.next_obses = np.empty((capacity, *obs_shape), dtype=self.obs_dtype)

        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)

        self.rewards = np.empty((capacity, reward_shape), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        if goal_shape is not None:
            self.xyz_state = np.empty((capacity, *goal_shape), dtype=self.obs_dtype)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.goal_shape = goal_shape
        self.reward_shape = reward_shape

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, xyz_state=None):
        if isinstance(obs, dict):
            for k, v in obs.items():
                np.copyto(self.obses[k][self.idx], v.cpu())

            for k, v in next_obs.items():
                np.copyto(self.next_obses[k][self.idx], v.cpu())   
        else:
            np.copyto(self.obses[self.idx], obs)
            np.copyto(self.next_obses[self.idx], next_obs)

        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        if xyz_state is not None:
            np.copyto(self.xyz_state[self.idx], xyz_state)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, random=True, idxes=False):
        if random:
            idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)
        else:
            idxs = np.arange(0, self.idx if self.idx < batch_size and self.idx > 0 else batch_size)

        if isinstance(self.obses, dict):
            obses = {}
            next_obses = {}
            for k, v in self.obses.items():
                obses[k] = torch.as_tensor(v[idxs], device=self.device).float()

            for k, v in self.next_obses.items():
                next_obses[k] = torch.as_tensor(v[idxs],
                                     device=self.device).float()
        else:
            obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
            next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        if idxes:
           return obses, actions, rewards, next_obses, not_dones, not_dones_no_max, idxs     
        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def sample_future_states(self, k, idx):
        future_obs_idxs = np.random.randint(idx + 1, self.capacity if self.full else self.idx, size=k)
        xyz_states = torch.as_tensor(self.xyz_state[future_obs_idxs], device=self.device).float()
        cur_xyz_state = self.xyz_state[idx]

        return xyz_states, cur_xyz_state

    def update_regrets(self, idxs, regrets, coeff=.9):
        old_regrets = self.rewards[idxs]
        new_regrets = coeff * old_regrets + (1-coeff) * regrets.cpu().numpy() 
        self.rewards[idxs] = new_regrets

    def clear(self):
        self.obses = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.next_obses = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, self.reward_shape), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((self.capacity, 1), dtype=np.float32)
        if self.goal_shape is not None:
            self.xyz_state = np.empty((self.capacity, *self.goal_shape), dtype=self.obs_dtype)
        
        self.idx = 0
        self.last_save = 0
        self.full = False

    
    def save(self, save_dir):
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        if isinstance(self.obses, dict):
            sliced_obses = {k:v[self.last_save:self.idx] for k, v in self.obses.items()}
            sliced_next_obses = {k:v[self.last_save:self.idx] for k, v in self.next_obses.items()}
        else:
            sliced_obses = self.obses[self.last_save:self.idx]
            sliced_next_obses = self.next_obses[self.last_save:self.idx]

        payload = [
            sliced_obses,
            sliced_next_obses,
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx],
            self.not_dones_no_max[self.last_save:self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            if self.idx != start or end <= start:
                continue
            try:
                payload = torch.load(path)
            except:
                print("Unable to load ", str(path))
                continue

            if isinstance(self.obses, dict):
                for k,v in payload[0].items():
                    self.obses[k][start:end] = v
                for k,v in payload[1].items():
                    self.next_obses[k][start:end] = v
            else:
                self.obses[start:end] = payload[0]
                self.next_obses[start:end] = payload[1]

            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.not_dones_no_max[start:end] = payload[5]
            self.idx = end
            self.last_save = end

class BCReplayBuffer(object):
    """Buffer to store environment transitions for behavioural cloning."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=self.obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.log_probs = np.empty((capacity, 1), dtype=np.float32)
        
        self.idx = 0
        self.last_save = 0
        self.full = False
        self.obs_shape = obs_shape
        self.action_shape = action_shape

    def __len__(self):
        return self.capacity if self.full else self.idx
    

    def add(self, obs, action, log_prob=None):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        if log_prob is not None:
            np.copyto(self.log_probs[self.idx], log_prob)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        log_probs = torch.as_tensor(self.log_probs[idxs], device=self.device)

        return obses, actions, log_probs

    def save(self, save_dir):
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))

        if isinstance(self.obses, dict):
            sliced_obses = {k:v[self.last_save:self.idx] for k, v in self.obses.items()}
        else:
            sliced_obses = self.obses[self.last_save:self.idx]

        payload = [
            sliced_obses,
            self.actions[self.last_save:self.idx],
            self.log_probs[self.last_save:self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            if self.idx != start or end <= start:
                continue
            try:
                payload = torch.load(path)
            except:
                print("Unable to load ", str(path))
                continue

            if isinstance(self.obses, dict):
                for k,v in payload[0].items():
                    self.obses[k][start:end] = v
            else:
                self.obses[start:end] = payload[0]

            self.actions[start:end] = payload[1]
            self.log_probs[start:end] = payload[2]
            self.idx = end
            self.last_save = end
    
    def clear(self):
        self.obses = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.log_probs = np.empty((self.capacity, 1), dtype=np.float32)
        
        self.idx = 0
        self.last_save = 0
        self.full = False