import gym
import numpy as np

import matplotlib.pyplot as plt
from gym.spaces import Dict, Box, Tuple
from utils.utils import flatten_space, unflatten, flatten, flatten_obs
import copy
import torch

ALICE = 0
BOB = 1

def gym_space_from_arrays(arrays, min=-np.inf, max=np.inf):
    """ Define environment observation space using an example observation """
    if isinstance(arrays, np.ndarray):
        ret = Box(min, max, arrays.shape, np.float32)
        ret.flatten_dim = np.prod(ret.shape)
    elif isinstance(arrays, (tuple, list)):
        ret = Tuple([gym_space_from_arrays(arr) for arr in arrays])
    elif isinstance(arrays, dict):
        ret = Dict(dict([(k, gym_space_from_arrays(v)) for k, v in arrays.items()]))
    else:
        raise TypeError(f"Array is of unsupported type: {type(arrays)}")
    return ret

def get_test_cases(env_name):
    test_cases = ['random_OOD', 'random']
    # test_cases = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'random',
    #               'random_OOD', 'top_left_OOD', 'top_right_OOD', 'bottom_left_OOD', 'bottom_right_OOD', 
    #               'top_mid_OOD', 'left_mid_OOD', 'right_mid_OOD', 'bottom_mid_OOD']
    if 'point_mass_maze' in env_name:
        test_cases.append('maze_center')
    elif env_name == 'fetch_push':
        test_cases.extend(['push_backward_mid', 'push_backward_left', 'push_backward_right'])
    elif 'walker' in env_name:
        test_cases.extend(['top_right_OOD', 'right_mid_OOD'])
    elif 'toss' in env_name:
        test_cases.extend(['top_right_OOD', 'top_mid_OOD', 'top_left_OOD'])
    # elif 'multi' in env_name or env_name == 'point_mass' or env_name == 'manipulator_toss' or env_name == 'manipulator_reach':
    #     test_cases = ['random_OOD', 'random']

    return test_cases 

def generate_eval_goals(num_processes, env_name, goal_shape, goal_space, eval_rng, test_case):
    if len(goal_shape) > 1:
        num_goals = goal_shape[0] 
        elements = num_goals * goal_shape[1]  * num_processes 
    else: 
        num_goals = 1
        elements = goal_shape[0]  * num_processes
    additional_dims = 0 
    if 'point_mass' in env_name:
        full_space = Box(np.array([-.3, -.3]), np.array([.3, .3]))
        additional_dims = goal_space.shape[-1] - full_space.shape[-1]
        if additional_dims > 0:
            full_space = Box(np.array([-.3, -.3] + [-1.3] * additional_dims), np.array([.3, .3] + [1.3] * additional_dims))
    elif env_name == 'walker':
        full_space = Box(np.array([12, -.2]), np.array([20, .2]))
        additional_dims = goal_space.shape[-1] - full_space.shape[-1]
        if additional_dims > 0:
            full_space = Box(np.array([12, -.2] + [-1.3] * additional_dims), np.array([20, .2] + [1.3] * additional_dims))
    elif env_name == 'manipulator_toss':
        full_space = Box(np.array([-.6, .5]), np.array([.6, 1.4]))
        additional_dims = goal_space.shape[-1] - full_space.shape[-1]
        if additional_dims > 0:
            full_space = Box(np.array([-.6, .5] + [-1.3] * additional_dims), np.array([.6, 1.4] + [1.3] * additional_dims))
    elif env_name == 'manipulator_reach':
        full_space = Box(np.array([-.4, .1]), np.array([.4, .8]))
        additional_dims = goal_space.shape[-1] - full_space.shape[-1]
        if additional_dims > 0:
            full_space = Box(np.array([-.4, .1] + [-1.3] * additional_dims), np.array([.4, .8] + [1.3] * additional_dims))
    elif env_name == 'manipulator_reach_multi':
        full_space = Box(np.array(5 * [-.4, .1]), np.array(5 * [.4, .8]))
    elif env_name == 'fetch_push':
        full_space = Box(np.array([1.15, .5]), np.array([1.5, .9]))
        additional_dims = goal_space.shape[-1] - full_space.shape[-1]
        if additional_dims > 0:
            full_space = Box(np.array([1.15, .5] + [-1.3] * additional_dims), np.array([.4, .8] + [1.3] * additional_dims))

    if test_case == 'top_left':
        return torch.FloatTensor(num_processes * [goal_space.low[0], goal_space.high[1]]).view(num_processes, -1)
    elif test_case == 'top_right':
        return torch.FloatTensor(num_processes * [goal_space.high[0], goal_space.high[1]]).view(num_processes, -1)
    elif test_case == 'bottom_left':
        return torch.FloatTensor(num_processes * [goal_space.low[0], goal_space.low[1]]).view(num_processes, -1)
    elif test_case == 'bottom_right':
        return torch.FloatTensor(num_processes * [goal_space.high[0], goal_space.low[1]]).view(num_processes, -1)
    elif test_case == 'random':
        return torch.FloatTensor(eval_rng.uniform(goal_space.low, goal_space.high, size=(num_processes, goal_space.shape[0])))
    elif test_case == 'random_OOD':
        if 'point_mass' in env_name:
            rand_ood_goals = eval_rng.uniform(.25, .3, size=(elements))
            return torch.FloatTensor([-g if eval_rng.random() > .5 else g for g in rand_ood_goals]).view(num_processes, -1)
        elif env_name == 'walker':
            # rand_ood_x = eval_rng.uniform(full_space.low[0], full_space.high[0])
            # rand_ood_z = eval_rng.uniform(full_space.low[1], full_space.high[1])
            # return torch.FloatTensor(num_processes * [rand_ood_x, rand_ood_z]).view(num_processes, -1)
            return torch.FloatTensor(eval_rng.uniform(num_goals * [full_space.low[0], full_space.low[1], *full_space.low[2:]], num_goals * [full_space.high[0], full_space.high[1], *full_space.high[2:]], size=(elements))).view(num_processes, -1)
        elif  'manipulator_toss' in env_name or 'manipulator_reach' in env_name:
            ood_rand = eval_rng.random()
            if 0 <= ood_rand < .33: #left 
                return torch.FloatTensor(eval_rng.uniform(num_goals * [full_space.low[0], goal_space.low[1], *full_space.low[2:]], num_goals * [goal_space.low[0], goal_space.high[1], *full_space.high[2:]], size=(elements))).view(num_processes, -1) 
            elif .33 <= ood_rand < .67: #right
                return torch.FloatTensor(eval_rng.uniform(num_goals * [goal_space.high[0], goal_space.low[1], *full_space.low[2:]], num_goals * [full_space.high[0], goal_space.high[1], *full_space.high[2:]], size=(elements))).view(num_processes, -1)
            else: #top
                return torch.FloatTensor(eval_rng.uniform(num_goals * [goal_space.low[0], goal_space.high[1], *full_space.low[2:]], num_goals * [goal_space.high[0], full_space.high[1], *full_space.high[2:]], size=(elements))).view(num_processes, -1)
        elif env_name == 'fetch_push':
            ood_rand = eval_rng.random()
            if 0 <= ood_rand < .25: #left 
                return torch.FloatTensor(eval_rng.uniform([goal_space.low[0], full_space.low[1], *full_space.low[2:]], [goal_space.high[0], goal_space.low[1], *full_space.high[2:]], size=(elements))).view(num_processes, -1) 
            elif .25 <= ood_rand < .5: #right 
                return torch.FloatTensor(eval_rng.uniform([goal_space.low[0], goal_space.high[1], *full_space.low[2:]], [goal_space.high[0], full_space.high[1], *full_space.high[2:]], size=(elements))).view(num_processes, -1)
            elif .5 <= ood_rand < .75: #top 
                return torch.FloatTensor(eval_rng.uniform([goal_space.high[0], goal_space.low[1], *full_space.low[2:]], [full_space.high[0], goal_space.high[1], *full_space.high[2:]], size=(elements))).view(num_processes, -1)
            else: #bottom
                return torch.FloatTensor(eval_rng.uniform([full_space.low[0], goal_space.low[1], *full_space.low[2:]], [goal_space.low[0], goal_space.high[1], *full_space.high[2:]], size=(elements))).view(num_processes, -1)

    elif test_case == 'top_left_OOD':
        if additional_dims > 0:
            rand_additional_dims = eval_rng.uniform(full_space.low[2:], full_space.high[2:])
            return torch.FloatTensor(num_processes * [full_space.low[0], full_space.high[1], *rand_additional_dims]).view(num_processes, -1)
        else:
            return torch.FloatTensor(num_processes * [full_space.low[0], full_space.high[1]]).view(num_processes, -1)
    elif test_case == 'top_right_OOD':
        if additional_dims > 0:
            rand_additional_dims = eval_rng.uniform(full_space.low[2:], full_space.high[2:])
            return torch.FloatTensor(num_processes * [full_space.high[0], full_space.high[1], *rand_additional_dims]).view(num_processes, -1)
        else:        
            return torch.FloatTensor(num_processes * [full_space.high[0], full_space.high[1]]).view(num_processes, -1)
    elif test_case == 'bottom_left_OOD':
        if additional_dims > 0:
            rand_additional_dims = eval_rng.uniform(full_space.low[2:], full_space.high[2:])
            return torch.FloatTensor(num_processes * [full_space.low[0], full_space.low[1], *rand_additional_dims]).view(num_processes, -1)
        else:                
            return torch.FloatTensor(num_processes * [full_space.low[0], full_space.low[1]]).view(num_processes, -1)
    elif test_case == 'bottom_right_OOD':   
        if additional_dims > 0:
            rand_additional_dims = eval_rng.uniform(full_space.low[2:], full_space.high[2:])
            return torch.FloatTensor(num_processes * [full_space.high[0], full_space.low[1], *rand_additional_dims]).view(num_processes, -1)
        else:        
            return torch.FloatTensor(num_processes * [full_space.high[0], full_space.low[1]]).view(num_processes, -1)
    elif test_case == 'top_mid_OOD':
        if additional_dims > 0:
            rand_additional_dims = eval_rng.uniform(full_space.low[2:], full_space.high[2:])
            return torch.FloatTensor(num_processes * [(full_space.low[0] + full_space.high[0])/2, full_space.high[1], *rand_additional_dims]).view(num_processes, -1)
        else:                
            return torch.FloatTensor(num_processes * [(full_space.low[0] + full_space.high[0])/2, full_space.high[1]]).view(num_processes, -1)
    elif test_case == 'left_mid_OOD':
        if additional_dims > 0:
            rand_additional_dims = eval_rng.uniform(full_space.low[2:], full_space.high[2:])
            return torch.FloatTensor(num_processes * [full_space.low[0], (full_space.low[1] + full_space.high[1])/2, *rand_additional_dims]).view(num_processes, -1)
        else:                        
            return torch.FloatTensor(num_processes * [full_space.low[0], (full_space.low[1] + full_space.high[1])/2]).view(num_processes, -1)
    elif test_case == 'right_mid_OOD':
        if additional_dims > 0:
            rand_additional_dims = eval_rng.uniform(full_space.low[2:], full_space.high[2:])
            return torch.FloatTensor(num_processes * [full_space.high[0], (full_space.low[1] + full_space.high[1])/2, *rand_additional_dims]).view(num_processes, -1)
        else:                                
            return torch.FloatTensor(num_processes * [full_space.high[0], (full_space.low[1] + full_space.high[1])/2]).view(num_processes, -1)
    elif test_case == 'bottom_mid_OOD':
        if additional_dims > 0:
            rand_additional_dims = eval_rng.uniform(full_space.low[2:], full_space.high[2:])
            return torch.FloatTensor(num_processes * [(full_space.low[0] + full_space.high[0])/2, full_space.low[1], *rand_additional_dims]).view(num_processes, -1)
        else:                                
            return torch.FloatTensor(num_processes * [(full_space.low[0] + full_space.high[0])/2, full_space.low[1]]).view(num_processes, -1)
    elif test_case == 'maze_center':
        if additional_dims > 0:
            rand_additional_dims = eval_rng.uniform(full_space.low[2:], full_space.high[2:])
            if env_name[-1] == '0':
                return torch.FloatTensor(num_processes * [.15, .15, *rand_additional_dims]).view(num_processes, -1)
            elif env_name[-1] == '1':
                return torch.FloatTensor(num_processes * [-.15, .15, *rand_additional_dims]).view(num_processes, -1)
            elif env_name[-1] == '2':
                return torch.FloatTensor(num_processes * [-.15, -.15, *rand_additional_dims]).view(num_processes, -1)
            elif env_name[-1] == '3':
                return torch.FloatTensor(num_processes * [.15, -.15, *rand_additional_dims]).view(num_processes, -1)
            else:
                raise NotImplementedError
        else:
            if env_name[-1] == '0':
                return torch.FloatTensor(num_processes * [.15, .15]).view(num_processes, -1)
            elif env_name[-1] == '1':
                return torch.FloatTensor(num_processes * [-.15, .15]).view(num_processes, -1)
            elif env_name[-1] == '2':
                return torch.FloatTensor(num_processes * [-.15, -.15]).view(num_processes, -1)
            elif env_name[-1] == '3':
                return torch.FloatTensor(num_processes * [.15, -.15]).view(num_processes, -1)
            else:
                raise NotImplementedError
    elif test_case == 'push_backward_mid':
        return torch.FloatTensor(num_processes * [1.15, .7]).view(num_processes, -1)
    elif test_case == 'push_backward_left':
        return torch.FloatTensor(num_processes * [1.15, .5]).view(num_processes, -1)
    elif test_case == 'push_backward_right':
        return torch.FloatTensor(num_processes * [1.15, .9]).view(num_processes, -1)
    else:
        raise NotImplementedError

class SelfPlayEnv(gym.Wrapper):
    # Base wrapper for resetting and goal setting
    def __init__(self, *args):
        super(SelfPlayEnv, self).__init__(*args)
        self.agent = ALICE
        self.goal = None

    def set_agent(self, agent):
        self.agent = agent
    
    def set_device(self, device):
        self.device = device

    def set_goal(self, goal):
        if not goal:
            return
        self.goal = goal

    def goal_shape(self):
        return self.goal.shape

    def goal_space(self):
        return self.observation_space['goal_state']

    def get_init_qpos_qvel(self):
        self.init_qpos = self.env.env._env.physics.data.qpos.ravel().copy()
        self.init_qvel = self.env.env._env.physics.data.qvel.ravel().copy()

    def sim_update_qpos_qvel(self):
        self.env.env._env.physics.data.qpos[:] = self.init_qpos
        self.env.env._env.physics.data.qvel[:] = self.init_qvel
        self.env.env._env.physics.forward()

    def set_init_qpos_qvel(self, qpos=None, qvel=None):
        self.init_qpos = qpos
        self.init_qvel = qvel

    def render(self, mode='rgb_array', height=420, width=420):
        return self.env.env.render(mode=mode, height=height, width=width)


class SelfPlayEnvPointMass(SelfPlayEnv):
    def __init__(self, maze=None, additional_dims=0, *args):
        super(SelfPlayEnvPointMass, self).__init__(*args)
        self.agent = ALICE

        self.additional_dims = additional_dims
        self.goal = np.zeros(2 + additional_dims) #yz pos
        self.torch_goal = torch.tensor(self.goal) #TODO: rename goal setting
        self.update_observation_space()
        self.device = None
        self.maze = maze

    def set_init_qpos_qvel(self, qpos=None, qvel=None):
        if self.maze is not None:
            # Place maze in specified quadrant
            if self.maze == '0':
                maze_center = [.15, .15]
            elif self.maze == '1':
                maze_center = [-.15, .15]
            elif self.maze == '2':
                maze_center = [-.15, -.15]
            elif self.maze == '3':
                maze_center = [.15, -.15]
            
            if np.random.rand() < .1:
                self.init_qpos = torch.FloatTensor(maze_center)
            else:
                self.init_qpos = torch.zeros(2) if qpos is None else qpos
        else:
            self.init_qpos = torch.zeros(2) if qpos is None else qpos
        self.init_qvel = torch.zeros(2) if qvel is None else qvel

    def set_goal(self, args, goal_space=None, goal_shape=None, goal=None):
        if goal is not None:
            self.torch_goal = goal
            self.goal = goal.detach().cpu().numpy()
            target_id = self.env.env._env.physics.model.name2id('target', 'geom')
            geom_xpos = self.env.env._env.physics.data.geom_xpos.copy()
            geom_xpos[target_id, :2] = self.goal[:2]
        else:
            self.goal = np.zeros(2)
            self.torch_goal = None

    def update_observation_space(self):
        obs_full = self.env.reset()
        xyz_state = obs_full[:2]

        self.observation_dict = {
            'full_state' : gym_space_from_arrays(obs_full),
            'full_state_with_goal' : gym_space_from_arrays(np.concatenate((obs_full, self.goal))),
            'xyz_state' : gym_space_from_arrays(xyz_state),
            'goal_state' : gym_space_from_arrays(self.goal, min=-.25, max=.25),
        }

        self.observation_space = Dict(self.observation_dict)

    def step(self, action):
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        obs_full, rew, done, info = self.env.step(action)
        xyz_state = obs_full[:2]

        obs = {
            'full_state' : obs_full,
            'full_state_with_goal' : np.concatenate((obs_full, self.goal)),
            'xyz_state': xyz_state,
            'goal_state' : self.goal if self.torch_goal is not None else None 
        }

        info['env_rew'] = rew
        
        if self.torch_goal is not None:
            dist_to_goal = np.linalg.norm(torch.FloatTensor(xyz_state) - self.goal[:2])
            reward = -dist_to_goal
            info['goal_rew'] = dist_to_goal
            if dist_to_goal < .05: 
                info['success'] = True
            else:
                info['success'] = False
        else:
            reward = -100 
            dist_to_goal = 0
            done = False
            info['success'] = False

        return obs, reward, done or info['success'], info

    def reset(self):
        obs_full = self.env.reset()

        self.set_init_qpos_qvel()
        self.sim_update_qpos_qvel()

        obs_full = self.env.env._task.get_observation(self.env.env._physics)
        obs_full = flatten_obs(obs_full)

        xyz_state = obs_full[:2]
        
        obs = {
            'full_state' : obs_full,
            'full_state_with_goal' : np.concatenate((obs_full, self.goal)),
            'xyz_state' : xyz_state,
            'goal_state' : self.goal 
        }

        return obs

    def check_valid_goal(self, info, init_obs, final_obs):
        assert self.agent == ALICE
        is_valid_goal = int(info[0]['success'])

        if np.allclose(init_obs['full_state'], final_obs['full_state']):
            is_valid_goal = 0

        return is_valid_goal

class SelfPlayEnvWalker(SelfPlayEnv):
    def __init__(self, small=False, additional_dims=0, *args):
        super(SelfPlayEnvWalker, self).__init__(*args)
        self.small = small
        self.agent = ALICE
        self.additional_dims = additional_dims
        self.goal = np.zeros(2 + additional_dims) #zx pos 
        self.torch_goal = torch.tensor(self.goal)
        self.update_observation_space()
        self.device = None

    def set_init_qpos_qvel(self, qpos=None, qvel=None):
        self.init_qpos = torch.zeros(9) if qpos is None else qpos
        self.init_qvel = torch.zeros(9) if qvel is None else qvel

    def set_goal(self, args, goal_space=None, goal_shape=None, goal=None):
        if goal is not None:
            # swap x, z
            x = goal[0]
            z = goal[1]
            goal = torch.FloatTensor([z, x, *goal[2:]])
            self.torch_goal = goal
            self.goal = goal.detach().cpu().numpy() #goal is z, x positions which is flipped in the env
        else:
            self.goal = np.zeros(2 + self.additional_dims) 
            self.torch_goal = None

    def get_observation(self, obs_full):
        xyz_state = np.asarray(self.env.env._env.physics.data.qpos.copy()[:2]) # xpos only 
        if not isinstance(obs_full, np.ndarray):
            obs_full = np.concatenate([xyz_state, obs_full['orientations'], [obs_full['height']], obs_full['velocity']]) 
        else:
            obs_full = np.concatenate([xyz_state, obs_full])

        return obs_full, xyz_state

    def update_observation_space(self):
        obs_full = self.env.reset() 
        obs_full, xyz_state = self.get_observation(obs_full)
        if self.small:
            goal_state_space = Box(np.array([0, -.2] + [-1] * self.additional_dims), np.array([3, 0.2]+ [1] * self.additional_dims)) #account for x limits, z limits
        else:
            goal_state_space = Box(np.array([-1, -.2]+ [-1] * self.additional_dims), np.array([10, 0.2]+ [1] * self.additional_dims)) #account for x limits, z limits

        self.observation_dict = {
            'full_state' : gym_space_from_arrays(obs_full), 
            'full_state_with_goal' : gym_space_from_arrays(np.concatenate((obs_full, self.goal))), 
            'xyz_state' : gym_space_from_arrays(xyz_state),
            'goal_state' : goal_state_space,
        }

        self.observation_space = Dict(self.observation_dict)

    def step(self, action):
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        obs_full, rew, done, info = self.env.step(action)
        obs_full, xyz_state = self.get_observation(obs_full)

        info['env_rew'] = rew

        if self.torch_goal is not None:
            dist_to_goal = np.abs(xyz_state - self.goal[:2]) 
            info['goal_rew'] = dist_to_goal

            if dist_to_goal[0] < .5 and dist_to_goal[1] < .1: 
                info['success'] = True
            else:
                info['success'] = False
        else:
            dist_to_goal = 0
            done = False
            info['success'] = False

        reward = -np.linalg.norm(xyz_state - self.goal[:2]) 
        obs = {
            'full_state' : obs_full,
            'full_state_with_goal' : np.concatenate((obs_full, self.goal)),
            'xyz_state': xyz_state,
            'goal_state' : self.goal if self.torch_goal is not None else None 
        }

        return obs, reward, done or info['success'], info

    def reset(self):
        obs_full = self.env.reset()
        self.set_init_qpos_qvel()
        self.sim_update_qpos_qvel()

        obs_full = self.env.env._task.get_observation(self.env.env._physics)
        obs_full, xyz_state = self.get_observation(obs_full)

        if self.torch_goal is not None:
            dist_to_goal = np.linalg.norm(torch.FloatTensor(xyz_state) - self.goal[:2])
        else:
            dist_to_goal = 0
        
        obs = {
            'full_state' : obs_full,
            'full_state_with_goal' : np.concatenate((obs_full, self.goal)),
            'xyz_state' : xyz_state,
            'goal_state' : self.goal
        }

        return obs

class SelfPlayEnvManipulator(SelfPlayEnv):
    def __init__(self, task, additional_dims, *args):
        super(SelfPlayEnvManipulator, self).__init__(*args)
        self.agent = ALICE
        self.task = task
        self.additional_dims = additional_dims
        self.goal = np.zeros(2 + additional_dims) #xz pos 
        self.torch_goal = torch.tensor(self.goal)
        self.update_observation_space()
        self.device = None

    def set_init_qpos_qvel(self, qpos=None, qvel=None):
        self.init_qpos = torch.zeros(11) if qpos is None else qpos #index 8,9,10 are x,z, theta coordinates of ball

        # 10% of the time initialize ball in gripper or at target
        object_init_probs = [1, 0, 0]
        init_type = np.random.choice(['in_hand', 'in_target', 'origin'],
                            p=object_init_probs)
        if init_type == 'in_target':
            object_x = self.goal[0]
            object_z = self.goal[1]
            object_angle = np.random.uniform(-np.pi, np.pi)
        elif init_type == 'in_hand':
            self.env.env._env.physics.data.qpos[:] = self.init_qpos
            self.env.env._env.physics.forward()
            object_x = self.env.env._env.physics.named.data.site_xpos['grasp', 'x']
            object_z = self.env.env._env.physics.named.data.site_xpos['grasp', 'z']
            grasp_direction = self.env.env._env.physics.named.data.site_xmat['grasp', ['xx', 'zx']]
            object_angle = np.pi-np.arctan2(grasp_direction[1], grasp_direction[0])
        else:
            object_x = 0
            object_z = 0
            object_angle = 0

        self.init_object_qpos = object_x, object_z, object_angle
        self.init_qvel = torch.zeros(11) if qvel is None else qvel
    
    def sim_update_qpos_qvel(self):
        self.env.env._env.physics.named.model.body_pos['target_ball', ['x', 'z']] = self.goal[...,:2]
        self.env.env._env.physics.named.model.body_quat['target_ball', ['qw', 'qy']] = [0, 0]
        self.env.env._env.physics.data.qpos[:] = self.init_qpos
        self.env.env._env.physics.named.data.qpos[self.env.env._env.task._object_joints] = self.init_object_qpos
        self.env.env._env.physics.data.qvel[:] = self.init_qvel
        self.env.env._env.physics.forward()

    def set_goal(self, args, goal_space=None, goal_shape=None, goal=None):
        if goal is not None:
            if self.task == 'toss':
                if np.random.rand() < .1:
                    goal = torch.FloatTensor([0, .915] + [0] * self.additional_dims) 
            
            self.torch_goal = goal
            self.goal = goal.detach().cpu().numpy() #goal is x y z positions
        else:
            self.goal = np.zeros(2)
            self.torch_goal = None

    def get_observation(self, obs_full):
        xyz_state = self.env.env._env.physics.body_2d_pose(self.env.env._env.task._object)[:2] #only take position, no orientation
        if not isinstance(obs_full, np.ndarray):
            obs_full = np.concatenate([v.flatten() for v in obs_full.values()])

        return obs_full, xyz_state

    def update_observation_space(self):
        obs_full = self.env.reset() 
        obs_full, xyz_state = self.get_observation(obs_full)
        if self.task == 'reach':
            goal_state_space = Box(np.array([-.25, .1] + [-1] * self.additional_dims), np.array([.25, .6] + [1] * self.additional_dims)) #account for horizonal, z limits
        elif self.task == 'toss':
            goal_state_space = Box(np.array([-.5, .5] + [-1] * self.additional_dims), np.array([.5, 1.3] + [1] * self.additional_dims))  # try larger goal space to learn throwing

        self.observation_dict = {
            'full_state' : gym_space_from_arrays(obs_full),
            'full_state_with_goal' : gym_space_from_arrays(np.concatenate((obs_full, self.goal))),
            'xyz_state' : gym_space_from_arrays(xyz_state),
            'goal_state' : goal_state_space,
        }

        self.observation_space = Dict(self.observation_dict)

    def step(self, action):
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        obs_full, rew, done, info = self.env.step(action)
        obs_full, xyz_state = self.get_observation(obs_full)

        obs = {
            'full_state' : obs_full,
            'full_state_with_goal' : np.concatenate((obs_full, self.goal)),
            'xyz_state': xyz_state,
            'goal_state' : self.goal if self.torch_goal is not None else None 
        }

        info['env_rew'] = rew
        if self.torch_goal is not None:
            dist_to_goal = np.linalg.norm(torch.FloatTensor(xyz_state) - self.goal[:2])
            reward = -dist_to_goal
            info['goal_rew'] = dist_to_goal
            if dist_to_goal < .1: 
                info['success'] = True
                # reward = 0
            else:
                info['success'] = False
                # reward = -1

        else:
            dist_to_goal = 0
            reward = -1
            done = False
            info['success'] = False

        return obs, reward, done or info['success'], info

    def reset(self):
        obs_full = self.env.reset()
        self.set_init_qpos_qvel()
        self.sim_update_qpos_qvel()

        obs_full = self.env.env._task.get_observation(self.env.env._physics)
        obs_full, xyz_state = self.get_observation(obs_full)
        
        obs = {
            'full_state' : obs_full,
            'full_state_with_goal' : np.concatenate((obs_full, self.goal)),
            'xyz_state' : xyz_state,
            'goal_state' : self.goal
        }

        return obs


class SelfPlayEnvManipulatorMultiGoal(SelfPlayEnv):
    def __init__(self, task, num_goals, *args):
        super(SelfPlayEnvManipulatorMultiGoal, self).__init__(*args)
        self.agent = ALICE
        self.task = task
        self.num_goals = num_goals
        self.goal = np.zeros((num_goals, 2)) #xz pos 
        self.torch_goal = torch.tensor(self.goal)
        self.update_observation_space()
        self.device = None
        self.done_goals = np.ones(self.num_goals)

    def set_init_qpos_qvel(self, qpos=None, qvel=None):
        self.init_qpos = torch.zeros(11) if qpos is None else qpos #index 8,9,10 are x,z, theta coordinates of ball

        # 10% of the time initialize ball in gripper or at target
        
        self.env.env._env.physics.data.qpos[:] = self.init_qpos
        self.env.env._env.physics.forward()
        object_x = self.env.env._env.physics.named.data.site_xpos['grasp', 'x']
        object_z = self.env.env._env.physics.named.data.site_xpos['grasp', 'z']
        grasp_direction = self.env.env._env.physics.named.data.site_xmat['grasp', ['xx', 'zx']]
        object_angle = np.pi-np.arctan2(grasp_direction[1], grasp_direction[0])

        self.init_object_qpos = object_x, object_z, object_angle
        self.init_qvel = torch.zeros(11) if qvel is None else qvel
    
    def sim_update_qpos_qvel(self):
        self.env.env._env.physics.named.model.body_quat['target_ball', ['qw', 'qy']] = [0, 0]
        self.env.env._env.physics.data.qpos[:] = self.init_qpos
        self.env.env._env.physics.named.data.qpos[self.env.env._env.task._object_joints] = self.init_object_qpos
        self.env.env._env.physics.data.qvel[:] = self.init_qvel
        self.env.env._env.physics.forward()

    def set_goal(self, args, goal_space=None, goal_shape=None, goal=None):
        if goal is not None:
            self.torch_goal = goal.reshape(self.goal.shape)
            self.goal = self.torch_goal.detach().cpu().numpy() #goal is x y z positions
        else:
            self.goal = np.zeros((self.num_goals, 2))
            self.torch_goal = None

    def get_observation(self, obs_full):
        xyz_state = self.env.env._env.physics.body_2d_pose(self.env.env._env.task._object)[:2] #only take position, no orientation
        if not isinstance(obs_full, np.ndarray):
            obs_full = np.concatenate([v.flatten() for v in obs_full.values()])

        return obs_full, xyz_state

    def update_observation_space(self):
        obs_full = self.env.reset() 
        obs_full, xyz_state = self.get_observation(obs_full)

        if self.task == 'reach':
            goal_state_space = Box(np.array(self.num_goals * [-.25, .1]), np.array(  self.num_goals * [.25, .6])) #account for horizonal, z limits
        elif self.task == 'toss':
            goal_state_space = Box(np.array(  self.num_goals * [-.5, .5]), np.array( self.num_goals * [.5, 1.3]))  # try larger goal space to learn throwing

        self.observation_dict = {
            'full_state' : gym_space_from_arrays(obs_full),
            'full_state_with_goal' : gym_space_from_arrays(np.concatenate((obs_full, self.goal.flatten()))),
            'xyz_state' : gym_space_from_arrays(xyz_state),
            'goal_state' : goal_state_space,
        }
        self.observation_space = Dict(self.observation_dict)

    def step(self, action):
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        obs_full, rew, done, info = self.env.step(action)
        obs_full, xyz_state = self.get_observation(obs_full)

        obs = {
            'full_state' : obs_full,
            'full_state_with_goal' : np.concatenate((obs_full, self.goal.flatten())),
            'xyz_state': xyz_state,
            'goal_state' : self.goal.flatten() if self.torch_goal is not None else None 
        }

        info['env_rew'] = rew

        if self.torch_goal is not None:
            reward = 0

            for goal_num, goal in enumerate(self.goal):
                dist_to_goal = np.linalg.norm(torch.FloatTensor(xyz_state) - goal)
                if dist_to_goal < .1 or self.done_goals[goal_num] == 0:
                    self.done_goals[goal_num] = 0
                else:
                    self.done_goals[goal_num] = 1
                    reward += -1
            if np.sum(self.done_goals) == 0:
                info['success'] = True
            else:
                info['success'] = False

        else:
            dist_to_goal = 0
            reward = -1 * self.num_goals
            done = False
            info['success'] = False

        return obs, reward, done or info['success'], info

    def reset(self):
        obs_full = self.env.reset()
        self.set_init_qpos_qvel()
        self.sim_update_qpos_qvel()
        self.done_goals = np.ones_like(self.done_goals)

        obs_full = self.env.env._task.get_observation(self.env.env._physics)
        obs_full, xyz_state = self.get_observation(obs_full)
        
        obs = {
            'full_state' : obs_full,
            'full_state_with_goal' : np.concatenate((obs_full, self.goal.flatten())),
            'xyz_state' : xyz_state,
            'goal_state' : self.goal.flatten()
        }

        return obs


class SelfPlayEnvFetchPush(SelfPlayEnv):
    def __init__(self, additional_dims, *args):
        super(SelfPlayEnvFetchPush, self).__init__(*args)
        self.agent = ALICE

        self.additional_dims = additional_dims
        self.goal = np.zeros(2 + additional_dims) #xy pos 
        self.torch_goal = torch.tensor(self.goal)
        self.update_observation_space()
        self.device = None

    def set_init_qpos_qvel(self, qpos=None, qvel=None):
        # fetch push already handles qpos, qvel of robot
        pass
    
    def sim_update_qpos_qvel(self):
        # Move end effector into position.
        gripper_target = self.initial_gripper_xpos
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(100):
            self.sim.step()

        self.sim.data.qvel[:] = 0
        object_xpos = self.sim.data.get_site_xpos('robot0:grip')[:2] #- .03 #start object close to gripper
        object_qpos = self.env.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.env.sim.data.set_joint_qpos('object0:joint', object_qpos)
            
        self.env.sim.step()

    def set_goal(self, args, goal_space=None, goal_shape=None, goal=None):
        if goal is not None:
            self.torch_goal = goal
            self.goal = goal.detach().cpu().numpy() #goal is x y z positions
            self.env.goal[:2] = self.goal[:2] #env goal is 3d
        else:
            self.goal = np.zeros(2)
            self.torch_goal = None

    def get_observation(self, obs_full):
        xyz_state = self.env.sim.data.get_joint_qpos('object0:joint')[:2] #object pos
        obs_full = obs_full['observation']
        return obs_full, xyz_state

    def update_observation_space(self):
        obs_full = self.env.reset() 
        obs_full, xyz_state = self.get_observation(obs_full)
  
        goal_state_space = Box(np.array([1.35, .6] + [-1] * self.additional_dims), np.array([1.45, .8]+ [1] * self.additional_dims)) #account for horizonal, z limits
        # goal_state_space = Box(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        self.observation_dict = {
            'full_state' : gym_space_from_arrays(obs_full),
            'full_state_with_goal' : gym_space_from_arrays(np.concatenate((obs_full, self.goal))),
            'xyz_state' : gym_space_from_arrays(xyz_state),
            'goal_state' : goal_state_space,
        }

        self.observation_space = Dict(self.observation_dict)

    def step(self, action):
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        
        #stop moving z up/down too much
        cur_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip')
        if np.abs(cur_gripper_xpos[-1] - self.env.initial_gripper_xpos[-1]) > 0.05:
            action[2] = 0
        if cur_gripper_xpos[0] > 1.5 or cur_gripper_xpos[0] < 1.2:
            action[0] = 0
        if cur_gripper_xpos[1] > .9 or cur_gripper_xpos[1] < .5:
            action[1] = 0

        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        try:
            obs_full, rew, done, info = self.env.step(action)
        except:
            import pdb; pdb.set_trace()
        obs_full, xyz_state = self.get_observation(obs_full)

        obs = {
            'full_state' : obs_full,
            'full_state_with_goal' : np.concatenate((obs_full, self.goal)),
            'xyz_state': xyz_state,
            'goal_state' : self.goal if self.torch_goal is not None else None 
        }

        info['env_rew'] = rew
        if self.torch_goal is not None:
            dist_to_goal = -np.linalg.norm(torch.FloatTensor(xyz_state) - self.goal[:2])
            info['goal_rew'] = dist_to_goal
            info['success'] = True if info['is_success'] else False
            dist_to_goal = 1 if not info['is_success'] else 0 #-1 if not at goal, else 0
        else:
            dist_to_goal = 1 
            done = False
            info['success'] = False

        return obs, -dist_to_goal, done or info['success'], info

    def reset(self):
        obs_full = self.env.reset()
        self.set_init_qpos_qvel()
        self.sim_update_qpos_qvel() #set object position
        self.env.goal[:2] = self.goal[:2] #set goal position
        obs_full = self.env.env._get_obs()
        obs_full, xyz_state = self.get_observation(obs_full)
        
        obs = {
            'full_state' : obs_full,
            'full_state_with_goal' : np.concatenate((obs_full, self.goal)),
            'xyz_state' : xyz_state,
            'goal_state' : self.goal
        }

        return obs
