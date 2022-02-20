import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution
from typing import List

from utils.distributions import Bernoulli, Categorical, DiagGaussian, MultiCategorical
from utils.utils import init
from gym.spaces import Dict

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                       constant_(x, 0), np.sqrt(2))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_space, action_space, device, latent_size=None, goal_history_shape=None,
                 base_kwargs=None, hidden_size=256, embed_dim=256, recurrent_hidden_size=2,
                 input_fc=(200, 200), output_fc=(200, 100), goals_embed_dim=256, goal_conditioned=True,
        ):
        super(Policy, self).__init__()
        self.separate_actor_critic = False
        self.device = device
        self.latent_size = latent_size

        if base_kwargs is None:
            base_kwargs = {}

        if latent_size is None:
            if isinstance(obs_space, Dict):
                self.actor = ASPBase(obs_space, goal_conditioned=goal_conditioned, hidden_size=hidden_size, embed_dim=embed_dim, recurrent_hidden_size=recurrent_hidden_size, device=device).to(device)
                self.critic = ASPBase(obs_space, goal_conditioned=goal_conditioned, hidden_size=hidden_size, embed_dim=embed_dim, recurrent_hidden_size=recurrent_hidden_size, device=device).to(device)
                self.critic_linear = init_(nn.Linear(output_fc[-1], 1)).to(device)
                self.separate_actor_critic = True
            else:
                obs_shape = obs_space.shape
                if len(obs_shape) == 3:
                    base = CNNBase
                elif len(obs_shape) == 1:
                    base = MLPBase
                else:
                    raise NotImplementedError

                self.base = base(obs_shape[0], **base_kwargs)
        elif isinstance(obs_space, Dict):
                self.actor = GoalGenerator(latent_dim=latent_size, goal_space=action_space, observation_space=obs_space, goal_history_shape=goal_history_shape, device=device, lstm_size=recurrent_hidden_size,
                                           input_fc=input_fc, output_fc=output_fc, goals_embed_dim=goals_embed_dim).to(device)
                self.critic = GoalGenerator(latent_dim=latent_size, goal_space=action_space, observation_space=obs_space, goal_history_shape=goal_history_shape, device=device, lstm_size=recurrent_hidden_size,
                                            input_fc=input_fc, output_fc=output_fc, goals_embed_dim=goals_embed_dim).to(device)
                self.critic_linear = init_(nn.Linear(output_fc[-1], 1)).to(device)
                self.separate_actor_critic = True
                self.goal_space = action_space
        else:
            raise NotImplementedError


        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.actor.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            num_outputs = action_space.shape[0]
            dists = []
            for i in range(num_outputs):
                dists.append(Categorical(self.actor.output_size, action_space.nvec[i]).to(device))
            self.dist = MultiCategorical(dists).to(device)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        if self.separate_actor_critic:
            return self.actor.is_recurrent
        else:
            return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""

        if self.separate_actor_critic:
            return (self.actor.recurrent_hidden_state_size, self.critic.recurrent_hidden_state_size)
        else:
            return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, goals=None, deterministic=False):
        ac_rnn_hxs, v_rnn_hxs = rnn_hxs
        actor_features, ac_rnn_hxs = self.actor(inputs, ac_rnn_hxs.to(self.device), masks.to(self.device), goals=goals)
        value, v_rnn_hxs = self.critic(inputs, v_rnn_hxs.to(self.device), masks.to(self.device), goals=goals)
        value = self.critic_linear(value)

        if self.dist.__class__.__name__ == "MultiCategorical":
            self.dist(actor_features)
            dist = self.dist
        else:
            dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        if self.latent_size is not None:
            for i, (low, high) in enumerate(zip(self.goal_space.low, self.goal_space.high)):
                space_range = (high - low)/2
                space_mid = (high + low)/2
                action[..., i] = torch.tanh(action[..., i]) * space_range + space_mid

            #torch does not support clipping with tensor max/mins currently
            action = torch.FloatTensor(np.clip(action.cpu().numpy(), a_min=self.goal_space.low, a_max=self.goal_space.high)).to(self.device)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action, action_log_probs, (ac_rnn_hxs, v_rnn_hxs)

    def get_value(self, inputs, rnn_hxs, masks, goals=None):
        ac_rnn_hxs, v_rnn_hxs = rnn_hxs

        value, _ = self.critic(inputs, v_rnn_hxs.to(self.device), masks.to(self.device), goals=goals)
        value = self.critic_linear(value)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, action, masks, goals=None):
        ac_rnn_hxs, v_rnn_hxs = rnn_hxs

        actor_features, ac_rnn_hxs = self.actor(inputs, ac_rnn_hxs.to(self.device), masks.to(self.device), goals=goals)
        value, v_rnn_hxs = self.critic(inputs, v_rnn_hxs.to(self.device), masks.to(self.device), goals=goals)
        value = self.critic_linear(value)


        if self.dist.__class__.__name__ == "MultiCategorical":
            self.dist(actor_features)
            dist = self.dist
        else:
            dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action.to(self.device))
        dist_entropy = dist.entropy().mean()
        value = value.squeeze(0)
        action_log_probs = action_log_probs.squeeze(0)

        return value, action_log_probs, dist_entropy, (ac_rnn_hxs, v_rnn_hxs)


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class ASPBase(NNBase):
    def __init__(self, obs_space, goal_conditioned=True, device='cpu', hidden_size=256, embed_dim=256,
                 recurrent_hidden_size=256, goals_embed_dim=256, input_fc=(200, 200), output_fc=(200, 100)):
        super(ASPBase, self).__init__(recurrent=True, recurrent_input_size=input_fc[1], hidden_size=recurrent_hidden_size)
        self._recurrent_hidden_state_size = recurrent_hidden_size
        self.device = device
        self.embed_dim = embed_dim
        self.obj_embed_dim = embed_dim * 2
        self.obs_space = obs_space
        self._output_size = output_fc[-1]

        # self.obs_filtered = {
        #     'robot_joint_pos' : ['robot_joint_pos'],
        #     'gripper_pos': ['gripper_pos', 'gripper_qpos'],
        #     'obj_state': [ 'obj_pos', 'obj_rot',  'obj_gripper_contact', 'obj_rel_pos',  'obj_vel_pos', 'obj_vel_rot'],
        #     'goal_state' : ['goal_obj_pos', 'goal_obj_rot', 'rel_goal_obj_pos', 'rel_goal_obj_rot',],
        #     'obj_state_simple' : [ 'obj_pos', 'obj_rot'] #only used for checking goal validity
        # }

        if 'robot_joint_pos' in obs_space.spaces.keys():

            self.num_objects = int((len(obs_space.spaces.keys()) - 3) / 2)
            self.input_dict = {}

            self.input_dict['robot_joint_pos'] = nn.Sequential(
                init_(nn.Linear(obs_space['robot_joint_pos'].shape[0], embed_dim)),
                nn.LayerNorm(embed_dim)
            ).to(device)

            self.input_dict['gripper_pos'] = nn.Sequential(
                init_(nn.Linear(obs_space['gripper_pos'].shape[0], embed_dim)),
                nn.LayerNorm(embed_dim)
            ).to(device)

            for obj in range(self.num_objects):
                input_dim = obs_space['obj_state_' + str(obj)].shape[-1] + obs_space['goal_state_' + str(obj)].shape[-1]

                self.input_dict['obj_' + str(obj)] = nn.Sequential(
                    init_(nn.Linear(input_dim, self.obj_embed_dim)),
                    nn.LayerNorm(self.obj_embed_dim)
                ).to(device)

            #TODO add in image input later

            self.base = nn.Sequential(
                nn.ReLU(),
                init_(nn.Linear(self.obj_embed_dim + self.embed_dim + self.embed_dim, hidden_size)),
                nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)),
                nn.Tanh(),
            ).to(device)
        else:
            self.input_dict = {}

            self.input_dict['full_state'] = nn.Sequential(
                init_(nn.Linear(obs_space['full_state'].shape[0], embed_dim)),
                nn.LayerNorm(embed_dim)
            ).to(device)
            if goal_conditioned:
                self.input_dict['goal_state'] = nn.Sequential(
                    init_(nn.Linear(obs_space['goal_state'].shape[0], embed_dim)),
                    nn.LayerNorm(embed_dim)
                ).to(device)

            embed_size = self.embed_dim * len(self.input_dict.keys())


            self.input_fc = nn.Sequential(
                init_(nn.Linear(embed_size, input_fc[0])),
                nn.ReLU(),
                init_(nn.Linear(input_fc[0], input_fc[1])),
            ).to(device)
            self.output_fc = nn.Sequential(
                init_(nn.Linear(recurrent_hidden_size, output_fc[0])),
                nn.ReLU(),
                init_(nn.Linear(output_fc[0], output_fc[1])),
            ).to(device)


        self.train()

    @property
    def recurrent_hidden_state_size(self):
        if self.is_recurrent:
            return self._recurrent_hidden_state_size
        return 1

    @property
    def output_size(self):
        return self._output_size

    def forward(self, inputs, hncn, masks, goals=None):
        x = inputs
        out = []

        for k, v in self.input_dict.items():
                if 'goal' in k and x[k] is None or torch.isnan(x[k]).any():  #goal state will be nan for ASP when alice has no goal
                    continue
                result = v(x[k].to(self.device))
                out.append(result)

        out = torch.cat(out, dim=-1)
        out = self.input_fc(out)

        out, hncn = self._forward_gru(out, hncn, masks)
        out = self.output_fc(out)

        return out, hncn


class GoalGenerator(nn.Module):
    def __init__(self, latent_dim, goal_space, observation_space, goal_history_shape=None, device='cpu',
                 embed_dim = 256, lstm_size=256,
                 input_fc=(200, 200), output_fc=(200, 100), goals_embed_dim=256):
        super(GoalGenerator, self).__init__()
        self.device = device
        self.goal_space = goal_space
        self.goals_embed_dim = goals_embed_dim
        self.goal_history_shape = goal_history_shape
        self._output_size = output_fc[-1]

        self.is_recurrent = True
        self._recurrent_hidden_state_size = lstm_size

        self.input_dict = {}

        self.input_dict['full_state'] = nn.Sequential(
            init_(nn.Linear(observation_space['full_state'].shape[0], embed_dim)),
            nn.LayerNorm(embed_dim)
        ).to(device)
        self.input_dict['latent'] = nn.Sequential(
            init_(nn.Linear(latent_dim, embed_dim)),
            nn.LayerNorm(embed_dim)
        ).to(device)
        embed_size = 2 * embed_dim

        if goal_history_shape is not None:
            goal_size, num_goals, components = goal_history_shape

            self.input_dict['goal_history'] = nn.Sequential(
                init_(nn.Linear(goal_size * num_goals, goals_embed_dim)),
                nn.LayerNorm(goals_embed_dim)
            ).to(device)
            embed_size += self.goals_embed_dim

        # base
        self.input_fc = nn.Sequential(
            init_(nn.Linear(embed_size, input_fc[0])),
            nn.ReLU(),
            init_(nn.Linear(input_fc[0], input_fc[1])),
        ).to(device)
        self.output_fc = nn.Sequential(
            init_(nn.Linear(lstm_size, output_fc[0])),
            nn.ReLU(),
            init_(nn.Linear(output_fc[0], output_fc[1])),
        ).to(device)

        self.rnn = nn.GRUCell(input_fc[1], lstm_size).to(device)

    @property
    def output_size(self):
        return self._output_size
    @property
    def recurrent_hidden_state_size(self):
        if self.is_recurrent:
            return self._recurrent_hidden_state_size
        return 1

    def forward(self, inputs, hncn, masks, goals=None):
        x = inputs
        out = []

        for k, v in self.input_dict.items():
            if k == 'goal_history' and goals is not None:
                
                goal_shape, num_goals, goal_components = self.goal_history_shape

                if len(goals.shape) == 3:
                    goal_buff, proc, full_goal_shape = goals.shape

                else:
                    TN, full_goal_shape = goals.shape
                    proc = TN // num_goals
                    goal_buff = num_goals

                goals = goals.view(goal_buff, proc, goal_shape, goal_components)
                goals = goals.permute(1,3,0,2).reshape(-1, goal_buff * goal_shape, goal_components)
                goals = torch.nn.MaxPool1d(goal_components)(goals).view(proc, goal_buff * goal_shape)

                goal_out =  v(goals).view(proc, -1)

                BS, dim = out[0].shape
                goal_out = torch.cat(BS // proc * [goal_out])

                out.append(goal_out)
                continue
            else:
                result = v(x[k].to(self.device))
                out.append(result)

        out = torch.cat(out, dim=-1) #TODO: check if summing or concat is better

        out_1 = self.input_fc(out)
        if torch.isnan(out_1).any():
            import pdb; pdb.set_trace()
        hidden = self.rnn(out_1, hncn)


        out = self.output_fc(hidden)
        if torch.isnan(out).any():
            import pdb; pdb.set_trace()

        return out, hidden
