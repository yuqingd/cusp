import numpy as np
import torch
import torch.nn as nn
from utils.distributions import Bernoulli, Categorical, DiagGaussian, MultiCategorical
from agents.model import NNBase, init_
import torch.optim as optim
from agents.actor import SquashedNormal

class GoalGenerator(NNBase):
    def __init__(self, latent_dim, goal_space, observation_space, device='cpu', hidden_size=128,
                 trunk_hidden_size=1024, goals_embed_dim=256, lr=None, eps=None, goal_algo='adam'):
        super(GoalGenerator, self).__init__(recurrent=False, recurrent_input_size=hidden_size, hidden_size=hidden_size)
        self.device = device
        self.goal_space = goal_space
        self.goals_embed_dim = goals_embed_dim
        self.goal_algo = goal_algo

        self.input_dict = {}

        self.input_dict['full_state'] = nn.Sequential(
                init_(nn.Linear(observation_space['full_state'].shape[0], hidden_size)),
                nn.LayerNorm(hidden_size)
            ).to(device)

        self.input_dict['latent'] = nn.Sequential(
                init_(nn.Linear(latent_dim, hidden_size)),
                nn.LayerNorm(hidden_size)
            ).to(device)
        
        self.embed_size = 2 * hidden_size

        self.input_dict = nn.ModuleDict(self.input_dict)
        #base
        if goal_algo == 'adam':
            self.base = nn.Sequential(
                nn.Tanh(),
                init_(nn.Linear(self.embed_size, trunk_hidden_size)),
                nn.Tanh(),
                init_(nn.Linear(trunk_hidden_size, goal_space.shape[0])),
                nn.Tanh(),
            ).to(device)

            self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=eps)
        
    def input_forward(self, init_obs):
        out = []
        
        for k, v in init_obs.items():
            out.append(self.input_dict[k](v))
                       
        out = torch.cat(out, dim=-1)
        return out

    def forward(self, init_obs):
        out = self.input_forward(init_obs)
        goals = self.base(out)
        scaled_goals = goals.clone()

        for i, (low, high) in enumerate(zip(self.goal_space.low, self.goal_space.high)):
            space_range = (high - low)/2
            space_mid = (high + low)/2
            scaled_goals[..., i] = scaled_goals[..., i] * space_range + space_mid
            
        return scaled_goals 

    def update(self, regret, L, step):
        self.optimizer.zero_grad()
        loss = -regret.mean()
        loss.backward()
        self.optimizer.step()

        L.log('train/goal_generator/loss', loss, step) #log the modified reward for alice
        return 

    def save(self, model_dir, step):
        torch.save(
            self.state_dict(), '%s/goal_gen_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.optimizer.state_dict(), '%s/goal_gen_optimizer_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.load_state_dict(
            torch.load('%s/goal_gen_%s.pt' % (model_dir, step))
        )
        self.optimizer.load_state_dict(
            torch.load('%s/goal_gen_optimizer_%s.pt' % (model_dir, step))
        )

