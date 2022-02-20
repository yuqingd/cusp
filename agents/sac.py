import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agents import Agent
import utils.utils as utils

from agents.actor import DiagGaussianActor
from agents.critic import DoubleQCritic
import hydra
from omegaconf.dictconfig import DictConfig

class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, goal_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, goal_actor_lr, actor_betas, actor_update_frequency, critic_lr, goal_critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, bc_actor_lr, avec_loss, goal_generator_cfg=None,
                 goal_space=None, goal_observation_space=None, goal_temperature=True):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.use_avec_loss = avec_loss

        if not goal_temperature:
            self.learnable_temperature = False
            init_temperature = 0
    
        self.goal_space = goal_space

        if isinstance(goal_generator_cfg, DictConfig):
            self.actor = DiagGaussianActor(goal_space=goal_space,
                    goal_observation_space=goal_observation_space,
                    goal_generator_cfg=goal_generator_cfg,
                    device=self.device,
                    **actor_cfg.params).to(self.device)
            self.critic = DoubleQCritic(goal_space=goal_space,
                    goal_observation_space=goal_observation_space,
                    goal_generator_cfg=goal_generator_cfg,
                    device=self.device,
                    **critic_cfg.params
                ).to(self.device)
            self.critic_target = DoubleQCritic(goal_space=goal_space,
                    goal_observation_space=goal_observation_space,
                    goal_generator_cfg=goal_generator_cfg,
                    device=self.device,
                    **critic_cfg.params
                ).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
        else:
            self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

            self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
            self.critic_target = hydra.utils.instantiate(critic_cfg).to(
                    self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

        if init_temperature > 0:
            self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
            self.log_alpha.requires_grad = True
        else:
            self.log_alpha = None
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=goal_actor_lr if isinstance(goal_generator_cfg, DictConfig) else actor_lr,
                                                betas=actor_betas)

        self.bc_actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=bc_actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=goal_critic_lr if isinstance(goal_generator_cfg, DictConfig) else critic_lr,
                                                 betas=critic_betas)
        if init_temperature > 0:
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)
        else:
            self.log_alpha_optimizer = None

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        if self.log_alpha:
            return self.log_alpha.exp()
        else:
            return torch.zeros(1).to(self.device)

    def act(self, obs, sample=False):
        if isinstance(obs, dict):
            reshape_init_obs = {}
            for k, v in obs.items():
                reshape_init_obs[k] = v.clone().unsqueeze(0)

            dist = self.actor(reshape_init_obs)
        else:
            obs = torch.FloatTensor(obs).to(self.device)
            reshape_obs = obs.clone().unsqueeze(0)
            dist = self.actor(reshape_obs)
        action = dist.sample() if sample else dist.mean
        if self.goal_space is not None and ((self.goal_space.high != float("inf")).all() and (self.goal_space.low != float("-inf")).all()):
            action, _ = self.scale_action(action)

        action = action.clamp(*self.action_range)
        # print(action, action.ndim, action.shape)
        assert action.ndim == 3 and action.shape[0] == 1 #and action.shape[1] == 1

        if torch.isnan(action[0]).any():
            import pdb; pdb.set_trace()

        return utils.to_np(action[0])

    def avec_loss(self, input, target):
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)

        avec_loss = torch.var(expanded_target - expanded_input)
        return avec_loss.mean()

    def update_critic(self, obs, action, reward, next_obs, not_done, logger,
                      step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

        if self.goal_space is not None  and ((self.goal_space.high != float("inf")).all() and (self.goal_space.low != float("-inf")).all()):
            next_action, log_prob = self.scale_action(next_action, log_prob)

        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob #v_backup
        target_Q = reward + (not_done * self.discount * target_V) 
        target_Q = target_Q.detach() # q_backup
        
        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        
        if self.use_avec_loss:
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + self.avec_loss(current_Q2, target_Q) + self.avec_loss(current_Q1, target_Q) #qf1_loss + qf2_loss
        else:
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        logger.log('train_critic/{}/loss'.format(self.agent_str), critic_loss, step)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        try:
            critic_loss.backward()
        except:
            import pdb; pdb.set_trace()
        self.critic_optimizer.step()
        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        if self.goal_space is not None  and ((self.goal_space.high != float("inf")).all() and (self.goal_space.low != float("-inf")).all()):
            action, log_prob = self.scale_action(action, log_prob)

        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/{}/loss'.format(self.agent_str), actor_loss, step)
        logger.log('train_actor/{}/target_entropy'.format(self.agent_str), self.target_entropy, step)
        logger.log('train_actor/{}/entropy'.format(self.agent_str), -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        if torch.isnan(actor_loss) or torch.isneginf(actor_loss.detach()) or torch.isposinf(actor_loss.detach()):
            import pdb; pdb.set_trace()
        actor_loss.backward()
        self.actor_optimizer.step()

        for layer in self.actor.trunk:
            try:
                if torch.isnan(layer.weight).any():
                    import pdb; pdb.set_trace()
            except:
                pass

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/{}/loss'.format(self.agent_str), alpha_loss, step)
            logger.log('train_alpha/{}/value'.format(self.agent_str), self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step, agent_str, alice_weight=None, agent='alice'):
        self.agent_str = agent_str

        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)
        if alice_weight is not None:
            assert reward.shape[-1] == 2
            if agent == 'alice':
                reward = reward[..., 0] * alice_weight - reward[..., 1]
            elif agent == 'bob':
                reward = reward[..., 1] * alice_weight - reward[..., 0]
            else:
                raise NotImplementedError
            assert len(reward.shape) == 1
            reward = reward.unsqueeze(-1)

        logger.log('train/{}/batch_reward'.format(self.agent_str), reward.mean(), step)
        self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                           logger, step)
        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

    def bc_update(self, replay_buffer, logger, step, agent_str):
        obs, action, old_log_probs = replay_buffer.sample(self.batch_size)

        dist = self.actor(obs)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        bc_loss = log_prob - old_log_probs
        bc_loss = -torch.clamp(bc_loss, -.1, .1).mean()

        logger.log('train_actor/{}/bc_loss'.format(agent_str), bc_loss, step)

        # optimize the actor
        self.bc_actor_optimizer.zero_grad()
        try:
            if torch.isnan(bc_loss) or torch.isneginf(bc_loss.detach()) or torch.isposinf(bc_loss.detach()):
                import pdb; pdb.set_trace()
            bc_loss.backward()
        except:
            pass
            #import pdb; pdb.set_trace()
        

        nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
        self.bc_actor_optimizer.step()

        self.actor.log(logger, step)

        for layer in self.actor.trunk:
            try:
                if torch.isnan(layer.weight).any():
                    import pdb; pdb.set_trace()
            except:
                pass

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.actor_optimizer.state_dict(), '%s/actor_optimizer_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic_optimizer.state_dict(), '%s/critic_optimizer_%s.pt' % (model_dir, step)
        )
        if self.log_alpha_optimizer:
            torch.save(
                self.log_alpha_optimizer.state_dict(), '%s/log_alpha_optimizer_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.actor_optimizer.load_state_dict(
            torch.load('%s/actor_optimizer_%s.pt' % (model_dir, step))
        )
        self.critic_optimizer.load_state_dict(
            torch.load('%s/critic_optimizer_%s.pt' % (model_dir, step))
        )
        if self.log_alpha_optimizer:
            self.log_alpha_optimizer.load_state_dict(
                torch.load('%s/log_alpha_optimizer_%s.pt' % (model_dir, step))
            )

    def scale_action(self, action, log_prob=None):
        scaled_action = action.clone()
        if log_prob is not None:
            scaled_log_prob = log_prob.clone()
        else:
            scaled_log_prob = None
        for i, (low, high) in enumerate(zip(self.goal_space.low, self.goal_space.high)):
            space_range = (high - low)/2
            space_mid = (high + low)/2
            scaled_action[..., i] = scaled_action[..., i] * space_range + space_mid

            if log_prob is not None and space_range > 0:
                scaled_log_prob -= np.log(space_range)

        return scaled_action, scaled_log_prob