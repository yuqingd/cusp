import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 diversity_bonus=False,
                 magnitude_bonus=False):
        
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.diversity_bonus = diversity_bonus
        self.magnitude_bonus = magnitude_bonus

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts, bc_rollouts=None, goals=None):
        advantages = rollouts.returns[1:] - rollouts.value_preds[1:]

        if len(advantages.squeeze().shape) > 0:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        bc_loss_epoch=0

        total_goal_diversity = total_goal_magnitude = 0

        bob_masks = None

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            if bc_rollouts is not None:
                data_generator_bc = bc_rollouts.recurrent_generator(None, self.num_mini_batch)

            for sample_num, sample in enumerate(data_generator):
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, goals_batch, T, N = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    actions_batch, masks_batch, goals_batch)

                # if torch.isposinf(action_log_probs.detach()).any() or torch.isneginf(action_log_probs.detach()).any():
                #     continue

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)

                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2)

                action_loss = action_loss.mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                if rollouts.goal_buffer is not None:
                    goal_buff_size, N, _  = rollouts.goal_buffer.shape
                    goal_diversity = goal_magnitude = 0
                    if self.diversity_bonus:
                        goal_diversity = torch.var(rollouts.goal_buffer.view(goal_buff_size * N, -1), dim=0).mean()
                    if self.magnitude_bonus:
                        goal_magnitude = torch.mean(rollouts.goal_buffer.view(goal_buff_size * N, -1), dim=0).mean()

                self.optimizer.zero_grad()

                loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)
                if rollouts.goal_buffer is not None:
                    loss = loss - goal_magnitude - goal_diversity
                    total_goal_diversity += 0.1 * goal_diversity
                    total_goal_magnitude += 0.01 * goal_magnitude

                loss.backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()


            if bc_rollouts is not None:
                for sample_num, sample in enumerate(data_generator_bc):
                    obs_batch_bc, recurrent_hidden_states_batch_bc, actions_batch_bc, \
                        value_preds_batch_bc, return_batch_bc, masks_batch_bc, old_action_log_probs_batch_bc, \
                    adv_targ_bc, goals_batch, T, N = sample

                    values_bc, action_log_probs_bc, dist_entropy_bc, _ = self.actor_critic.evaluate_actions( obs_batch_bc, recurrent_hidden_states_batch_bc, actions_batch_bc, masks_batch_bc,  goals=goals)
                    # if torch.isposinf(action_log_probs_bc.detach()).any() or torch.isneginf(action_log_probs_bc.detach()).any():
                    #     continue
                    
                    bc_loss = action_log_probs_bc - old_action_log_probs_batch_bc
                    # if torch.isposinf(bc_loss.detach()).any() or torch.isneginf(bc_loss.detach()).any():
                    #     continue

                    #add in loss clipping
                    bc_loss = -torch.clamp(bc_loss, -.1, .1).mean()

                    self.optimizer.zero_grad()
                    bc_loss.backward()

                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                             self.max_grad_norm)

                    self.optimizer.step()
                    bc_loss_epoch += bc_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        bc_loss_epoch /= num_updates
        total_goal_diversity /= num_updates
        total_goal_magnitude /= num_updates

        return value_loss_epoch, action_loss_epoch, bc_loss_epoch, dist_entropy_epoch, total_goal_diversity, total_goal_magnitude