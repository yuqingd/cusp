import torch
import os
import re
from utils.utils import *
from envs.SelfPlay import ALICE, BOB
from agents.ppo import PPO
import hydra
from agents.replay_buffer import ReplayBuffer
from agents.sac import SACAgent
from agents.gan import StateGAN
from gym.spaces import Box
from torch.autograd import Variable

class GoalGeneratorTrainer:
    def __init__(self, cfg, L, envs, device, model_save_path, reload_model=False):
        self.cfg = cfg
        self.L = L
        self.env = envs
        self.device = device
        self.goal_algo = cfg.goal_algo
        self.step = -1
        
        # Goal specification
        self.single_goal_shape = envs.env_method('goal_shape')[0][0]
        if self.cfg.num_subgoals > 1:
            self.single_goal_shape = envs.env_method('goal_shape')[0][0] * envs.env_method('goal_shape')[0][1]

        self.goal_shape = self.single_goal_shape * cfg.num_goals         
        self.goal_components = 1

        goal_action_space = envs.env_method('goal_space')[0]  # specify bounds for goal

        # Specify goal generator action space (ie. goal space)
        goal_obs_space = {}
        goal_obs_space['full_state'] = envs.observation_space['full_state']
        goal_obs_space['latent'] = Box(-np.inf, np.inf, (cfg.latent_dim, ))
        goal_obs_space = Dict(goal_obs_space)
        goal_history_shape = (self.goal_shape, cfg.goal_buffer_size, self.goal_components)

        if reload_model:
            for checkpoint in os.listdir(model_save_path):
                if checkpoint is None or 'ep_' not in checkpoint:
                    continue 
                cur_episode_step = int(re.search(r'\d+', checkpoint).group())
                if self.step < cur_episode_step:
                    self.step = cur_episode_step

        if self.goal_algo == 'cusp':
            self.goal_history_shape = goal_history_shape   

            cfg.goal_generator.params.hidden_size = 4 # SAC agent does worse with larger input dims
            cfg.agent.params.action_dim = goal_action_space.shape[0] 
            cfg.agent.params.action_range = [
                float(goal_action_space.low.min()),
                float(goal_action_space.high.max())
            ]    

            if cfg.symmetrize:
                self.goal_agent = [
                    SACAgent(
                        goal_space=goal_action_space,
                        goal_observation_space=goal_obs_space,
                        goal_generator_cfg=cfg.goal_generator.params,
                        goal_temperature=cfg.goal_temperature,
                        **cfg.agent.params),
                    SACAgent(
                        goal_space=goal_action_space,
                        goal_observation_space=goal_obs_space,
                        goal_generator_cfg=cfg.goal_generator.params,
                        goal_temperature=cfg.goal_temperature,
                        **cfg.agent.params)
                ]
                self.cfg.num_goals = 2 # generate two goals per round
                
            else:
                self.goal_agent = SACAgent(
                    goal_space=goal_action_space,
                    goal_observation_space=goal_obs_space,
                    goal_generator_cfg=cfg.goal_generator.params,
                    goal_temperature=cfg.goal_temperature,
                    **cfg.agent.params)

            try:
                if self.step >= 0:
                    if cfg.symmetrize:
                        self.goal_agent[0].load(model_save_path + '/ep_' + str(self.step) + '/goal_gen_alice', self.step)
                        self.goal_agent[1].load(model_save_path + '/ep_' + str(self.step) + '/goal_gen_bob', self.step)
                    
                    else:
                        self.goal_agent.load(model_save_path + '/ep_' + str(self.step) + '/goal_gen', self.step)
                    self.step += 1
            except:
                print("No Goal Generator found for reloading.")

            self.sac_goal_buffer = ReplayBuffer(goal_obs_space,
                                        goal_action_space.shape,
                                        int(cfg.goal_replay_buffer_capacity),
                                        self.device,
                                        reward_shape=2) #store alice, bob utilities separately so reward shape = 2
            
            if not os.path.exists(model_save_path + '/sac_goal_buffer'):
                os.mkdir(model_save_path + '/sac_goal_buffer')

            if self.step >= 1:
                self.sac_goal_buffer.load(model_save_path + '/sac_goal_buffer')

        elif self.goal_algo == 'goalgan':
            #goal gan uses replay buffer of goals
            self.goal_buffer = torch.zeros(cfg.goal_buffer_size, cfg.num_processes, self.single_goal_shape * self.goal_components).to(device)
            self.goal_idx = 0
            if reload_model and os.path.exists(model_save_path+'/goal_hist.pt'.format(self.step)):
                self.goal_buffer = torch.load(model_save_path+'/goal_hist.pt'.format(self.step))

            self.goal_agent = StateGAN(
                evaluater_size=1,
                goal_space=goal_action_space,
                device=device
            )

            self.cur_goals = {} #dict to store goals : success

        if self.step == -1:
            self.step = 0

        
    def generate_goal(self):
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.cfg.num_processes, self.cfg.latent_dim)))).to(self.device)
        init_obs = self.env.reset()
        init_goal_obs = {}
        init_goal_obs["full_state"] = torch.FloatTensor(init_obs["full_state"]).to(self.device)
        init_goal_obs["latent"] = z

        if self.goal_algo == 'goalgan': 
            if self.step > 0 and dict(self.cur_goals): 
                idx = np.random.choice(self.goal_idx, size=100)
                old_goals = self.goal_buffer[idx]
                new_goals, _ = self.goal_agent.sample_states_with_noise(200)
                new_goals = new_goals.reshape(200, self.cfg.num_processes, -1)
                goals = torch.cat([raw_goals, old_goals], dim=0)
            else: 
                new_goals, _ = self.goal_agent.sample_states_with_noise(200)
                new_goals = new_goals.reshape(200, self.cfg.num_processes, -1)
                goals = new_goals
            
            old_buff = self.goal_buffer[:-200].clone().detach()
            self.goal_buffer[200:].copy_(old_buff)
            self.goal_buffer[:200].copy_(new_goals.detach())
            self.goal_idx = min(self.goal_idx + 200, len(self.goal_buffer))

            self.cur_goals = {tuple(goal.detach().clone()): [] for goal in goals}
            self.step += 1
            return goals
        
        elif self.goal_algo == 'cusp':
            self.cur_obs = init_goal_obs
            if self.cfg.symmetrize:
                self.goals = []
                for agent in self.goal_agent:
                    with eval_mode(agent):
                        self.goals.append(agent.act((init_goal_obs), sample=True))
                goals = torch.stack([
                    torch.as_tensor(goal).to(self.device) for goal in self.goals
                ])

            else:
                with eval_mode(self.goal_agent):
                    self.goals = self.goal_agent.act((init_goal_obs), sample=True)
                    goals = torch.as_tensor(self.goals).to(self.device)
        
        self.step += 1
        goals = goals.view(self.cfg.num_goals, self.cfg.num_processes, -1)
        return goals

    def update(self, goals, alice_utilities, bob_utilities):
        alice_weight = max(self.cfg.annealing_start_weight * (self.cfg.annealing_length - self.step) / self.cfg.annealing_length, self.cfg.annealing_end_weight) # anneal to 1 
        regrets = alice_utilities * alice_weight - bob_utilities
        self.L.log('train/mean_regret', regrets.mean(), self.step) 
        self.L.log('train/total_regret', regrets.sum(), self.step) 
        self.L.log('train/alice_utility_weight', alice_weight, self.step) 
        print("{} - Mean Regret: {} ".format(self.step, regrets.mean()))
        print("{} - Total Regret: {} ".format(self.step, regrets.sum()))
        print()

        #Goal update 
        if self.goal_algo == 'cusp':
            for i, goal in enumerate(goals):   
                reward = torch.FloatTensor([alice_utilities[i].sum(), bob_utilities[i].sum()]) #store alice, bob utilities separately

                self.sac_goal_buffer.add(self.cur_obs, goal.cpu(), reward.cpu(), self.cur_obs, 1.0, 0.0)
            for _ in range(self.cfg.num_goal_gen_updates):
                
                if self.cfg.symmetrize:
                    self.goal_agent[0].update(self.sac_goal_buffer, self.L, self.step, 
                            agent_str='goal_gen', alice_weight = alice_weight, agent='alice')
                    self.goal_agent[1].update(self.sac_goal_buffer, self.L, self.step, 
                            agent_str='goal_gen', alice_weight = alice_weight, agent='bob')
                else:
                    self.goal_agent.update(self.sac_goal_buffer, self.L, self.step, 
                            agent_str='goal_gen', alice_weight = alice_weight, agent='alice') #update using current weights

    def check_regrets(self, alice, bob):
        assert self.goal_algo == 'cusp'
        self.sample_regrets(self.sac_goal_buffer, alice, bob)

    def sample_regrets(self, goal_buffer, alice, bob):
        obs, rand_goals, old_regret, next_obs, not_done, not_done_no_max, idxs = goal_buffer.sample(int(self.cfg.num_stale_updates), idxes=True)
        if self.cfg.num_goals > 1:
            rand_goals = rand_goals[:, :self.single_goal_shape]
        obs = torch.cat([obs['full_state'], rand_goals], dim=-1).cpu() #alice bob observations are state & goal
        with torch.no_grad():
            alice_act = alice.act(obs, sample=False)
            alice_actor_Q1, alice_actor_Q2 = alice.critic(obs.to(self.device), torch.FloatTensor(alice_act).to(self.device))
            alice_actor_Q = torch.min(alice_actor_Q1, alice_actor_Q2)

            self.L.log('eval/goal_generator/alice_Q/mean', alice_actor_Q.mean(), self.step)
            self.L.log('eval/goal_generator/alice_Q/var', alice_actor_Q.var(), self.step)

            bob_act = bob.act(obs, sample=False)
            bob_actor_Q1, bob_actor_Q2 = bob.critic(obs.to(self.device), torch.FloatTensor(bob_act).to(self.device))
            bob_actor_Q = torch.min(bob_actor_Q1, bob_actor_Q2)

            self.L.log('eval/goal_generator/bob_Q/mean', bob_actor_Q.mean(), self.step)
            self.L.log('eval/goal_generator/bob_Q/var', bob_actor_Q.var(), self.step)

            actor_Q = torch.cat([alice_actor_Q, bob_actor_Q], dim=-1)

        goal_buffer.update_regrets(idxs, actor_Q.cpu(), self.cfg.stale_regret_coeff) #replace old regrets with Q_value estimates

        self.L.log('eval/goal_generator/old_regret/mean', old_regret.mean(), self.step)
        self.L.log('eval/goal_generator/old_regret/var', old_regret.var(), self.step)

        return old_regret.mean(), actor_Q.mean(), rand_goals

    def save(self, model_save_path, step):
        if self.goal_algo == 'cusp':
            assert os.path.isdir(model_save_path+'/ep_'+str(step))
            
            if self.cfg.symmetrize:
                os.mkdir(model_save_path+'/ep_'+str(step) + '/goal_gen_alice')
                os.mkdir(model_save_path+'/ep_'+str(step) + '/goal_gen_bob')
                self.goal_agent[0].save(model_save_path+'/ep_' + str(step) +'/goal_gen_alice', step)
                self.goal_agent[1].save(model_save_path+'/ep_' + str(step) +'/goal_gen_bob', step)
            else:
                os.mkdir(model_save_path+'/ep_'+str(step) + '/goal_gen')
                self.goal_agent.save(model_save_path+'/ep_' + str(step) +'/goal_gen', step)

            self.sac_goal_buffer.save(model_save_path+'/sac_goal_buffer')
        return 
        

    ## for GoalGAN baseline
    def update_gan(self):
        label_dict = {}
        goals = []
        labels = []

        for k, v in self.cur_goals.items():
            if len(v) == 0: 
                continue
            mean_reward = torch.stack(v).mean()
            goals.append(k[0].clone())
            labels.append(torch.FloatTensor([ float(mean_reward > .1 and mean_reward < .9)  ]))

        goals = torch.stack(goals)
        labels = torch.stack(labels)
        self.goal_agent.train(
            goals, labels, self.L, 250, 
        )
        self.cur_goals = {}


